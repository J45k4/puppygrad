//! Context planning uses source metadata and the compiler's allocation plan.
//! It never knows attention heads, cache layout or model arithmetic.
use crate::{
    compiler::{
        cuda,
        pop::{Error, Result, Scalar},
        source::{self, Context, TensorSpec},
    },
    models::pup_llm::Checkpoint,
};
use std::path::PathBuf;

pub const DEFAULT_RESERVE_MIB: usize = 512;
pub const DEFAULT_PREFILL_CHUNK: usize = 128;

#[derive(clap::Args, Debug)]
pub struct Options {
    pub source: PathBuf,
    #[arg(long, default_value = "models/gpt2")]
    pub model_dir: PathBuf,
    #[arg(long, default_value = "cuda:0")]
    pub device: String,
    /// Fixed prefill chunk size for retained programs; defaults to 128.
    #[arg(long, conflicts_with = "single_shot")]
    pub prefill_chunk: Option<usize>,
    /// Plan whole-prompt prefill instead of retained chunk execution.
    #[arg(long)]
    pub single_shot: bool,
    /// Override available device memory for offline planning without a GPU.
    #[arg(long)]
    pub memory_budget_mib: Option<usize>,
    /// Space for driver modules, graphs, transient growth and other GPU work.
    #[arg(long, default_value_t = DEFAULT_RESERVE_MIB)]
    pub reserve_mib: usize,
    #[arg(long)]
    pub json: bool,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Report {
    pub model_position_limit: usize,
    pub max_context_tokens: usize,
    pub prefill_chunk_tokens: Option<usize>,
    pub retained: bool,
    pub available_bytes: usize,
    pub reserve_bytes: usize,
    pub budget_bytes: usize,
    pub input_bytes: usize,
    pub state_bytes: usize,
    pub workspace_bytes: usize,
    pub output_bytes: usize,
    pub planned_device_bytes: usize,
    pub estimate: bool,
    pub limiting_reason: String,
}

pub fn position_limit(context: &Context) -> Result<usize> {
    match context.constants.get("n_positions") {
        Some(Scalar::Int(n)) if *n > 0 => usize::try_from(*n).map_err(|e| Error(e.to_string())),
        _ => Err(Error("missing positive model position limit".into())),
    }
}

pub fn retained(text: &str, context: &Context) -> Result<bool> {
    Ok(!source::parse_with_context(text, context)?.states.is_empty())
}

fn shape_plan(
    text: &str,
    context: &Context,
    capacity: usize,
    tokens: usize,
) -> Result<(cuda::MemoryPlan, Vec<source::StateSpec>)> {
    let mut context = context.clone();
    context
        .constants
        .insert("buffer_capacity".into(), Scalar::Int(capacity as i64));
    let input = context
        .tensors
        .get("tokens")
        .ok_or_else(|| Error("missing tokens input metadata".into()))?;
    context.tensors.insert(
        "tokens".into(),
        TensorSpec {
            slot: input.slot,
            dtype: input.dtype,
            shape: vec![tokens],
        },
    );
    let p = source::parse_with_context(text, &context)?;
    Ok((cuda::memory_plan(&p.graph, p.root)?, p.states))
}

/// Union the actual prefill shapes and decode into one retained allocation plan.
pub fn request_plan(
    text: &str,
    context: &Context,
    capacity: usize,
    prompt_tokens: usize,
    prefill_chunk: Option<usize>,
    is_retained: bool,
) -> Result<cuda::MemoryPlan> {
    if prompt_tokens == 0 || capacity == 0 || prefill_chunk == Some(0) {
        return Err(Error("capacity, prompt and chunk must be positive".into()));
    }
    if prompt_tokens > capacity {
        return Err(Error("prompt exceeds planned context capacity".into()));
    }
    let chunk = prefill_chunk.unwrap_or(prompt_tokens).min(prompt_tokens);
    let (mut plan, states) = shape_plan(text, context, capacity, chunk)?;
    let mut merge_shape = |tokens| -> Result<()> {
        let (next, next_states) = shape_plan(text, context, capacity, tokens)?;
        if next_states != states {
            return Err(Error(
                "retained state layout changes between execution shapes".into(),
            ));
        }
        plan.merge(&next)
    };
    let tail = prompt_tokens % chunk;
    if tail > 0 {
        merge_shape(tail)?;
    }
    if is_retained && chunk != 1 && tail != 1 {
        merge_shape(1)?;
    }
    Ok(plan)
}

pub fn determine(
    text: &str,
    context: &Context,
    available_bytes: usize,
    reserve_bytes: usize,
    explicit_chunk: Option<usize>,
    single_shot: bool,
) -> Result<Report> {
    if explicit_chunk == Some(0) {
        return Err(Error("prefill chunk must be positive".into()));
    }
    let limit = position_limit(context)?;
    let is_retained = retained(text, context)?;
    if explicit_chunk.is_some() && !is_retained {
        return Err(Error(
            "chunked prefill requires a program with retained state".into(),
        ));
    }
    let chunk = if single_shot || !is_retained {
        None
    } else {
        Some(explicit_chunk.unwrap_or(DEFAULT_PREFILL_CHUNK).min(limit))
    };
    let budget = available_bytes
        .checked_sub(reserve_bytes)
        .ok_or_else(|| Error("memory reserve exceeds available budget".into()))?;
    let evaluate = |n: usize| request_plan(text, context, n, n, chunk, is_retained);
    let budget_reason =
        |bytes| format!("planned device buffers need {bytes} bytes, budget is {budget}");
    let mut best =
        evaluate(1).map_err(|e| Error(format!("even minimum context cannot fit: {e}")))?;
    if best.total_bytes() > budget {
        return Err(Error(format!(
            "even minimum context cannot fit: {}",
            budget_reason(best.total_bytes())
        )));
    }
    let (mut low, mut high) = (1, limit);
    let mut above: Option<(usize, usize)> = None;
    let mut reason = "model position limit".to_owned();
    if limit > 1 {
        match evaluate(limit) {
            Ok(plan) if plan.total_bytes() <= budget => {
                low = limit;
                best = plan;
            }
            Ok(plan) => {
                above = Some((limit, plan.total_bytes()));
                reason = budget_reason(plan.total_bytes());
                high = limit - 1;
            }
            Err(e) => {
                high = limit - 1;
                reason = e.to_string();
            }
        }
    }
    while low < high {
        // Retained allocation plans are often affine in capacity. Interpolate
        // between measured costs to avoid many expensive source lowerings;
        // keep bisection when compiler guards prevent a cost measurement.
        let mid = if let Some((n, bytes)) = above.filter(|(_, bytes)| *bytes > best.total_bytes()) {
            let offset = (budget - best.total_bytes()) as u128 * (n - low) as u128
                / (bytes - best.total_bytes()) as u128;
            low + (offset as usize).clamp(1, high - low)
        } else {
            low + (high - low).div_ceil(2)
        };
        match evaluate(mid) {
            Ok(plan) if plan.total_bytes() <= budget => {
                low = mid;
                best = plan;
            }
            Ok(plan) => {
                above = Some((mid, plan.total_bytes()));
                high = mid - 1;
                reason = budget_reason(plan.total_bytes());
            }
            Err(e) => {
                high = mid - 1;
                above = None;
                reason = e.to_string();
            }
        }
    }
    if low < limit {
        reason = match evaluate(low + 1) {
            Ok(plan) => budget_reason(plan.total_bytes()),
            Err(e) => e.to_string(),
        };
    }
    Ok(Report {
        model_position_limit: limit,
        max_context_tokens: low,
        prefill_chunk_tokens: chunk,
        retained: is_retained,
        available_bytes,
        reserve_bytes,
        budget_bytes: budget,
        input_bytes: best.input_bytes(),
        state_bytes: best.state_bytes(),
        workspace_bytes: best.workspace_bytes,
        output_bytes: best.output_bytes(),
        planned_device_bytes: best.total_bytes(),
        estimate: true,
        limiting_reason: reason,
    })
}

pub fn run(options: Options) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let device = cuda::device_index(&options.device)?;
    if options.source.extension().is_none_or(|e| e != "pup") {
        return Err("capacity planning currently requires a .pup source program".into());
    }
    let context = Checkpoint::metadata_context(&options.model_dir, 1)?;
    let text = std::fs::read_to_string(&options.source)?;
    let available = if let Some(mib) = options.memory_budget_mib {
        mib.checked_mul(1024 * 1024)
            .ok_or("memory budget overflow")?
    } else {
        cuda::Runtime::new(device)?.memory_info()?.free_bytes
    };
    let reserve = options
        .reserve_mib
        .checked_mul(1024 * 1024)
        .ok_or("memory reserve overflow")?;
    let report = determine(
        &text,
        &context,
        available,
        reserve,
        options.prefill_chunk,
        options.single_shot,
    )?;
    if options.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "Estimated maximum context: {} tokens (model position limit {})",
            report.max_context_tokens, report.model_position_limit
        );
        println!(
            "Prefill: {}",
            report
                .prefill_chunk_tokens
                .map_or_else(|| "whole prompt".into(), |n| format!("{n}-token chunks"))
        );
        println!(
            "Device buffers: {:.2} GiB; state {:.2} GiB; scratch {:.2} GiB; reserved {:.0} MiB",
            report.planned_device_bytes as f64 / 2f64.powi(30),
            report.state_bytes as f64 / 2f64.powi(30),
            report.workspace_bytes as f64 / 2f64.powi(30),
            report.reserve_bytes as f64 / 2f64.powi(20)
        );
        println!("Limit: {}", report.limiting_reason);
    }
    Ok(())
}
