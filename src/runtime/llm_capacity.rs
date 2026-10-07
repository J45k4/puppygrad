//! Context planning uses source metadata and the compiler's allocation plan.
//! It never knows attention heads, cache layout or model arithmetic.
use crate::{
    compiler::{
        gpu,
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
    #[arg(skip)]
    pub max_memory: Option<super::memory_limit::MemoryLimit>,
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

#[derive(Clone, Copy, serde::Deserialize, serde::Serialize)]
pub(crate) struct ContextRequest {
    pub capacity: usize,
    pub prompt_tokens: usize,
    /// The full prompt and requested generation must fit; capacity is a preference.
    /// Omitted by existing callers that require an exact capacity.
    #[serde(default)]
    pub minimum_capacity: Option<usize>,
}

pub(crate) struct PreparedRequest {
    pub plan: gpu::MemoryPlan,
    pub shapes: Vec<(usize, gpu::Lowered)>,
    pub states: Vec<source::StateSpec>,
}

pub(crate) struct PreparedContext {
    pub capacity: usize,
    pub prefill_chunk: Option<usize>,
    pub prepared: PreparedRequest,
}

/// Prefer the usual context bucket and chunk, then reduce working memory.
/// The caller supplies either an initial budget or a live retained-allocation check.
pub(crate) fn prepare_adaptive_request(
    backend: gpu::Backend,
    text: &str,
    context: &Context,
    request: ContextRequest,
    prefill_chunk: Option<usize>,
    is_retained: bool,
    adaptive_chunk: bool,
    check: impl Fn(&gpu::MemoryPlan) -> Result<()>,
) -> Result<PreparedContext> {
    prepare_adaptive(
        backend,
        text,
        context,
        request,
        prefill_chunk,
        is_retained,
        adaptive_chunk,
        false,
        check,
    )
}

pub(crate) fn prepare_adaptive_reusable_request(
    backend: gpu::Backend,
    text: &str,
    context: &Context,
    request: ContextRequest,
    prefill_chunk: Option<usize>,
    is_retained: bool,
    adaptive_chunk: bool,
    check: impl Fn(&gpu::MemoryPlan) -> Result<()>,
) -> Result<PreparedContext> {
    prepare_adaptive(
        backend,
        text,
        context,
        request,
        prefill_chunk,
        is_retained,
        adaptive_chunk,
        true,
        check,
    )
}

fn prepare_adaptive(
    backend: gpu::Backend,
    text: &str,
    context: &Context,
    request: ContextRequest,
    prefill_chunk: Option<usize>,
    is_retained: bool,
    adaptive_chunk: bool,
    reusable: bool,
    check: impl Fn(&gpu::MemoryPlan) -> Result<()>,
) -> Result<PreparedContext> {
    let capacity = request.capacity.min(position_limit(context)?);
    let minimum = request.minimum_capacity.unwrap_or(capacity);
    if minimum == 0
        || minimum > capacity
        || request.prompt_tokens == 0
        || request.prompt_tokens > minimum
    {
        return Err(Error(
            "request length exceeds the model's context limit".into(),
        ));
    }
    let chunk = is_retained.then_some(prefill_chunk.unwrap_or(DEFAULT_PREFILL_CHUNK).min(capacity));
    if chunk == Some(0) {
        return Err(Error("prefill chunk must be positive".into()));
    }
    let prepare = |capacity, chunk| {
        let prepare = if reusable && is_retained {
            prepare_reusable_request
        } else {
            prepare_request
        };
        prepare(
            backend,
            text,
            context,
            capacity,
            if is_retained {
                request.prompt_tokens
            } else {
                capacity
            },
            chunk,
            is_retained,
        )
    };
    let preferred = prepare(capacity, chunk)?;
    if check(&preferred.plan).is_ok() {
        return Ok(PreparedContext {
            capacity,
            prefill_chunk: chunk,
            prepared: preferred,
        });
    }
    // Check the smallest permissible plan first. A model that cannot fit its
    // weights should fail promptly rather than lowering every candidate shape.
    let smallest_chunk = if adaptive_chunk && is_retained {
        Some(1)
    } else {
        chunk
    };
    let smallest = if minimum == capacity && smallest_chunk == chunk {
        preferred
    } else {
        prepare(minimum, smallest_chunk)?
    };
    check(&smallest.plan)?;
    for (index, candidate_capacity) in [capacity, minimum].into_iter().enumerate() {
        if index == 1 && minimum == capacity {
            // The first pass already considered this capacity.
            break;
        }
        let mut candidate_chunk = chunk;
        loop {
            let is_preferred = candidate_capacity == capacity && candidate_chunk == chunk;
            let is_smallest = candidate_capacity == minimum && candidate_chunk == smallest_chunk;
            if !is_preferred && !is_smallest {
                let prepared = prepare(candidate_capacity, candidate_chunk)?;
                if check(&prepared.plan).is_ok() {
                    return Ok(PreparedContext {
                        capacity: candidate_capacity,
                        prefill_chunk: candidate_chunk,
                        prepared,
                    });
                }
            }
            if !adaptive_chunk || !is_retained || candidate_chunk == Some(1) {
                break;
            }
            // Chunks larger than this prompt have the same allocation plan.
            candidate_chunk = candidate_chunk.map(|n| {
                if reusable {
                    if n > 8 {
                        8
                    } else {
                        1
                    }
                } else {
                    (n.min(request.prompt_tokens) / 2).max(1)
                }
            });
        }
    }
    Ok(PreparedContext {
        capacity: minimum,
        prefill_chunk: smallest_chunk,
        prepared: smallest,
    })
}

pub(crate) fn shape_plan(
    backend: gpu::Backend,
    text: &str,
    context: &Context,
    capacity: usize,
    tokens: usize,
) -> Result<(gpu::Lowered, Vec<source::StateSpec>)> {
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
    Ok((
        crate::compiler::cpu::cuda_lower::emit_backend(&p.graph, p.root, backend)?,
        p.states,
    ))
}

/// Union the actual prefill shapes and decode into one retained allocation plan.
pub fn request_plan(
    text: &str,
    context: &Context,
    capacity: usize,
    prompt_tokens: usize,
    prefill_chunk: Option<usize>,
    is_retained: bool,
) -> Result<gpu::MemoryPlan> {
    request_plan_for_backend(
        gpu::Backend::Cuda,
        text,
        context,
        capacity,
        prompt_tokens,
        prefill_chunk,
        is_retained,
    )
}
pub fn request_plan_for_backend(
    backend: gpu::Backend,
    text: &str,
    context: &Context,
    capacity: usize,
    prompt_tokens: usize,
    prefill_chunk: Option<usize>,
    is_retained: bool,
) -> Result<gpu::MemoryPlan> {
    Ok(prepare_request(
        backend,
        text,
        context,
        capacity,
        prompt_tokens,
        prefill_chunk,
        is_retained,
    )?
    .plan)
}

/// Keep the kernels from allocation planning so execution does not lower them again.
pub(crate) fn prepare_request(
    backend: gpu::Backend,
    text: &str,
    context: &Context,
    capacity: usize,
    prompt_tokens: usize,
    prefill_chunk: Option<usize>,
    is_retained: bool,
) -> Result<PreparedRequest> {
    if prompt_tokens == 0 || capacity == 0 || prefill_chunk == Some(0) {
        return Err(Error("capacity, prompt and chunk must be positive".into()));
    }
    if prompt_tokens > capacity {
        return Err(Error("prompt exceeds planned context capacity".into()));
    }
    let chunk = prefill_chunk.unwrap_or(prompt_tokens).min(prompt_tokens);
    let (lowered, states) = shape_plan(backend, text, context, capacity, chunk)?;
    let mut plan = gpu::MemoryPlan::from_lowered(&lowered)?;
    let mut shapes = vec![(chunk, lowered)];
    let mut merge_shape = |tokens| -> Result<()> {
        let (next, next_states) = shape_plan(backend, text, context, capacity, tokens)?;
        if next_states != states {
            return Err(Error(
                "retained state layout changes between execution shapes".into(),
            ));
        }
        plan.merge(&gpu::MemoryPlan::from_lowered(&next)?)?;
        shapes.push((tokens, next));
        Ok(())
    };
    let tail = prompt_tokens % chunk;
    if tail > 0 {
        merge_shape(tail)?;
    }
    if is_retained && chunk != 1 && tail != 1 {
        merge_shape(1)?;
    }
    Ok(PreparedRequest {
        plan,
        shapes,
        states,
    })
}

/// Fixed programs consume only real tokens. The final short tail uses decode,
/// so no padding can advance source-owned positions or enter retained state.
pub(crate) fn reusable_prefill_chunk(remaining: usize, limit: usize) -> usize {
    if remaining >= DEFAULT_PREFILL_CHUNK && limit >= DEFAULT_PREFILL_CHUNK {
        DEFAULT_PREFILL_CHUNK
    } else if remaining >= 8 && limit >= 8 {
        8
    } else {
        1
    }
}

pub(crate) fn reusable_prefill_shapes(tokens: usize, limit: usize) -> Vec<usize> {
    let mut remaining = tokens;
    let mut shapes = vec![];
    while remaining > 0 {
        let chunk = reusable_prefill_chunk(remaining, limit);
        shapes.push(chunk);
        remaining %= chunk;
    }
    // The same retained state must also support subsequent decode.
    if !shapes.contains(&1) {
        shapes.push(1);
    }
    shapes
}

fn prepare_reusable_request(
    backend: gpu::Backend,
    text: &str,
    context: &Context,
    capacity: usize,
    prompt_tokens: usize,
    prefill_chunk: Option<usize>,
    is_retained: bool,
) -> Result<PreparedRequest> {
    if !is_retained || prompt_tokens == 0 || prompt_tokens > capacity || prefill_chunk == Some(0) {
        return Err(Error("invalid reusable retained prefill request".into()));
    }
    let sizes = reusable_prefill_shapes(
        prompt_tokens,
        prefill_chunk.unwrap_or(DEFAULT_PREFILL_CHUNK).min(capacity),
    );
    let (first, states) = shape_plan(backend, text, context, capacity, sizes[0])?;
    let mut plan = gpu::MemoryPlan::from_lowered(&first)?;
    let mut shapes = vec![(sizes[0], first)];
    for &tokens in &sizes[1..] {
        let (next, next_states) = shape_plan(backend, text, context, capacity, tokens)?;
        if next_states != states {
            return Err(Error(
                "retained state layout changes between execution shapes".into(),
            ));
        }
        plan.merge(&gpu::MemoryPlan::from_lowered(&next)?)?;
        shapes.push((tokens, next));
    }
    Ok(PreparedRequest {
        plan,
        shapes,
        states,
    })
}

pub fn determine(
    text: &str,
    context: &Context,
    available_bytes: usize,
    reserve_bytes: usize,
    explicit_chunk: Option<usize>,
    single_shot: bool,
) -> Result<Report> {
    determine_for_backend(
        gpu::Backend::Cuda,
        text,
        context,
        available_bytes,
        reserve_bytes,
        explicit_chunk,
        single_shot,
    )
}
pub fn determine_for_backend(
    backend: gpu::Backend,
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
    let evaluate =
        |n: usize| request_plan_for_backend(backend, text, context, n, n, chunk, is_retained);
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
    let (backend, device) = gpu::device(&options.device)?;
    if options.source.extension().is_none_or(|e| e != "pup") {
        return Err("capacity planning currently requires a .pup source program".into());
    }
    let context = Checkpoint::metadata_context(&options.model_dir, 1)?;
    let text = std::fs::read_to_string(&options.source)?;
    let mut available = if let Some(mib) = options.memory_budget_mib {
        mib.checked_mul(1024 * 1024)
            .ok_or("memory budget overflow")?
    } else {
        gpu::Runtime::new(backend, device)?
            .memory_info()?
            .free_bytes
    };
    let reserve = options
        .reserve_mib
        .checked_mul(1024 * 1024)
        .ok_or("memory reserve overflow")?;
    if let Some(limit) = options.max_memory {
        available = available.min(limit.0.saturating_add(reserve));
    }
    let report = determine_for_backend(
        backend,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::pop::DType;

    const PROGRAM: &str = "x = input(\"tokens\")\ns = state(\"history\", i32, [config(\"buffer_capacity\")])\nw = store(index(s, arange(dim(x, 0))), x)\noutput after(s, w)";

    fn context() -> Context {
        let mut c = Context::default();
        c.constants.insert("n_positions".into(), Scalar::Int(1024));
        c.tensors.insert(
            "tokens".into(),
            TensorSpec {
                slot: 0,
                dtype: DType::I32,
                shape: vec![1],
            },
        );
        c
    }

    #[test]
    fn reusable_prefill_respects_adaptive_capacity_and_buffer_budget() {
        let c = context();
        let budget =
            prepare_reusable_request(gpu::Backend::Cuda, PROGRAM, &c, 32, 17, Some(1), true)
                .unwrap()
                .plan
                .total_bytes();
        let selected = prepare_adaptive_reusable_request(
            gpu::Backend::Cuda,
            PROGRAM,
            &c,
            ContextRequest {
                capacity: 512,
                prompt_tokens: 17,
                minimum_capacity: Some(32),
            },
            None,
            true,
            true,
            |plan| {
                if plan.total_bytes() <= budget {
                    Ok(())
                } else {
                    Err(Error("budget exceeded".into()))
                }
            },
        )
        .unwrap();
        assert_eq!(selected.capacity, 32);
        assert_eq!(selected.prefill_chunk, Some(1));
        assert_eq!(
            selected
                .prepared
                .shapes
                .iter()
                .map(|(n, _)| *n)
                .collect::<Vec<_>>(),
            [1]
        );
    }

    #[test]
    fn reusable_chunks_cover_only_real_tokens_and_do_not_exceed_limits() {
        for limit in [1, 2, 7, 8, 32, 128] {
            for tokens in [1, 7, 8, 13, 17, 127, 128, 129, 137, 512] {
                let mut left = tokens;
                while left > 0 {
                    let n = reusable_prefill_chunk(left, limit);
                    assert!([1, 8, 128].contains(&n));
                    assert!(n <= left && n <= limit);
                    left -= n;
                }
            }
        }
        assert_eq!(reusable_prefill_shapes(17, 128), [8, 1]);
        assert_eq!(reusable_prefill_shapes(137, 128), [128, 8, 1]);
        assert_eq!(reusable_prefill_shapes(16, 128), [8, 1]);
    }

    #[test]
    fn adaptive_request_responds_to_budget_without_shortening_generation() {
        let c = context();
        let request = ContextRequest {
            capacity: 512,
            prompt_tokens: 8,
            minimum_capacity: Some(32),
        };
        let preferred = request_plan(PROGRAM, &c, 512, 8, Some(8), true).unwrap();
        let minimum = request_plan(PROGRAM, &c, 32, 8, Some(8), true).unwrap();
        let budget = std::cell::Cell::new(preferred.total_bytes());
        let select = || {
            prepare_adaptive_request(
                gpu::Backend::Cuda,
                PROGRAM,
                &c,
                request,
                Some(8),
                true,
                false,
                |plan| {
                    if plan.total_bytes() <= budget.get() {
                        Ok(())
                    } else {
                        Err(Error("memory budget exceeded".into()))
                    }
                },
            )
        };
        assert_eq!(select().unwrap().capacity, 512);
        budget.set(minimum.total_bytes());
        let limited = select().unwrap();
        assert_eq!(limited.capacity, 32);
        assert_eq!(limited.prefill_chunk, Some(8));
        assert!(limited.prepared.plan.total_bytes() <= budget.get());
        budget.set(minimum.total_bytes() - 1);
        assert!(select().is_err());
        budget.set(preferred.total_bytes());
        assert_eq!(select().unwrap().capacity, 512);
    }

    #[test]
    fn adaptive_prefill_reduces_chunks_and_preserves_explicit_chunk_setting() {
        let c = context();
        let request = ContextRequest {
            capacity: 32,
            prompt_tokens: 8,
            minimum_capacity: Some(32),
        };
        let budget = request_plan(PROGRAM, &c, 32, 8, Some(2), true)
            .unwrap()
            .total_bytes();
        assert!(
            request_plan(PROGRAM, &c, 32, 8, Some(8), true)
                .unwrap()
                .total_bytes()
                > budget
        );
        let select = |adaptive| {
            prepare_adaptive_request(
                gpu::Backend::Cuda,
                PROGRAM,
                &c,
                request,
                Some(8),
                true,
                adaptive,
                |plan| {
                    if plan.total_bytes() <= budget {
                        Ok(())
                    } else {
                        Err(Error("memory budget exceeded".into()))
                    }
                },
            )
        };
        let fitted = select(true).unwrap();
        assert_eq!(fitted.capacity, 32);
        assert!(fitted.prefill_chunk.unwrap() <= 2);
        assert!(select(false).is_err());
    }
}
