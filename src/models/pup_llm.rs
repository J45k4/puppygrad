//! Transitional .pup provider for the LLM ABI. Model math executes through the
//! compiler's CPU, CUDA or HIP backend; checkpoint binding and generation stay outside compiler.
mod checkpoint;
pub use checkpoint::Checkpoint;

use super::generation::{LogitsSampler, TextGenerationConfig};
use crate::runtime::llm_capacity;
use crate::{
    compiler::{cpu, gpu, pop::Scalar, source},
    runtime::llm_ffi::{
        self as ffi, Api, Callbacks, ErrorBuffer, Generation, Info, DONE_CONTEXT, DONE_EOS,
        DONE_ERROR, DONE_LIMIT, DONE_MEMORY, NO_EOS,
    },
};
use std::{
    collections::HashMap,
    ffi::c_void,
    panic::{catch_unwind, AssertUnwindSafe},
    path::PathBuf,
    time::Instant,
};
#[derive(serde::Deserialize)]
struct Config {
    source: PathBuf,
    model_dir: PathBuf,
    device: String,
    #[serde(default)]
    verify_reference: bool,
    #[serde(default)]
    threads: Option<usize>,
    #[serde(default)]
    cpu_target: cpu::CpuTarget,
    #[serde(default = "default_cache_dir")]
    cache_dir: PathBuf,
    /// GPU providers plan their context from live device memory by default.
    #[serde(default)]
    auto_context: Option<bool>,
    #[serde(default)]
    prefill_chunk: Option<usize>,
    #[serde(default)]
    context_budget_mib: Option<usize>,
    #[serde(default = "default_context_reserve")]
    context_reserve_mib: usize,
    /// Interactive callers can plan one request instead of probing maximum capacity.
    #[serde(default)]
    context_request: Option<llm_capacity::ContextRequest>,
    #[serde(default)]
    max_memory: Option<usize>,
    #[serde(default)]
    grow_context: bool,
}
fn default_context_reserve() -> usize {
    llm_capacity::DEFAULT_RESERVE_MIB
}
fn default_cache_dir() -> PathBuf {
    PathBuf::from(".cache/pup")
}
struct State {
    source: String,
    source_path: PathBuf,
    cache_dir: PathBuf,
    checkpoint: Checkpoint,
    threads: usize,
    build_options: cpu::BuildOptions,
    info: Info,
    context_capacity: usize,
    grow_context: bool,
    gpu_runtime: Option<gpu::Runtime>,
    cpu_runtime: cpu::Runtime,
    retained: Option<bool>,
    prefill_chunk: Option<usize>,
    auto_context: bool,
    adaptive_prefill: bool,
    reusable_prefill: bool,
    shape_plans: HashMap<(usize, usize), (gpu::MemoryPlan, Vec<source::StateSpec>)>,
    request_plans: HashMap<(usize, usize, usize), gpu::MemoryPlan>,
    prepared_shapes: HashMap<(usize, usize), gpu::Lowered>,
    executables: HashMap<(usize, usize), Executable>,
    output: Vec<u32>,
    reference: Option<super::gpt2::Gpt2Runtime>,
}
enum Executable {
    Cpu(cpu::Executable),
    Gpu(gpu::Executable),
}
impl Executable {
    fn run(
        &self,
        inputs: &[cpu::Tensor],
        threads: usize,
    ) -> crate::compiler::pop::Result<Vec<cpu::Tensor>> {
        match self {
            Self::Cpu(e) => e.run_with_threads(inputs, threads),
            Self::Gpu(e) => e.run(inputs),
        }
    }
}
pub static API: Api = Api {
    abi_version: ffi::ABI_VERSION,
    struct_size: std::mem::size_of::<Api>() as u32,
    build_model: Some(build_model),
    infer: Some(infer),
    read_output: Some(read_output),
    free_model: Some(free_model),
};
fn boundary<T>(f: impl FnOnce() -> ffi::Result<T>) -> ffi::Result<T> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| Err(".pup LLM provider panicked".into()))
}
fn profile_startup(stage: &str, started: Instant) {
    if std::env::var_os("PUPPYGRAD_STARTUP_PROFILE").is_some() {
        eprintln!("startup {stage}: {:.3}s", started.elapsed().as_secs_f64());
    }
}
unsafe fn error_to(error: *mut ErrorBuffer, message: &str) {
    if let Some(error) = error.as_mut() {
        error.set(message);
    }
}
unsafe extern "C" fn build_model(
    config: *const u8,
    len: usize,
    info: *mut Info,
    error: *mut ErrorBuffer,
) -> *mut c_void {
    let result = boundary(|| {
        if config.is_null() || info.is_null() {
            return Err("null build_model argument".into());
        }
        let config: Config = serde_json::from_slice(std::slice::from_raw_parts(config, len))
            .map_err(|e| e.to_string())?;
        if config.context_request.is_some_and(|r| {
            r.capacity == 0
                || r.prompt_tokens == 0
                || r.prompt_tokens > r.capacity
                || r.minimum_capacity
                    .is_some_and(|n| n < r.prompt_tokens || n > r.capacity)
        }) {
            return Err("invalid requested context capacity or prompt length".into());
        }
        let device = if matches!(config.device.as_str(), "cpu" | "cpu:0" | "c" | "c:0") {
            None
        } else {
            Some(gpu::device(&config.device).map_err(|e| e.to_string())?)
        };
        if config.max_memory == Some(0) {
            return Err("memory limit must be positive".into());
        }
        if config.max_memory.is_some() && device.is_none() {
            return Err(
                "--max-memory currently limits GPU model buffers; select CUDA or HIP".into(),
            );
        }
        if device.is_some() && config.cpu_target != cpu::CpuTarget::Generic {
            return Err("--cpu-target applies to the CPU backend".into());
        }
        let threads = config.threads.unwrap_or_else(cpu::default_threads);
        if threads == 0 {
            return Err("threads must be greater than zero".into());
        }
        config.cpu_target.validate().map_err(|e| e.to_string())?;
        let source = std::fs::read_to_string(&config.source).map_err(|e| e.to_string())?;
        let started = Instant::now();
        let mut checkpoint = Checkpoint::load(&config.model_dir).map_err(|e| e.to_string())?;
        profile_startup("checkpoint", started);
        let integer = |key: &str| match checkpoint.context.constants.get(key) {
            Some(Scalar::Int(n)) if *n > 0 => Ok(*n as u64),
            _ => Err(format!("missing positive {key} in checkpoint config")),
        };
        let vocab_size =
            u32::try_from(integer("vocab_size")?).map_err(|_| "vocabulary exceeds u32")?;
        let context_length = integer("n_positions")?;
        let eos_token = match checkpoint.context.constants.get("eos_token_id") {
            Some(Scalar::Int(n)) => u32::try_from(*n).map_err(|_| "invalid eos_token_id")?,
            _ => NO_EOS,
        };
        if eos_token != NO_EOS && eos_token >= vocab_size {
            return Err("EOS token is outside vocabulary".into());
        }
        let mut metadata = Info {
            vocab_size,
            eos_token,
            context_length,
        };
        if let Some(request) = config.context_request {
            if !config.grow_context {
                metadata.context_length = metadata.context_length.min(request.capacity as u64);
            }
            if request.prompt_tokens as u64 > metadata.context_length
                || request
                    .minimum_capacity
                    .is_some_and(|n| n as u64 > metadata.context_length)
            {
                return Err("requested prompt and generation exceed model position limit".into());
            }
        }
        let mut context_capacity = config.context_request.map_or(context_length as usize, |r| {
            r.capacity.min(context_length as usize)
        });
        let auto_context = config.auto_context.unwrap_or(device.is_some());
        if auto_context && device.is_none() {
            return Err("automatic context planning currently requires CUDA or HIP".into());
        }
        if config.prefill_chunk == Some(0) {
            return Err("prefill chunk must be positive".into());
        }
        let gpu_runtime = device
            .map(|(backend, index)| gpu::Runtime::new(backend, index))
            .transpose()
            .map_err(|e| e.to_string())?;
        if let Some(runtime) = &gpu_runtime {
            runtime
                .set_memory_limit(config.max_memory)
                .map_err(|e| e.to_string())?;
        }
        let mut retained = None;
        let mut prefill_chunk = config.prefill_chunk;
        let mut request_plans = HashMap::new();
        let mut prepared_shapes = HashMap::new();
        let mut shape_plans = HashMap::new();
        let started = Instant::now();
        if auto_context || prefill_chunk.is_some() {
            checkpoint.bind_tokens(&[0]).map_err(|e| e.to_string())?;
            let has_state =
                llm_capacity::retained(&source, &checkpoint.context).map_err(|e| e.to_string())?;
            if prefill_chunk.is_some() && !has_state {
                return Err("chunked prefill requires a program with retained state".into());
            }
            retained = Some(has_state);
            if auto_context {
                let live_free = gpu_runtime
                    .as_ref()
                    .unwrap()
                    .memory_info()
                    .map_err(|e| e.to_string())?
                    .free_bytes;
                let mut available = config
                    .context_budget_mib
                    .map(|n| n.checked_mul(1024 * 1024).ok_or("context budget overflow"))
                    .transpose()?
                    .unwrap_or(live_free)
                    .min(live_free);
                let reserve = config
                    .context_reserve_mib
                    .checked_mul(1024 * 1024)
                    .ok_or("context reserve overflow")?;
                if let Some(limit) = config.max_memory {
                    available = available.min(limit.saturating_add(reserve));
                }
                if let Some(request) = config.context_request {
                    let budget = available.saturating_sub(reserve);
                    let prepare = if config.grow_context && config.prefill_chunk.is_none() {
                        llm_capacity::prepare_adaptive_reusable_request
                    } else {
                        llm_capacity::prepare_adaptive_request
                    };
                    let selected = prepare(
                        gpu_runtime.as_ref().unwrap().backend(),
                        &source,
                        &checkpoint.context,
                        request,
                        prefill_chunk,
                        has_state,
                        config.prefill_chunk.is_none(),
                        |plan: &gpu::MemoryPlan| {
                            if plan.total_bytes() <= budget {
                                return Ok(());
                            }
                            let advice = if plan.input_bytes() > budget {
                                "Model inputs/weights alone exceed this budget; free GPU memory or use a smaller model."
                            } else {
                                "Free GPU memory or reduce the prompt or requested output length."
                            };
                            Err(crate::compiler::pop::Error(format!(
                                "Not enough GPU memory: requested context needs {:.2} GiB of model buffers, but only {:.2} GiB is usable (budget is {budget} bytes; {:.2} GiB free, {} MiB reserved). {advice}",
                                plan.total_bytes() as f64 / 2f64.powi(30), budget as f64 / 2f64.powi(30),
                                live_free as f64 / 2f64.powi(30), config.context_reserve_mib,
                            )))
                        },
                    )
                    .map_err(|e| e.to_string())?;
                    let capacity = selected.capacity;
                    context_capacity = capacity;
                    if !config.grow_context {
                        metadata.context_length = capacity as u64;
                    }
                    prefill_chunk = selected.prefill_chunk;
                    let prepared = selected.prepared;
                    let tokens = if has_state {
                        request.prompt_tokens
                    } else {
                        capacity
                    };
                    let main = prefill_chunk.unwrap_or(tokens).min(tokens);
                    eprintln!("request context: {capacity} tokens (model positions {context_length}), {} prompt tokens; planned buffers {:.2} GiB, reserve {} MiB", request.prompt_tokens, prepared.plan.total_bytes() as f64 / 2f64.powi(30), config.context_reserve_mib);
                    request_plans.insert((capacity, main, tokens % main), prepared.plan);
                    for (tokens, lowered) in prepared.shapes {
                        shape_plans.insert(
                            (tokens, capacity),
                            (
                                gpu::MemoryPlan::from_lowered(&lowered)
                                    .map_err(|e| e.to_string())?,
                                prepared.states.clone(),
                            ),
                        );
                        prepared_shapes.insert((tokens, capacity), lowered);
                    }
                } else {
                    let report = llm_capacity::determine_for_backend(
                        gpu_runtime.as_ref().unwrap().backend(),
                        &source,
                        &checkpoint.context,
                        available,
                        reserve,
                        prefill_chunk,
                        false,
                    )
                    .map_err(|e| e.to_string())?;
                    context_capacity = report.max_context_tokens;
                    if !config.grow_context {
                        metadata.context_length = context_capacity as u64;
                    }
                    prefill_chunk = report.prefill_chunk_tokens;
                    eprintln!("automatic context: {} tokens (model positions {}), prefill {}; planned buffers {:.2} GiB, reserve {} MiB", report.max_context_tokens, report.model_position_limit, prefill_chunk.map_or_else(|| "whole prompt".into(), |n| format!("{n}-token chunks")), report.planned_device_bytes as f64 / 2f64.powi(30), config.context_reserve_mib);
                }
            }
        }
        profile_startup("context planning", started);
        let reference = if config.verify_reference {
            if checkpoint
                .model_type
                .as_deref()
                .is_some_and(|kind| kind != "gpt2")
            {
                return Err("--verify-reference currently supports GPT-2 only".into());
            }
            Some(super::gpt2::Gpt2Runtime::from_dir(&config.model_dir).map_err(|e| e.to_string())?)
        } else {
            None
        };
        let state = Box::new(State {
            source,
            source_path: config.source,
            cache_dir: config.cache_dir,
            checkpoint,
            threads,
            build_options: cpu::BuildOptions {
                cpu_target: config.cpu_target,
            },
            info: metadata,
            context_capacity,
            grow_context: config.grow_context,
            gpu_runtime,
            cpu_runtime: cpu::Runtime::default(),
            retained,
            prefill_chunk,
            auto_context,
            adaptive_prefill: auto_context && config.prefill_chunk.is_none(),
            reusable_prefill: config.grow_context && config.prefill_chunk.is_none(),
            shape_plans,
            request_plans,
            prepared_shapes,
            executables: HashMap::new(),
            output: vec![],
            reference,
        });
        *info = metadata;
        Ok(Box::into_raw(state).cast())
    });
    match result {
        Ok(state) => state,
        Err(e) => {
            error_to(error, &e);
            std::ptr::null_mut()
        }
    }
}
impl State {
    fn execute(
        &mut self,
        chunk: &[usize],
        capacity: usize,
        first_program: Option<source::Program>,
        announce: bool,
    ) -> ffi::Result<(Vec<cpu::Tensor>, std::time::Duration)> {
        self.checkpoint
            .bind_tokens(chunk)
            .map_err(|e| e.to_string())?;
        let key = (chunk.len(), capacity);
        if !self.executables.contains_key(&key) {
            let started = Instant::now();
            let lowered = self.prepared_shapes.remove(&key);
            let program = if lowered.is_some() {
                None
            } else {
                Some(if let Some(program) = first_program {
                    program
                } else {
                    source::parse_with_context(&self.source, &self.checkpoint.context)
                        .map_err(|e| format!("{}:{e}", self.source_path.display()))?
                })
            };
            let exe = if let Some(runtime) = &self.gpu_runtime {
                let cache = self.cache_dir.join(runtime.backend().tag());
                let e = if let Some(lowered) = lowered {
                    gpu::compile_lowered_with_runtime(lowered, &cache, runtime)
                } else {
                    let program = program.as_ref().unwrap();
                    gpu::compile_with_runtime(&program.graph, program.root, &cache, runtime)
                }
                .map_err(|e| e.to_string())?;
                if announce {
                    eprintln!(
                        "compiled {}: {} contractions, {}; source: {}",
                        runtime.backend().label(),
                        e.gemm_count,
                        e.device_name,
                        e.source_path.display()
                    );
                }
                Executable::Gpu(e)
            } else {
                let program = program.unwrap();
                let mut e = cpu::compile_with_options(
                    &program.graph,
                    program.root,
                    &self.cache_dir.join("cpu"),
                    &self.build_options,
                )
                .map_err(|e| e.to_string())?;
                if announce {
                    eprintln!(
                        "compiled {} Pops, {} GEMM contractions, CPU target {}; C source: {}",
                        program
                            .graph
                            .toposort(program.root)
                            .map_err(|e| e.to_string())?
                            .len(),
                        e.gemm_count,
                        e.build_info.cpu_target,
                        e.source_path.display()
                    );
                }
                e.share_runtime(&self.cpu_runtime);
                Executable::Cpu(e)
            };
            self.executables.insert(key, exe);
            profile_startup(&format!("compile {}-token shape", chunk.len()), started);
        }
        let started = Instant::now();
        let outputs = self.executables[&key]
            .run(&self.checkpoint.inputs, self.threads)
            .map_err(|e| e.to_string())?;
        Ok((outputs, started.elapsed()))
    }
    /// Merge cached allocation plans; only unseen fixed shapes parse and lower.
    fn reusable_plan(
        &mut self,
        backend: gpu::Backend,
        capacity: usize,
        tokens: usize,
        limit: usize,
    ) -> ffi::Result<gpu::MemoryPlan> {
        let mut combined: Option<gpu::MemoryPlan> = None;
        let mut states = None;
        for n in llm_capacity::reusable_prefill_shapes(tokens, limit) {
            let key = (n, capacity);
            if !self.shape_plans.contains_key(&key) {
                let (lowered, layout) = llm_capacity::shape_plan(
                    backend,
                    &self.source,
                    &self.checkpoint.context,
                    capacity,
                    n,
                )
                .map_err(|e| e.to_string())?;
                let plan = gpu::MemoryPlan::from_lowered(&lowered).map_err(|e| e.to_string())?;
                self.shape_plans.insert(key, (plan, layout));
                if !self.executables.contains_key(&key) {
                    self.prepared_shapes.insert(key, lowered);
                }
            }
            let (plan, layout) = &self.shape_plans[&key];
            if states.as_ref().is_some_and(|s| s != layout) {
                return Err("retained state layout changes between execution shapes".into());
            }
            states = Some(layout.clone());
            if let Some(combined) = &mut combined {
                combined.merge(plan).map_err(|e| e.to_string())?;
            } else {
                combined = Some(plan.clone());
            }
        }
        combined.ok_or_else(|| "empty prefill plan".into())
    }

    fn fit_reusable_request(
        &mut self,
        capacity: usize,
        minimum: usize,
        tokens: usize,
        limit: usize,
    ) -> ffi::Result<(usize, usize)> {
        let backend = self.gpu_runtime.as_ref().unwrap().backend();
        let mut seen = std::collections::HashSet::new();
        let mut last_error = String::new();
        for candidate in [capacity, minimum] {
            for chunk in [limit, limit.min(8), 1] {
                let shapes = llm_capacity::reusable_prefill_shapes(tokens, chunk);
                if !seen.insert((candidate, shapes)) {
                    continue;
                }
                let plan = self.reusable_plan(backend, candidate, tokens, chunk)?;
                match self.gpu_runtime.as_ref().unwrap().check_memory(&plan, 0) {
                    Ok(()) => return Ok((candidate, chunk)),
                    Err(error) => last_error = error.to_string(),
                }
            }
        }
        Err(last_error)
    }

    fn check_decode_memory(&mut self, capacity: usize, tokens: usize) -> ffi::Result<bool> {
        let Some(runtime) = &self.gpu_runtime else {
            return Ok(true);
        };
        let key = (capacity, tokens, 0);
        if !self.request_plans.contains_key(&key) {
            let prepared = llm_capacity::prepare_request(
                runtime.backend(),
                &self.source,
                &self.checkpoint.context,
                capacity,
                tokens,
                None,
                false,
            )
            .map_err(|e| e.to_string())?;
            self.request_plans.insert(key, prepared.plan);
            for (n, lowered) in prepared.shapes {
                self.prepared_shapes.entry((n, capacity)).or_insert(lowered);
            }
        }
        Ok(runtime.check_memory(&self.request_plans[&key], 0).is_ok())
    }

    fn grow_buffers(
        &mut self,
        old_capacity: usize,
        required: usize,
        retained: bool,
    ) -> ffi::Result<bool> {
        let mut capacity = old_capacity
            .saturating_mul(2)
            .max(required)
            .min(self.info.context_length as usize);
        if let Some(runtime) = &self.gpu_runtime {
            // Check the smallest legal allocation before selecting growth headroom.
            let minimum = llm_capacity::prepare_request(
                runtime.backend(),
                &self.source,
                &self.checkpoint.context,
                required,
                if retained { 1 } else { required },
                None,
                retained,
            )
            .map_err(|e| e.to_string())?;
            if runtime.check_memory(&minimum.plan, 0).is_err() {
                return Ok(false);
            }
            let selected = llm_capacity::prepare_adaptive_request(
                runtime.backend(),
                &self.source,
                &self.checkpoint.context,
                llm_capacity::ContextRequest {
                    capacity,
                    prompt_tokens: if retained { 1 } else { required },
                    minimum_capacity: Some(required),
                },
                None,
                retained,
                false,
                |plan| runtime.check_memory(plan, 0),
            )
            .map_err(|e| e.to_string())?;
            capacity = selected.capacity;
            let tokens = if retained { 1 } else { capacity };
            self.request_plans
                .insert((capacity, tokens, 0), selected.prepared.plan);
            for (n, lowered) in selected.prepared.shapes {
                self.shape_plans.insert(
                    (n, capacity),
                    (
                        gpu::MemoryPlan::from_lowered(&lowered).map_err(|e| e.to_string())?,
                        selected.prepared.states.clone(),
                    ),
                );
                self.prepared_shapes.insert((n, capacity), lowered);
            }
        }
        let old = source::parse_with_context(&self.source, &self.checkpoint.context)
            .map_err(|e| e.to_string())?
            .states;
        let mut context = self.checkpoint.context.clone();
        context
            .constants
            .insert("buffer_capacity".into(), Scalar::Int(capacity as i64));
        let new = source::parse_with_context(&self.source, &context)
            .map_err(|e| e.to_string())?
            .states;
        if old.len() != new.len()
            || old
                .iter()
                .zip(&new)
                .any(|(a, b)| a.name != b.name || a.slot != b.slot || a.dtype != b.dtype)
        {
            return Err("retained state identity changed during context growth".into());
        }
        for (a, b) in old.iter().zip(&new) {
            if let Some(runtime) = &self.gpu_runtime {
                runtime
                    .grow_state(a.slot, a.dtype, &a.shape, &b.shape)
                    .map_err(|e| e.to_string())?;
            } else {
                self.cpu_runtime
                    .grow_state(a.slot, a.dtype, &a.shape, &b.shape)
                    .map_err(|e| e.to_string())?;
            }
        }
        self.checkpoint.context = context;
        self.context_capacity = capacity;
        Ok(true)
    }

    fn generate(
        &mut self,
        input: &[u32],
        generation: &Generation,
        callbacks: &Callbacks,
    ) -> ffi::Result<u32> {
        self.output.clear();
        generation.validate()?;
        if input.is_empty() {
            return Err("prompt must contain at least one token".into());
        }
        if input.iter().any(|&id| id >= self.info.vocab_size) {
            return Err("input token is outside vocabulary".into());
        }
        let required = (input.len() as u64)
            .checked_add(generation.max_new_tokens.saturating_sub(1))
            .ok_or("context length overflow")?;
        if required > self.info.context_length {
            return Err(format!(
                "generation would exceed context length {}",
                self.info.context_length
            ));
        }
        let minimum = if self.grow_context {
            input.len()
        } else {
            required as usize
        };
        let mut capacity = if self.grow_context {
            self.context_capacity
                .max(minimum)
                .min(self.info.context_length as usize)
        } else {
            (required as usize)
                .max(512)
                .checked_next_power_of_two()
                .ok_or("buffer capacity overflow")?
                .min(self.info.context_length as usize)
        };
        self.checkpoint
            .context
            .constants
            .insert("buffer_capacity".into(), Scalar::Int(capacity as i64));
        self.checkpoint
            .bind_tokens(&input.iter().map(|&id| id as usize).collect::<Vec<_>>())
            .map_err(|e| e.to_string())?;
        // State declarations opt into the generic retained-input contract. The
        // model source owns all state layout, positions, writes, and attention.
        let mut first_program = if self.retained.is_none() {
            Some(
                source::parse_with_context(&self.source, &self.checkpoint.context)
                    .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };
        if let Some(program) = &first_program {
            self.retained = Some(!program.states.is_empty());
        }
        let retained = self.retained.unwrap();
        let reusable = retained && self.reusable_prefill;
        if reusable {
            first_program = None;
        }
        let mut prefill_chunk = if retained && (self.adaptive_prefill || reusable) {
            Some(llm_capacity::DEFAULT_PREFILL_CHUNK.min(capacity))
        } else {
            self.prefill_chunk
        };
        if reusable && self.auto_context {
            let started = Instant::now();
            let selected =
                self.fit_reusable_request(capacity, minimum, input.len(), prefill_chunk.unwrap())?;
            capacity = selected.0;
            prefill_chunk = Some(selected.1);
            profile_startup("reusable request planning", started);
            self.checkpoint
                .context
                .constants
                .insert("buffer_capacity".into(), Scalar::Int(capacity as i64));
        } else if let Some(runtime) = self.gpu_runtime.as_ref().filter(|_| self.auto_context) {
            let planned_tokens = if retained { input.len() } else { minimum };
            let main = prefill_chunk.unwrap_or(planned_tokens).min(planned_tokens);
            let key = (capacity, main, planned_tokens % main);
            if !self.request_plans.contains_key(&key) {
                let started = Instant::now();
                let prepared = llm_capacity::prepare_request(
                    runtime.backend(),
                    &self.source,
                    &self.checkpoint.context,
                    capacity,
                    planned_tokens,
                    prefill_chunk,
                    retained,
                )
                .map_err(|e| e.to_string())?;
                self.request_plans.insert(key, prepared.plan);
                for (tokens, lowered) in prepared.shapes {
                    let shape_key = (tokens, capacity);
                    if !self.executables.contains_key(&shape_key) {
                        self.prepared_shapes.insert(shape_key, lowered);
                    }
                }
                profile_startup("request planning", started);
            }
            // Capacity selection reserved module/graph headroom. Those loaded
            // resources now already reduce live free memory, so do not charge
            // that initial overhead twice. This check includes resize peaks.
            if runtime.check_memory(&self.request_plans[&key], 0).is_err() {
                let selected = llm_capacity::prepare_adaptive_request(
                    runtime.backend(),
                    &self.source,
                    &self.checkpoint.context,
                    llm_capacity::ContextRequest {
                        capacity,
                        prompt_tokens: input.len(),
                        minimum_capacity: Some(minimum),
                    },
                    prefill_chunk,
                    retained,
                    self.adaptive_prefill,
                    |plan| runtime.check_memory(plan, 0),
                )
                .map_err(|e| e.to_string())?;
                capacity = selected.capacity;
                prefill_chunk = selected.prefill_chunk;
                self.checkpoint
                    .context
                    .constants
                    .insert("buffer_capacity".into(), Scalar::Int(capacity as i64));
                let tokens = if retained { input.len() } else { capacity };
                let main = prefill_chunk.unwrap_or(tokens).min(tokens);
                self.request_plans
                    .insert((capacity, main, tokens % main), selected.prepared.plan);
                for (tokens, lowered) in selected.prepared.shapes {
                    let shape_key = (tokens, capacity);
                    if !self.executables.contains_key(&shape_key) {
                        self.prepared_shapes.insert(shape_key, lowered);
                    }
                }
                eprintln!(
                    "memory pressure: context {capacity}, prefill {}",
                    prefill_chunk
                        .map_or_else(|| "whole prompt".into(), |n| format!("{n}-token chunks"))
                );
            }
        }
        self.context_capacity = capacity;
        if let Some(runtime) = &self.gpu_runtime {
            runtime.reset_state().map_err(|e| e.to_string())?;
        }
        self.cpu_runtime.reset_state().map_err(|e| e.to_string())?;
        let mut history: Vec<usize> = input.iter().map(|&id| id as usize).collect();
        let mut sampling = TextGenerationConfig::new(generation.max_new_tokens as usize);
        sampling.temperature = generation.temperature;
        sampling.seed = generation.seed;
        let mut sampler = LogitsSampler::new(generation.seed);
        for step in 0..generation.max_new_tokens {
            if self.grow_context && history.len() > capacity {
                if !self.grow_buffers(capacity, history.len(), retained)? {
                    return Ok(DONE_MEMORY);
                }
                capacity = self.context_capacity;
                first_program = None;
            }
            if self.grow_context
                && !retained
                && step > 0
                && !self.check_decode_memory(capacity, history.len())?
            {
                return Ok(DONE_MEMORY);
            }
            let chunk = if retained && step > 0 {
                &history[history.len() - 1..]
            } else {
                &history[..]
            };
            let mut execution_time = std::time::Duration::ZERO;
            let mut outputs = vec![];
            let chunk_size = if step == 0 && retained {
                prefill_chunk.unwrap_or(chunk.len())
            } else {
                chunk.len()
            };
            let mut offset = 0;
            while offset < chunk.len() {
                let size = if step == 0 && reusable {
                    llm_capacity::reusable_prefill_chunk(chunk.len() - offset, chunk_size)
                } else {
                    chunk_size.min(chunk.len() - offset)
                };
                let part = &chunk[offset..offset + size];
                let program = if reusable { None } else { first_program.take() };
                let (result, elapsed) = self.execute(part, capacity, program, step == 0)?;
                outputs = result;
                execution_time += elapsed;
                offset += size;
            }
            if outputs.len() != 1 {
                return Err("LLM .pup provider expects one logits output".into());
            }
            let logits = outputs[0].f32().map_err(|e| e.to_string())?;
            if logits.len() != self.info.vocab_size as usize
                || logits.iter().any(|x| !x.is_finite())
            {
                return Err("expected one finite vocabulary-sized logits row".into());
            }
            if let Some(reference) = &self.reference {
                let expected = reference
                    .model
                    .forward(&history)
                    .map_err(|e| e.to_string())?;
                let expected = expected.last_logits().map_err(|e| e.to_string())?;
                let error = logits
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let greedy = TextGenerationConfig::new(1);
                if expected.len() != logits.len()
                    || error > 0.003
                    || sampler
                        .select_next_token(logits, &history, &greedy)
                        .map_err(|e| e.to_string())?
                        != sampler
                            .select_next_token(expected, &history, &greedy)
                            .map_err(|e| e.to_string())?
                {
                    return Err(format!(
                        "reference mismatch: max absolute logit error {error}"
                    ));
                }
                eprintln!(
                    "reference passed: max absolute logit error {error:.7}, greedy token matches"
                );
            }
            let next = sampler
                .select_next_token(logits, &history, &sampling)
                .map_err(|e| e.to_string())? as u32;
            eprintln!(
                "step {}: {} tokens, {:.3}s execution, next token {next}",
                step + 1,
                history.len(),
                execution_time.as_secs_f64()
            );
            history.push(next as usize);
            self.output.push(next);
            if let Some(callback) = callbacks.on_tokens {
                unsafe { callback(callbacks.user, &next, 1) };
                if let Some(error) = unsafe { ffi::callback_error(callbacks) } {
                    return Err(error);
                }
            }
            if next == self.info.eos_token {
                return Ok(DONE_EOS);
            }
        }
        Ok(
            if self.grow_context
                && required == self.info.context_length
                && generation.max_new_tokens > 0
            {
                DONE_CONTEXT
            } else {
                DONE_LIMIT
            },
        )
    }
}
unsafe extern "C" fn infer(
    state: *mut c_void,
    input: *const u32,
    count: usize,
    generation: *const Generation,
    callbacks: *const Callbacks,
    error: *mut ErrorBuffer,
) -> i32 {
    if callbacks.is_null() {
        error_to(error, "null callbacks argument");
        return 1;
    }
    let result = boundary(|| {
        if state.is_null() || input.is_null() || generation.is_null() {
            return Err("null infer argument".into());
        }
        let state = &mut *state.cast::<State>();
        state.generate(
            std::slice::from_raw_parts(input, count),
            &*generation,
            &*callbacks,
        )
    });
    let (status, reason) = match result {
        Ok(reason) => (0, reason),
        Err(e) => {
            error_to(error, &e);
            (1, DONE_ERROR)
        }
    };
    if let Some(done) = (*callbacks).on_done {
        done((*callbacks).user, reason);
    }
    status
}
unsafe extern "C" fn read_output(
    state: *mut c_void,
    destination: *mut u32,
    capacity: usize,
    count: *mut usize,
    error: *mut ErrorBuffer,
) -> i32 {
    let result = boundary(|| {
        if state.is_null() || count.is_null() {
            return Err("null read_output argument".into());
        }
        let output = &(*state.cast::<State>()).output;
        *count = output.len();
        if destination.is_null() && capacity == 0 {
            return Ok(());
        }
        if capacity < output.len() || destination.is_null() {
            return Err("output destination is too small or null".into());
        }
        if !output.is_empty() {
            std::ptr::copy_nonoverlapping(output.as_ptr(), destination, output.len());
        }
        Ok(())
    });
    match result {
        Ok(()) => 0,
        Err(e) => {
            error_to(error, &e);
            1
        }
    }
}
unsafe extern "C" fn free_model(state: *mut c_void) {
    if !state.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(Box::from_raw(state.cast::<State>()))
        }));
    }
}

#[cfg(test)]
mod prefill_tests {
    use super::*;
    fn tiny_state() -> Box<State> {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "pup-prefill-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("model.pup");
        std::fs::write(&source, "tokens = input(\"tokens\")\nposition = state(\"position\", i32, [1])\ni = index(position, cast(0, i32))\nadvance = store(i, load(i) + dim(tokens, 0))\noutput after(weight(\"scores\"), advance)\n").unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"vocab_size":2,"n_positions":512}"#,
        )
        .unwrap();
        let bytes = [0f32, 1.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2], &bytes).unwrap();
        std::fs::write(
            dir.join("model.safetensors"),
            safetensors::tensor::serialize([("scores", view)], None).unwrap(),
        )
        .unwrap();
        let config = serde_json::to_vec(&serde_json::json!({"source":source,"model_dir":dir,"cache_dir":dir.join("compiled"),"threads":1,"device":"cpu","grow_context":true,"context_request":{"capacity":512,"prompt_tokens":1}})).unwrap();
        let mut info = Info::default();
        let pointer = unsafe {
            build_model(
                config.as_ptr(),
                config.len(),
                &mut info,
                std::ptr::null_mut(),
            )
        };
        assert!(!pointer.is_null());
        unsafe { Box::from_raw(pointer.cast::<State>()) }
    }
    fn cleanup(state: Box<State>) {
        let dir = state.source_path.parent().unwrap().to_owned();
        drop(state);
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn warmed_prefill_reuses_programs_for_different_prompt_lengths() {
        let mut state = tiny_state();
        let callbacks = Callbacks {
            on_tokens: None,
            on_done: None,
            user: std::ptr::null_mut(),
        };
        let generation = Generation {
            max_new_tokens: 2,
            temperature: 0.,
            reserved: 0,
            seed: 1,
        };
        for length in [1, 8, 13, 16, 17, 21] {
            state
                .generate(&vec![0; length], &generation, &callbacks)
                .unwrap();
            assert_eq!(state.output, [1, 1]);
            assert!(state
                .executables
                .keys()
                .all(|&(n, c)| [1, 8].contains(&n) && c == 512));
        }
        assert_eq!(state.executables.len(), 2);
        for length in [128, 129, 137] {
            state
                .generate(&vec![0; length], &generation, &callbacks)
                .unwrap();
            assert!(state
                .executables
                .keys()
                .all(|&(n, _)| [1, 8, 128].contains(&n)));
        }
        assert_eq!(state.executables.len(), 3);
        cleanup(state);
    }
    #[test]
    fn reusable_gpu_plans_are_cached_per_shape_without_a_gpu() {
        let mut state = tiny_state();
        state.checkpoint.bind_tokens(&[0]).unwrap();
        let a = state
            .reusable_plan(gpu::Backend::Cuda, 512, 17, 128)
            .unwrap();
        assert_eq!(state.shape_plans.len(), 2);
        let b = state
            .reusable_plan(gpu::Backend::Cuda, 512, 21, 128)
            .unwrap();
        assert_eq!(state.shape_plans.len(), 2);
        assert_eq!(a.total_bytes(), b.total_bytes());
        state
            .reusable_plan(gpu::Backend::Cuda, 512, 129, 128)
            .unwrap();
        state
            .reusable_plan(gpu::Backend::Cuda, 512, 137, 128)
            .unwrap();
        assert_eq!(state.shape_plans.len(), 3);
        assert!(state
            .shape_plans
            .keys()
            .all(|&(n, c)| [1, 8, 128].contains(&n) && c == 512));
        cleanup(state);
    }
}
