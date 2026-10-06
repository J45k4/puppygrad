//! Transitional .pup provider for the LLM ABI. Model math executes through the
//! compiler's CPU or CUDA backend; checkpoint binding and generation stay outside compiler.
mod checkpoint;
pub use checkpoint::Checkpoint;

use super::generation::{LogitsSampler, TextGenerationConfig};
use crate::{
    compiler::{cpu, cuda, pop::Scalar, source},
    runtime::llm_ffi::{
        self as ffi, Api, Callbacks, ErrorBuffer, Generation, Info, DONE_EOS, DONE_ERROR,
        DONE_LIMIT, NO_EOS,
    },
};
use std::{
    collections::HashMap,
    ffi::c_void,
    panic::{catch_unwind, AssertUnwindSafe},
    path::{Path, PathBuf},
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
}
struct State {
    source: String,
    source_path: PathBuf,
    checkpoint: Checkpoint,
    threads: usize,
    build_options: cpu::BuildOptions,
    info: Info,
    cuda_runtime: Option<cuda::Runtime>,
    cpu_runtime: cpu::Runtime,
    retained: Option<bool>,
    executables: HashMap<(usize, usize), Executable>,
    output: Vec<u32>,
    reference: Option<super::gpt2::Gpt2Runtime>,
}
enum Executable {
    Cpu(cpu::Executable),
    Cuda(cuda::Executable),
}
impl Executable {
    fn run(
        &self,
        inputs: &[cpu::Tensor],
        threads: usize,
    ) -> crate::compiler::pop::Result<Vec<cpu::Tensor>> {
        match self {
            Self::Cpu(e) => e.run_with_threads(inputs, threads),
            Self::Cuda(e) => e.run(inputs),
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
        let device = if matches!(config.device.as_str(), "cpu" | "cpu:0" | "c" | "c:0") {
            None
        } else if config.device == "cuda" || config.device.starts_with("cuda:") {
            Some(cuda::device_index(&config.device).map_err(|e| e.to_string())?)
        } else {
            return Err(format!(
                "unsupported device {:?}; use --device cpu or cuda:<index>",
                config.device
            ));
        };
        if device.is_some() && config.cpu_target != cpu::CpuTarget::Generic {
            return Err("--cpu-target applies to the CPU backend".into());
        }
        let threads = config.threads.unwrap_or_else(cpu::default_threads);
        if threads == 0 {
            return Err("threads must be greater than zero".into());
        }
        config.cpu_target.validate().map_err(|e| e.to_string())?;
        let source = std::fs::read_to_string(&config.source).map_err(|e| e.to_string())?;
        let checkpoint = Checkpoint::load(&config.model_dir).map_err(|e| e.to_string())?;
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
        let metadata = Info {
            vocab_size,
            eos_token,
            context_length,
        };
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
            checkpoint,
            threads,
            build_options: cpu::BuildOptions {
                cpu_target: config.cpu_target,
            },
            info: metadata,
            cuda_runtime: device
                .map(cuda::Runtime::new)
                .transpose()
                .map_err(|e| e.to_string())?,
            cpu_runtime: cpu::Runtime::default(),
            retained: None,
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
        if let Some(runtime) = &self.cuda_runtime {
            runtime.reset_state().map_err(|e| e.to_string())?;
        }
        self.cpu_runtime.reset_state().map_err(|e| e.to_string())?;
        let capacity = (required as usize)
            .max(512)
            .checked_next_power_of_two()
            .ok_or("buffer capacity overflow")?
            .min(self.info.context_length as usize);
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
        let mut history: Vec<usize> = input.iter().map(|&id| id as usize).collect();
        let mut sampling = TextGenerationConfig::new(generation.max_new_tokens as usize);
        sampling.temperature = generation.temperature;
        sampling.seed = generation.seed;
        let mut sampler = LogitsSampler::new(generation.seed);
        for step in 0..generation.max_new_tokens {
            let chunk = if retained && step > 0 {
                &history[history.len() - 1..]
            } else {
                &history[..]
            };
            self.checkpoint
                .bind_tokens(chunk)
                .map_err(|e| e.to_string())?;
            let key = (chunk.len(), capacity);
            if !self.executables.contains_key(&key) {
                let program = if let Some(program) = first_program.take() {
                    program
                } else {
                    source::parse_with_context(&self.source, &self.checkpoint.context)
                        .map_err(|e| format!("{}:{e}", self.source_path.display()))?
                };
                let exe = if let Some(runtime) = &self.cuda_runtime {
                    let e = cuda::compile_with_runtime(
                        &program.graph,
                        program.root,
                        Path::new(".cache/pup/cuda"),
                        runtime,
                    )
                    .map_err(|e| e.to_string())?;
                    if step == 0 {
                        eprintln!(
                            "compiled CUDA: {} contractions, {}; source: {}",
                            e.gemm_count,
                            e.device_name,
                            e.source_path.display()
                        );
                    }
                    Executable::Cuda(e)
                } else {
                    let mut e = cpu::compile_with_options(
                        &program.graph,
                        program.root,
                        Path::new(".cache/pup/cpu"),
                        &self.build_options,
                    )
                    .map_err(|e| e.to_string())?;
                    if step == 0 {
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
            }
            let started = Instant::now();
            let outputs = self.executables[&key]
                .run(&self.checkpoint.inputs, self.threads)
                .map_err(|e| e.to_string())?;
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
                started.elapsed().as_secs_f64()
            );
            history.push(next as usize);
            self.output.push(next);
            if let Some(callback) = callbacks.on_tokens {
                unsafe { callback(callbacks.user, &next, 1) };
            }
            if next == self.info.eos_token {
                return Ok(DONE_EOS);
            }
        }
        Ok(DONE_LIMIT)
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
