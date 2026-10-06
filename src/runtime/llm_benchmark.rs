//! Repeated LLM inference measurements, with a fresh process per CPU thread count.
use super::{llm, llm_ffi::Generation};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(clap::Args, Debug, Serialize)]
pub struct Options {
    /// .pup program or shared library implementing the LLM contract.
    #[arg(default_value = "examples/llm.pup")]
    pub source: PathBuf,
    #[arg(long, default_value = "models/gpt2")]
    pub model_dir: PathBuf,
    #[arg(long, default_value = "cpu")]
    pub device: String,
    /// CPU instruction target for .pup compilation, forwarded to each worker.
    #[arg(long, value_enum, default_value_t = crate::compiler::cpu::CpuTarget::Generic)]
    pub cpu_target: crate::compiler::cpu::CpuTarget,
    #[arg(long, default_value = "The meaning of life is")]
    pub prompt: String,
    /// Sweep every count from 1 through N; defaults to available CPU parallelism.
    #[arg(long)]
    pub max_threads: Option<usize>,
    #[arg(long, default_value_t = 3)]
    pub runs: usize,
    #[arg(long, default_value_t = 1)]
    pub warmups: usize,
    #[arg(long, alias = "tokens", default_value_t = 8)]
    pub max_new_tokens: usize,
    /// New results directory; defaults to .cache/benchmarks/llm-threads-TIMESTAMP-PID.
    #[arg(long)]
    pub output_dir: Option<PathBuf>,
    // Re-executed by the parent to isolate each provider and its worker threads.
    #[arg(long, hide = true)]
    #[serde(skip)]
    pub worker_threads: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Trial {
    cpu_build: Option<crate::compiler::cpu::BuildInfo>,
    threads: usize,
    worker_pid: u32,
    prompt_tokens: usize,
    generated_tokens: usize,
    warmups: usize,
    runs: usize,
    seconds: Vec<f64>,
    median_seconds: f64,
    min_seconds: f64,
    max_seconds: f64,
    tokens_per_second: f64,
    load_seconds_excluded: f64,
    output_token_ids: Vec<u32>,
    text: String,
    host_start: serde_json::Value,
    host_end: serde_json::Value,
    speedup_vs_one_thread: Option<f64>,
}

fn host_snapshot() -> serde_json::Value {
    serde_json::json!({
        "unix_seconds": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64(),
        "load_average": fs::read_to_string("/proc/loadavg").ok().map(|s|
            s.split_whitespace().take(3).filter_map(|v| v.parse::<f64>().ok()).collect::<Vec<_>>()),
        "cpu_pressure": fs::read_to_string("/proc/pressure/cpu").ok(),
    })
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    fs::write(path, serde_json::to_string_pretty(value)? + "\n")?;
    Ok(())
}

pub fn run(options: Options) -> Result<()> {
    let available = std::thread::available_parallelism()?.get();
    let max_threads = options.max_threads.unwrap_or(available);
    if max_threads == 0 || options.runs == 0 || options.warmups == 0 || options.max_new_tokens == 0
    {
        return Err("max-threads, runs, warmups, and max-new-tokens must all be positive".into());
    }
    if max_threads > available {
        return Err(format!(
            "max-threads {max_threads} exceeds available CPU parallelism ({available})"
        )
        .into());
    }
    if options.device != "cpu" {
        return Err("the thread benchmark currently supports --device cpu only".into());
    }
    options.cpu_target.validate()?;
    if options.cpu_target != crate::compiler::cpu::CpuTarget::Generic
        && !options.source.extension().is_some_and(|e| e == "pup")
    {
        return Err(
            "--cpu-target applies to .pup compilation; shared libraries are already compiled"
                .into(),
        );
    }
    if let Some(threads) = options.worker_threads {
        if threads == 0 || threads > max_threads {
            return Err("worker thread count is outside the requested sweep".into());
        }
        return trial(&options, threads);
    }
    if cfg!(debug_assertions) {
        eprintln!(
            "Note: this is a debug build; use cargo run --release for performance measurements."
        );
    }
    let output = options.output_dir.clone().unwrap_or_else(|| {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        PathBuf::from(format!(
            ".cache/benchmarks/llm-threads-{stamp}-{}",
            std::process::id()
        ))
    });
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    // Never overwrite earlier measurements, including incomplete sweeps.
    fs::create_dir(&output).map_err(|e| {
        format!(
            "cannot create new results directory {}: {e}",
            output.display()
        )
    })?;
    let output = output.canonicalize()?;
    let executable = std::env::current_exe()?;
    let mut order: Vec<_> = (1..=max_threads).collect();
    // Fixed-seed shuffle reduces ordering bias and remains reproducible.
    let mut rng = 42u64;
    for i in (1..order.len()).rev() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        order.swap(i, (rng % (i as u64 + 1)) as usize);
    }
    let mut metadata = serde_json::json!({
        "options": options, "available_parallelism": available, "thread_trial_order": order,
        "executable": executable, "debug_build": cfg!(debug_assertions),
        "compute_backend": if options.source.extension().is_some_and(|e| e == "pup") { "self-contained C ABI 3, pthreads" } else { "external LLM provider" },
        "cpu_info": fs::read_to_string("/proc/cpuinfo").ok().and_then(|s|
            s.lines().find(|line| line.starts_with("model name")).map(str::to_owned)),
        "host_start": host_snapshot(),
        "timing": "warm complete infer call; includes sampling and output copy; excludes model/tokenizer loading, compilation/warmup, and text decoding",
        "sampling": "greedy, seed 42, no streaming, no reference verification",
        "status": "running",
    });
    write_json(&output.join("metadata.json"), &metadata)?;
    let mut results: Vec<Trial> = Vec::new();
    for (index, &threads) in order.iter().enumerate() {
        eprintln!("Starting {threads} threads ({}/{max_threads})", index + 1);
        let log_path = output.join(format!("threads-{threads:02}.log"));
        let log = fs::File::create(&log_path)?;
        let status = Command::new(&executable)
            .args(["llm", "benchmark"])
            .arg(&options.source)
            .arg("--model-dir")
            .arg(&options.model_dir)
            .arg("--device")
            .arg(&options.device)
            .arg("--cpu-target")
            .arg(options.cpu_target.to_string())
            .arg("--prompt")
            .arg(&options.prompt)
            .arg("--max-threads")
            .arg(max_threads.to_string())
            .arg("--runs")
            .arg(options.runs.to_string())
            .arg("--warmups")
            .arg(options.warmups.to_string())
            .arg("--max-new-tokens")
            .arg(options.max_new_tokens.to_string())
            .arg("--output-dir")
            .arg(&output)
            .arg("--worker-threads")
            .arg(threads.to_string())
            .env("TOKENIZERS_PARALLELISM", "false")
            .stdin(Stdio::null())
            .stderr(log.try_clone()?)
            .stdout(log)
            .status()?;
        if !status.success() {
            metadata["status"] = "failed".into();
            metadata["failed_threads"] = threads.into();
            metadata["host_end"] = host_snapshot();
            write_json(&output.join("metadata.json"), &metadata)?;
            return Err(format!(
                "benchmark at {threads} threads failed ({status}); see {}",
                log_path.display()
            )
            .into());
        }
        let result: Trial = serde_json::from_slice(&fs::read(
            output.join(format!("threads-{threads:02}.json")),
        )?)?;
        if results
            .first()
            .is_some_and(|first| first.output_token_ids != result.output_token_ids)
        {
            metadata["status"] = "output_mismatch".into();
            metadata["host_end"] = host_snapshot();
            write_json(&output.join("metadata.json"), &metadata)?;
            return Err(format!(
                "thread count changed generated tokens at {threads} threads; results in {}",
                output.display()
            )
            .into());
        }
        eprintln!(
            "{threads:2} threads: {:.3} s, {:.3} tokens/s",
            result.median_seconds, result.tokens_per_second
        );
        results.push(result);
    }
    results.sort_by_key(|r| r.threads);
    let baseline = results[0].median_seconds;
    let mut table = format!("| Threads | Median seconds / {} tokens | Min–max seconds | Tokens/s | Speedup |\n|---:|---:|---:|---:|---:|\n", options.max_new_tokens);
    let mut csv = String::from(
        "threads,median_seconds,min_seconds,max_seconds,tokens_per_second,speedup_vs_one_thread\n",
    );
    for row in &mut results {
        let speedup = baseline / row.median_seconds;
        row.speedup_vs_one_thread = Some(speedup);
        table += &format!(
            "| {} | {:.3} | {:.3}–{:.3} | {:.3} | {:.2}× |\n",
            row.threads,
            row.median_seconds,
            row.min_seconds,
            row.max_seconds,
            row.tokens_per_second,
            speedup
        );
        csv += &format!(
            "{},{},{},{},{},{}\n",
            row.threads,
            row.median_seconds,
            row.min_seconds,
            row.max_seconds,
            row.tokens_per_second,
            speedup
        );
    }
    write_json(&output.join("results.json"), &results)?;
    fs::write(output.join("results.csv"), csv)?;
    fs::write(output.join("results.md"), &table)?;
    metadata["status"] = "complete".into();
    metadata["host_end"] = host_snapshot();
    write_json(&output.join("metadata.json"), &metadata)?;
    print!("{table}\nResults: {}\n", output.display());
    Ok(())
}

fn trial(options: &Options, threads: usize) -> Result<()> {
    let cpu_build = if options.source.extension().is_some_and(|e| e == "pup") {
        Some(crate::compiler::cpu::BuildInfo::resolve(
            &crate::compiler::cpu::BuildOptions {
                cpu_target: options.cpu_target,
            },
        )?)
    } else {
        None
    };
    let output = options
        .output_dir
        .as_ref()
        .ok_or("worker requires an output directory")?;
    let host_start = host_snapshot();
    let load = Instant::now();
    let mut model = llm::load_model(
        &options.source,
        &options.model_dir,
        &options.device,
        Some(threads),
        false,
        options.cpu_target,
    )?;
    let load_seconds_excluded = load.elapsed().as_secs_f64();
    let (tokenizer, input) = llm::encode_prompt(&model, &options.model_dir, &options.prompt)?;
    let generation = Generation {
        max_new_tokens: options.max_new_tokens as u64,
        temperature: 0.,
        reserved: 0,
        seed: 42,
    };
    let mut expected = None;
    for _ in 0..options.warmups {
        let tokens = model.infer(&input, generation, None)?.tokens;
        if tokens.len() != options.max_new_tokens {
            return Err(
                "EOS shortened benchmark workload; choose a different prompt or fewer tokens"
                    .into(),
            );
        }
        if expected.as_ref().is_some_and(|ids| *ids != tokens) {
            return Err("warmup output changed".into());
        }
        expected = Some(tokens);
    }
    let expected = expected.unwrap();
    let mut seconds = Vec::with_capacity(options.runs);
    for run in 0..options.runs {
        let start = Instant::now();
        let tokens = model.infer(&input, generation, None)?.tokens;
        let duration = start.elapsed().as_secs_f64();
        if tokens != expected {
            return Err("measured output differs from warmup".into());
        }
        seconds.push(duration);
        eprintln!(
            "BENCH threads={threads} run={} seconds={duration:.6}",
            run + 1
        );
    }
    let mut sorted = seconds.clone();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    let median_seconds = if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.
    } else {
        sorted[middle]
    };
    let mut full = input.clone();
    full.extend_from_slice(&expected);
    let result = Trial {
        cpu_build,
        threads,
        worker_pid: std::process::id(),
        prompt_tokens: input.len(),
        generated_tokens: expected.len(),
        warmups: options.warmups,
        runs: options.runs,
        seconds,
        median_seconds,
        min_seconds: sorted[0],
        max_seconds: *sorted.last().unwrap(),
        tokens_per_second: expected.len() as f64 / median_seconds,
        load_seconds_excluded,
        output_token_ids: expected,
        text: tokenizer.decode(&full, true).map_err(|e| e.to_string())?,
        host_start,
        host_end: host_snapshot(),
        speedup_vs_one_thread: None,
    };
    write_json(&output.join(format!("threads-{threads:02}.json")), &result)
}
