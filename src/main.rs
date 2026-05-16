use clap::{Parser, Subcommand, ValueEnum};
use puppygrad::engine::Tensor;
use puppygrad::models::gpt2::{
    default_gpt2_small_dir, download_gpt2_small_assets, download_huggingface_gpt2_assets,
    Gpt2BackendConfig, Gpt2GenerationConfig, Gpt2GenerationStats, Gpt2Runtime,
};
use serde::Serialize;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser, Debug)]
#[command(name = "puppygrad")]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run GPT-2 small through puppygrad's native reference model.
    Gpt2 {
        /// Local directory containing config.json, tokenizer.json, and model.safetensors.
        #[arg(long)]
        model_dir: Option<PathBuf>,

        /// Hugging Face model id used with --download.
        #[arg(long, default_value = "gpt2")]
        model_id: String,

        /// Hugging Face revision used with --download.
        #[arg(long, default_value = "main")]
        revision: String,

        /// Download missing model assets into --model-dir before running.
        #[arg(long)]
        download: bool,

        /// Prompt text.
        #[arg(long)]
        prompt: String,

        /// Max new tokens to generate.
        #[arg(long, default_value_t = 32)]
        max_new_tokens: usize,

        /// Execution backend.
        #[arg(long, value_enum, default_value_t = Gpt2BackendArg::Rust)]
        backend: Gpt2BackendArg,

        /// Number of worker threads for CPU-style backends.
        #[arg(long, default_value_t = 1)]
        threads: usize,

        /// Print generation timing and token throughput to stderr.
        #[arg(long)]
        stats: bool,
    },

    /// Placeholder for the future in-house Qwen runtime.
    Qwen {
        /// Local model directory reserved for future native weight loading.
        #[arg(long)]
        model_dir: Option<PathBuf>,

        /// Model id reserved for future metadata/download tooling.
        #[arg(long, default_value = "Qwen/Qwen2.5-0.5B-Instruct")]
        model_id: String,

        /// Revision reserved for future metadata/download tooling.
        #[arg(long, default_value = "main")]
        revision: String,

        /// Reserved flag for future native model asset handling.
        #[arg(long)]
        download: bool,

        /// Prompt text
        #[arg(long)]
        prompt: String,

        /// Max new tokens to generate
        #[arg(long, default_value_t = 128)]
        max_new_tokens: usize,

        /// Temperature (<= 0 => greedy)
        #[arg(long, default_value_t = 0.8)]
        temperature: f64,

        /// Top-p nucleus sampling (optional)
        #[arg(long)]
        top_p: Option<f64>,

        /// Top-k sampling (optional)
        #[arg(long)]
        top_k: Option<usize>,

        /// RNG seed
        #[arg(long, default_value_t = 299792458)]
        seed: u64,

        /// Repeat penalty (1.0 = disabled)
        #[arg(long, default_value_t = 1.1)]
        repeat_penalty: f32,

        /// How many last tokens are considered for repeat penalty
        #[arg(long, default_value_t = 128)]
        repeat_last_n: usize,

        /// Reserved dtype selector for the future native runtime.
        #[arg(long)]
        dtype: Option<String>,

        /// Reserved flag for future prompt templating.
        #[arg(long)]
        instruct: bool,
    },

    /// Run reproducible performance sweeps.
    Experiment {
        #[command(subcommand)]
        cmd: ExperimentCommand,
    },

    /// Train y = 2x + 3 with scalar parameters using the in-house autograd engine.
    DemoLinear {
        /// Number of SGD steps.
        #[arg(long, default_value_t = 300)]
        steps: usize,

        /// SGD learning rate.
        #[arg(long, default_value_t = 0.1)]
        lr: f32,

        /// Print progress every N steps.
        #[arg(long, default_value_t = 25)]
        log_every: usize,
    },

    /// Quick matrix multiply + backward sanity check.
    MatmulCheck,
}

#[derive(Subcommand, Debug)]
enum ExperimentCommand {
    /// Sweep GPT-2 runtime settings and print timing rows.
    Gpt2 {
        /// Local directory containing config.json, tokenizer.json, and model.safetensors.
        #[arg(long)]
        model_dir: Option<PathBuf>,

        /// Hugging Face model id used with --download.
        #[arg(long, default_value = "gpt2")]
        model_id: String,

        /// Hugging Face revision used with --download.
        #[arg(long, default_value = "main")]
        revision: String,

        /// Download missing model assets into --model-dir before running.
        #[arg(long)]
        download: bool,

        /// Prompt text.
        #[arg(long)]
        prompt: String,

        /// Comma-separated worker-thread counts, for example 1,2,4,8.
        #[arg(long, default_value = "1")]
        threads: String,

        /// Comma-separated max-new-token counts, for example 8,16,32.
        #[arg(long, default_value = "32")]
        max_new_tokens: String,

        /// Measured runs per setting.
        #[arg(long, default_value_t = 3)]
        runs: usize,

        /// Warmup runs per setting, excluded from output.
        #[arg(long, default_value_t = 1)]
        warmup_runs: usize,

        /// Output format.
        #[arg(long, value_enum, default_value_t = ExperimentFormatArg::Table)]
        format: ExperimentFormatArg,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Gpt2 {
            model_dir,
            model_id,
            revision,
            download,
            prompt,
            max_new_tokens,
            backend,
            threads,
            stats,
        } => run_gpt2(RunGpt2Args {
            model_dir,
            model_id,
            revision,
            download,
            prompt,
            max_new_tokens,
            backend,
            threads,
            stats,
        }),
        Command::Qwen {
            model_dir,
            model_id,
            revision,
            download,
            prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            seed,
            repeat_penalty,
            repeat_last_n,
            dtype,
            instruct,
        } => run_qwen(RunQwenArgs {
            model_dir,
            model_id,
            revision,
            download,
            prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            seed,
            repeat_penalty,
            repeat_last_n,
            dtype,
            instruct,
        }),
        Command::Experiment { cmd } => match cmd {
            ExperimentCommand::Gpt2 {
                model_dir,
                model_id,
                revision,
                download,
                prompt,
                threads,
                max_new_tokens,
                runs,
                warmup_runs,
                format,
            } => run_gpt2_experiment(RunGpt2ExperimentArgs {
                model_dir,
                model_id,
                revision,
                download,
                prompt,
                threads,
                max_new_tokens,
                runs,
                warmup_runs,
                format,
            }),
        },
        Command::DemoLinear {
            steps,
            lr,
            log_every,
        } => run_demo_linear(steps, lr, log_every),
        Command::MatmulCheck => run_matmul_check(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Gpt2BackendArg {
    Rust,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ExperimentFormatArg {
    Table,
    Csv,
    Json,
}

struct RunGpt2Args {
    model_dir: Option<PathBuf>,
    model_id: String,
    revision: String,
    download: bool,
    prompt: String,
    max_new_tokens: usize,
    backend: Gpt2BackendArg,
    threads: usize,
    stats: bool,
}

struct RunGpt2ExperimentArgs {
    model_dir: Option<PathBuf>,
    model_id: String,
    revision: String,
    download: bool,
    prompt: String,
    threads: String,
    max_new_tokens: String,
    runs: usize,
    warmup_runs: usize,
    format: ExperimentFormatArg,
}

struct RunQwenArgs {
    model_dir: Option<PathBuf>,
    model_id: String,
    revision: String,
    download: bool,
    prompt: String,
    max_new_tokens: usize,
    temperature: f64,
    top_p: Option<f64>,
    top_k: Option<usize>,
    seed: u64,
    repeat_penalty: f32,
    repeat_last_n: usize,
    dtype: Option<String>,
    instruct: bool,
}

fn run_gpt2(args: RunGpt2Args) -> Result<()> {
    let generation = Gpt2GenerationConfig::new(args.max_new_tokens);
    generation.validate()?;
    let backend = match args.backend {
        Gpt2BackendArg::Rust => Gpt2BackendConfig::rust(args.threads)?,
    };

    let model_dir = args.model_dir.unwrap_or_else(default_gpt2_small_dir);
    if args.download {
        let download_start = Instant::now();
        eprintln!(
            "downloading missing GPT-2 assets into {}",
            model_dir.display()
        );
        if args.model_id == "gpt2" && args.revision == "main" {
            download_gpt2_small_assets(&model_dir)?;
        } else {
            download_huggingface_gpt2_assets(&args.model_id, &args.revision, &model_dir)?;
        }
        if args.stats {
            eprintln!(
                "download/check time: {}",
                format_duration(download_start.elapsed())
            );
        }
    }

    eprintln!("backend: {}", backend.describe());
    eprintln!("loading GPT-2 from {}", model_dir.display());
    let load_start = Instant::now();
    let runtime = Gpt2Runtime::from_dir_with_backend(&model_dir, backend)?;
    let load_time = load_start.elapsed();
    let mut stdout = std::io::stdout().lock();
    let (_, generation_stats) =
        runtime.stream_greedy_text_with_stats(&args.prompt, generation.max_new_tokens, |text| {
            write!(stdout, "{text}")?;
            stdout.flush()?;
            Ok::<(), Box<dyn std::error::Error>>(())
        })?;
    writeln!(stdout)?;
    stdout.flush()?;
    if args.stats {
        print_gpt2_stats(load_time, &generation_stats);
    } else {
        print_gpt2_speed(&generation_stats);
    }
    Ok(())
}

fn print_gpt2_speed(stats: &Gpt2GenerationStats) {
    eprintln!(
        "\ntokens/sec: {:.2} tok/s ({} generated tokens)",
        stats.decode_tokens_per_second(),
        stats.generated_tokens
    );
}

fn print_gpt2_stats(load_time: Duration, stats: &Gpt2GenerationStats) {
    eprintln!("\nstats:");
    eprintln!("  load: {}", format_duration(load_time));
    eprintln!("  tokenize: {}", format_duration(stats.tokenize_time));
    eprintln!(
        "  prefill: {} ({} prompt tokens, {:.2} tok/s)",
        format_duration(stats.prefill_time),
        stats.prompt_tokens,
        stats.prefill_tokens_per_second()
    );
    if let Some(first_token_time) = stats.first_token_time {
        eprintln!(
            "  time to first token: {}",
            format_duration(first_token_time)
        );
    }
    eprintln!(
        "  decode: {} ({} generated tokens, {:.2} tok/s)",
        format_duration(stats.decode_time),
        stats.generated_tokens,
        stats.decode_tokens_per_second()
    );
    if let Some(avg_decode_token_time) = stats.average_decode_token_time() {
        eprintln!(
            "  avg decode token: {}",
            format_duration(avg_decode_token_time)
        );
    }
    eprintln!(
        "  generation total: {} ({} model tokens, {:.2} tok/s)",
        format_duration(stats.total_generation_time),
        stats.total_model_tokens(),
        stats.total_tokens_per_second()
    );
}

fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs_f64();
    if seconds >= 1.0 {
        return format!("{seconds:.3}s");
    }
    let milliseconds = seconds * 1_000.0;
    if milliseconds >= 1.0 {
        return format!("{milliseconds:.2}ms");
    }
    let microseconds = seconds * 1_000_000.0;
    format!("{microseconds:.2}us")
}

#[derive(Clone, Debug, Serialize)]
struct Gpt2ExperimentRow {
    backend: String,
    threads: usize,
    max_new_tokens: usize,
    runs: usize,
    prompt_tokens: usize,
    generated_tokens: f64,
    load_ms: f64,
    tokenize_ms: f64,
    prefill_ms: f64,
    time_to_first_token_ms: Option<f64>,
    decode_ms: f64,
    total_generation_ms: f64,
    prefill_tokens_per_second: f64,
    decode_tokens_per_second: f64,
    total_tokens_per_second: f64,
}

fn run_gpt2_experiment(args: RunGpt2ExperimentArgs) -> Result<()> {
    if args.runs == 0 {
        return Err("experiment --runs must be > 0".into());
    }
    let thread_counts = parse_usize_list("threads", &args.threads)?;
    let token_counts = parse_usize_list("max-new-tokens", &args.max_new_tokens)?;
    let model_dir = args.model_dir.unwrap_or_else(default_gpt2_small_dir);

    if args.download {
        eprintln!(
            "downloading missing GPT-2 assets into {}",
            model_dir.display()
        );
        if args.model_id == "gpt2" && args.revision == "main" {
            download_gpt2_small_assets(&model_dir)?;
        } else {
            download_huggingface_gpt2_assets(&args.model_id, &args.revision, &model_dir)?;
        }
    }

    let mut rows = Vec::new();
    for threads in thread_counts {
        let backend = Gpt2BackendConfig::rust(threads)?;
        eprintln!(
            "loading GPT-2 from {} with {}",
            model_dir.display(),
            backend.describe()
        );
        let load_start = Instant::now();
        let runtime = Gpt2Runtime::from_dir_with_backend(&model_dir, backend)?;
        let load_time = load_start.elapsed();

        for max_new_tokens in token_counts.iter().copied() {
            let generation = Gpt2GenerationConfig::new(max_new_tokens);
            generation.validate()?;

            for _ in 0..args.warmup_runs {
                let _ = runtime.stream_greedy_text_with_stats(
                    &args.prompt,
                    generation.max_new_tokens,
                    |_| Ok::<(), Box<dyn std::error::Error>>(()),
                )?;
            }

            let mut stats = Vec::with_capacity(args.runs);
            for _ in 0..args.runs {
                let (_, run_stats) = runtime.stream_greedy_text_with_stats(
                    &args.prompt,
                    generation.max_new_tokens,
                    |_| Ok::<(), Box<dyn std::error::Error>>(()),
                )?;
                stats.push(run_stats);
            }

            rows.push(average_gpt2_experiment_row(
                "rust",
                threads,
                max_new_tokens,
                args.runs,
                load_time,
                &stats,
            ));
        }
    }

    match args.format {
        ExperimentFormatArg::Table => print_gpt2_experiment_table(&rows),
        ExperimentFormatArg::Csv => print_gpt2_experiment_csv(&rows),
        ExperimentFormatArg::Json => println!("{}", serde_json::to_string_pretty(&rows)?),
    }

    Ok(())
}

fn average_gpt2_experiment_row(
    backend: &str,
    threads: usize,
    max_new_tokens: usize,
    runs: usize,
    load_time: Duration,
    stats: &[Gpt2GenerationStats],
) -> Gpt2ExperimentRow {
    let runs_f64 = runs as f64;
    let prompt_tokens = stats.first().map_or(0, |stats| stats.prompt_tokens);
    let generated_tokens = stats
        .iter()
        .map(|stats| stats.generated_tokens as f64)
        .sum::<f64>()
        / runs_f64;
    let tokenize_time = average_duration(stats.iter().map(|stats| stats.tokenize_time));
    let prefill_time = average_duration(stats.iter().map(|stats| stats.prefill_time));
    let decode_time = average_duration(stats.iter().map(|stats| stats.decode_time));
    let total_generation_time =
        average_duration(stats.iter().map(|stats| stats.total_generation_time));
    let first_token_times: Vec<Duration> = stats
        .iter()
        .filter_map(|stats| stats.first_token_time)
        .collect();
    let first_token_time = if first_token_times.is_empty() {
        None
    } else {
        Some(average_duration(first_token_times))
    };

    Gpt2ExperimentRow {
        backend: backend.to_string(),
        threads,
        max_new_tokens,
        runs,
        prompt_tokens,
        generated_tokens,
        load_ms: duration_ms(load_time),
        tokenize_ms: duration_ms(tokenize_time),
        prefill_ms: duration_ms(prefill_time),
        time_to_first_token_ms: first_token_time.map(duration_ms),
        decode_ms: duration_ms(decode_time),
        total_generation_ms: duration_ms(total_generation_time),
        prefill_tokens_per_second: rate(prompt_tokens as f64, prefill_time),
        decode_tokens_per_second: rate(generated_tokens, decode_time),
        total_tokens_per_second: rate(
            prompt_tokens as f64 + generated_tokens,
            total_generation_time,
        ),
    }
}

fn print_gpt2_experiment_table(rows: &[Gpt2ExperimentRow]) {
    println!(
        "{:<7} {:>7} {:>8} {:>5} {:>7} {:>7} {:>8} {:>8} {:>8} {:>10} {:>10}",
        "backend",
        "threads",
        "new_tok",
        "runs",
        "prompt",
        "gen",
        "load_ms",
        "prefill",
        "decode",
        "tok/s",
        "total/s"
    );
    for row in rows {
        println!(
            "{:<7} {:>7} {:>8} {:>5} {:>7} {:>7.1} {:>8.1} {:>8.1} {:>8.1} {:>10.2} {:>10.2}",
            row.backend,
            row.threads,
            row.max_new_tokens,
            row.runs,
            row.prompt_tokens,
            row.generated_tokens,
            row.load_ms,
            row.prefill_ms,
            row.decode_ms,
            row.decode_tokens_per_second,
            row.total_tokens_per_second
        );
    }
}

fn print_gpt2_experiment_csv(rows: &[Gpt2ExperimentRow]) {
    println!(
        "backend,threads,max_new_tokens,runs,prompt_tokens,generated_tokens,load_ms,tokenize_ms,prefill_ms,time_to_first_token_ms,decode_ms,total_generation_ms,prefill_tokens_per_second,decode_tokens_per_second,total_tokens_per_second"
    );
    for row in rows {
        let first_token = row
            .time_to_first_token_ms
            .map(|time| format!("{time:.3}"))
            .unwrap_or_default();
        println!(
            "{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{},{:.3},{:.3},{:.3},{:.3},{:.3}",
            row.backend,
            row.threads,
            row.max_new_tokens,
            row.runs,
            row.prompt_tokens,
            row.generated_tokens,
            row.load_ms,
            row.tokenize_ms,
            row.prefill_ms,
            first_token,
            row.decode_ms,
            row.total_generation_ms,
            row.prefill_tokens_per_second,
            row.decode_tokens_per_second,
            row.total_tokens_per_second
        );
    }
}

fn parse_usize_list(name: &str, values: &str) -> Result<Vec<usize>> {
    let parsed: std::result::Result<Vec<_>, _> = values
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::parse::<usize>)
        .collect();
    let parsed = parsed.map_err(|err| format!("invalid --{name} list: {err}"))?;
    if parsed.is_empty() {
        return Err(format!("--{name} must contain at least one value").into());
    }
    if parsed.contains(&0) {
        return Err(format!("--{name} values must be > 0").into());
    }
    Ok(parsed)
}

fn average_duration<I>(durations: I) -> Duration
where
    I: IntoIterator<Item = Duration>,
{
    let mut count = 0usize;
    let total_seconds = durations
        .into_iter()
        .map(|duration| {
            count += 1;
            duration.as_secs_f64()
        })
        .sum::<f64>();
    if count == 0 {
        return Duration::ZERO;
    }
    Duration::from_secs_f64(total_seconds / count as f64)
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn rate(tokens: f64, duration: Duration) -> f64 {
    let seconds = duration.as_secs_f64();
    if seconds == 0.0 {
        return 0.0;
    }
    tokens / seconds
}

fn run_qwen(args: RunQwenArgs) -> Result<()> {
    println!("qwen runtime is not implemented yet.");
    println!(
        "The external runtime was removed so transformer work can target puppygrad's native engine."
    );
    println!("requested model: {}@{}", args.model_id, args.revision);
    if let Some(model_dir) = args.model_dir {
        println!("model dir: {}", model_dir.display());
    }
    println!("prompt: {}", args.prompt);
    println!(
        "generation args: max_new_tokens={} temperature={} top_k={:?} top_p={:?} seed={} repeat_penalty={} repeat_last_n={} dtype={:?} instruct={} download={}",
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        args.seed,
        args.repeat_penalty,
        args.repeat_last_n,
        args.dtype,
        args.instruct,
        args.download
    );
    Ok(())
}

fn run_demo_linear(steps: usize, lr: f32, log_every: usize) -> Result<()> {
    let x = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], vec![4], false)?;
    let y = Tensor::from_vec(vec![1.0, 3.0, 5.0, 7.0], vec![4], false)?;

    let w = Tensor::scalar(-0.25, true);
    let b = Tensor::scalar(0.5, true);

    let initial_loss = mse(&x, &y, &w, &b)?.item()?;
    println!(
        "init: w={:.5} b={:.5} loss={:.6}",
        w.item()?,
        b.item()?,
        initial_loss
    );

    let log_every = log_every.max(1);
    for step in 0..steps {
        w.zero_grad();
        b.zero_grad();

        let loss = mse(&x, &y, &w, &b)?;
        loss.backward()?;

        w.step(lr)?;
        b.step(lr)?;

        if (step + 1) % log_every == 0 || step + 1 == steps {
            println!(
                "step {:>4}: loss={:.6} w={:.5} b={:.5}",
                step + 1,
                loss.item()?,
                w.item()?,
                b.item()?
            );
        }
    }

    let final_loss = mse(&x, &y, &w, &b)?.item()?;
    println!(
        "done: w={:.5} b={:.5} loss={:.6}",
        w.item()?,
        b.item()?,
        final_loss
    );
    Ok(())
}

fn run_matmul_check() -> Result<()> {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true)?;
    let b = Tensor::from_vec(vec![5.0, 6.0], vec![2, 1], true)?;
    let out = a.matmul(&b)?.mean()?;
    out.backward()?;

    println!("out={:.5}", out.item()?);
    println!("grad(a)={:?}", a.grad().unwrap_or_default());
    println!("grad(b)={:?}", b.grad().unwrap_or_default());
    Ok(())
}

fn mse(x: &Tensor, y: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    let pred = x.mul(w)?.add(b)?;
    let diff = pred.sub(y)?;
    Ok(diff.mul(&diff)?.mean()?)
}
