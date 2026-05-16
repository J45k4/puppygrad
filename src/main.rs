use clap::{Parser, Subcommand, ValueEnum};
use puppygrad::engine::Tensor;
use puppygrad::models::gpt2::{
    default_gpt2_small_dir, download_gpt2_small_assets, download_huggingface_gpt2_assets,
    Gpt2BackendConfig, Gpt2GenerationConfig, Gpt2GenerationStats, Gpt2Runtime, Gpt2RustConfig,
};
use serde::Serialize;
use std::fs;
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

        /// Minimum dense multiply-add work items before parallel execution.
        #[arg(long)]
        dense_parallel_threshold: Option<usize>,

        /// Dense output chunk size for QKV projection jobs.
        #[arg(long)]
        qkv_chunk_size: Option<usize>,

        /// Dense output chunk size for attention projection jobs.
        #[arg(long)]
        attention_projection_chunk_size: Option<usize>,

        /// Dense output chunk size for MLP expansion jobs.
        #[arg(long)]
        mlp_fc_chunk_size: Option<usize>,

        /// Dense output chunk size for MLP projection jobs.
        #[arg(long)]
        mlp_projection_chunk_size: Option<usize>,

        /// Dense output chunk size for final logits jobs.
        #[arg(long)]
        logits_chunk_size: Option<usize>,

        /// Minimum attention work items before parallelizing across heads.
        #[arg(long)]
        attention_head_parallel_threshold: Option<usize>,

        /// Use experimental row-wise int8 dense/logit weights.
        #[arg(long)]
        quantized_weights: bool,

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

        /// Prompt text. Ignored when --prompt-file is set.
        #[arg(long)]
        prompt: Option<String>,

        /// Text file with one benchmark prompt per non-empty line.
        #[arg(long)]
        prompt_file: Option<PathBuf>,

        /// Comma-separated worker-thread counts, for example 1,2,4,8.
        #[arg(long, default_value = "1")]
        threads: String,

        /// Comma-separated dense parallel thresholds to sweep.
        #[arg(long, default_value = "262144")]
        dense_parallel_thresholds: String,

        /// Dense output chunk size for QKV projection jobs.
        #[arg(long)]
        qkv_chunk_size: Option<usize>,

        /// Dense output chunk size for attention projection jobs.
        #[arg(long)]
        attention_projection_chunk_size: Option<usize>,

        /// Dense output chunk size for MLP expansion jobs.
        #[arg(long)]
        mlp_fc_chunk_size: Option<usize>,

        /// Dense output chunk size for MLP projection jobs.
        #[arg(long)]
        mlp_projection_chunk_size: Option<usize>,

        /// Dense output chunk size for final logits jobs.
        #[arg(long)]
        logits_chunk_size: Option<usize>,

        /// Minimum attention work items before parallelizing across heads.
        #[arg(long)]
        attention_head_parallel_threshold: Option<usize>,

        /// Use experimental row-wise int8 dense/logit weights.
        #[arg(long)]
        quantized_weights: bool,

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
            dense_parallel_threshold,
            qkv_chunk_size,
            attention_projection_chunk_size,
            mlp_fc_chunk_size,
            mlp_projection_chunk_size,
            logits_chunk_size,
            attention_head_parallel_threshold,
            quantized_weights,
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
            tuning: RustTuning {
                dense_parallel_threshold,
                qkv_chunk_size,
                attention_projection_chunk_size,
                mlp_fc_chunk_size,
                mlp_projection_chunk_size,
                logits_chunk_size,
                attention_head_parallel_threshold,
                quantized_weights,
            },
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
                prompt_file,
                threads,
                dense_parallel_thresholds,
                qkv_chunk_size,
                attention_projection_chunk_size,
                mlp_fc_chunk_size,
                mlp_projection_chunk_size,
                logits_chunk_size,
                attention_head_parallel_threshold,
                quantized_weights,
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
                prompt_file,
                threads,
                dense_parallel_thresholds,
                tuning: RustTuning {
                    dense_parallel_threshold: None,
                    qkv_chunk_size,
                    attention_projection_chunk_size,
                    mlp_fc_chunk_size,
                    mlp_projection_chunk_size,
                    logits_chunk_size,
                    attention_head_parallel_threshold,
                    quantized_weights,
                },
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
    tuning: RustTuning,
    stats: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct RustTuning {
    dense_parallel_threshold: Option<usize>,
    qkv_chunk_size: Option<usize>,
    attention_projection_chunk_size: Option<usize>,
    mlp_fc_chunk_size: Option<usize>,
    mlp_projection_chunk_size: Option<usize>,
    logits_chunk_size: Option<usize>,
    attention_head_parallel_threshold: Option<usize>,
    quantized_weights: bool,
}

struct RunGpt2ExperimentArgs {
    model_dir: Option<PathBuf>,
    model_id: String,
    revision: String,
    download: bool,
    prompt: Option<String>,
    prompt_file: Option<PathBuf>,
    threads: String,
    dense_parallel_thresholds: String,
    tuning: RustTuning,
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
        Gpt2BackendArg::Rust => Gpt2BackendConfig::Rust(rust_config(args.threads, args.tuning)?),
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

fn rust_config(
    threads: usize,
    tuning: RustTuning,
) -> puppygrad::models::gpt2::Result<Gpt2RustConfig> {
    let defaults = Gpt2RustConfig::default();
    let config = Gpt2RustConfig {
        threads,
        dense_parallel_threshold: tuning
            .dense_parallel_threshold
            .unwrap_or(defaults.dense_parallel_threshold),
        qkv_chunk_size: tuning.qkv_chunk_size.unwrap_or(defaults.qkv_chunk_size),
        attention_projection_chunk_size: tuning
            .attention_projection_chunk_size
            .unwrap_or(defaults.attention_projection_chunk_size),
        mlp_fc_chunk_size: tuning
            .mlp_fc_chunk_size
            .unwrap_or(defaults.mlp_fc_chunk_size),
        mlp_projection_chunk_size: tuning
            .mlp_projection_chunk_size
            .unwrap_or(defaults.mlp_projection_chunk_size),
        logits_chunk_size: tuning
            .logits_chunk_size
            .unwrap_or(defaults.logits_chunk_size),
        attention_head_parallel_threshold: tuning
            .attention_head_parallel_threshold
            .unwrap_or(defaults.attention_head_parallel_threshold),
        quantized_weights: tuning.quantized_weights,
    };
    config.validate()?;
    Ok(config)
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
    eprintln!("  profile:");
    eprintln!(
        "    layernorm: {}",
        format_duration(stats.operation_profile.layer_norm)
    );
    eprintln!(
        "    qkv projection: {}",
        format_duration(stats.operation_profile.qkv_projection)
    );
    eprintln!(
        "    attention: {}",
        format_duration(stats.operation_profile.attention)
    );
    eprintln!(
        "    attention projection: {}",
        format_duration(stats.operation_profile.attention_projection)
    );
    eprintln!(
        "    mlp fc projection: {}",
        format_duration(stats.operation_profile.mlp_fc_projection)
    );
    eprintln!(
        "    mlp projection: {}",
        format_duration(stats.operation_profile.mlp_projection)
    );
    eprintln!(
        "    final logits: {}",
        format_duration(stats.operation_profile.final_logits)
    );
    eprintln!(
        "    tokenization: {}",
        format_duration(stats.operation_profile.tokenization)
    );
    eprintln!(
        "    decoding: {}",
        format_duration(stats.operation_profile.decoding)
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
    prompt_index: Option<usize>,
    prompt: String,
    threads: usize,
    dense_parallel_threshold: usize,
    qkv_chunk_size: usize,
    attention_projection_chunk_size: usize,
    mlp_fc_chunk_size: usize,
    mlp_projection_chunk_size: usize,
    logits_chunk_size: usize,
    attention_head_parallel_threshold: usize,
    quantized_weights: bool,
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
    decode_ms_min: f64,
    decode_ms_median: f64,
    decode_ms_p95: f64,
    decode_ms_max: f64,
    decode_ms_stddev: f64,
    total_generation_ms_min: f64,
    total_generation_ms_median: f64,
    total_generation_ms_p95: f64,
    total_generation_ms_max: f64,
    total_generation_ms_stddev: f64,
    prefill_tokens_per_second: f64,
    decode_tokens_per_second: f64,
    total_tokens_per_second: f64,
    profile_tokenization_ms: f64,
    profile_layer_norm_ms: f64,
    profile_qkv_projection_ms: f64,
    profile_attention_ms: f64,
    profile_attention_projection_ms: f64,
    profile_mlp_fc_projection_ms: f64,
    profile_mlp_projection_ms: f64,
    profile_final_logits_ms: f64,
    profile_decoding_ms: f64,
}

#[derive(Clone, Copy, Debug)]
struct DistributionSummary {
    min: f64,
    median: f64,
    p95: f64,
    max: f64,
    stddev: f64,
}

fn run_gpt2_experiment(args: RunGpt2ExperimentArgs) -> Result<()> {
    if args.runs == 0 {
        return Err("experiment --runs must be > 0".into());
    }
    let thread_counts = parse_usize_list("threads", &args.threads)?;
    let dense_parallel_thresholds =
        parse_usize_list("dense-parallel-thresholds", &args.dense_parallel_thresholds)?;
    let token_counts = parse_usize_list("max-new-tokens", &args.max_new_tokens)?;
    let prompts = load_experiment_prompts(args.prompt.as_deref(), args.prompt_file.as_ref())?;
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
        for dense_parallel_threshold in dense_parallel_thresholds.iter().copied() {
            let mut tuning = args.tuning;
            tuning.dense_parallel_threshold = Some(dense_parallel_threshold);
            let rust_config = rust_config(threads, tuning)?;
            let backend = Gpt2BackendConfig::Rust(rust_config.clone());
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

                let mut aggregate_stats = Vec::with_capacity(args.runs * prompts.len());
                for (prompt_index, prompt) in prompts.iter().enumerate() {
                    for _ in 0..args.warmup_runs {
                        let _ = runtime.stream_greedy_text_with_stats(
                            prompt,
                            generation.max_new_tokens,
                            |_| Ok::<(), Box<dyn std::error::Error>>(()),
                        )?;
                    }

                    let mut stats = Vec::with_capacity(args.runs);
                    for _ in 0..args.runs {
                        let (_, run_stats) = runtime.stream_greedy_text_with_stats(
                            prompt,
                            generation.max_new_tokens,
                            |_| Ok::<(), Box<dyn std::error::Error>>(()),
                        )?;
                        aggregate_stats.push(run_stats.clone());
                        stats.push(run_stats);
                    }

                    rows.push(average_gpt2_experiment_row(
                        "rust",
                        &rust_config,
                        Some(prompt_index),
                        prompt,
                        threads,
                        max_new_tokens,
                        args.runs,
                        load_time,
                        &stats,
                    ));
                }

                if prompts.len() > 1 {
                    rows.push(average_gpt2_experiment_row(
                        "rust",
                        &rust_config,
                        None,
                        "<aggregate>",
                        threads,
                        max_new_tokens,
                        aggregate_stats.len(),
                        load_time,
                        &aggregate_stats,
                    ));
                }
            }
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
    rust_config: &Gpt2RustConfig,
    prompt_index: Option<usize>,
    prompt: &str,
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
    let decode_summary = duration_distribution(stats.iter().map(|stats| stats.decode_time));
    let total_summary =
        duration_distribution(stats.iter().map(|stats| stats.total_generation_time));
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
        prompt_index,
        prompt: prompt.to_string(),
        threads,
        dense_parallel_threshold: rust_config.dense_parallel_threshold,
        qkv_chunk_size: rust_config.qkv_chunk_size,
        attention_projection_chunk_size: rust_config.attention_projection_chunk_size,
        mlp_fc_chunk_size: rust_config.mlp_fc_chunk_size,
        mlp_projection_chunk_size: rust_config.mlp_projection_chunk_size,
        logits_chunk_size: rust_config.logits_chunk_size,
        attention_head_parallel_threshold: rust_config.attention_head_parallel_threshold,
        quantized_weights: rust_config.quantized_weights,
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
        decode_ms_min: decode_summary.min,
        decode_ms_median: decode_summary.median,
        decode_ms_p95: decode_summary.p95,
        decode_ms_max: decode_summary.max,
        decode_ms_stddev: decode_summary.stddev,
        total_generation_ms_min: total_summary.min,
        total_generation_ms_median: total_summary.median,
        total_generation_ms_p95: total_summary.p95,
        total_generation_ms_max: total_summary.max,
        total_generation_ms_stddev: total_summary.stddev,
        prefill_tokens_per_second: rate(prompt_tokens as f64, prefill_time),
        decode_tokens_per_second: rate(generated_tokens, decode_time),
        total_tokens_per_second: rate(
            prompt_tokens as f64 + generated_tokens,
            total_generation_time,
        ),
        profile_tokenization_ms: duration_ms(average_duration(
            stats
                .iter()
                .map(|stats| stats.operation_profile.tokenization),
        )),
        profile_layer_norm_ms: duration_ms(average_duration(
            stats.iter().map(|stats| stats.operation_profile.layer_norm),
        )),
        profile_qkv_projection_ms: duration_ms(average_duration(
            stats
                .iter()
                .map(|stats| stats.operation_profile.qkv_projection),
        )),
        profile_attention_ms: duration_ms(average_duration(
            stats.iter().map(|stats| stats.operation_profile.attention),
        )),
        profile_attention_projection_ms: duration_ms(average_duration(
            stats
                .iter()
                .map(|stats| stats.operation_profile.attention_projection),
        )),
        profile_mlp_fc_projection_ms: duration_ms(average_duration(
            stats
                .iter()
                .map(|stats| stats.operation_profile.mlp_fc_projection),
        )),
        profile_mlp_projection_ms: duration_ms(average_duration(
            stats
                .iter()
                .map(|stats| stats.operation_profile.mlp_projection),
        )),
        profile_final_logits_ms: duration_ms(average_duration(
            stats
                .iter()
                .map(|stats| stats.operation_profile.final_logits),
        )),
        profile_decoding_ms: duration_ms(average_duration(
            stats.iter().map(|stats| stats.operation_profile.decoding),
        )),
    }
}

fn print_gpt2_experiment_table(rows: &[Gpt2ExperimentRow]) {
    println!(
        "{:<7} {:>6} {:>7} {:>9} {:>7} {:>8} {:>5} {:>7} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
        "backend",
        "prompt",
        "threads",
        "dense_th",
        "weights",
        "new_tok",
        "runs",
        "prompt",
        "gen",
        "load_ms",
        "prefill",
        "dec_avg",
        "dec_p95",
        "dec_sd",
        "tok/s",
        "total/s"
    );
    for row in rows {
        let prompt_index = row
            .prompt_index
            .map(|index| index.to_string())
            .unwrap_or_else(|| "all".to_string());
        println!(
            "{:<7} {:>6} {:>7} {:>9} {:>7} {:>8} {:>5} {:>7} {:>7.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>10.2} {:>10.2}",
            row.backend,
            prompt_index,
            row.threads,
            row.dense_parallel_threshold,
            if row.quantized_weights { "int8" } else { "f32" },
            row.max_new_tokens,
            row.runs,
            row.prompt_tokens,
            row.generated_tokens,
            row.load_ms,
            row.prefill_ms,
            row.decode_ms,
            row.decode_ms_p95,
            row.decode_ms_stddev,
            row.decode_tokens_per_second,
            row.total_tokens_per_second
        );
    }
}

fn print_gpt2_experiment_csv(rows: &[Gpt2ExperimentRow]) {
    println!(
        "backend,prompt_index,prompt,threads,dense_parallel_threshold,qkv_chunk_size,attention_projection_chunk_size,mlp_fc_chunk_size,mlp_projection_chunk_size,logits_chunk_size,attention_head_parallel_threshold,quantized_weights,max_new_tokens,runs,prompt_tokens,generated_tokens,load_ms,tokenize_ms,prefill_ms,time_to_first_token_ms,decode_ms,total_generation_ms,decode_ms_min,decode_ms_median,decode_ms_p95,decode_ms_max,decode_ms_stddev,total_generation_ms_min,total_generation_ms_median,total_generation_ms_p95,total_generation_ms_max,total_generation_ms_stddev,prefill_tokens_per_second,decode_tokens_per_second,total_tokens_per_second,profile_tokenization_ms,profile_layer_norm_ms,profile_qkv_projection_ms,profile_attention_ms,profile_attention_projection_ms,profile_mlp_fc_projection_ms,profile_mlp_projection_ms,profile_final_logits_ms,profile_decoding_ms"
    );
    for row in rows {
        let first_token = row
            .time_to_first_token_ms
            .map(|time| format!("{time:.3}"))
            .unwrap_or_default();
        let prompt_index = row
            .prompt_index
            .map(|index| index.to_string())
            .unwrap_or_default();
        println!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            row.backend,
            prompt_index,
            csv_escape(&row.prompt),
            row.threads,
            row.dense_parallel_threshold,
            row.qkv_chunk_size,
            row.attention_projection_chunk_size,
            row.mlp_fc_chunk_size,
            row.mlp_projection_chunk_size,
            row.logits_chunk_size,
            row.attention_head_parallel_threshold,
            row.quantized_weights,
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
            row.decode_ms_min,
            row.decode_ms_median,
            row.decode_ms_p95,
            row.decode_ms_max,
            row.decode_ms_stddev,
            row.total_generation_ms_min,
            row.total_generation_ms_median,
            row.total_generation_ms_p95,
            row.total_generation_ms_max,
            row.total_generation_ms_stddev,
            row.prefill_tokens_per_second,
            row.decode_tokens_per_second,
            row.total_tokens_per_second,
            row.profile_tokenization_ms,
            row.profile_layer_norm_ms,
            row.profile_qkv_projection_ms,
            row.profile_attention_ms,
            row.profile_attention_projection_ms,
            row.profile_mlp_fc_projection_ms,
            row.profile_mlp_projection_ms,
            row.profile_final_logits_ms,
            row.profile_decoding_ms
        );
    }
}

fn load_experiment_prompts(
    prompt: Option<&str>,
    prompt_file: Option<&PathBuf>,
) -> Result<Vec<String>> {
    if let Some(path) = prompt_file {
        let text = fs::read_to_string(path)
            .map_err(|err| format!("failed to read --prompt-file {}: {err}", path.display()))?;
        let prompts: Vec<String> = text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_string)
            .collect();
        if prompts.is_empty() {
            return Err(format!("--prompt-file {} contains no prompts", path.display()).into());
        }
        return Ok(prompts);
    }

    let prompt = prompt.ok_or("experiment gpt2 requires --prompt or --prompt-file")?;
    Ok(vec![prompt.to_string()])
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
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

fn duration_distribution<I>(durations: I) -> DistributionSummary
where
    I: IntoIterator<Item = Duration>,
{
    let mut values: Vec<f64> = durations.into_iter().map(duration_ms).collect();
    if values.is_empty() {
        return DistributionSummary {
            min: 0.0,
            median: 0.0,
            p95: 0.0,
            max: 0.0,
            stddev: 0.0,
        };
    }
    values.sort_by(f64::total_cmp);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    DistributionSummary {
        min: values[0],
        median: percentile(&values, 0.5),
        p95: percentile(&values, 0.95),
        max: values[values.len() - 1],
        stddev: variance.sqrt(),
    }
}

fn percentile(sorted_values: &[f64], percentile: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let rank = percentile.clamp(0.0, 1.0) * (sorted_values.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        return sorted_values[lower];
    }
    let weight = rank - lower as f64;
    sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
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
