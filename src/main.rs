use clap::{Parser, Subcommand, ValueEnum};
use puppygrad::audio::{
    inspect_wav, list_input_devices, record_input_device, resample_linear,
    start_input_device_stream, write_wav_pcm16, AudioDropPolicy as RuntimeAudioDropPolicy,
    PcmAudio as SharedPcmAudio,
};
use puppygrad::engine::Tensor;
use puppygrad::models::autotune::{autotune, AutoTuneOptions, AutoTuneTarget};
use puppygrad::models::generation::{TextGenerationArgs, TextGenerationConfig};
use puppygrad::models::gpt2::{
    default_gpt2_small_dir, download_gpt2_small_assets, download_huggingface_gpt2_assets,
    Gpt2BackendConfig, Gpt2GenerationConfig, Gpt2GenerationStats, Gpt2Runtime, Gpt2RustConfig,
};
use puppygrad::models::resnet::{default_resnet18_dir, download_resnet18_assets, ResNetRuntime};
use puppygrad::models::streaming::{escape_raw_token, RawTokenDecoder};
use puppygrad::models::whisper::{
    default_whisper_dir, is_silence, load_wav_pcm, load_wav_pcm_bytes, log_mel_spectrogram,
    normalize_transcript, seconds_to_samples, PartialCommitState, PartialObservation,
    RollingAudioBuffer, TranscriptionEvent, TranscriptionJob, TranscriptionJobPurpose,
    WhisperBackendConfig, WhisperOperationProfile, WhisperRuntime, WhisperRustConfig, WhisperSize,
    WhisperTask as RuntimeWhisperTask, WHISPER_SAMPLE_RATE,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
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
    /// Shared audio utilities for microphone capture and WAV inspection.
    Audio {
        #[command(subcommand)]
        cmd: AudioCommand,
    },

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

        #[command(flatten)]
        generation: TextGenerationArgs,

        /// Token id that stops generation.
        #[arg(long)]
        eos_token_id: Option<usize>,

        /// Do not stop when the GPT-2 EOS token is generated.
        #[arg(long)]
        no_eos_stop: bool,

        /// Execution backend.
        #[arg(long, value_enum, default_value_t = Gpt2BackendArg::Rust)]
        backend: Gpt2BackendArg,

        /// Number of worker threads for CPU-style backends.
        #[arg(long)]
        threads: Option<usize>,

        /// Load tuning config from this JSON file.
        #[arg(long)]
        tuning_file: Option<PathBuf>,

        /// Do not auto-load model-dir/puppygrad-tune.json.
        #[arg(long)]
        no_tuning: bool,

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

        /// Stream raw tokenizer tokens as TSV rows instead of decoded text.
        #[arg(long)]
        stream_raw_tokens: bool,
    },

    /// Run an ImageNet ResNet classifier through puppygrad's native reference model.
    Resnet {
        /// RGB image path to classify.
        #[arg(long)]
        image: PathBuf,

        /// ResNet variant.
        #[arg(long, value_enum, default_value_t = ResNetVariantArg::Resnet18)]
        variant: ResNetVariantArg,

        /// Local directory containing model.safetensors and labels.
        #[arg(long)]
        model_dir: Option<PathBuf>,

        /// Download missing ResNet-18 assets into --model-dir before running.
        #[arg(long)]
        download: bool,

        /// Label file override. Supports ImageNet id2label JSON or one label per line.
        #[arg(long)]
        labels: Option<PathBuf>,

        /// Number of classes to print.
        #[arg(long, default_value_t = 5)]
        top_k: usize,

        /// Reserved worker-thread count for the future optimized CPU path.
        #[arg(long)]
        threads: Option<usize>,
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

        #[command(flatten)]
        generation: TextGenerationArgs,

        /// Reserved dtype selector for the future native runtime.
        #[arg(long)]
        dtype: Option<String>,

        /// Reserved flag for future prompt templating.
        #[arg(long)]
        instruct: bool,
    },

    /// Prepare assets and run the native Whisper runtime.
    Whisper {
        /// Audio WAV path to transcribe, or "-" to read WAV bytes from stdin. Use --download/--print-config without audio to prepare assets.
        #[arg(long, conflicts_with = "mic")]
        audio: Option<PathBuf>,

        /// Record from the default microphone and transcribe fixed-size chunks.
        #[arg(long)]
        mic: bool,

        /// Record and transcribe one microphone chunk, preserving the old smoke-test behavior.
        #[arg(long, requires = "mic")]
        once: bool,

        /// Input device index for --mic. Indices come from `puppygrad audio list-input-devices`.
        #[arg(long, requires = "mic")]
        input_device: Option<usize>,

        /// Microphone decode mode.
        #[arg(long, requires = "mic", value_enum, default_value_t = WhisperMicModeArg::Chunks)]
        mic_mode: WhisperMicModeArg,

        /// Seconds per microphone chunk in chunk mode.
        #[arg(long, default_value_t = 4.0)]
        chunk_seconds: f32,

        /// Rolling microphone window size in seconds.
        #[arg(long, requires = "mic", default_value_t = 8.0)]
        window_seconds: f32,

        /// Rolling partial decode interval in milliseconds.
        #[arg(long, requires = "mic", default_value_t = 1000)]
        partial_interval_ms: u64,

        /// Commit rolling partial text after silence.
        #[arg(long, requires = "mic", default_value_t = true, action = clap::ArgAction::Set)]
        commit_on_silence: bool,

        /// Silence duration required before committing rolling text.
        #[arg(long, requires = "mic", default_value_t = 900)]
        silence_ms: u64,

        /// RMS threshold used by the cheap microphone silence gate.
        #[arg(long, requires = "mic", default_value_t = 0.01)]
        silence_threshold: f32,

        /// Captured-audio queue size before decode begins dropping or blocking chunks.
        #[arg(long, requires = "mic", default_value_t = 2)]
        max_queued_chunks: usize,

        /// Queue overflow policy for continuous microphone capture.
        #[arg(long, requires = "mic", value_enum, default_value_t = AudioDropPolicyArg::Oldest)]
        drop_policy: AudioDropPolicyArg,

        /// Whisper checkpoint size.
        #[arg(long, value_enum, default_value_t = WhisperSize::TinyEn)]
        size: WhisperSize,

        /// Local directory containing config.json, tokenizer.json, preprocessor_config.json, and model.safetensors.
        #[arg(long)]
        model_dir: Option<PathBuf>,

        /// Hugging Face model id used with --download.
        #[arg(long)]
        model_id: Option<String>,

        /// Hugging Face revision used with --download.
        #[arg(long, default_value = "main")]
        revision: String,

        /// Download missing model assets into --model-dir before running.
        #[arg(long)]
        download: bool,

        /// Print loaded model and preprocessor dimensions.
        #[arg(long)]
        print_config: bool,

        /// Whisper task.
        #[arg(long, value_enum, default_value_t = WhisperTaskArg::Transcribe)]
        task: WhisperTaskArg,

        /// Language code, for example en.
        #[arg(long)]
        language: Option<String>,

        /// Enable timestamp token generation and decode timestamp-token segment timings.
        #[arg(long, conflicts_with = "no_timestamps")]
        timestamps: bool,

        /// Disable timestamp token generation.
        #[arg(long)]
        no_timestamps: bool,

        /// Do not prepend previous segment text to later segment prompts.
        #[arg(long)]
        no_condition_on_previous_text: bool,

        /// Number of worker threads for Whisper CPU projections and attention heads.
        #[arg(long)]
        threads: Option<usize>,

        /// Execution backend.
        #[arg(long, value_enum, default_value_t = WhisperBackendArg::Rust)]
        backend: WhisperBackendArg,

        /// Use experimental row-wise int8 logits weights.
        #[arg(long)]
        quantized_weights: bool,

        /// Output format.
        #[arg(long, value_enum, default_value_t = WhisperOutputFormatArg::Text)]
        output: WhisperOutputFormatArg,

        /// Stream raw decoder tokens as TSV rows: segment, phase, token id, raw token.
        #[arg(long)]
        stream_raw_tokens: bool,

        /// Skip a segment if the decoder no-speech probability is at least this threshold.
        #[arg(long)]
        no_speech_threshold: Option<f32>,

        #[command(flatten)]
        generation: TextGenerationArgs,

        /// Write first decoder-step logits as little-endian f32 and exit.
        #[arg(long, hide = true)]
        first_logits_out: Option<PathBuf>,

        /// Write the first encoded-audio values as little-endian f32 and exit.
        #[arg(long, hide = true)]
        encoder_slice_out: Option<PathBuf>,

        /// Print timing/profile information to stderr.
        #[arg(long)]
        stats: bool,
    },

    /// Run reproducible performance sweeps.
    Experiment {
        #[command(subcommand)]
        cmd: ExperimentCommand,
    },

    /// Search runtime settings and report the fastest candidate.
    Autotune {
        #[command(subcommand)]
        cmd: AutoTuneCommand,
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
enum AudioCommand {
    /// List available audio input devices.
    ListInputDevices,

    /// Record a fixed-duration clip from an input device to PCM WAV.
    Record {
        /// Input device index from `audio list-input-devices`; omitted means OS default.
        #[arg(long)]
        input_device: Option<usize>,

        /// Recording duration in seconds.
        #[arg(long)]
        seconds: f32,

        /// Output PCM WAV path.
        #[arg(long)]
        out: PathBuf,
    },

    /// Inspect a PCM WAV file.
    Inspect {
        /// WAV path to inspect.
        path: PathBuf,
    },
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

    /// Time Whisper preprocessing, encode, and decode stages.
    Whisper {
        /// Audio WAV path to transcribe.
        #[arg(long)]
        audio: PathBuf,

        /// Whisper checkpoint size.
        #[arg(long, value_enum, default_value_t = WhisperSize::TinyEn)]
        size: WhisperSize,

        /// Local directory containing Whisper assets.
        #[arg(long)]
        model_dir: Option<PathBuf>,

        /// Hugging Face model id used with --download.
        #[arg(long)]
        model_id: Option<String>,

        /// Hugging Face revision used with --download.
        #[arg(long, default_value = "main")]
        revision: String,

        /// Download missing model assets into --model-dir before running.
        #[arg(long)]
        download: bool,

        /// Whisper task.
        #[arg(long, value_enum, default_value_t = WhisperTaskArg::Transcribe)]
        task: WhisperTaskArg,

        /// Language code, for example en.
        #[arg(long)]
        language: Option<String>,

        /// Disable timestamp token generation.
        #[arg(long)]
        no_timestamps: bool,

        /// Number of worker threads for Whisper CPU projections and attention heads.
        #[arg(long)]
        threads: Option<usize>,

        /// New tokens per measured decode.
        #[arg(long, default_value_t = 8)]
        max_new_tokens: usize,

        /// Measured runs.
        #[arg(long, default_value_t = 3)]
        runs: usize,

        /// Warmup runs, excluded from output.
        #[arg(long, default_value_t = 1)]
        warmup_runs: usize,

        /// Output format.
        #[arg(long, value_enum, default_value_t = ExperimentFormatArg::Table)]
        format: ExperimentFormatArg,
    },
}

#[allow(clippy::large_enum_variant)]
#[derive(Subcommand, Debug)]
enum AutoTuneCommand {
    /// Autotune GPT-2 Rust backend settings.
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

        /// Text file with one tuning prompt per non-empty line.
        #[arg(long)]
        prompt_file: Option<PathBuf>,

        /// Comma-separated worker-thread candidates.
        #[arg(long, default_value = "1,2,4,8,12,16,24,32")]
        threads: String,

        /// Comma-separated dense parallel threshold candidates.
        #[arg(long, default_value = "65536,131072,262144,524288")]
        dense_parallel_thresholds: String,

        /// Comma-separated QKV chunk-size candidates.
        #[arg(long, default_value = "32,48,64")]
        qkv_chunk_sizes: String,

        /// Comma-separated attention projection chunk-size candidates.
        #[arg(long, default_value = "48,64,96")]
        attention_projection_chunk_sizes: String,

        /// Comma-separated MLP expansion chunk-size candidates.
        #[arg(long, default_value = "64,128,192")]
        mlp_fc_chunk_sizes: String,

        /// Comma-separated MLP projection chunk-size candidates.
        #[arg(long, default_value = "48,64,96")]
        mlp_projection_chunk_sizes: String,

        /// Comma-separated final logits chunk-size candidates.
        #[arg(long, default_value = "128,256,512")]
        logits_chunk_sizes: String,

        /// Comma-separated attention head parallel threshold candidates.
        #[arg(long, default_value = "1024,4096,16384")]
        attention_head_parallel_thresholds: String,

        /// Also try experimental row-wise int8 weights.
        #[arg(long)]
        include_quantized: bool,

        /// New tokens per tuning trial.
        #[arg(long, default_value_t = 16)]
        max_new_tokens: usize,

        /// Measured runs per candidate.
        #[arg(long, default_value_t = 5)]
        runs: usize,

        /// Warmup runs per candidate.
        #[arg(long, default_value_t = 2)]
        warmup_runs: usize,

        /// Extra measured runs for the selected best config before saving.
        #[arg(long, default_value_t = 7)]
        validation_runs: usize,

        /// Stop after this many candidate configs.
        #[arg(long, default_value_t = 48)]
        max_trials: usize,

        /// Save the best tuning config to this JSON path.
        #[arg(long)]
        save_tuning: Option<PathBuf>,
    },

    /// Autotune Whisper decode length candidates for the current reference path.
    Whisper {
        /// Audio WAV path to transcribe.
        #[arg(long)]
        audio: PathBuf,

        /// Whisper checkpoint size.
        #[arg(long, value_enum, default_value_t = WhisperSize::TinyEn)]
        size: WhisperSize,

        /// Local directory containing Whisper assets.
        #[arg(long)]
        model_dir: Option<PathBuf>,

        /// Hugging Face model id used with --download.
        #[arg(long)]
        model_id: Option<String>,

        /// Hugging Face revision used with --download.
        #[arg(long, default_value = "main")]
        revision: String,

        /// Download missing model assets into --model-dir before running.
        #[arg(long)]
        download: bool,

        /// Whisper task.
        #[arg(long, value_enum, default_value_t = WhisperTaskArg::Transcribe)]
        task: WhisperTaskArg,

        /// Language code, for example en.
        #[arg(long)]
        language: Option<String>,

        /// Disable timestamp token generation.
        #[arg(long)]
        no_timestamps: bool,

        /// Number of worker threads for Whisper CPU projections and attention heads.
        #[arg(long)]
        threads: Option<usize>,

        /// Comma-separated max-new-token candidates.
        #[arg(long, default_value = "1,2,4,8")]
        max_new_tokens: String,

        /// Measured runs per candidate.
        #[arg(long, default_value_t = 2)]
        runs: usize,

        /// Warmup runs per candidate.
        #[arg(long, default_value_t = 1)]
        warmup_runs: usize,

        /// Stop after this many candidate configs.
        #[arg(long, default_value_t = 4)]
        max_trials: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Audio { cmd } => run_audio(cmd),
        Command::Gpt2 {
            model_dir,
            model_id,
            revision,
            download,
            prompt,
            generation,
            eos_token_id,
            no_eos_stop,
            backend,
            threads,
            tuning_file,
            no_tuning,
            dense_parallel_threshold,
            qkv_chunk_size,
            attention_projection_chunk_size,
            mlp_fc_chunk_size,
            mlp_projection_chunk_size,
            logits_chunk_size,
            attention_head_parallel_threshold,
            quantized_weights,
            stats,
            stream_raw_tokens,
        } => run_gpt2(RunGpt2Args {
            model_dir,
            model_id,
            revision,
            download,
            prompt,
            generation,
            eos_token_id,
            no_eos_stop,
            backend,
            threads,
            tuning_file,
            no_tuning,
            tuning: RustTuning {
                dense_parallel_threshold,
                qkv_chunk_size,
                attention_projection_chunk_size,
                mlp_fc_chunk_size,
                mlp_projection_chunk_size,
                logits_chunk_size,
                attention_head_parallel_threshold,
                quantized_weights: quantized_weights.then_some(true),
            },
            stats,
            stream_raw_tokens,
        }),
        Command::Resnet {
            image,
            variant,
            model_dir,
            download,
            labels,
            top_k,
            threads,
        } => run_resnet(RunResNetArgs {
            image,
            variant,
            model_dir,
            download,
            labels,
            top_k,
            threads,
        }),
        Command::Qwen {
            model_dir,
            model_id,
            revision,
            download,
            prompt,
            generation,
            dtype,
            instruct,
        } => run_qwen(RunQwenArgs {
            model_dir,
            model_id,
            revision,
            download,
            prompt,
            generation,
            dtype,
            instruct,
        }),
        Command::Whisper {
            audio,
            mic,
            once,
            input_device,
            mic_mode,
            chunk_seconds,
            window_seconds,
            partial_interval_ms,
            commit_on_silence,
            silence_ms,
            silence_threshold,
            max_queued_chunks,
            drop_policy,
            size,
            model_dir,
            model_id,
            revision,
            download,
            print_config,
            task,
            language,
            timestamps,
            no_timestamps,
            no_condition_on_previous_text,
            threads,
            backend,
            quantized_weights,
            output,
            stream_raw_tokens,
            no_speech_threshold,
            generation,
            first_logits_out,
            encoder_slice_out,
            stats,
        } => run_whisper(RunWhisperArgs {
            audio,
            mic,
            once,
            input_device,
            mic_mode,
            chunk_seconds,
            window_seconds,
            partial_interval_ms,
            commit_on_silence,
            silence_ms,
            silence_threshold,
            max_queued_chunks,
            drop_policy,
            size,
            model_dir,
            model_id,
            revision,
            download,
            print_config,
            task,
            language,
            timestamps,
            no_timestamps,
            no_condition_on_previous_text,
            threads,
            backend,
            quantized_weights,
            output,
            stream_raw_tokens,
            no_speech_threshold,
            generation,
            first_logits_out,
            encoder_slice_out,
            stats,
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
                    quantized_weights: Some(quantized_weights),
                },
                max_new_tokens,
                runs,
                warmup_runs,
                format,
            }),
            ExperimentCommand::Whisper {
                audio,
                size,
                model_dir,
                model_id,
                revision,
                download,
                task,
                language,
                no_timestamps,
                threads,
                max_new_tokens,
                runs,
                warmup_runs,
                format,
            } => run_whisper_experiment(RunWhisperExperimentArgs {
                audio,
                size,
                model_dir,
                model_id,
                revision,
                download,
                task,
                language,
                no_timestamps,
                threads,
                max_new_tokens,
                runs,
                warmup_runs,
                format,
            }),
        },
        Command::Autotune { cmd } => match cmd {
            AutoTuneCommand::Gpt2 {
                model_dir,
                model_id,
                revision,
                download,
                prompt,
                prompt_file,
                threads,
                dense_parallel_thresholds,
                qkv_chunk_sizes,
                attention_projection_chunk_sizes,
                mlp_fc_chunk_sizes,
                mlp_projection_chunk_sizes,
                logits_chunk_sizes,
                attention_head_parallel_thresholds,
                include_quantized,
                max_new_tokens,
                runs,
                warmup_runs,
                validation_runs,
                max_trials,
                save_tuning,
            } => run_gpt2_autotune(RunGpt2AutoTuneArgs {
                model_dir,
                model_id,
                revision,
                download,
                prompt,
                prompt_file,
                threads,
                dense_parallel_thresholds,
                qkv_chunk_sizes,
                attention_projection_chunk_sizes,
                mlp_fc_chunk_sizes,
                mlp_projection_chunk_sizes,
                logits_chunk_sizes,
                attention_head_parallel_thresholds,
                include_quantized,
                max_new_tokens,
                runs,
                warmup_runs,
                validation_runs,
                max_trials,
                save_tuning,
            }),
            AutoTuneCommand::Whisper {
                audio,
                size,
                model_dir,
                model_id,
                revision,
                download,
                task,
                language,
                no_timestamps,
                threads,
                max_new_tokens,
                runs,
                warmup_runs,
                max_trials,
            } => run_whisper_autotune(RunWhisperAutoTuneArgs {
                audio,
                size,
                model_dir,
                model_id,
                revision,
                download,
                task,
                language,
                no_timestamps,
                threads,
                max_new_tokens,
                runs,
                warmup_runs,
                max_trials,
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
enum ResNetVariantArg {
    Resnet18,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum WhisperBackendArg {
    Rust,
    Gpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ExperimentFormatArg {
    Table,
    Csv,
    Json,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum WhisperTaskArg {
    Transcribe,
    Translate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum WhisperOutputFormatArg {
    Text,
    Json,
    EventsJson,
    Srt,
    Vtt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum WhisperMicModeArg {
    Chunks,
    Rolling,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum AudioDropPolicyArg {
    Oldest,
    Newest,
    Block,
}

impl From<AudioDropPolicyArg> for RuntimeAudioDropPolicy {
    fn from(value: AudioDropPolicyArg) -> Self {
        match value {
            AudioDropPolicyArg::Oldest => RuntimeAudioDropPolicy::Oldest,
            AudioDropPolicyArg::Newest => RuntimeAudioDropPolicy::Newest,
            AudioDropPolicyArg::Block => RuntimeAudioDropPolicy::Block,
        }
    }
}

impl From<WhisperTaskArg> for RuntimeWhisperTask {
    fn from(value: WhisperTaskArg) -> Self {
        match value {
            WhisperTaskArg::Transcribe => RuntimeWhisperTask::Transcribe,
            WhisperTaskArg::Translate => RuntimeWhisperTask::Translate,
        }
    }
}

struct RunGpt2Args {
    model_dir: Option<PathBuf>,
    model_id: String,
    revision: String,
    download: bool,
    prompt: String,
    generation: TextGenerationArgs,
    eos_token_id: Option<usize>,
    no_eos_stop: bool,
    backend: Gpt2BackendArg,
    threads: Option<usize>,
    tuning_file: Option<PathBuf>,
    no_tuning: bool,
    tuning: RustTuning,
    stats: bool,
    stream_raw_tokens: bool,
}

struct RunResNetArgs {
    image: PathBuf,
    variant: ResNetVariantArg,
    model_dir: Option<PathBuf>,
    download: bool,
    labels: Option<PathBuf>,
    top_k: usize,
    threads: Option<usize>,
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
    quantized_weights: Option<bool>,
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

struct RunWhisperExperimentArgs {
    audio: PathBuf,
    size: WhisperSize,
    model_dir: Option<PathBuf>,
    model_id: Option<String>,
    revision: String,
    download: bool,
    task: WhisperTaskArg,
    language: Option<String>,
    no_timestamps: bool,
    threads: Option<usize>,
    max_new_tokens: usize,
    runs: usize,
    warmup_runs: usize,
    format: ExperimentFormatArg,
}

struct RunGpt2AutoTuneArgs {
    model_dir: Option<PathBuf>,
    model_id: String,
    revision: String,
    download: bool,
    prompt: Option<String>,
    prompt_file: Option<PathBuf>,
    threads: String,
    dense_parallel_thresholds: String,
    qkv_chunk_sizes: String,
    attention_projection_chunk_sizes: String,
    mlp_fc_chunk_sizes: String,
    mlp_projection_chunk_sizes: String,
    logits_chunk_sizes: String,
    attention_head_parallel_thresholds: String,
    include_quantized: bool,
    max_new_tokens: usize,
    runs: usize,
    warmup_runs: usize,
    validation_runs: usize,
    max_trials: usize,
    save_tuning: Option<PathBuf>,
}

struct RunWhisperAutoTuneArgs {
    audio: PathBuf,
    size: WhisperSize,
    model_dir: Option<PathBuf>,
    model_id: Option<String>,
    revision: String,
    download: bool,
    task: WhisperTaskArg,
    language: Option<String>,
    no_timestamps: bool,
    threads: Option<usize>,
    max_new_tokens: String,
    runs: usize,
    warmup_runs: usize,
    max_trials: usize,
}

struct RunQwenArgs {
    model_dir: Option<PathBuf>,
    model_id: String,
    revision: String,
    download: bool,
    prompt: String,
    generation: TextGenerationArgs,
    dtype: Option<String>,
    instruct: bool,
}

struct RunWhisperArgs {
    audio: Option<PathBuf>,
    mic: bool,
    once: bool,
    input_device: Option<usize>,
    mic_mode: WhisperMicModeArg,
    chunk_seconds: f32,
    window_seconds: f32,
    partial_interval_ms: u64,
    commit_on_silence: bool,
    silence_ms: u64,
    silence_threshold: f32,
    max_queued_chunks: usize,
    drop_policy: AudioDropPolicyArg,
    size: WhisperSize,
    model_dir: Option<PathBuf>,
    model_id: Option<String>,
    revision: String,
    download: bool,
    print_config: bool,
    task: WhisperTaskArg,
    language: Option<String>,
    timestamps: bool,
    no_timestamps: bool,
    no_condition_on_previous_text: bool,
    threads: Option<usize>,
    backend: WhisperBackendArg,
    quantized_weights: bool,
    output: WhisperOutputFormatArg,
    stream_raw_tokens: bool,
    no_speech_threshold: Option<f32>,
    generation: TextGenerationArgs,
    first_logits_out: Option<PathBuf>,
    encoder_slice_out: Option<PathBuf>,
    stats: bool,
}

#[derive(Debug, Serialize)]
struct WhisperSegmentOutput {
    start: f32,
    end: f32,
    text: String,
}

fn run_audio(cmd: AudioCommand) -> Result<()> {
    match cmd {
        AudioCommand::ListInputDevices => {
            let devices = list_input_devices()?;
            if devices.is_empty() {
                println!("no input devices found");
                return Ok(());
            }
            for device in devices {
                let default = if device.is_default { " default" } else { "" };
                println!("{}\t{}{}", device.index, device.name, default);
            }
            Ok(())
        }
        AudioCommand::Record {
            input_device,
            seconds,
            out,
        } => {
            let duration = audio_duration(seconds, "seconds")?;
            eprintln!(
                "recording {} from {}",
                format_duration(duration),
                input_device
                    .map(|index| format!("input device {index}"))
                    .unwrap_or_else(|| "default input device".to_string())
            );
            let audio = record_input_device(input_device, duration)?;
            write_wav_pcm16(&out, &audio)?;
            eprintln!(
                "wrote {}: {} Hz, {} channel(s), {:.3}s, {} samples",
                out.display(),
                audio.sample_rate,
                audio.channels,
                audio.duration_seconds(),
                audio.samples.len()
            );
            Ok(())
        }
        AudioCommand::Inspect { path } => {
            let info = inspect_wav(&path)?;
            println!("path: {}", path.display());
            println!("format: {}", info.format);
            println!("sample_rate: {}", info.sample_rate);
            println!("channels: {}", info.channels);
            println!("duration_seconds: {:.3}", info.duration_seconds);
            println!("sample_count: {}", info.sample_count);
            Ok(())
        }
    }
}

fn run_gpt2(args: RunGpt2Args) -> Result<()> {
    let mut generation = Gpt2GenerationConfig::from_args(&args.generation);
    if args.no_eos_stop {
        generation.eos_token_id = None;
    } else if let Some(eos_token_id) = args.eos_token_id {
        generation.eos_token_id = Some(eos_token_id);
    }
    generation.validate()?;
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

    let tuning_base = if args.no_tuning {
        Gpt2RustConfig::default()
    } else {
        load_gpt2_tuning(args.tuning_file.as_ref(), &model_dir)?.unwrap_or_default()
    };
    let backend = match args.backend {
        Gpt2BackendArg::Rust => {
            Gpt2BackendConfig::Rust(rust_config_from(tuning_base, args.threads, args.tuning)?)
        }
    };

    eprintln!("backend: {}", backend.describe());
    eprintln!("loading GPT-2 from {}", model_dir.display());
    let load_start = Instant::now();
    let runtime = Gpt2Runtime::from_dir_with_backend(&model_dir, backend)?;
    let load_time = load_start.elapsed();
    let mut stdout = std::io::stdout().lock();
    let generation_stats = if args.stream_raw_tokens {
        let prompt_tokens = runtime.tokenizer.encode(&args.prompt)?;
        for token_id in &prompt_tokens {
            write_raw_token_event(&mut stdout, None, "prompt", *token_id, &runtime.tokenizer)?;
        }
        stdout.flush()?;
        let (_, generation_stats) =
            runtime.stream_token_ids_with_stats(&args.prompt, &generation, |token_id| {
                write_raw_token_event(
                    &mut stdout,
                    None,
                    "generated",
                    token_id,
                    &runtime.tokenizer,
                )?;
                stdout.flush()?;
                Ok::<(), Box<dyn std::error::Error>>(())
            })?;
        generation_stats
    } else {
        let (_, generation_stats) =
            runtime.stream_text_with_stats(&args.prompt, &generation, |text| {
                write!(stdout, "{text}")?;
                stdout.flush()?;
                Ok::<(), Box<dyn std::error::Error>>(())
            })?;
        writeln!(stdout)?;
        generation_stats
    };
    if args.stream_raw_tokens {
        stdout.flush()?;
    }
    if args.stats {
        print_gpt2_stats(load_time, &generation_stats);
    } else {
        print_gpt2_speed(&generation_stats);
    }
    Ok(())
}

fn run_resnet(args: RunResNetArgs) -> Result<()> {
    match args.variant {
        ResNetVariantArg::Resnet18 => {}
    }
    if let Some(threads) = args.threads {
        eprintln!(
            "--threads {threads} is accepted but the current ResNet CPU path is single-threaded"
        );
    }
    if args.top_k == 0 {
        return Err("--top-k must be greater than 0".into());
    }
    let model_dir = args.model_dir.unwrap_or_else(default_resnet18_dir);
    if args.download {
        eprintln!(
            "downloading missing ResNet-18 assets into {}",
            model_dir.display()
        );
        download_resnet18_assets(&model_dir)?;
    }

    eprintln!("loading ResNet-18 from {}", model_dir.display());
    let runtime = ResNetRuntime::from_dir(&model_dir, args.labels.as_deref())?;
    let classifications = runtime.classify_image(&args.image, args.top_k)?;
    let mut stdout = std::io::stdout().lock();
    for item in classifications {
        writeln!(
            stdout,
            "{}\t{:.6}\t{:.6}\t{}",
            item.class_index, item.probability, item.logit, item.label
        )?;
    }
    Ok(())
}

fn run_whisper(args: RunWhisperArgs) -> Result<()> {
    let model_dir = args
        .model_dir
        .clone()
        .unwrap_or_else(|| default_whisper_dir(args.size));
    if args.download {
        eprintln!(
            "downloading/checking Whisper assets for {} into {}",
            args.size,
            model_dir.display()
        );
    }

    let backend_threads = if args.mic && args.threads.is_none() {
        std::thread::available_parallelism()
            .ok()
            .map(|threads| threads.get().clamp(1, 8))
    } else {
        args.threads
    };
    let backend = whisper_backend_config(
        args.size,
        backend_threads,
        args.backend,
        args.quantized_weights,
    )?;
    let load_start = Instant::now();
    let runtime = WhisperRuntime::prepare_from_huggingface_with_backend(
        args.size,
        args.model_id.as_deref(),
        &args.revision,
        &model_dir,
        args.download,
        backend,
    )?;
    let load_time = load_start.elapsed();

    if args.print_config || (args.audio.is_none() && !args.mic) {
        for line in runtime.metadata_lines() {
            println!("{line}");
        }
        if args.audio.is_none() && !args.mic {
            if args.stats {
                eprintln!("stats:");
                eprintln!("  load: {}", format_duration(load_time));
            }
            return Ok(());
        }
    }

    let no_timestamps = args.no_timestamps || !args.timestamps;
    let prefix = runtime.tokenizer.prompt_prefix(
        args.task.into(),
        args.language.as_deref(),
        no_timestamps,
        args.size,
    )?;

    if args.mic {
        return run_whisper_mic(&args, &runtime, prefix, load_time, no_timestamps);
    }

    let audio_start = Instant::now();
    let audio_path = args.audio.as_ref().expect("checked above");
    let audio = load_audio_arg(audio_path)?;
    if audio.sample_rate != WHISPER_SAMPLE_RATE {
        return Err(format!(
            "{} has sample rate {} Hz; native Whisper currently requires {WHISPER_SAMPLE_RATE} Hz WAV input",
            audio_path.display(),
            audio.sample_rate
        )
        .into());
    }
    let audio_time = audio_start.elapsed();

    let mut generation_template = args.generation.to_config_with_default(1);
    generation_template.eos_token_id = Some(runtime.tokenizer.special_tokens().eos);
    generation_template.validate()?;
    let mut profile = WhisperOperationProfile::default();
    let mut features_time = Duration::ZERO;
    let mut encode_time = Duration::ZERO;
    let mut prefill_time = Duration::ZERO;
    let mut decode_time = Duration::ZERO;
    let mut generated_tokens = 0usize;
    let mut segments = Vec::new();
    let mut previous_text_tokens = Vec::new();
    let chunk_samples = runtime.preprocessor.n_samples;
    let total_samples = audio.samples.len().max(1);
    let mut stdout = std::io::stdout().lock();

    for (segment_index, start_sample) in (0..total_samples).step_by(chunk_samples).enumerate() {
        let end_sample = (start_sample + chunk_samples).min(audio.samples.len());
        let chunk = &audio.samples[start_sample..end_sample];
        let features_start = Instant::now();
        let features = log_mel_spectrogram(chunk, &runtime.preprocessor)?;
        features_time += features_start.elapsed();

        let encode_start = Instant::now();
        let encoded = runtime.encode_audio(&features, &mut profile)?;
        encode_time += encode_start.elapsed();

        if let Some(path) = args.encoder_slice_out.as_ref() {
            if segment_index == 0 {
                let take = encoded.values.len().min(64);
                let bytes = encoded.values[..take]
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>();
                fs::write(path, bytes)?;
                if args.stats {
                    eprintln!("stats:");
                    eprintln!("  load: {}", format_duration(load_time));
                    eprintln!(
                        "  audio preprocessing: {}",
                        format_duration(audio_time + features_time)
                    );
                    eprintln!("    wav decode: {}", format_duration(audio_time));
                    eprintln!("    log-mel: {}", format_duration(features_time));
                    eprintln!("  encode: {}", format_duration(encode_time));
                    eprintln!("  encoder slice: {}", path.display());
                }
                return Ok(());
            }
        }

        let mut segment_prompt = prefix.clone();
        let mut segment_generation = generation_template.clone();
        let default_max_new_tokens = runtime
            .config
            .n_text_ctx
            .saturating_sub(segment_prompt.len());
        let default_segment_max_new_tokens = if args.mic {
            default_max_new_tokens.min(2)
        } else {
            default_max_new_tokens
        };
        segment_generation.max_new_tokens = args
            .generation
            .max_new_tokens
            .unwrap_or(default_segment_max_new_tokens);
        segment_generation.max_new_tokens = segment_generation
            .max_new_tokens
            .min(default_max_new_tokens);

        if !args.no_condition_on_previous_text && !previous_text_tokens.is_empty() {
            let available = runtime
                .config
                .n_text_ctx
                .saturating_sub(segment_generation.max_new_tokens)
                .saturating_sub(segment_prompt.len());
            let keep = available.min(previous_text_tokens.len()).min(224);
            segment_prompt
                .extend_from_slice(&previous_text_tokens[previous_text_tokens.len() - keep..]);
        }
        segment_generation.max_new_tokens = segment_generation.max_new_tokens.min(
            runtime
                .config
                .n_text_ctx
                .saturating_sub(segment_prompt.len()),
        );
        segment_generation.validate()?;

        let prefill_start = Instant::now();
        let first_logits = runtime.decoder_logits(&encoded, &segment_prompt, &mut profile)?;
        prefill_time += prefill_start.elapsed();

        if let Some(path) = args.first_logits_out.as_ref() {
            if segment_index == 0 {
                let bytes = first_logits
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>();
                fs::write(path, bytes)?;
                if args.stats {
                    eprintln!("stats:");
                    eprintln!("  load: {}", format_duration(load_time));
                    eprintln!(
                        "  audio preprocessing: {}",
                        format_duration(audio_time + features_time)
                    );
                    eprintln!("    wav decode: {}", format_duration(audio_time));
                    eprintln!("    log-mel: {}", format_duration(features_time));
                    eprintln!("  encode: {}", format_duration(encode_time));
                    eprintln!(
                        "  prefill: {} ({} prompt tokens)",
                        format_duration(prefill_time),
                        segment_prompt.len()
                    );
                    eprintln!("  first logits: {}", path.display());
                }
                return Ok(());
            }
        }

        let no_speech_probability = runtime
            .tokenizer
            .special_tokens()
            .no_speech
            .map(|token_id| logits_probability(&first_logits, token_id))
            .unwrap_or(0.0);
        if args
            .no_speech_threshold
            .is_some_and(|threshold| no_speech_probability >= threshold)
        {
            segments.push(WhisperSegmentOutput {
                start: start_sample as f32 / audio.sample_rate as f32,
                end: end_sample as f32 / audio.sample_rate as f32,
                text: String::new(),
            });
            continue;
        }

        let decode_start = Instant::now();
        let output_tokens = if args.stream_raw_tokens {
            let raw_tokenizer = runtime.tokenizer.clone();
            for token_id in &segment_prompt {
                write_raw_token_event(
                    &mut stdout,
                    Some(segment_index),
                    "prompt",
                    *token_id,
                    &raw_tokenizer,
                )?;
            }
            stdout.flush()?;
            runtime.stream_greedy_tokens(
                &encoded,
                &segment_prompt,
                &segment_generation,
                no_timestamps,
                &mut profile,
                |token_id| {
                    write_raw_token_event(
                        &mut stdout,
                        Some(segment_index),
                        "generated",
                        token_id,
                        &raw_tokenizer,
                    )?;
                    stdout.flush()?;
                    Ok::<(), Box<dyn std::error::Error>>(())
                },
            )?
        } else {
            runtime.generate_greedy(
                &encoded,
                &segment_prompt,
                &segment_generation,
                no_timestamps,
                &mut profile,
            )?
        };
        decode_time += decode_start.elapsed();
        let segment_generated = output_tokens.len().saturating_sub(segment_prompt.len());
        generated_tokens += segment_generated;
        let generated = &output_tokens[segment_prompt.len()..];
        let window_start = start_sample as f32 / audio.sample_rate as f32;
        let window_end = end_sample as f32 / audio.sample_rate as f32;
        let mut decoded_segments = if no_timestamps {
            let transcript = runtime.tokenizer.decode(generated)?;
            vec![WhisperSegmentOutput {
                start: window_start,
                end: window_end,
                text: transcript.trim().to_string(),
            }]
        } else {
            decode_timestamped_segments(&runtime, generated, window_start, window_end)?
        };
        for segment in &decoded_segments {
            if !segment.text.is_empty() {
                previous_text_tokens.extend(runtime.tokenizer.encode(&segment.text)?);
            }
        }
        if args.mic && !args.stream_raw_tokens && args.output == WhisperOutputFormatArg::Text {
            let transcript = decoded_segments
                .iter()
                .map(|segment| segment.text.as_str())
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            if !transcript.is_empty() {
                writeln!(stdout, "{transcript}")?;
                stdout.flush()?;
            }
        }
        segments.append(&mut decoded_segments);
    }

    if !args.stream_raw_tokens && !(args.mic && args.output == WhisperOutputFormatArg::Text) {
        match args.output {
            WhisperOutputFormatArg::Text => {
                let transcript = segments
                    .iter()
                    .map(|segment| segment.text.as_str())
                    .filter(|text| !text.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ");
                writeln!(stdout, "{transcript}")?;
            }
            WhisperOutputFormatArg::Json => {
                serde_json::to_writer_pretty(&mut stdout, &segments)?;
                writeln!(stdout)?;
            }
            WhisperOutputFormatArg::EventsJson => {
                for segment in segments.iter().filter(|segment| !segment.text.is_empty()) {
                    write_event(
                        &mut stdout,
                        &TranscriptionEvent::Commit {
                            text: segment.text.clone(),
                        },
                    )?;
                }
            }
            WhisperOutputFormatArg::Srt => {
                for (i, segment) in segments
                    .iter()
                    .filter(|segment| !segment.text.is_empty())
                    .enumerate()
                {
                    writeln!(stdout, "{}", i + 1)?;
                    writeln!(
                        stdout,
                        "{} --> {}",
                        format_srt_timestamp(segment.start),
                        format_srt_timestamp(segment.end)
                    )?;
                    writeln!(stdout, "{}\n", segment.text)?;
                }
            }
            WhisperOutputFormatArg::Vtt => {
                writeln!(stdout, "WEBVTT\n")?;
                for segment in segments.iter().filter(|segment| !segment.text.is_empty()) {
                    writeln!(
                        stdout,
                        "{} --> {}",
                        format_vtt_timestamp(segment.start),
                        format_vtt_timestamp(segment.end)
                    )?;
                    writeln!(stdout, "{}\n", segment.text)?;
                }
            }
        }
    }
    stdout.flush()?;

    if args.stats {
        eprintln!("stats:");
        eprintln!("  load: {}", format_duration(load_time));
        eprintln!(
            "  audio preprocessing: {}",
            format_duration(audio_time + features_time)
        );
        eprintln!("    wav decode: {}", format_duration(audio_time));
        eprintln!("    log-mel: {}", format_duration(features_time));
        eprintln!("  encode: {}", format_duration(encode_time));
        eprintln!(
            "  prefill: {} ({} prompt tokens)",
            format_duration(prefill_time),
            prefix.len()
        );
        eprintln!(
            "  decode: {} ({} generated tokens)",
            format_duration(decode_time),
            generated_tokens
        );
        eprintln!("  profile:");
        eprintln!(
            "    audio projection: {}",
            format_duration(profile.audio_projection)
        );
        eprintln!(
            "    encoder attention: {}",
            format_duration(profile.encoder_attention)
        );
        eprintln!("    encoder mlp: {}", format_duration(profile.encoder_mlp));
        eprintln!(
            "    encoder layernorm: {}",
            format_duration(profile.encoder_layer_norm)
        );
        eprintln!(
            "    decoder self-attention: {}",
            format_duration(profile.decoder_self_attention)
        );
        eprintln!(
            "    decoder cross-attention: {}",
            format_duration(profile.decoder_cross_attention)
        );
        eprintln!("    decoder mlp: {}", format_duration(profile.decoder_mlp));
        eprintln!(
            "    decoder layernorm: {}",
            format_duration(profile.decoder_layer_norm)
        );
        eprintln!(
            "    final logits: {}",
            format_duration(profile.final_logits)
        );
        eprintln!(
            "  total: {}",
            format_duration(
                load_time + audio_time + features_time + encode_time + prefill_time + decode_time
            )
        );
    }
    Ok(())
}

struct RealtimeDecodeOutput {
    text: String,
    generated_tokens: usize,
    no_speech_probability: f32,
    features_time: Duration,
    encode_time: Duration,
    prefill_time: Duration,
    decode_time: Duration,
}

fn run_whisper_mic(
    args: &RunWhisperArgs,
    runtime: &WhisperRuntime,
    prefix: Vec<usize>,
    load_time: Duration,
    no_timestamps: bool,
) -> Result<()> {
    validate_whisper_mic_args(args)?;

    let running = Arc::new(AtomicBool::new(true));
    let signal_running = running.clone();
    ctrlc::set_handler(move || {
        signal_running.store(false, Ordering::SeqCst);
    })?;

    let mut generation_template = args.generation.to_config_with_default(64);
    generation_template.eos_token_id = Some(runtime.tokenizer.special_tokens().eos);
    generation_template.validate()?;

    let mut stdout = std::io::stdout().lock();
    let mut profile = WhisperOperationProfile::default();
    let mut totals = RealtimeDecodeOutput {
        text: String::new(),
        generated_tokens: 0,
        no_speech_probability: 0.0,
        features_time: Duration::ZERO,
        encode_time: Duration::ZERO,
        prefill_time: Duration::ZERO,
        decode_time: Duration::ZERO,
    };

    if args.once {
        let duration = audio_duration(args.chunk_seconds, "chunk-seconds")?;
        eprintln!(
            "recording one {} chunk from {}",
            format_duration(duration),
            args.input_device
                .map(|index| format!("input device {index}"))
                .unwrap_or_else(|| "default input device".to_string())
        );
        let recorded = record_input_device(args.input_device, duration)?;
        let resampled = resample_linear(&recorded, WHISPER_SAMPLE_RATE)?;
        let output = decode_mic_samples(
            runtime,
            &prefix,
            args,
            &generation_template,
            no_timestamps,
            &mut profile,
            &mut stdout,
            0,
            &resampled.samples,
            0.0,
            TranscriptionJobPurpose::Final,
        )?;
        emit_commit(&mut stdout, args, &output.text, &mut 0)?;
        totals = output;
        print_realtime_stats(args, load_time, &totals, &profile, 0, 0, Duration::ZERO);
        return Ok(());
    }

    let stream = start_input_device_stream(
        args.input_device,
        args.max_queued_chunks,
        args.drop_policy.into(),
    )?;
    eprintln!(
        "listening continuously from {}; input {} Hz, {} channel(s), mode {:?}",
        args.input_device
            .map(|index| format!("input device {index}"))
            .unwrap_or_else(|| "default input device".to_string()),
        stream.sample_rate(),
        stream.channels(),
        args.mic_mode
    );

    match args.mic_mode {
        WhisperMicModeArg::Chunks => run_whisper_mic_chunks(
            args,
            runtime,
            &prefix,
            &generation_template,
            no_timestamps,
            &running,
            &stream,
            &mut stdout,
            &mut profile,
            &mut totals,
        )?,
        WhisperMicModeArg::Rolling => run_whisper_mic_rolling(
            args,
            runtime,
            &prefix,
            &generation_template,
            no_timestamps,
            &running,
            &stream,
            &mut stdout,
            &mut profile,
            &mut totals,
        )?,
    }

    print_realtime_stats(
        args,
        load_time,
        &totals,
        &profile,
        stream.dropped_chunks(),
        stream.queued_chunks(),
        Duration::ZERO,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_whisper_mic_chunks<W: Write>(
    args: &RunWhisperArgs,
    runtime: &WhisperRuntime,
    prefix: &[usize],
    generation_template: &TextGenerationConfig,
    no_timestamps: bool,
    running: &AtomicBool,
    stream: &puppygrad::audio::ContinuousInputStream,
    stdout: &mut W,
    profile: &mut WhisperOperationProfile,
    totals: &mut RealtimeDecodeOutput,
) -> Result<()> {
    let target_native_samples = seconds_to_samples(stream.sample_rate(), args.chunk_seconds);
    let mut segment_index = 0usize;
    let mut collected = Vec::with_capacity(target_native_samples);
    let mut window_start = 0.0f32;
    while running.load(Ordering::SeqCst) {
        match stream.recv_timeout(Duration::from_millis(200))? {
            Some(chunk) => collected.extend(chunk),
            None => continue,
        }
        emit_queue_warnings(stdout, args, stream)?;
        while collected.len() >= target_native_samples {
            let native = collected
                .drain(..target_native_samples)
                .collect::<Vec<f32>>();
            let audio = SharedPcmAudio {
                path: PathBuf::from("<microphone>"),
                sample_rate: stream.sample_rate(),
                channels: 1,
                samples: native,
            };
            let resampled = resample_linear(&audio, WHISPER_SAMPLE_RATE)?;
            let decode_start = Instant::now();
            let output = decode_mic_samples(
                runtime,
                prefix,
                args,
                generation_template,
                no_timestamps,
                profile,
                stdout,
                segment_index,
                &resampled.samples,
                window_start,
                TranscriptionJobPurpose::Final,
            )?;
            if decode_start.elapsed().as_secs_f32() > args.chunk_seconds {
                emit_warning(
                    stdout,
                    args,
                    "decode is falling behind realtime; captured audio may queue or drop",
                )?;
            }
            if output.no_speech_probability < args.no_speech_threshold.unwrap_or(f32::INFINITY) {
                emit_commit(stdout, args, &output.text, &mut 0)?;
            } else {
                emit_silence(stdout, args)?;
            }
            add_realtime_totals(totals, &output);
            segment_index += 1;
            window_start += args.chunk_seconds;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_whisper_mic_rolling<W: Write>(
    args: &RunWhisperArgs,
    runtime: &WhisperRuntime,
    prefix: &[usize],
    generation_template: &TextGenerationConfig,
    no_timestamps: bool,
    running: &AtomicBool,
    stream: &puppygrad::audio::ContinuousInputStream,
    stdout: &mut W,
    profile: &mut WhisperOperationProfile,
    totals: &mut RealtimeDecodeOutput,
) -> Result<()> {
    let mut rolling = RollingAudioBuffer::new(WHISPER_SAMPLE_RATE, args.window_seconds);
    let mut partial_state = PartialCommitState::new(3);
    let mut last_decode = Instant::now();
    let partial_interval = Duration::from_millis(args.partial_interval_ms);
    let silence_duration = Duration::from_millis(args.silence_ms);
    let mut last_voice = Instant::now();
    let mut segment_index = 0usize;
    let mut visible_partial_width = 0usize;
    let mut last_committed_normalized = String::new();

    while running.load(Ordering::SeqCst) {
        if let Some(chunk) = stream.recv_timeout(Duration::from_millis(100))? {
            let audio = SharedPcmAudio {
                path: PathBuf::from("<microphone>"),
                sample_rate: stream.sample_rate(),
                channels: 1,
                samples: chunk,
            };
            let resampled = resample_linear(&audio, WHISPER_SAMPLE_RATE)?;
            let silent = is_silence(&resampled.samples, args.silence_threshold);
            if !silent {
                last_voice = Instant::now();
            }
            rolling.append(&resampled.samples);
        }
        emit_queue_warnings(stdout, args, stream)?;

        if args.commit_on_silence
            && last_voice.elapsed() >= silence_duration
            && rolling.duration_seconds() > 0.0
        {
            if let Some(text) = partial_state.commit_active() {
                clear_visible_partial(stdout, args, &mut visible_partial_width)?;
                emit_rolling_commit_once(
                    stdout,
                    args,
                    &text,
                    &mut visible_partial_width,
                    &mut last_committed_normalized,
                )?;
            } else {
                emit_silence(stdout, args)?;
            }
            rolling.clear();
            last_voice = Instant::now();
            continue;
        }

        if rolling.samples().is_empty() || last_decode.elapsed() < partial_interval {
            continue;
        }
        last_decode = Instant::now();
        if is_silence(rolling.samples(), args.silence_threshold) {
            emit_silence(stdout, args)?;
            continue;
        }
        let output = decode_mic_samples(
            runtime,
            prefix,
            args,
            generation_template,
            no_timestamps,
            profile,
            stdout,
            segment_index,
            rolling.samples(),
            rolling.window_start_seconds(),
            TranscriptionJobPurpose::Partial,
        )?;
        add_realtime_totals(totals, &output);
        segment_index += 1;
        if args
            .no_speech_threshold
            .is_some_and(|threshold| output.no_speech_probability >= threshold)
        {
            emit_silence(stdout, args)?;
            continue;
        }
        match partial_state.observe_partial(output.text.trim()) {
            PartialObservation::Partial(text) => {
                emit_partial(stdout, args, &text, &mut visible_partial_width)?;
            }
            PartialObservation::Commit(text) => {
                clear_visible_partial(stdout, args, &mut visible_partial_width)?;
                emit_rolling_commit_once(
                    stdout,
                    args,
                    &text,
                    &mut visible_partial_width,
                    &mut last_committed_normalized,
                )?;
                rolling.clear();
                partial_state.clear();
            }
            PartialObservation::Empty | PartialObservation::Duplicate => {}
        }
    }

    if let Some(text) = partial_state.commit_active() {
        clear_visible_partial(stdout, args, &mut visible_partial_width)?;
        emit_rolling_commit_once(
            stdout,
            args,
            &text,
            &mut visible_partial_width,
            &mut last_committed_normalized,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_mic_samples<W: Write>(
    runtime: &WhisperRuntime,
    prefix: &[usize],
    args: &RunWhisperArgs,
    generation_template: &TextGenerationConfig,
    no_timestamps: bool,
    profile: &mut WhisperOperationProfile,
    stdout: &mut W,
    segment_index: usize,
    samples: &[f32],
    window_start: f32,
    purpose: TranscriptionJobPurpose,
) -> Result<RealtimeDecodeOutput> {
    let _job = TranscriptionJob {
        id: segment_index as u64,
        stream_id: 0,
        window_start_seconds: window_start,
        window_end_seconds: window_start + samples.len() as f32 / WHISPER_SAMPLE_RATE as f32,
        samples: samples.to_vec(),
        purpose,
        max_new_tokens: generation_template.max_new_tokens,
    };

    let features_start = Instant::now();
    let features = log_mel_spectrogram(samples, &runtime.preprocessor)?;
    let features_time = features_start.elapsed();

    let encode_start = Instant::now();
    let encoded = runtime.encode_audio(&features, profile)?;
    let encode_time = encode_start.elapsed();

    let segment_prompt = prefix.to_vec();
    let mut segment_generation = generation_template.clone();
    let default_max_new_tokens = runtime
        .config
        .n_text_ctx
        .saturating_sub(segment_prompt.len());
    segment_generation.max_new_tokens = args
        .generation
        .max_new_tokens
        .unwrap_or(64)
        .min(default_max_new_tokens);
    segment_generation.validate()?;

    let prefill_start = Instant::now();
    let first_logits = runtime.decoder_logits(&encoded, &segment_prompt, profile)?;
    let prefill_time = prefill_start.elapsed();
    let no_speech_probability = runtime
        .tokenizer
        .special_tokens()
        .no_speech
        .map(|token_id| logits_probability(&first_logits, token_id))
        .unwrap_or(0.0);

    let decode_start = Instant::now();
    let output_tokens = if args.stream_raw_tokens {
        for token_id in &segment_prompt {
            emit_raw_token(stdout, args, "prompt", *token_id, &runtime.tokenizer)?;
        }
        stdout.flush()?;
        runtime.stream_greedy_tokens(
            &encoded,
            &segment_prompt,
            &segment_generation,
            no_timestamps,
            profile,
            |token_id| {
                emit_raw_token(stdout, args, "generated", token_id, &runtime.tokenizer)?;
                stdout.flush()?;
                Ok::<(), Box<dyn std::error::Error>>(())
            },
        )?
    } else {
        runtime.generate_greedy(
            &encoded,
            &segment_prompt,
            &segment_generation,
            no_timestamps,
            profile,
        )?
    };
    let decode_time = decode_start.elapsed();
    let generated = &output_tokens[segment_prompt.len()..];
    let text = runtime.tokenizer.decode(generated)?.trim().to_string();

    Ok(RealtimeDecodeOutput {
        text,
        generated_tokens: generated.len(),
        no_speech_probability,
        features_time,
        encode_time,
        prefill_time,
        decode_time,
    })
}

fn validate_whisper_mic_args(args: &RunWhisperArgs) -> Result<()> {
    audio_duration(args.chunk_seconds, "chunk-seconds")?;
    audio_duration(args.window_seconds, "window-seconds")?;
    if args.partial_interval_ms == 0 {
        return Err("--partial-interval-ms must be greater than zero".into());
    }
    if args.silence_ms == 0 {
        return Err("--silence-ms must be greater than zero".into());
    }
    if !args.silence_threshold.is_finite() || args.silence_threshold < 0.0 {
        return Err("--silence-threshold must be a finite non-negative number".into());
    }
    if args.max_queued_chunks == 0 {
        return Err("--max-queued-chunks must be greater than zero".into());
    }
    Ok(())
}

fn add_realtime_totals(total: &mut RealtimeDecodeOutput, next: &RealtimeDecodeOutput) {
    total.generated_tokens += next.generated_tokens;
    total.features_time += next.features_time;
    total.encode_time += next.encode_time;
    total.prefill_time += next.prefill_time;
    total.decode_time += next.decode_time;
    total.no_speech_probability = next.no_speech_probability;
}

fn emit_partial<W: Write>(
    writer: &mut W,
    args: &RunWhisperArgs,
    text: &str,
    visible_partial_width: &mut usize,
) -> Result<()> {
    if text.trim().is_empty() {
        return Ok(());
    }
    if args.stream_raw_tokens && args.output != WhisperOutputFormatArg::EventsJson {
        return Ok(());
    }
    match args.output {
        WhisperOutputFormatArg::EventsJson => write_event(
            writer,
            &TranscriptionEvent::Partial {
                text: text.to_string(),
            },
        ),
        WhisperOutputFormatArg::Text => {
            let clear = " ".repeat((*visible_partial_width).saturating_sub(text.len()));
            write!(writer, "\r{text}{clear}")?;
            writer.flush()?;
            *visible_partial_width = text.len();
            Ok(())
        }
        _ => Ok(()),
    }
}

fn emit_commit<W: Write>(
    writer: &mut W,
    args: &RunWhisperArgs,
    text: &str,
    visible_partial_width: &mut usize,
) -> Result<()> {
    if text.trim().is_empty() {
        return Ok(());
    }
    if args.stream_raw_tokens && args.output != WhisperOutputFormatArg::EventsJson {
        return Ok(());
    }
    match args.output {
        WhisperOutputFormatArg::EventsJson => write_event(
            writer,
            &TranscriptionEvent::Commit {
                text: text.to_string(),
            },
        ),
        WhisperOutputFormatArg::Text => {
            if *visible_partial_width > 0 {
                writeln!(writer)?;
                *visible_partial_width = 0;
            }
            writeln!(writer, "{text}")?;
            writer.flush()?;
            Ok(())
        }
        _ => Ok(()),
    }
}

fn emit_rolling_commit_once<W: Write>(
    writer: &mut W,
    args: &RunWhisperArgs,
    text: &str,
    visible_partial_width: &mut usize,
    last_committed_normalized: &mut String,
) -> Result<()> {
    let normalized = normalize_transcript(text);
    if normalized.is_empty() || normalized == *last_committed_normalized {
        return Ok(());
    }
    emit_commit(writer, args, text, visible_partial_width)?;
    *last_committed_normalized = normalized;
    Ok(())
}

fn clear_visible_partial<W: Write>(
    writer: &mut W,
    args: &RunWhisperArgs,
    visible_partial_width: &mut usize,
) -> Result<()> {
    if args.output == WhisperOutputFormatArg::Text && *visible_partial_width > 0 {
        write!(writer, "\r{}\r", " ".repeat(*visible_partial_width))?;
        writer.flush()?;
        *visible_partial_width = 0;
    }
    Ok(())
}

fn emit_silence<W: Write>(writer: &mut W, args: &RunWhisperArgs) -> Result<()> {
    if args.output == WhisperOutputFormatArg::EventsJson {
        write_event(writer, &TranscriptionEvent::Silence)?;
    }
    Ok(())
}

fn emit_warning<W: Write>(writer: &mut W, args: &RunWhisperArgs, message: &str) -> Result<()> {
    eprintln!("warning: {message}");
    if args.output == WhisperOutputFormatArg::EventsJson {
        write_event(
            writer,
            &TranscriptionEvent::Warning {
                message: message.to_string(),
            },
        )?;
    }
    Ok(())
}

fn emit_queue_warnings<W: Write>(
    writer: &mut W,
    args: &RunWhisperArgs,
    stream: &puppygrad::audio::ContinuousInputStream,
) -> Result<()> {
    let dropped = stream.take_dropped_chunks();
    if dropped > 0 {
        emit_warning(
            writer,
            args,
            &format!("audio capture queue overflowed; dropped {dropped} chunk(s)"),
        )?;
    }
    Ok(())
}

fn emit_raw_token<W, D>(
    writer: &mut W,
    args: &RunWhisperArgs,
    phase: &str,
    token_id: usize,
    decoder: &D,
) -> Result<()>
where
    W: Write,
    D: RawTokenDecoder,
    D::Error: std::error::Error + 'static,
{
    let token = decoder.raw_token(token_id)?;
    if args.output == WhisperOutputFormatArg::EventsJson {
        write_event(
            writer,
            &TranscriptionEvent::RawToken {
                phase: phase.to_string(),
                token_id,
                token,
            },
        )
    } else {
        writeln!(writer, "{phase}\t{token_id}\t{}", escape_raw_token(&token))?;
        Ok(())
    }
}

fn write_event<W: Write>(writer: &mut W, event: &TranscriptionEvent) -> Result<()> {
    serde_json::to_writer(&mut *writer, event)?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(())
}

fn print_realtime_stats(
    args: &RunWhisperArgs,
    load_time: Duration,
    totals: &RealtimeDecodeOutput,
    profile: &WhisperOperationProfile,
    dropped_chunks: usize,
    queue_depth: usize,
    capture_lag: Duration,
) {
    if !args.stats {
        return;
    }
    eprintln!("stats:");
    eprintln!("  load: {}", format_duration(load_time));
    eprintln!("  capture lag: {}", format_duration(capture_lag));
    eprintln!("  queue depth: {queue_depth}");
    eprintln!("  dropped chunks: {dropped_chunks}");
    eprintln!("  log-mel: {}", format_duration(totals.features_time));
    eprintln!("  encode: {}", format_duration(totals.encode_time));
    eprintln!("  prefill: {}", format_duration(totals.prefill_time));
    eprintln!(
        "  decode: {} ({} generated tokens)",
        format_duration(totals.decode_time),
        totals.generated_tokens
    );
    eprintln!("  final logits: {}", format_duration(profile.final_logits));
}

fn write_raw_token_event<W, D>(
    writer: &mut W,
    segment_index: Option<usize>,
    phase: &str,
    token_id: usize,
    decoder: &D,
) -> Result<()>
where
    W: Write,
    D: RawTokenDecoder,
    D::Error: std::error::Error + 'static,
{
    let token = decoder.raw_token(token_id)?;
    if let Some(segment_index) = segment_index {
        writeln!(
            writer,
            "{segment_index}\t{phase}\t{token_id}\t{}",
            escape_raw_token(&token)
        )?;
    } else {
        writeln!(writer, "{phase}\t{token_id}\t{}", escape_raw_token(&token))?;
    }
    Ok(())
}

fn load_audio_arg(audio_path: &PathBuf) -> Result<puppygrad::models::whisper::PcmAudio> {
    if audio_path.as_os_str() == "-" {
        let mut data = Vec::new();
        std::io::stdin().lock().read_to_end(&mut data)?;
        return Ok(load_wav_pcm_bytes(PathBuf::from("<stdin>"), &data)?);
    }
    Ok(load_wav_pcm(audio_path)?)
}

fn audio_duration(seconds: f32, arg_name: &str) -> Result<Duration> {
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(format!("--{arg_name} must be a positive finite number of seconds").into());
    }
    Ok(Duration::from_secs_f32(seconds))
}

fn rust_config(
    threads: usize,
    tuning: RustTuning,
) -> puppygrad::models::gpt2::Result<Gpt2RustConfig> {
    rust_config_from(Gpt2RustConfig::default(), Some(threads), tuning)
}

fn whisper_rust_config(
    size: WhisperSize,
    threads: Option<usize>,
) -> puppygrad::models::whisper::Result<WhisperRustConfig> {
    let mut config = WhisperRustConfig::for_size(size);
    if let Some(threads) = threads {
        config = config.with_threads(threads);
    }
    config.validate()?;
    Ok(config)
}

fn whisper_backend_config(
    size: WhisperSize,
    threads: Option<usize>,
    backend: WhisperBackendArg,
    quantized_weights: bool,
) -> puppygrad::models::whisper::Result<WhisperBackendConfig> {
    match backend {
        WhisperBackendArg::Rust => Ok(WhisperBackendConfig::Rust(
            whisper_rust_config(size, threads)?.with_quantized_weights(quantized_weights),
        )),
        WhisperBackendArg::Gpu => Ok(WhisperBackendConfig::Gpu),
    }
}

fn rust_config_from(
    base: Gpt2RustConfig,
    threads: Option<usize>,
    tuning: RustTuning,
) -> puppygrad::models::gpt2::Result<Gpt2RustConfig> {
    let config = Gpt2RustConfig {
        threads: threads.unwrap_or(base.threads),
        dense_parallel_threshold: tuning
            .dense_parallel_threshold
            .unwrap_or(base.dense_parallel_threshold),
        qkv_chunk_size: tuning.qkv_chunk_size.unwrap_or(base.qkv_chunk_size),
        attention_projection_chunk_size: tuning
            .attention_projection_chunk_size
            .unwrap_or(base.attention_projection_chunk_size),
        mlp_fc_chunk_size: tuning.mlp_fc_chunk_size.unwrap_or(base.mlp_fc_chunk_size),
        mlp_projection_chunk_size: tuning
            .mlp_projection_chunk_size
            .unwrap_or(base.mlp_projection_chunk_size),
        logits_chunk_size: tuning.logits_chunk_size.unwrap_or(base.logits_chunk_size),
        attention_head_parallel_threshold: tuning
            .attention_head_parallel_threshold
            .unwrap_or(base.attention_head_parallel_threshold),
        quantized_weights: tuning.quantized_weights.unwrap_or(base.quantized_weights),
    };
    config.validate()?;
    Ok(config)
}

fn load_gpt2_tuning(
    tuning_file: Option<&PathBuf>,
    model_dir: &std::path::Path,
) -> Result<Option<Gpt2RustConfig>> {
    let path = tuning_file
        .cloned()
        .unwrap_or_else(|| model_dir.join("puppygrad-tune.json"));
    if !path.exists() {
        if tuning_file.is_some() {
            return Err(format!("tuning file {} does not exist", path.display()).into());
        }
        return Ok(None);
    }

    let text = fs::read_to_string(&path)
        .map_err(|err| format!("failed to read tuning file {}: {err}", path.display()))?;
    let tuning: SavedGpt2Tuning = serde_json::from_str(&text)
        .map_err(|err| format!("failed to parse tuning file {}: {err}", path.display()))?;
    tuning.rust.validate()?;
    eprintln!("loaded tuning config from {}", path.display());
    Ok(Some(tuning.rust))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Gpt2AutoTuneConfig {
    rust: Gpt2RustConfig,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Gpt2AutoTuneMeasurement {
    load_ms: f64,
    prompt_count: usize,
    runs_per_prompt: usize,
    generated_tokens: usize,
    decode_tokens_per_second: f64,
    decode_tokens_per_second_p25: f64,
    decode_tokens_per_second_median: f64,
    decode_tokens_per_second_p95: f64,
    decode_tokens_per_second_stddev: f64,
    total_tokens_per_second: f64,
    total_tokens_per_second_median: f64,
    prefill_ms: f64,
    decode_ms: f64,
    total_generation_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SavedGpt2Tuning {
    model: String,
    backend: String,
    score_name: String,
    score: f64,
    search_score: f64,
    max_new_tokens: usize,
    warmup_runs: usize,
    measured_runs: usize,
    prompt_count: usize,
    rust: Gpt2RustConfig,
    measurement: Gpt2AutoTuneMeasurement,
}

struct Gpt2AutoTuneTarget {
    model_dir: PathBuf,
    prompts: Vec<String>,
    max_new_tokens: usize,
    candidates: Vec<Gpt2AutoTuneConfig>,
}

impl AutoTuneTarget for Gpt2AutoTuneTarget {
    type Config = Gpt2AutoTuneConfig;
    type Measurement = Gpt2AutoTuneMeasurement;
    type Error = Box<dyn std::error::Error>;

    fn candidate_configs(&self) -> Vec<Self::Config> {
        self.candidates.clone()
    }

    fn evaluate_config(
        &mut self,
        config: &Self::Config,
        options: &AutoTuneOptions,
    ) -> std::result::Result<Self::Measurement, Self::Error> {
        eprintln!(
            "autotune trial: threads={} dense_threshold={} chunks=({},{},{},{},{}) attn_threshold={} weights={}",
            config.rust.threads,
            config.rust.dense_parallel_threshold,
            config.rust.qkv_chunk_size,
            config.rust.attention_projection_chunk_size,
            config.rust.mlp_fc_chunk_size,
            config.rust.mlp_projection_chunk_size,
            config.rust.logits_chunk_size,
            config.rust.attention_head_parallel_threshold,
            if config.rust.quantized_weights { "int8" } else { "f32" }
        );
        let backend = Gpt2BackendConfig::Rust(config.rust.clone());
        let load_start = Instant::now();
        let runtime = Gpt2Runtime::from_dir_with_backend(&self.model_dir, backend)?;
        let load_time = load_start.elapsed();

        let generation = Gpt2GenerationConfig::new(self.max_new_tokens);
        generation.validate()?;

        let mut stats = Vec::with_capacity(options.measured_runs * self.prompts.len());
        for prompt in &self.prompts {
            for _ in 0..options.warmup_runs {
                let _ = runtime.stream_greedy_text_with_stats(
                    prompt,
                    generation.max_new_tokens,
                    |_| Ok::<(), Box<dyn std::error::Error>>(()),
                )?;
            }
            for _ in 0..options.measured_runs {
                let (_, run_stats) = runtime.stream_greedy_text_with_stats(
                    prompt,
                    generation.max_new_tokens,
                    |_| Ok::<(), Box<dyn std::error::Error>>(()),
                )?;
                stats.push(run_stats);
            }
        }

        let generated_tokens = stats
            .iter()
            .map(|stats| stats.generated_tokens)
            .sum::<usize>();
        let prompt_tokens = stats.iter().map(|stats| stats.prompt_tokens).sum::<usize>();
        let prefill_time = sum_duration(stats.iter().map(|stats| stats.prefill_time));
        let decode_time = sum_duration(stats.iter().map(|stats| stats.decode_time));
        let total_generation_time =
            sum_duration(stats.iter().map(|stats| stats.total_generation_time));
        let decode_tps_summary = value_distribution(
            stats
                .iter()
                .map(|stats| rate(stats.generated_tokens as f64, stats.decode_time)),
        );
        let total_tps_summary = value_distribution(stats.iter().map(|stats| {
            rate(
                (stats.prompt_tokens + stats.generated_tokens) as f64,
                stats.total_generation_time,
            )
        }));

        Ok(Gpt2AutoTuneMeasurement {
            load_ms: duration_ms(load_time),
            prompt_count: self.prompts.len(),
            runs_per_prompt: options.measured_runs,
            generated_tokens,
            decode_tokens_per_second: rate(generated_tokens as f64, decode_time),
            decode_tokens_per_second_p25: decode_tps_summary.p25,
            decode_tokens_per_second_median: decode_tps_summary.median,
            decode_tokens_per_second_p95: decode_tps_summary.p95,
            decode_tokens_per_second_stddev: decode_tps_summary.stddev,
            total_tokens_per_second: rate(
                (prompt_tokens + generated_tokens) as f64,
                total_generation_time,
            ),
            total_tokens_per_second_median: total_tps_summary.median,
            prefill_ms: duration_ms(prefill_time) / stats.len().max(1) as f64,
            decode_ms: duration_ms(decode_time) / stats.len().max(1) as f64,
            total_generation_ms: duration_ms(total_generation_time) / stats.len().max(1) as f64,
        })
    }

    fn score(&self, measurement: &Self::Measurement) -> f64 {
        measurement.decode_tokens_per_second_median
    }
}

fn run_gpt2_autotune(args: RunGpt2AutoTuneArgs) -> Result<()> {
    if args.runs == 0 {
        return Err("autotune gpt2 --runs must be > 0".into());
    }
    if args.max_trials == 0 {
        return Err("autotune gpt2 --max-trials must be > 0".into());
    }

    let candidates = gpt2_autotune_candidates(&args)?;
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

    let prompts = load_experiment_prompts(args.prompt.as_deref(), args.prompt_file.as_ref())?;
    eprintln!(
        "autotune: {} candidates, evaluating up to {}",
        candidates.len(),
        args.max_trials
    );

    let mut target = Gpt2AutoTuneTarget {
        model_dir,
        prompts,
        max_new_tokens: args.max_new_tokens,
        candidates,
    };
    let options = AutoTuneOptions {
        warmup_runs: args.warmup_runs,
        measured_runs: args.runs,
        max_trials: Some(args.max_trials),
    };
    let result = autotune(&mut target, &options)?;

    print_gpt2_autotune_result(&result);
    let validation_options = AutoTuneOptions {
        warmup_runs: args.warmup_runs,
        measured_runs: args.validation_runs,
        max_trials: None,
    };
    eprintln!(
        "validating best config with {} measured runs",
        args.validation_runs
    );
    let validation_measurement =
        target.evaluate_config(&result.best_config, &validation_options)?;
    let validation_score = target.score(&validation_measurement);
    println!();
    println!(
        "validated best: {:.2} tok/s median ({:.2} tok/s mean)",
        validation_score, validation_measurement.decode_tokens_per_second
    );

    let save_path = args
        .save_tuning
        .unwrap_or_else(|| target.model_dir.join("puppygrad-tune.json"));
    save_gpt2_tuning(
        &save_path,
        &result.best_config,
        result.best_score,
        validation_score,
        validation_measurement,
        args.max_new_tokens,
        &validation_options,
    )?;
    eprintln!("saved tuning config to {}", save_path.display());
    Ok(())
}

fn gpt2_autotune_candidates(args: &RunGpt2AutoTuneArgs) -> Result<Vec<Gpt2AutoTuneConfig>> {
    let threads = parse_usize_list("threads", &args.threads)?;
    let dense_thresholds =
        parse_usize_list("dense-parallel-thresholds", &args.dense_parallel_thresholds)?;
    let qkv_chunks = parse_usize_list("qkv-chunk-sizes", &args.qkv_chunk_sizes)?;
    let attention_projection_chunks = parse_usize_list(
        "attention-projection-chunk-sizes",
        &args.attention_projection_chunk_sizes,
    )?;
    let mlp_fc_chunks = parse_usize_list("mlp-fc-chunk-sizes", &args.mlp_fc_chunk_sizes)?;
    let mlp_projection_chunks = parse_usize_list(
        "mlp-projection-chunk-sizes",
        &args.mlp_projection_chunk_sizes,
    )?;
    let logits_chunks = parse_usize_list("logits-chunk-sizes", &args.logits_chunk_sizes)?;
    let attention_thresholds = parse_usize_list(
        "attention-head-parallel-thresholds",
        &args.attention_head_parallel_thresholds,
    )?;
    let quantized_options = if args.include_quantized {
        [false, true].as_slice()
    } else {
        [false].as_slice()
    };

    let mut candidates = Vec::new();
    for quantized_weights in quantized_options {
        for dense_parallel_threshold in &dense_thresholds {
            for qkv_chunk_size in &qkv_chunks {
                for attention_projection_chunk_size in &attention_projection_chunks {
                    for mlp_fc_chunk_size in &mlp_fc_chunks {
                        for mlp_projection_chunk_size in &mlp_projection_chunks {
                            for logits_chunk_size in &logits_chunks {
                                for attention_head_parallel_threshold in &attention_thresholds {
                                    for threads in &threads {
                                        let rust = Gpt2RustConfig {
                                            threads: *threads,
                                            dense_parallel_threshold: *dense_parallel_threshold,
                                            qkv_chunk_size: *qkv_chunk_size,
                                            attention_projection_chunk_size:
                                                *attention_projection_chunk_size,
                                            mlp_fc_chunk_size: *mlp_fc_chunk_size,
                                            mlp_projection_chunk_size: *mlp_projection_chunk_size,
                                            logits_chunk_size: *logits_chunk_size,
                                            attention_head_parallel_threshold:
                                                *attention_head_parallel_threshold,
                                            quantized_weights: *quantized_weights,
                                        };
                                        rust.validate()?;
                                        candidates.push(Gpt2AutoTuneConfig { rust });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if candidates.is_empty() {
        return Err("autotune gpt2 generated no candidates".into());
    }
    Ok(candidates)
}

fn print_gpt2_autotune_result(
    result: &puppygrad::models::autotune::AutoTuneResult<
        Gpt2AutoTuneConfig,
        Gpt2AutoTuneMeasurement,
    >,
) {
    println!("best:");
    print_gpt2_autotune_config(&result.best_config);
    println!("  median score: {:.2} tok/s", result.best_score);
    println!();
    println!(
        "{:>5} {:>7} {:>9} {:>5} {:>5} {:>5} {:>5} {:>6} {:>9} {:>7} {:>10} {:>10} {:>10}",
        "trial",
        "threads",
        "dense_th",
        "qkv",
        "attn",
        "fc",
        "proj",
        "logits",
        "attn_th",
        "weights",
        "med tok/s",
        "mean tok/s",
        "total/s"
    );
    for (index, trial) in result.trials.iter().enumerate() {
        let config = &trial.config.rust;
        let measurement = &trial.measurement;
        println!(
            "{:>5} {:>7} {:>9} {:>5} {:>5} {:>5} {:>5} {:>6} {:>9} {:>7} {:>10.2} {:>10.2} {:>10.2}",
            index + 1,
            config.threads,
            config.dense_parallel_threshold,
            config.qkv_chunk_size,
            config.attention_projection_chunk_size,
            config.mlp_fc_chunk_size,
            config.mlp_projection_chunk_size,
            config.logits_chunk_size,
            config.attention_head_parallel_threshold,
            if config.quantized_weights {
                "int8"
            } else {
                "f32"
            },
            measurement.decode_tokens_per_second_median,
            measurement.decode_tokens_per_second,
            measurement.total_tokens_per_second
        );
    }
}

fn save_gpt2_tuning(
    path: &PathBuf,
    best_config: &Gpt2AutoTuneConfig,
    search_score: f64,
    validation_score: f64,
    validation_measurement: Gpt2AutoTuneMeasurement,
    max_new_tokens: usize,
    options: &AutoTuneOptions,
) -> Result<()> {
    let saved = SavedGpt2Tuning {
        model: "gpt2".to_string(),
        backend: "rust".to_string(),
        score_name: "median_decode_tokens_per_second".to_string(),
        score: validation_score,
        search_score,
        max_new_tokens,
        warmup_runs: options.warmup_runs,
        measured_runs: options.measured_runs,
        prompt_count: validation_measurement.prompt_count,
        rust: best_config.rust.clone(),
        measurement: validation_measurement,
    };

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create tuning dir {}: {err}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(&saved)?;
    fs::write(path, format!("{json}\n"))
        .map_err(|err| format!("failed to write tuning file {}: {err}", path.display()))?;
    Ok(())
}

fn print_gpt2_autotune_config(config: &Gpt2AutoTuneConfig) {
    let config = &config.rust;
    println!("  --threads {}", config.threads);
    println!(
        "  --dense-parallel-threshold {}",
        config.dense_parallel_threshold
    );
    println!("  --qkv-chunk-size {}", config.qkv_chunk_size);
    println!(
        "  --attention-projection-chunk-size {}",
        config.attention_projection_chunk_size
    );
    println!("  --mlp-fc-chunk-size {}", config.mlp_fc_chunk_size);
    println!(
        "  --mlp-projection-chunk-size {}",
        config.mlp_projection_chunk_size
    );
    println!("  --logits-chunk-size {}", config.logits_chunk_size);
    println!(
        "  --attention-head-parallel-threshold {}",
        config.attention_head_parallel_threshold
    );
    if config.quantized_weights {
        println!("  --quantized-weights");
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct WhisperAutoTuneConfig {
    max_new_tokens: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct WhisperAutoTuneMeasurement {
    segments: usize,
    generated_tokens: usize,
    decode_tokens_per_second: f64,
    total_ms: f64,
    log_mel_ms: f64,
    encode_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
}

struct WhisperAutoTuneTarget {
    runtime: WhisperRuntime,
    samples: Vec<f32>,
    prefix: Vec<usize>,
    no_timestamps: bool,
    candidates: Vec<WhisperAutoTuneConfig>,
}

impl AutoTuneTarget for WhisperAutoTuneTarget {
    type Config = WhisperAutoTuneConfig;
    type Measurement = WhisperAutoTuneMeasurement;
    type Error = Box<dyn std::error::Error>;

    fn candidate_configs(&self) -> Vec<Self::Config> {
        self.candidates.clone()
    }

    fn evaluate_config(
        &mut self,
        config: &Self::Config,
        options: &AutoTuneOptions,
    ) -> std::result::Result<Self::Measurement, Self::Error> {
        eprintln!(
            "autotune whisper trial: max_new_tokens={}",
            config.max_new_tokens
        );
        let mut generation = TextGenerationConfig::new(config.max_new_tokens);
        generation.eos_token_id = Some(self.runtime.tokenizer.special_tokens().eos);
        generation.validate()?;

        for _ in 0..options.warmup_runs {
            let _ = whisper_experiment_run(
                &self.runtime,
                &self.samples,
                &self.prefix,
                &generation,
                self.no_timestamps,
            )?;
        }

        let mut measurements = Vec::with_capacity(options.measured_runs);
        for _ in 0..options.measured_runs {
            measurements.push(whisper_experiment_run(
                &self.runtime,
                &self.samples,
                &self.prefix,
                &generation,
                self.no_timestamps,
            )?);
        }

        let generated_tokens = measurements
            .iter()
            .map(|measurement| measurement.generated_tokens)
            .sum::<usize>();
        let log_mel_time = sum_duration(measurements.iter().map(|m| m.log_mel_time));
        let encode_time = sum_duration(measurements.iter().map(|m| m.encode_time));
        let prefill_time = sum_duration(measurements.iter().map(|m| m.prefill_time));
        let decode_time = sum_duration(measurements.iter().map(|m| m.decode_time));
        let total_time = log_mel_time + encode_time + prefill_time + decode_time;
        let runs = measurements.len().max(1) as f64;
        Ok(WhisperAutoTuneMeasurement {
            segments: measurements.first().map_or(0, |m| m.segments),
            generated_tokens,
            decode_tokens_per_second: rate(generated_tokens as f64, decode_time),
            total_ms: duration_ms(total_time) / runs,
            log_mel_ms: duration_ms(log_mel_time) / runs,
            encode_ms: duration_ms(encode_time) / runs,
            prefill_ms: duration_ms(prefill_time) / runs,
            decode_ms: duration_ms(decode_time) / runs,
        })
    }

    fn score(&self, measurement: &Self::Measurement) -> f64 {
        measurement.decode_tokens_per_second
    }
}

fn run_whisper_autotune(args: RunWhisperAutoTuneArgs) -> Result<()> {
    if args.runs == 0 {
        return Err("autotune whisper --runs must be > 0".into());
    }
    if args.max_trials == 0 {
        return Err("autotune whisper --max-trials must be > 0".into());
    }
    let token_candidates = parse_usize_list("max-new-tokens", &args.max_new_tokens)?;
    let candidates = token_candidates
        .into_iter()
        .map(|max_new_tokens| WhisperAutoTuneConfig { max_new_tokens })
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Err("autotune whisper generated no candidates".into());
    }

    let model_dir = args
        .model_dir
        .clone()
        .unwrap_or_else(|| default_whisper_dir(args.size));
    if args.download {
        eprintln!(
            "downloading/checking Whisper assets for {} into {}",
            args.size,
            model_dir.display()
        );
    }
    let rust_config = whisper_rust_config(args.size, args.threads)?;
    let runtime = WhisperRuntime::prepare_from_huggingface_with_rust_config(
        args.size,
        args.model_id.as_deref(),
        &args.revision,
        &model_dir,
        args.download,
        rust_config,
    )?;
    let audio = load_wav_pcm(&args.audio)?;
    if audio.sample_rate != WHISPER_SAMPLE_RATE {
        return Err(format!(
            "{} has sample rate {} Hz; native Whisper currently requires {WHISPER_SAMPLE_RATE} Hz WAV input",
            args.audio.display(),
            audio.sample_rate
        )
        .into());
    }
    let prefix = runtime.tokenizer.prompt_prefix(
        args.task.into(),
        args.language.as_deref(),
        args.no_timestamps,
        args.size,
    )?;
    let mut target = WhisperAutoTuneTarget {
        runtime,
        samples: audio.samples,
        prefix,
        no_timestamps: args.no_timestamps,
        candidates,
    };
    let options = AutoTuneOptions {
        warmup_runs: args.warmup_runs,
        measured_runs: args.runs,
        max_trials: Some(args.max_trials),
    };
    let result = autotune(&mut target, &options)?;

    println!("best:");
    println!("  --max-new-tokens {}", result.best_config.max_new_tokens);
    println!("  score: {:.2} decode tok/s", result.best_score);
    println!();
    println!(
        "{:>5} {:>14} {:>10} {:>10} {:>10} {:>10}",
        "trial", "max_new", "tok/s", "encode", "decode", "total"
    );
    for (index, trial) in result.trials.iter().enumerate() {
        println!(
            "{:>5} {:>14} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            index + 1,
            trial.config.max_new_tokens,
            trial.measurement.decode_tokens_per_second,
            trial.measurement.encode_ms,
            trial.measurement.decode_ms,
            trial.measurement.total_ms
        );
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

fn logits_probability(logits: &[f32], token_id: usize) -> f32 {
    if token_id >= logits.len() {
        return 0.0;
    }
    let max = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, value| acc.max(value));
    let denominator = logits.iter().map(|value| (*value - max).exp()).sum::<f32>();
    if denominator == 0.0 {
        return 0.0;
    }
    (logits[token_id] - max).exp() / denominator
}

fn decode_timestamped_segments(
    runtime: &WhisperRuntime,
    token_ids: &[usize],
    window_start: f32,
    window_end: f32,
) -> Result<Vec<WhisperSegmentOutput>> {
    let Some(timestamp_begin) = runtime.tokenizer.special_tokens().timestamp_begin else {
        let text = runtime.tokenizer.decode(token_ids)?.trim().to_string();
        return Ok(vec![WhisperSegmentOutput {
            start: window_start,
            end: window_end,
            text,
        }]);
    };

    let mut segments = Vec::new();
    let mut text_tokens = Vec::new();
    let mut segment_start = window_start;
    for token_id in token_ids.iter().copied() {
        if token_id >= timestamp_begin {
            let timestamp = timestamp_token_seconds(timestamp_begin, token_id, window_start);
            if !text_tokens.is_empty() {
                let text = runtime.tokenizer.decode(&text_tokens)?.trim().to_string();
                if !text.is_empty() {
                    segments.push(WhisperSegmentOutput {
                        start: segment_start,
                        end: timestamp.max(segment_start),
                        text,
                    });
                }
                text_tokens.clear();
            }
            segment_start = timestamp;
        } else {
            text_tokens.push(token_id);
        }
    }
    if !text_tokens.is_empty() {
        let text = runtime.tokenizer.decode(&text_tokens)?.trim().to_string();
        if !text.is_empty() {
            segments.push(WhisperSegmentOutput {
                start: segment_start,
                end: window_end.max(segment_start),
                text,
            });
        }
    }
    if segments.is_empty() {
        segments.push(WhisperSegmentOutput {
            start: window_start,
            end: window_end,
            text: String::new(),
        });
    }
    Ok(segments)
}

fn timestamp_token_seconds(timestamp_begin: usize, token_id: usize, window_start: f32) -> f32 {
    window_start + (token_id.saturating_sub(timestamp_begin) as f32 * 0.02)
}

fn format_srt_timestamp(seconds: f32) -> String {
    format_timestamp(seconds, ',')
}

fn format_vtt_timestamp(seconds: f32) -> String {
    format_timestamp(seconds, '.')
}

fn format_timestamp(seconds: f32, decimal_separator: char) -> String {
    let millis = (seconds.max(0.0) * 1_000.0).round() as u64;
    let hours = millis / 3_600_000;
    let minutes = (millis / 60_000) % 60;
    let seconds = (millis / 1_000) % 60;
    let millis = millis % 1_000;
    format!("{hours:02}:{minutes:02}:{seconds:02}{decimal_separator}{millis:03}")
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
    p25: f64,
    median: f64,
    p95: f64,
    max: f64,
    stddev: f64,
}

#[derive(Clone, Debug, Serialize)]
struct WhisperExperimentRow {
    size: String,
    audio: String,
    run: usize,
    segments: usize,
    max_new_tokens: usize,
    generated_tokens: usize,
    load_ms: f64,
    audio_decode_ms: f64,
    log_mel_ms: f64,
    encode_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    total_ms: f64,
    audio_projection_ms: f64,
    encoder_attention_ms: f64,
    encoder_mlp_ms: f64,
    encoder_layer_norm_ms: f64,
    decoder_self_attention_ms: f64,
    decoder_cross_attention_ms: f64,
    decoder_mlp_ms: f64,
    decoder_layer_norm_ms: f64,
    final_logits_ms: f64,
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

fn run_whisper_experiment(args: RunWhisperExperimentArgs) -> Result<()> {
    if args.runs == 0 {
        return Err("experiment whisper --runs must be > 0".into());
    }
    if args.max_new_tokens == 0 {
        return Err("experiment whisper --max-new-tokens must be > 0".into());
    }
    let model_dir = args
        .model_dir
        .clone()
        .unwrap_or_else(|| default_whisper_dir(args.size));
    if args.download {
        eprintln!(
            "downloading/checking Whisper assets for {} into {}",
            args.size,
            model_dir.display()
        );
    }

    let load_start = Instant::now();
    let rust_config = whisper_rust_config(args.size, args.threads)?;
    let runtime = WhisperRuntime::prepare_from_huggingface_with_rust_config(
        args.size,
        args.model_id.as_deref(),
        &args.revision,
        &model_dir,
        args.download,
        rust_config,
    )?;
    let load_time = load_start.elapsed();

    let audio_decode_start = Instant::now();
    let audio = load_wav_pcm(&args.audio)?;
    if audio.sample_rate != WHISPER_SAMPLE_RATE {
        return Err(format!(
            "{} has sample rate {} Hz; native Whisper currently requires {WHISPER_SAMPLE_RATE} Hz WAV input",
            args.audio.display(),
            audio.sample_rate
        )
        .into());
    }
    let audio_decode_time = audio_decode_start.elapsed();

    let prefix = runtime.tokenizer.prompt_prefix(
        args.task.into(),
        args.language.as_deref(),
        args.no_timestamps,
        args.size,
    )?;
    let mut generation = TextGenerationConfig::new(args.max_new_tokens);
    generation.eos_token_id = Some(runtime.tokenizer.special_tokens().eos);
    generation.validate()?;

    for _ in 0..args.warmup_runs {
        let _ = whisper_experiment_run(
            &runtime,
            &audio.samples,
            &prefix,
            &generation,
            args.no_timestamps,
        )?;
    }

    let mut rows = Vec::with_capacity(args.runs);
    for run in 1..=args.runs {
        let measurement = whisper_experiment_run(
            &runtime,
            &audio.samples,
            &prefix,
            &generation,
            args.no_timestamps,
        )?;
        rows.push(WhisperExperimentRow {
            size: args.size.to_string(),
            audio: args.audio.display().to_string(),
            run,
            segments: measurement.segments,
            max_new_tokens: args.max_new_tokens,
            generated_tokens: measurement.generated_tokens,
            load_ms: duration_ms(load_time),
            audio_decode_ms: duration_ms(audio_decode_time),
            log_mel_ms: duration_ms(measurement.log_mel_time),
            encode_ms: duration_ms(measurement.encode_time),
            prefill_ms: duration_ms(measurement.prefill_time),
            decode_ms: duration_ms(measurement.decode_time),
            total_ms: duration_ms(
                measurement.log_mel_time
                    + measurement.encode_time
                    + measurement.prefill_time
                    + measurement.decode_time,
            ),
            audio_projection_ms: duration_ms(measurement.profile.audio_projection),
            encoder_attention_ms: duration_ms(measurement.profile.encoder_attention),
            encoder_mlp_ms: duration_ms(measurement.profile.encoder_mlp),
            encoder_layer_norm_ms: duration_ms(measurement.profile.encoder_layer_norm),
            decoder_self_attention_ms: duration_ms(measurement.profile.decoder_self_attention),
            decoder_cross_attention_ms: duration_ms(measurement.profile.decoder_cross_attention),
            decoder_mlp_ms: duration_ms(measurement.profile.decoder_mlp),
            decoder_layer_norm_ms: duration_ms(measurement.profile.decoder_layer_norm),
            final_logits_ms: duration_ms(measurement.profile.final_logits),
        });
    }

    match args.format {
        ExperimentFormatArg::Table => print_whisper_experiment_table(&rows),
        ExperimentFormatArg::Csv => print_whisper_experiment_csv(&rows),
        ExperimentFormatArg::Json => println!("{}", serde_json::to_string_pretty(&rows)?),
    }

    Ok(())
}

#[derive(Clone, Debug)]
struct WhisperExperimentMeasurement {
    segments: usize,
    generated_tokens: usize,
    log_mel_time: Duration,
    encode_time: Duration,
    prefill_time: Duration,
    decode_time: Duration,
    profile: WhisperOperationProfile,
}

fn whisper_experiment_run(
    runtime: &WhisperRuntime,
    samples: &[f32],
    prefix: &[usize],
    generation: &TextGenerationConfig,
    no_timestamps: bool,
) -> Result<WhisperExperimentMeasurement> {
    let mut profile = WhisperOperationProfile::default();
    let mut log_mel_time = Duration::ZERO;
    let mut encode_time = Duration::ZERO;
    let mut prefill_time = Duration::ZERO;
    let mut decode_time = Duration::ZERO;
    let mut generated_tokens = 0usize;
    let mut segments = 0usize;
    let chunk_samples = runtime.preprocessor.n_samples;
    let total_samples = samples.len().max(1);

    for start_sample in (0..total_samples).step_by(chunk_samples) {
        let end_sample = (start_sample + chunk_samples).min(samples.len());
        let chunk = &samples[start_sample..end_sample];
        let start = Instant::now();
        let features = log_mel_spectrogram(chunk, &runtime.preprocessor)?;
        log_mel_time += start.elapsed();

        let start = Instant::now();
        let encoded = runtime.encode_audio(&features, &mut profile)?;
        encode_time += start.elapsed();

        let start = Instant::now();
        let _ = runtime.decoder_logits(&encoded, prefix, &mut profile)?;
        prefill_time += start.elapsed();

        let start = Instant::now();
        let output =
            runtime.generate_greedy(&encoded, prefix, generation, no_timestamps, &mut profile)?;
        decode_time += start.elapsed();
        generated_tokens += output.len().saturating_sub(prefix.len());
        segments += 1;
    }

    Ok(WhisperExperimentMeasurement {
        segments,
        generated_tokens,
        log_mel_time,
        encode_time,
        prefill_time,
        decode_time,
        profile,
    })
}

fn print_whisper_experiment_table(rows: &[WhisperExperimentRow]) {
    println!(
        "{:>4} {:>8} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "run",
        "segments",
        "tokens",
        "logmel",
        "encode",
        "prefill",
        "decode",
        "enc_attn",
        "xattn",
        "total"
    );
    for row in rows {
        println!(
            "{:>4} {:>8} {:>8} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
            row.run,
            row.segments,
            row.generated_tokens,
            row.log_mel_ms,
            row.encode_ms,
            row.prefill_ms,
            row.decode_ms,
            row.encoder_attention_ms,
            row.decoder_cross_attention_ms,
            row.total_ms
        );
    }
}

fn print_whisper_experiment_csv(rows: &[WhisperExperimentRow]) {
    println!(
        "size,audio,run,segments,max_new_tokens,generated_tokens,load_ms,audio_decode_ms,log_mel_ms,encode_ms,prefill_ms,decode_ms,total_ms,audio_projection_ms,encoder_attention_ms,encoder_mlp_ms,encoder_layer_norm_ms,decoder_self_attention_ms,decoder_cross_attention_ms,decoder_mlp_ms,decoder_layer_norm_ms,final_logits_ms"
    );
    for row in rows {
        println!(
            "{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            row.size,
            row.audio,
            row.run,
            row.segments,
            row.max_new_tokens,
            row.generated_tokens,
            row.load_ms,
            row.audio_decode_ms,
            row.log_mel_ms,
            row.encode_ms,
            row.prefill_ms,
            row.decode_ms,
            row.total_ms,
            row.audio_projection_ms,
            row.encoder_attention_ms,
            row.encoder_mlp_ms,
            row.encoder_layer_norm_ms,
            row.decoder_self_attention_ms,
            row.decoder_cross_attention_ms,
            row.decoder_mlp_ms,
            row.decoder_layer_norm_ms,
            row.final_logits_ms
        );
    }
}

#[allow(clippy::too_many_arguments)]
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
    value_distribution(durations.into_iter().map(duration_ms))
}

fn value_distribution<I>(values: I) -> DistributionSummary
where
    I: IntoIterator<Item = f64>,
{
    let mut values: Vec<f64> = values.into_iter().collect();
    if values.is_empty() {
        return DistributionSummary {
            min: 0.0,
            p25: 0.0,
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
        p25: percentile(&values, 0.25),
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

fn sum_duration<I>(durations: I) -> Duration
where
    I: IntoIterator<Item = Duration>,
{
    Duration::from_secs_f64(
        durations
            .into_iter()
            .map(|duration| duration.as_secs_f64())
            .sum(),
    )
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
        args.generation.max_new_tokens_or(32),
        args.generation.temperature,
        args.generation.top_k,
        args.generation.top_p,
        args.generation.seed,
        args.generation.repeat_penalty,
        args.generation.repeat_last_n,
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
