use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use puppygrad::audio::{write_wav_pcm16, PcmAudio};
use puppygrad::models::bark::{
    build_coarse_input, build_coarse_window_input, decode_encodec_audio, generate_coarse_codes,
    generate_fine_codes_with_history, load_bark_causal_transformer_weights, load_bark_config,
    load_bark_encodec_decoder_weights, load_bark_fine_transformer_weights,
    load_bark_generation_config, mask_coarse_logits_for_codebook, BarkAssetPaths,
    BarkCausalTransformer, BarkFineTransformer, BarkRuntimeOptions, Result,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Fixture {
    metadata: Metadata,
    semantic: Semantic,
    coarse: Coarse,
    fine: Fine,
}

#[derive(Debug, Deserialize)]
struct Metadata {
    text: String,
    seed: u64,
    sample_rate: usize,
}

#[derive(Debug, Deserialize)]
struct Semantic {
    generated_tokens: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct Coarse {
    first_logits: Option<FirstLogits>,
    generated_codebooks: Vec<Vec<usize>>,
}

#[derive(Debug, Deserialize)]
struct FirstLogits {
    slice: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct Fine {
    generated_codebooks: Vec<Vec<usize>>,
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let fixture = load_fixture(&args.fixture)?;
    let paths = BarkAssetPaths::new(&args.model_dir);
    let config = load_bark_config(&paths.config)?;
    let generation_config = load_bark_generation_config(&paths.generation_config)?;
    let mut options = BarkRuntimeOptions::from_generation_config(
        fixture.metadata.text.clone(),
        None,
        fixture.metadata.seed,
        &generation_config,
    );
    options.threads = args.threads;
    if let Some(temperature) = args.coarse_temperature {
        options.coarse_sampling.temperature = temperature;
    }

    if args.mode == "coarse-first-logits" {
        print_coarse_first_logits(&fixture, &paths, &config, &generation_config)?;
        return Ok(());
    }
    if args.mode == "coarse-cache-check" {
        print_coarse_cache_check(&fixture, &paths, &config, &generation_config)?;
        return Ok(());
    }
    if args.mode == "coarse-prefix-logits-check" {
        print_coarse_prefix_logits_check(&fixture, &args, &paths, &config, &generation_config)?;
        return Ok(());
    }

    let samples = match args.mode.as_str() {
        "decode-python-fine" => {
            let weights = load_bark_encodec_decoder_weights(&paths, &config.codec_config)?;
            decode_encodec_audio(
                &fixture.fine.generated_codebooks,
                &config.codec_config,
                &weights,
            )?
        }
        "rust-fine-from-python-coarse" => {
            let fine_weights =
                load_bark_fine_transformer_weights(&paths, &config.fine_acoustics_config)?;
            let fine_model = BarkFineTransformer::new_with_threads(
                config.fine_acoustics_config.clone(),
                fine_weights,
                args.threads,
            )?;
            let fine_codes = generate_fine_codes_with_history(
                &fine_model,
                &fixture.coarse.generated_codebooks,
                &generation_config,
                &options,
                None,
            )?;
            let weights = load_bark_encodec_decoder_weights(&paths, &config.codec_config)?;
            decode_encodec_audio(&fine_codes.codebooks, &config.codec_config, &weights)?
        }
        "rust-coarse-fine-from-python-semantic" => {
            let coarse_weights = load_bark_causal_transformer_weights(
                &paths,
                "coarse_acoustics",
                &config.coarse_acoustics_config,
            )?;
            let coarse_model = BarkCausalTransformer::new_with_threads(
                config.coarse_acoustics_config.clone(),
                coarse_weights,
                args.threads,
            )?;
            let semantic_tokens = semantic_tokens_for_rust_coarse(
                &fixture.semantic.generated_tokens,
                &generation_config,
            );
            let coarse_input = build_coarse_input(&semantic_tokens, &generation_config)?;
            let coarse_codes =
                generate_coarse_codes(&coarse_model, &coarse_input, &generation_config, &options)?;
            let fine_weights =
                load_bark_fine_transformer_weights(&paths, &config.fine_acoustics_config)?;
            let fine_model = BarkFineTransformer::new_with_threads(
                config.fine_acoustics_config.clone(),
                fine_weights,
                args.threads,
            )?;
            let fine_codes = generate_fine_codes_with_history(
                &fine_model,
                &coarse_codes,
                &generation_config,
                &options,
                None,
            )?;
            let weights = load_bark_encodec_decoder_weights(&paths, &config.codec_config)?;
            decode_encodec_audio(&fine_codes.codebooks, &config.codec_config, &weights)?
        }
        other => {
            return Err(puppygrad::models::bark::BarkError::InvalidInput(format!(
                "unknown mode {other}"
            )));
        }
    };

    let audio = PcmAudio {
        path: args.out.clone(),
        sample_rate: fixture.metadata.sample_rate,
        channels: 1,
        samples,
    };
    write_wav_pcm16(&args.out, &audio).map_err(|err| {
        puppygrad::models::bark::BarkError::InvalidInput(format!(
            "failed to write {}: {err}",
            args.out.display()
        ))
    })?;
    eprintln!(
        "wrote {} ({:.3}s)",
        args.out.display(),
        audio.duration_seconds()
    );
    Ok(())
}

fn print_coarse_first_logits(
    fixture: &Fixture,
    paths: &BarkAssetPaths,
    config: &puppygrad::models::bark::BarkConfig,
    generation_config: &puppygrad::models::bark::BarkGenerationConfig,
) -> Result<()> {
    let weights = load_bark_causal_transformer_weights(
        paths,
        "coarse_acoustics",
        &config.coarse_acoustics_config,
    )?;
    let model = BarkCausalTransformer::new(config.coarse_acoustics_config.clone(), weights)?;
    let coarse = &generation_config.coarse_acoustics_config;
    let semantic = &generation_config.semantic_config;
    let ratio = coarse.semantic_to_coarse_token_ratio(semantic.semantic_rate_hz);
    let max_semantic_history = (coarse.max_coarse_history as f32 / ratio).floor() as usize;
    let semantic_output =
        semantic_tokens_for_rust_coarse(&fixture.semantic.generated_tokens, generation_config)
            .into_iter()
            .map(|token| {
                if token == semantic.semantic_pad_token {
                    coarse.coarse_semantic_pad_token
                } else {
                    token
                }
            })
            .collect::<Vec<_>>();
    let first_window = build_coarse_window_input(
        &semantic_output,
        &[],
        0,
        max_semantic_history,
        generation_config,
    );
    let mut logits = model.last_logits(&first_window)?;
    mask_coarse_logits_for_codebook(
        &mut logits,
        0,
        generation_config.codebook_size,
        semantic.semantic_vocab_size,
    )?;

    let start = semantic.semantic_vocab_size;
    let reference = fixture.coarse.first_logits.as_ref();
    if let Some(reference) = reference {
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f32;
        for (idx, expected) in reference.slice.iter().copied().enumerate() {
            let diff = (logits[start + idx] - expected).abs();
            max_abs = max_abs.max(diff);
            sum_abs += diff;
        }
        eprintln!(
            "coarse first logits slice diff: mean_abs={:.6} max_abs={:.6}",
            sum_abs / reference.slice.len() as f32,
            max_abs
        );
    }
    let mut top = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .collect::<Vec<_>>();
    top.sort_by(|left, right| right.1.total_cmp(&left.1));
    for (rank, (token, logit)) in top.into_iter().take(10).enumerate() {
        eprintln!(
            "{rank}\ttoken={token}\tcode={}\tlogit={logit:.6}",
            token - start
        );
    }
    Ok(())
}

fn print_coarse_cache_check(
    fixture: &Fixture,
    paths: &BarkAssetPaths,
    config: &puppygrad::models::bark::BarkConfig,
    generation_config: &puppygrad::models::bark::BarkGenerationConfig,
) -> Result<()> {
    let weights = load_bark_causal_transformer_weights(
        paths,
        "coarse_acoustics",
        &config.coarse_acoustics_config,
    )?;
    let model = BarkCausalTransformer::new(config.coarse_acoustics_config.clone(), weights)?;
    let coarse = &generation_config.coarse_acoustics_config;
    let semantic = &generation_config.semantic_config;
    let ratio = coarse.semantic_to_coarse_token_ratio(semantic.semantic_rate_hz);
    let max_semantic_history = (coarse.max_coarse_history as f32 / ratio).floor() as usize;
    let semantic_output =
        semantic_tokens_for_rust_coarse(&fixture.semantic.generated_tokens, generation_config);
    let mut context = build_coarse_window_input(
        &semantic_output,
        &[],
        0,
        max_semantic_history,
        generation_config,
    );
    let first_token = semantic.semantic_vocab_size + fixture.coarse.generated_codebooks[0][0];
    let mut cache = model.new_kv_cache();
    let _ = model.cached_last_logits(&context, &mut cache)?;
    let cached = model.cached_last_logits(&[first_token], &mut cache)?;
    context.push(first_token);
    let full = model.last_logits(&context)?;
    let start = semantic.semantic_vocab_size + generation_config.codebook_size;
    let end = start + generation_config.codebook_size;
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    for idx in start..end {
        let diff = (cached[idx] - full[idx]).abs();
        max_abs = max_abs.max(diff);
        sum_abs += diff;
    }
    eprintln!(
        "coarse cached-vs-full second-token logits: mean_abs={:.6} max_abs={:.6}",
        sum_abs / generation_config.codebook_size as f32,
        max_abs
    );
    Ok(())
}

fn load_fixture(path: &Path) -> Result<Fixture> {
    let text = fs::read_to_string(path).map_err(|err| {
        puppygrad::models::bark::BarkError::Asset(format!(
            "failed to read {}: {err}",
            path.display()
        ))
    })?;
    serde_json::from_str(&text).map_err(|err| {
        puppygrad::models::bark::BarkError::Asset(format!(
            "failed to parse {}: {err}",
            path.display()
        ))
    })
}

fn semantic_tokens_for_rust_coarse(
    tokens: &[usize],
    generation_config: &puppygrad::models::bark::BarkGenerationConfig,
) -> Vec<usize> {
    let semantic = &generation_config.semantic_config;
    tokens
        .iter()
        .copied()
        .take_while(|token| {
            *token != semantic.eos_token_id && *token != semantic.semantic_pad_token
        })
        .collect()
}

#[derive(Debug)]
struct Args {
    mode: String,
    fixture: PathBuf,
    model_dir: PathBuf,
    out: PathBuf,
    threads: usize,
    coarse_temperature: Option<f32>,
    reference_logits: Option<PathBuf>,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut mode = None;
        let mut fixture = None;
        let mut model_dir = PathBuf::from("models/bark-small");
        let mut out = None;
        let mut threads = 8usize;
        let mut coarse_temperature = None;
        let mut reference_logits = None;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--mode" => mode = args.next(),
                "--fixture" => fixture = args.next().map(PathBuf::from),
                "--model-dir" => {
                    model_dir = args.next().map(PathBuf::from).ok_or_else(|| {
                        puppygrad::models::bark::BarkError::InvalidInput(
                            "--model-dir requires a value".to_string(),
                        )
                    })?
                }
                "--out" => out = args.next().map(PathBuf::from),
                "--threads" => {
                    let value = args.next().ok_or_else(|| {
                        puppygrad::models::bark::BarkError::InvalidInput(
                            "--threads requires a value".to_string(),
                        )
                    })?;
                    threads = value.parse().map_err(|err| {
                        puppygrad::models::bark::BarkError::InvalidInput(format!(
                            "invalid --threads value {value}: {err}"
                        ))
                    })?;
                }
                "--coarse-temperature" => {
                    let value = args.next().ok_or_else(|| {
                        puppygrad::models::bark::BarkError::InvalidInput(
                            "--coarse-temperature requires a value".to_string(),
                        )
                    })?;
                    coarse_temperature = Some(value.parse().map_err(|err| {
                        puppygrad::models::bark::BarkError::InvalidInput(format!(
                            "invalid --coarse-temperature value {value}: {err}"
                        ))
                    })?);
                }
                "--reference-logits" => reference_logits = args.next().map(PathBuf::from),
                "--help" | "-h" => {
                    eprintln!(
                        "usage: bark_stage_probe --mode MODE --fixture PATH --out PATH [--model-dir DIR] [--threads N]"
                    );
                    std::process::exit(0);
                }
                other => {
                    return Err(puppygrad::models::bark::BarkError::InvalidInput(format!(
                        "unknown arg {other}"
                    )));
                }
            }
        }

        Ok(Self {
            mode: mode.ok_or_else(|| {
                puppygrad::models::bark::BarkError::InvalidInput("--mode is required".to_string())
            })?,
            fixture: fixture.ok_or_else(|| {
                puppygrad::models::bark::BarkError::InvalidInput(
                    "--fixture is required".to_string(),
                )
            })?,
            model_dir,
            out: out.ok_or_else(|| {
                puppygrad::models::bark::BarkError::InvalidInput("--out is required".to_string())
            })?,
            threads,
            coarse_temperature,
            reference_logits,
        })
    }
}

#[derive(Debug, Deserialize)]
struct PrefixLogits {
    steps: Vec<PrefixLogitStep>,
}

#[derive(Debug, Deserialize)]
struct PrefixLogitStep {
    step: usize,
    token: usize,
    slice_start: usize,
    slice: Vec<f32>,
}

fn print_coarse_prefix_logits_check(
    fixture: &Fixture,
    args: &Args,
    paths: &BarkAssetPaths,
    config: &puppygrad::models::bark::BarkConfig,
    generation_config: &puppygrad::models::bark::BarkGenerationConfig,
) -> Result<()> {
    let path = args.reference_logits.as_ref().ok_or_else(|| {
        puppygrad::models::bark::BarkError::InvalidInput(
            "--reference-logits is required for coarse-prefix-logits-check".to_string(),
        )
    })?;
    let text = fs::read_to_string(path).map_err(|err| {
        puppygrad::models::bark::BarkError::Asset(format!(
            "failed to read {}: {err}",
            path.display()
        ))
    })?;
    let reference: PrefixLogits = serde_json::from_str(&text).map_err(|err| {
        puppygrad::models::bark::BarkError::Asset(format!(
            "failed to parse {}: {err}",
            path.display()
        ))
    })?;

    let weights = load_bark_causal_transformer_weights(
        paths,
        "coarse_acoustics",
        &config.coarse_acoustics_config,
    )?;
    let model = BarkCausalTransformer::new(config.coarse_acoustics_config.clone(), weights)?;
    let coarse = &generation_config.coarse_acoustics_config;
    let semantic = &generation_config.semantic_config;
    let ratio = coarse.semantic_to_coarse_token_ratio(semantic.semantic_rate_hz);
    let max_semantic_history = (coarse.max_coarse_history as f32 / ratio).floor() as usize;
    let semantic_output =
        semantic_tokens_for_rust_coarse(&fixture.semantic.generated_tokens, generation_config);
    let mut context = build_coarse_window_input(
        &semantic_output,
        &[],
        0,
        max_semantic_history,
        generation_config,
    );
    for step in &reference.steps {
        let mut logits = model.last_logits(&context)?;
        mask_coarse_logits_for_codebook(
            &mut logits,
            step.step % coarse.n_coarse_codebooks,
            generation_config.codebook_size,
            semantic.semantic_vocab_size,
        )?;
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f32;
        for (idx, expected) in step.slice.iter().copied().enumerate() {
            let diff = (logits[step.slice_start + idx] - expected).abs();
            max_abs = max_abs.max(diff);
            sum_abs += diff;
        }
        eprintln!(
            "step {}\tmean_abs={:.6}\tmax_abs={:.6}\ttoken_logit={:.6}",
            step.step,
            sum_abs / step.slice.len() as f32,
            max_abs,
            logits[step.token]
        );
        context.push(step.token);
    }
    Ok(())
}
