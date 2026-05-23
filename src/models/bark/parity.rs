use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::{
    build_coarse_window_input, build_semantic_input, build_semantic_input_embeddings,
    decode_encodec_audio, generate_bark_rust_trace, load_bark_causal_transformer_weights,
    load_bark_config, load_bark_encodec_decoder_weights, load_bark_generation_config,
    mask_coarse_logits_for_codebook, mask_semantic_logits, BarkAssetPaths, BarkCausalTransformer,
    BarkRuntimeOptions, BarkTokenizer, Result,
};

const FIXTURE_DIR: &str = "tests/data/bark";
const WAVEFORM_ABS_TOLERANCE: f32 = 2.0e-3;
const WAVEFORM_REL_TOLERANCE: f32 = 5.0e-2;
// Rust and Torch accumulate long f32 transformer reductions in different orders.
// Keep this tight enough to catch layout/weight bugs while allowing CPU drift.
const LOGITS_ABS_TOLERANCE: f32 = 1.0e-2;
const LOGITS_REL_TOLERANCE: f32 = 2.0e-3;

#[derive(Debug, Deserialize)]
struct BarkReferenceFixture {
    metadata: BarkReferenceMetadata,
    tokenizer: BarkReferenceTokenizer,
    semantic: BarkReferenceSemantic,
    coarse: BarkReferenceCoarse,
    fine: BarkReferenceFine,
    decoder: BarkReferenceDecoder,
}

#[derive(Debug, Deserialize)]
struct BarkReferenceMetadata {
    model_dir: String,
    #[allow(dead_code)]
    revision: Option<String>,
    #[allow(dead_code)]
    transformers_version: String,
    #[allow(dead_code)]
    torch_version: String,
    #[allow(dead_code)]
    numpy_version: String,
    #[allow(dead_code)]
    seed: u64,
    text: String,
    #[allow(dead_code)]
    voice_preset: Option<String>,
    #[allow(dead_code)]
    generation_settings: serde_json::Value,
    sample_rate: usize,
    codebook_size: usize,
}

#[derive(Debug, Deserialize)]
struct BarkReferenceTokenizer {
    #[allow(dead_code)]
    normalized_text: Option<String>,
    input_ids: Vec<usize>,
    #[allow(dead_code)]
    attention_mask: Vec<usize>,
    #[allow(dead_code)]
    tokens: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct BarkReferenceSemantic {
    model_input_ids_after_text_offset: Vec<usize>,
    generated_tokens: Vec<usize>,
    #[allow(dead_code)]
    eos_position: Option<usize>,
    #[allow(dead_code)]
    first_logits: Option<BarkReferenceFirstLogits>,
    shape: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct BarkReferenceCoarse {
    #[serde(default)]
    first_window_input_ids: Vec<usize>,
    #[allow(dead_code)]
    first_logits: Option<BarkReferenceFirstLogits>,
    #[allow(dead_code)]
    generated_tokens_flat: Vec<usize>,
    generated_codebooks: Vec<Vec<usize>>,
    output_lengths: Vec<usize>,
    shape: Vec<usize>,
    codebook_shape: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct BarkReferenceFirstLogits {
    shape: Vec<usize>,
    slice: Vec<f32>,
    #[allow(dead_code)]
    top_k: Option<BarkReferenceTopK>,
}

#[derive(Debug, Deserialize)]
struct BarkReferenceTopK {
    #[allow(dead_code)]
    token_ids: Vec<usize>,
    #[allow(dead_code)]
    logits: Vec<f32>,
    #[allow(dead_code)]
    probabilities: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct BarkReferenceFine {
    #[serde(default)]
    input_code_matrix: Vec<Vec<usize>>,
    generated_codebooks: Vec<Vec<usize>>,
    shape: Vec<usize>,
    #[allow(dead_code)]
    first_logits: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct BarkReferenceDecoder {
    quantized_latent_shape: Vec<usize>,
    sample_count: usize,
    min: f32,
    max: f32,
    mean: f32,
    rms: f32,
    first_samples: Vec<f32>,
}

fn fixture_paths() -> Vec<PathBuf> {
    let dir = Path::new(FIXTURE_DIR);
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut paths = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

fn load_fixture(path: &Path) -> Result<BarkReferenceFixture> {
    let text = fs::read_to_string(path).map_err(|err| {
        super::BarkError::Asset(format!("failed to read {}: {err}", path.display()))
    })?;
    serde_json::from_str(&text).map_err(|err| {
        super::BarkError::Asset(format!(
            "failed to parse Bark fixture {}: {err}",
            path.display()
        ))
    })
}

fn read_f32_slice_le(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path).map_err(|err| {
        super::BarkError::Asset(format!("failed to read {}: {err}", path.display()))
    })?;
    if !bytes.len().is_multiple_of(4) {
        return Err(super::BarkError::Asset(format!(
            "float fixture {} has byte length {}, not divisible by 4",
            path.display(),
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn read_u32_code_slice_le(path: &Path) -> Result<Vec<usize>> {
    let bytes = fs::read(path).map_err(|err| {
        super::BarkError::Asset(format!("failed to read {}: {err}", path.display()))
    })?;
    if !bytes.len().is_multiple_of(4) {
        return Err(super::BarkError::Asset(format!(
            "code fixture {} has byte length {}, not divisible by 4",
            path.display(),
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as usize)
        .collect())
}

fn all_fixtures() -> Result<Vec<BarkReferenceFixture>> {
    fixture_paths()
        .iter()
        .map(|path| load_fixture(path))
        .collect::<Result<Vec<_>>>()
}

fn fixture_asset_paths(fixture: &BarkReferenceFixture) -> BarkAssetPaths {
    BarkAssetPaths::new(&fixture.metadata.model_dir)
}

fn waveform_stats(samples: &[f32]) -> (f32, f32, f32, f32) {
    let min = samples.iter().copied().fold(f32::INFINITY, f32::min);
    let max = samples.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    let rms =
        (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32).sqrt();
    (min, max, mean, rms)
}

fn assert_close(actual: f32, expected: f32, label: &str) {
    let diff = (actual - expected).abs();
    let allowed = WAVEFORM_ABS_TOLERANCE.max(expected.abs() * WAVEFORM_REL_TOLERANCE);
    assert!(
        diff <= allowed,
        "{label}: actual {actual} differs from expected {expected} by {diff}, tolerance {allowed}"
    );
}

fn apply_fixture_generation_settings(
    options: &mut BarkRuntimeOptions,
    settings: &serde_json::Value,
) {
    if settings
        .get("greedy")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        options.semantic_sampling.temperature = 0.0;
        options.coarse_sampling.temperature = 0.0;
        options.fine_sampling.temperature = 1.0;
    }
    if let Some(value) = settings
        .get("semantic_temperature")
        .and_then(serde_json::Value::as_f64)
    {
        options.semantic_sampling.temperature = value as f32;
    }
    if let Some(value) = settings
        .get("coarse_temperature")
        .and_then(serde_json::Value::as_f64)
    {
        options.coarse_sampling.temperature = value as f32;
    }
    if let Some(value) = settings
        .get("fine_temperature")
        .and_then(serde_json::Value::as_f64)
    {
        options.fine_sampling.temperature = value as f32;
    }
    if let Some(value) = settings.get("top_k").and_then(serde_json::Value::as_u64) {
        let value = value as usize;
        options.semantic_sampling.top_k = Some(value);
        options.coarse_sampling.top_k = Some(value);
        options.fine_sampling.top_k = Some(value);
    }
    if let Some(value) = settings.get("top_p").and_then(serde_json::Value::as_f64) {
        let value = value as f32;
        options.semantic_sampling.top_p = Some(value);
        options.coarse_sampling.top_p = Some(value);
        options.fine_sampling.top_p = Some(value);
    }
    if let Some(value) = settings
        .get("max_semantic_tokens")
        .and_then(serde_json::Value::as_u64)
    {
        options.max_semantic_tokens = Some(value as usize);
    }
}

fn assert_logits_close(actual: f32, expected: f32, label: &str) {
    if expected.is_infinite() || actual.is_infinite() {
        assert_eq!(
            actual.is_sign_negative(),
            expected.is_sign_negative(),
            "{label} infinite sign mismatch: actual={actual} expected={expected}"
        );
        return;
    }
    let diff = (actual - expected).abs();
    let allowed = LOGITS_ABS_TOLERANCE.max(expected.abs() * LOGITS_REL_TOLERANCE);
    assert!(
        diff <= allowed,
        "{label} mismatch: actual={actual} expected={expected} diff={diff} allowed={allowed}"
    );
}

#[test]
fn parses_compact_bark_reference_fixture() {
    let fixture: BarkReferenceFixture = serde_json::from_str(
        r#"{
          "metadata": {
            "model_dir": "models/bark-small",
            "revision": null,
            "transformers_version": "0.0",
            "torch_version": "0.0",
            "numpy_version": "0.0",
            "seed": 1,
            "text": "hello",
            "voice_preset": null,
            "generation_settings": {},
            "sample_rate": 24000,
            "codebook_size": 1024
          },
          "tokenizer": {
            "input_ids": [101, 7592, 102],
            "attention_mask": [1, 1, 1],
            "tokens": ["[CLS]", "hello", "[SEP]"]
          },
          "semantic": {
            "model_input_ids_after_text_offset": [10149, 17640, 10150],
            "generated_tokens": [1, 2],
            "eos_position": null,
            "first_logits": null,
            "shape": [1, 2]
          },
          "coarse": {
            "first_window_input_ids": [1, 2, 99, 77],
            "first_logits": null,
            "generated_tokens_flat": [10000, 11024],
            "generated_codebooks": [[0], [0]],
            "output_lengths": [2],
            "shape": [1, 2],
            "codebook_shape": [1, 2, 1]
          },
          "fine": {
            "input_code_matrix": [[0], [0]],
            "generated_codebooks": [[0], [0]],
            "shape": [1, 2, 1],
            "first_logits": null
          },
          "decoder": {
            "quantized_latent_shape": [1, 128, 1],
            "sample_count": 2,
            "min": -0.1,
            "max": 0.2,
            "mean": 0.05,
            "rms": 0.15811388,
            "first_samples": [-0.1, 0.2]
          }
        }"#,
    )
    .unwrap();

    assert_eq!(fixture.metadata.text, "hello");
    assert_eq!(fixture.tokenizer.input_ids, [101, 7592, 102]);
    assert_eq!(fixture.coarse.first_window_input_ids, [1, 2, 99, 77]);
    assert_eq!(fixture.fine.input_code_matrix, [vec![0], vec![0]]);
    assert_eq!(fixture.fine.generated_codebooks.len(), 2);
}

#[test]
fn reads_binary_float_and_code_fixture_slices() -> Result<()> {
    let base = std::env::temp_dir().join(format!(
        "puppygrad-bark-binary-fixture-{}",
        std::process::id()
    ));
    let floats = base.with_extension("f32");
    let codes = base.with_extension("u32");
    fs::write(
        &floats,
        [0.25f32, -0.5]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .map_err(|err| super::BarkError::Asset(err.to_string()))?;
    fs::write(
        &codes,
        [7u32, 11]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .map_err(|err| super::BarkError::Asset(err.to_string()))?;

    assert_eq!(read_f32_slice_le(&floats)?, vec![0.25, -0.5]);
    assert_eq!(read_u32_code_slice_le(&codes)?, vec![7, 11]);

    fs::remove_file(&floats).ok();
    fs::remove_file(&codes).ok();
    Ok(())
}

#[test]
fn rejects_misaligned_binary_fixture_slices() {
    let path = std::env::temp_dir().join(format!(
        "puppygrad-bark-bad-binary-fixture-{}.f32",
        std::process::id()
    ));
    fs::write(&path, [1u8, 2, 3]).unwrap();

    let err = read_f32_slice_le(&path).unwrap_err();

    fs::remove_file(&path).ok();
    assert!(err.to_string().contains("not divisible by 4"));
}

#[test]
fn tokenizer_reference_fixtures_match_when_available() -> Result<()> {
    for fixture in all_fixtures()? {
        let paths = fixture_asset_paths(&fixture);
        if !paths.vocab.exists()
            || !paths.tokenizer_config.exists()
            || !paths.generation_config.exists()
        {
            continue;
        }
        let generation_config = load_bark_generation_config(&paths.generation_config)?;
        let tokenizer = BarkTokenizer::from_model_files(
            &paths.vocab,
            &paths.tokenizer_config,
            &paths.special_tokens_map,
            generation_config.semantic_config.max_input_semantic_length,
        )?;

        let encoding = tokenizer.encode_for_processor(&fixture.metadata.text)?;
        assert_eq!(encoding.ids, fixture.tokenizer.input_ids);

        let semantic_input =
            build_semantic_input(&tokenizer, &fixture.metadata.text, &generation_config)?;
        assert_eq!(
            semantic_input.text_semantic_ids,
            fixture.semantic.model_input_ids_after_text_offset
        );
    }
    Ok(())
}

#[test]
fn reference_codebook_layouts_are_valid_when_available() -> Result<()> {
    for fixture in all_fixtures()? {
        if fixture.coarse.generated_codebooks.is_empty()
            || fixture.fine.generated_codebooks.is_empty()
        {
            panic!("Bark fixture contains empty acoustic codebook data");
        }
        let coarse_frames = fixture.coarse.generated_codebooks[0].len();
        assert!(coarse_frames > 0);
        assert!(fixture
            .coarse
            .generated_codebooks
            .iter()
            .all(|row| row.len() == coarse_frames));
        assert_eq!(
            fixture.coarse.codebook_shape,
            vec![1, fixture.coarse.generated_codebooks.len(), coarse_frames]
        );
        assert_eq!(fixture.coarse.output_lengths, vec![fixture.coarse.shape[1]]);
        assert!(fixture
            .coarse
            .generated_codebooks
            .iter()
            .flatten()
            .all(|code| *code < fixture.metadata.codebook_size));

        let fine_frames = fixture.fine.generated_codebooks[0].len();
        assert!(fine_frames > 0);
        assert!(fixture
            .fine
            .generated_codebooks
            .iter()
            .all(|row| row.len() == fine_frames));
        assert_eq!(
            fixture.fine.shape,
            vec![1, fixture.fine.generated_codebooks.len(), fine_frames]
        );
        assert!(fixture
            .fine
            .generated_codebooks
            .iter()
            .flatten()
            .all(|code| *code < fixture.metadata.codebook_size));
        assert_eq!(fixture.semantic.shape[0], 1);
        assert_eq!(
            fixture.semantic.shape[1],
            fixture.semantic.generated_tokens.len()
        );
    }
    Ok(())
}

#[test]
fn coarse_reference_fixtures_match_first_window_layout_when_available() -> Result<()> {
    for fixture in all_fixtures()? {
        if fixture.coarse.first_window_input_ids.is_empty() {
            continue;
        }
        let paths = fixture_asset_paths(&fixture);
        if !paths.generation_config.exists() {
            continue;
        }
        let generation_config = load_bark_generation_config(&paths.generation_config)?;
        let coarse = &generation_config.coarse_acoustics_config;
        let ratio = coarse
            .semantic_to_coarse_token_ratio(generation_config.semantic_config.semantic_rate_hz);
        let max_semantic_history = (coarse.max_coarse_history as f32 / ratio).floor() as usize;
        let semantic_output = fixture
            .semantic
            .generated_tokens
            .iter()
            .map(|token| {
                if *token == generation_config.semantic_config.semantic_pad_token {
                    coarse.coarse_semantic_pad_token
                } else {
                    *token
                }
            })
            .collect::<Vec<_>>();

        let first_window = build_coarse_window_input(
            &semantic_output,
            &[],
            0,
            max_semantic_history,
            &generation_config,
        );

        assert_eq!(first_window, fixture.coarse.first_window_input_ids);
    }
    Ok(())
}

#[test]
fn semantic_logits_reference_fixtures_match_when_available() -> Result<()> {
    for fixture in all_fixtures()? {
        let Some(reference) = &fixture.semantic.first_logits else {
            continue;
        };
        let paths = fixture_asset_paths(&fixture);
        if !paths.native_weights.exists()
            || !paths.config.exists()
            || !paths.generation_config.exists()
            || !paths.vocab.exists()
            || !paths.tokenizer_config.exists()
            || !paths.special_tokens_map.exists()
        {
            continue;
        }
        let config = load_bark_config(&paths.config)?;
        let generation_config = load_bark_generation_config(&paths.generation_config)?;
        let tokenizer = BarkTokenizer::from_model_files(
            &paths.vocab,
            &paths.tokenizer_config,
            &paths.special_tokens_map,
            generation_config.semantic_config.max_input_semantic_length,
        )?;
        let weights =
            load_bark_causal_transformer_weights(&paths, "semantic", &config.semantic_config)?;
        let model = BarkCausalTransformer::new(config.semantic_config.clone(), weights)?;
        let input = build_semantic_input(&tokenizer, &fixture.metadata.text, &generation_config)?;
        let embeddings = build_semantic_input_embeddings(&model, &input, &generation_config)?;
        let mut logits =
            model.last_logits_from_embeddings(&embeddings, input.model_input_ids.len(), 0)?;
        mask_semantic_logits(
            &mut logits,
            generation_config.semantic_config.semantic_vocab_size,
        );

        assert_eq!(reference.shape, vec![1, logits.len()]);
        assert!(reference.slice.len() <= logits.len());
        for (idx, expected) in reference.slice.iter().copied().enumerate() {
            assert_logits_close(
                logits[idx],
                expected,
                &format!("semantic first logit {idx}"),
            );
        }
    }
    Ok(())
}

#[test]
fn coarse_logits_reference_fixtures_match_when_available() -> Result<()> {
    for fixture in all_fixtures()? {
        let Some(reference) = &fixture.coarse.first_logits else {
            continue;
        };
        let paths = fixture_asset_paths(&fixture);
        if !paths.native_weights.exists()
            || !paths.config.exists()
            || !paths.generation_config.exists()
        {
            continue;
        }
        let config = load_bark_config(&paths.config)?;
        let generation_config = load_bark_generation_config(&paths.generation_config)?;
        let weights = load_bark_causal_transformer_weights(
            &paths,
            "coarse_acoustics",
            &config.coarse_acoustics_config,
        )?;
        let model = BarkCausalTransformer::new(config.coarse_acoustics_config.clone(), weights)?;
        let coarse = &generation_config.coarse_acoustics_config;
        let semantic = &generation_config.semantic_config;
        let ratio = coarse.semantic_to_coarse_token_ratio(semantic.semantic_rate_hz);
        let max_semantic_history = (coarse.max_coarse_history as f32 / ratio).floor() as usize;
        let semantic_output = fixture
            .semantic
            .generated_tokens
            .iter()
            .map(|token| {
                if *token == semantic.semantic_pad_token {
                    coarse.coarse_semantic_pad_token
                } else {
                    *token
                }
            })
            .collect::<Vec<_>>();
        let first_window = build_coarse_window_input(
            &semantic_output,
            &[],
            0,
            max_semantic_history,
            &generation_config,
        );
        let mut logits = model.last_logits(&first_window)?;
        mask_coarse_logits_for_codebook(
            &mut logits,
            0,
            generation_config.codebook_size,
            semantic.semantic_vocab_size,
        )?;

        assert_eq!(reference.shape, vec![1, logits.len()]);
        let start = semantic.semantic_vocab_size;
        let end = (start + reference.slice.len()).min(logits.len());
        assert_eq!(end - start, reference.slice.len());
        for (idx, expected) in reference.slice.iter().copied().enumerate() {
            assert_logits_close(
                logits[start + idx],
                expected,
                &format!("coarse first logit {idx}"),
            );
        }
    }
    Ok(())
}

#[test]
fn encodec_decode_reference_fixtures_match_when_available() -> Result<()> {
    for fixture in all_fixtures()? {
        let paths = fixture_asset_paths(&fixture);
        if !paths.native_weights.exists() || !paths.config.exists() {
            continue;
        }
        let config = load_bark_config(&paths.config)?;
        let weights = load_bark_encodec_decoder_weights(&paths, &config.codec_config)?;
        let samples = decode_encodec_audio(
            &fixture.fine.generated_codebooks,
            &config.codec_config,
            &weights,
        )?;

        assert_eq!(
            fixture.metadata.sample_rate,
            config.codec_config.sampling_rate
        );
        assert_eq!(samples.len(), fixture.decoder.sample_count);
        assert!(samples.iter().all(|sample| sample.is_finite()));

        let (min, max, mean, rms) = waveform_stats(&samples);
        assert_close(min, fixture.decoder.min, "waveform min");
        assert_close(max, fixture.decoder.max, "waveform max");
        assert_close(mean, fixture.decoder.mean, "waveform mean");
        assert_close(rms, fixture.decoder.rms, "waveform rms");

        for (idx, expected) in fixture.decoder.first_samples.iter().copied().enumerate() {
            assert_close(samples[idx], expected, &format!("waveform sample {idx}"));
        }
        assert_eq!(
            fixture.decoder.quantized_latent_shape[2],
            fixture.fine.generated_codebooks[0].len()
        );
    }
    Ok(())
}

#[test]
fn voice_preset_reference_fixtures_match_when_explicitly_enabled() -> Result<()> {
    if std::env::var_os("PUPPYGRAD_BARK_RUN_GENERATION_PARITY").is_none() {
        return Ok(());
    }

    for fixture in all_fixtures()?
        .into_iter()
        .filter(|fixture| fixture.metadata.voice_preset.is_some())
    {
        let paths = fixture_asset_paths(&fixture);
        if !paths.native_weights.exists()
            || !paths.config.exists()
            || !paths.generation_config.exists()
        {
            continue;
        }
        let config = load_bark_config(&paths.config)?;
        let generation_config = load_bark_generation_config(&paths.generation_config)?;
        let mut options = BarkRuntimeOptions::from_generation_config(
            &fixture.metadata.text,
            fixture.metadata.voice_preset.clone(),
            fixture.metadata.seed,
            &generation_config,
        );
        apply_fixture_generation_settings(&mut options, &fixture.metadata.generation_settings);
        let trace = generate_bark_rust_trace(&paths, &config, &generation_config, &options)?;

        assert_eq!(trace.semantic_tokens, fixture.semantic.generated_tokens);
        assert_eq!(trace.coarse_codes, fixture.coarse.generated_codebooks);
        assert_eq!(trace.fine_codes, fixture.fine.generated_codebooks);
        assert_eq!(trace.audio.sample_rate, fixture.metadata.sample_rate);
        assert_eq!(trace.audio.samples.len(), fixture.decoder.sample_count);
        assert!(trace.audio.samples.iter().all(|sample| sample.is_finite()));
    }
    Ok(())
}

#[test]
fn full_generation_reference_fixtures_match_when_explicitly_enabled() -> Result<()> {
    if std::env::var_os("PUPPYGRAD_BARK_RUN_GENERATION_PARITY").is_none() {
        return Ok(());
    }

    for fixture in all_fixtures()? {
        let paths = fixture_asset_paths(&fixture);
        if !paths.native_weights.exists()
            || !paths.config.exists()
            || !paths.generation_config.exists()
        {
            continue;
        }
        let config = load_bark_config(&paths.config)?;
        let generation_config = load_bark_generation_config(&paths.generation_config)?;
        let mut options = BarkRuntimeOptions::from_generation_config(
            &fixture.metadata.text,
            fixture.metadata.voice_preset.clone(),
            fixture.metadata.seed,
            &generation_config,
        );
        apply_fixture_generation_settings(&mut options, &fixture.metadata.generation_settings);
        let trace = generate_bark_rust_trace(&paths, &config, &generation_config, &options)?;

        assert_eq!(trace.semantic_tokens, fixture.semantic.generated_tokens);
        assert_eq!(trace.coarse_codes, fixture.coarse.generated_codebooks);
        assert_eq!(trace.fine_codes, fixture.fine.generated_codebooks);
        assert_eq!(trace.audio.sample_rate, fixture.metadata.sample_rate);
        assert_eq!(trace.audio.samples.len(), fixture.decoder.sample_count);
        assert!(!trace.audio.samples.is_empty());
        assert!(trace.audio.samples.iter().all(|sample| sample.is_finite()));
        let duration = trace.audio.samples.len() as f32 / trace.audio.sample_rate as f32;
        let fixture_duration =
            fixture.decoder.sample_count as f32 / fixture.metadata.sample_rate as f32;
        assert_close(
            duration,
            fixture_duration,
            "full generation waveform duration",
        );

        let (min, max, mean, rms) = waveform_stats(&trace.audio.samples);
        assert_close(min, fixture.decoder.min, "full generation waveform min");
        assert_close(max, fixture.decoder.max, "full generation waveform max");
        assert_close(mean, fixture.decoder.mean, "full generation waveform mean");
        assert_close(rms, fixture.decoder.rms, "full generation waveform rms");
    }
    Ok(())
}
