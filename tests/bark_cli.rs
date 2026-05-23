use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use puppygrad::audio::load_wav_pcm;

#[test]
fn bark_print_config_smoke_uses_metadata_only_assets() {
    let dir = make_bark_fixture_dir("config");

    let output = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .args([
            "bark",
            "--model-dir",
            dir.to_str().unwrap(),
            "--print-config",
        ])
        .output()
        .unwrap();

    fs::remove_dir_all(&dir).ok();
    assert!(
        output.status.success(),
        "status: {}\nstdout: {}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("model_dir"));
    assert!(stdout.contains("sample_rate"));
    assert!(stdout.contains("semantic"));
}

#[test]
fn bark_print_tokens_smoke_uses_tokenizer_metadata() {
    let dir = make_bark_fixture_dir("tokens");

    let output = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .args([
            "bark",
            "--model-dir",
            dir.to_str().unwrap(),
            "--text",
            "Hello!",
            "--print-tokens",
        ])
        .output()
        .unwrap();

    fs::remove_dir_all(&dir).ok();
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("<s>"));
    assert!(stdout.contains("hello"));
    assert!(stdout.contains("!"));
}

#[test]
fn bark_rust_backend_reports_missing_native_weights() {
    let dir = make_bark_fixture_dir("missing-native");
    let out = dir.join("out.wav");

    let output = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .args([
            "bark",
            "--model-dir",
            dir.to_str().unwrap(),
            "--backend",
            "rust",
            "--text",
            "Hello!",
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    fs::remove_dir_all(&dir).ok();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("bark: starting Rust generation"),
        "{stderr}"
    );
    assert!(stderr.contains("bark: inspect native weights"), "{stderr}");
    assert!(stderr.contains("model.safetensors"), "{stderr}");
    assert!(stderr.contains("pytorch_model.bin"), "{stderr}");
    assert!(stderr.contains("convert"), "{stderr}");
    assert!(
        stderr.contains(dir.join("model.safetensors").to_str().unwrap()),
        "{stderr}"
    );
}

#[test]
fn bark_rust_backend_does_not_execute_python_backend() {
    let dir = make_bark_fixture_dir("rust-no-python");
    let out = dir.join("out.wav");

    let output = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .args([
            "bark",
            "--model-dir",
            dir.to_str().unwrap(),
            "--backend",
            "rust",
            "--python",
            "/definitely/missing/python-for-bark-test",
            "--text",
            "Hello!",
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    fs::remove_dir_all(&dir).ok();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("model.safetensors"), "{stderr}");
    assert!(!stderr.contains("failed to start"), "{stderr}");
    assert!(
        !stderr.contains("python-transformers backend failed"),
        "{stderr}"
    );
}

#[test]
fn bark_generation_overrides_validate_before_loading_weights() {
    let dir = make_bark_fixture_dir("settings-validation");
    let out = dir.join("out.wav");

    let output = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .args([
            "bark",
            "--model-dir",
            dir.to_str().unwrap(),
            "--backend",
            "rust",
            "--text",
            "Hello!",
            "--out",
            out.to_str().unwrap(),
            "--greedy",
            "--top-k",
            "0",
            "--threads",
            "2",
        ])
        .output()
        .unwrap();

    fs::remove_dir_all(&dir).ok();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--top-k must be > 0"), "{stderr}");
    assert!(!stderr.contains("model.safetensors"), "{stderr}");
}

#[test]
fn bark_generation_overrides_are_accepted_until_native_weights_are_needed() {
    let dir = make_bark_fixture_dir("settings-valid");
    let out = dir.join("out.wav");

    let output = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .args([
            "bark",
            "--model-dir",
            dir.to_str().unwrap(),
            "--backend",
            "rust",
            "--text",
            "Hello!",
            "--out",
            out.to_str().unwrap(),
            "--greedy",
            "--semantic-temperature",
            "0.2",
            "--coarse-temperature",
            "0.3",
            "--fine-temperature",
            "0.0",
            "--top-k",
            "1",
            "--top-p",
            "1.0",
            "--max-semantic-tokens",
            "1",
            "--threads",
            "2",
        ])
        .output()
        .unwrap();

    fs::remove_dir_all(&dir).ok();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("model.safetensors"), "{stderr}");
}

#[test]
fn bark_rust_native_wav_smoke_if_model_exists() {
    if std::env::var_os("PUPPYGRAD_BARK_RUN_NATIVE_WAV_SMOKE").is_none() {
        return;
    }
    let model_dir = PathBuf::from("models/bark-small");
    if !model_dir.join("model.safetensors").exists() {
        return;
    }
    let out = std::env::temp_dir().join(format!("puppygrad-bark-rust-{}.wav", std::process::id()));

    let output = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .args([
            "bark",
            "--model-dir",
            model_dir.to_str().unwrap(),
            "--backend",
            "rust",
            "--text",
            "hello",
            "--seed",
            "299792458",
            "--out",
            out.to_str().unwrap(),
            "--greedy",
            "--max-semantic-tokens",
            "1",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let audio = load_wav_pcm(&out).unwrap();
    fs::remove_file(&out).ok();
    assert_eq!(audio.channels, 1);
    assert_eq!(audio.sample_rate, 24_000);
    assert!(!audio.samples.is_empty());
    assert!(audio.samples.iter().all(|sample| sample.is_finite()));
}

fn make_bark_fixture_dir(label: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("puppygrad-bark-cli-{label}-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    write(dir.join("config.json"), CONFIG_JSON);
    write(dir.join("generation_config.json"), GENERATION_CONFIG_JSON);
    write(dir.join("tokenizer_config.json"), TOKENIZER_CONFIG_JSON);
    write(dir.join("special_tokens_map.json"), SPECIAL_TOKENS_JSON);
    write(dir.join("vocab.txt"), VOCAB_TXT);
    dir
}

fn write(path: impl AsRef<Path>, text: &str) {
    fs::write(path, text).unwrap();
}

const CONFIG_JSON: &str = r#"{
  "model_type": "bark",
  "initializer_range": 0.02,
  "semantic_config": {
    "block_size": 8,
    "input_vocab_size": 1000,
    "output_vocab_size": 16,
    "num_layers": 1,
    "num_heads": 1,
    "hidden_size": 4,
    "dropout": 0.0,
    "bias": true,
    "use_cache": true,
    "model_type": "semantic"
  },
  "coarse_acoustics_config": {
    "block_size": 8,
    "input_vocab_size": 1000,
    "output_vocab_size": 32,
    "num_layers": 1,
    "num_heads": 1,
    "hidden_size": 4,
    "dropout": 0.0,
    "bias": true,
    "use_cache": true,
    "model_type": "coarse"
  },
  "fine_acoustics_config": {
    "block_size": 8,
    "input_vocab_size": 1000,
    "output_vocab_size": 16,
    "num_layers": 1,
    "num_heads": 1,
    "hidden_size": 4,
    "dropout": 0.0,
    "bias": true,
    "use_cache": false,
    "model_type": "fine",
    "n_codes_total": 4,
    "n_codes_given": 2
  },
  "codec_config": {
    "sampling_rate": 24000,
    "audio_channels": 1,
    "hidden_size": 4,
    "num_filters": 1,
    "num_residual_layers": 1,
    "codebook_size": 4,
    "num_quantizers": 4,
    "codebook_dim": 4,
    "upsampling_ratios": [2],
    "kernel_size": 3,
    "last_kernel_size": 3,
    "residual_kernel_size": 3,
    "dilation_growth_rate": 2,
    "compress": 1,
    "num_lstm_layers": 1,
    "use_causal_conv": true,
    "trim_right_ratio": 1.0,
    "norm_type": "weight_norm",
    "pad_mode": "reflect",
    "use_conv_shortcut": true,
    "model_type": "encodec"
  }
}"#;

const GENERATION_CONFIG_JSON: &str = r#"{
  "sample_rate": 24000,
  "codebook_size": 4,
  "model_type": "bark",
  "semantic_config": {
    "eos_token_id": 10,
    "max_input_semantic_length": 8,
    "max_new_tokens": 8,
    "semantic_infer_token": 555,
    "semantic_pad_token": 10,
    "semantic_rate_hz": 49.9,
    "semantic_vocab_size": 10,
    "text_encoding_offset": 100,
    "text_pad_token": 999,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 1.0
  },
  "coarse_acoustics_config": {
    "coarse_infer_token": 77,
    "coarse_rate_hz": 75,
    "coarse_semantic_pad_token": 99,
    "max_coarse_history": 8,
    "max_coarse_input_length": 8,
    "n_coarse_codebooks": 2,
    "sliding_window_len": 2,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 1.0
  },
  "fine_acoustics_config": {
    "max_fine_history_length": 8,
    "max_fine_input_length": 8,
    "n_fine_codebooks": 4,
    "temperature": 0.0,
    "top_k": 50,
    "top_p": 1.0
  }
}"#;

const TOKENIZER_CONFIG_JSON: &str = r#"{
  "do_lower_case": true,
  "cls_token": "[CLS]",
  "sep_token": "[SEP]",
  "unk_token": "[UNK]",
  "pad_token": "[PAD]"
}"#;

const SPECIAL_TOKENS_JSON: &str = r#"{
  "cls_token": {"content": "<s>"},
  "sep_token": "</s>",
  "unk_token": "[UNK]",
  "pad_token": "[PAD]"
}"#;

const VOCAB_TXT: &str = "[PAD]\n[UNK]\n<s>\n</s>\nhello\n!\n";
