use puppygrad::runtime::{
    llm,
    llm_ffi::{Generation, Model, DONE_EOS, DONE_LIMIT},
};
use std::{
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Fixture {
    dir: PathBuf,
    library: PathBuf,
}
impl Fixture {
    fn new(flags: &[&str]) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "puppygrad-llm-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let library = dir.join("model.so");
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let result = Command::new("cc")
            .args(["-shared", "-fPIC", "-O1"])
            .args(flags)
            .arg("-I")
            .arg(root.join("include"))
            .arg(root.join("tests/data/llm/provider.c"))
            .arg("-o")
            .arg(&library)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        Self { dir, library }
    }
    fn model(&self, config: &str) -> Model {
        unsafe { Model::load(&self.library, config.as_bytes()) }.unwrap()
    }
    fn tokenizer(&self) {
        std::fs::write(self.dir.join("tokenizer.json"),r###"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":{"type":"WordPiece","prefix":"##","cleanup":false},"model":{"type":"WordLevel","vocab":{"prompt":0,"hello":1,"world":2,"foo":3,"bar":4,"[UNK]":5},"unk_token":"[UNK]"}}"###).unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}
fn generation() -> Generation {
    Generation {
        max_new_tokens: 3,
        temperature: 0.,
        reserved: 0,
        seed: 42,
    }
}
#[test]
fn c_provider_streams_and_retains_identical_output_and_frees_once() {
    let f = Fixture::new(&[]);
    let mut model = f.model("{}");
    let observer = unsafe { libloading::Library::new(&f.library) }.unwrap();
    let count = |symbol: &[u8]| unsafe {
        observer
            .get::<unsafe extern "C" fn() -> u32>(symbol)
            .unwrap()()
    };
    assert_eq!(count(b"test_live_count\0"), 1);
    let buffered = model.infer(&[0], generation(), None).unwrap();
    assert_eq!(buffered.tokens, [1, 2, 1]);
    assert_eq!(buffered.reason, DONE_LIMIT);
    assert_eq!(count(b"test_callback_count\0"), 0);
    let mut seen = vec![];
    let mut callback = |tokens: &[u32]| {
        seen.extend_from_slice(tokens);
        Ok(())
    };
    let streamed = model
        .infer(&[0], generation(), Some(&mut callback))
        .unwrap();
    assert_eq!(streamed.tokens, buffered.tokens);
    assert_eq!(seen, buffered.tokens);
    assert_eq!(count(b"test_callback_count\0"), 1);
    drop(model);
    assert_eq!(count(b"test_live_count\0"), 0);
    assert_eq!(count(b"test_free_count\0"), 1);
}
#[test]
fn abi_versions_and_table_sizes_are_checked_before_calling_functions() {
    for flag in ["-DTEST_ABI_VERSION=99", "-DTEST_TABLE_SIZE=8"] {
        let f = Fixture::new(&[flag]);
        let error = unsafe { Model::load(&f.library, b"{}") }.err().unwrap();
        assert!(
            error.contains("ABI version") || error.contains("too small"),
            "{error}"
        );
    }
}
#[test]
fn callback_failures_and_contract_violations_return_errors() {
    let f = Fixture::new(&[]);
    for (case, expected) in [
        ("double_done", "more than once"),
        ("no_done", "without on_done"),
        ("after_done", "after completion"),
        ("infer_fail", "deliberate infer failure"),
        ("wrong_output", "differ from retained"),
    ] {
        let mut model = f.model(&format!("{{\"case\":\"{case}\"}}"));
        let mut callback = |_: &[u32]| Ok(());
        let error = model
            .infer(&[0], generation(), Some(&mut callback))
            .unwrap_err();
        assert!(error.contains(expected), "{case}: {error}");
    }
    let mut model = f.model("{}");
    let mut callback = |_: &[u32]| Err("sink failed".into());
    assert_eq!(
        model
            .infer(&[0], generation(), Some(&mut callback))
            .unwrap_err(),
        "sink failed"
    );
    let mut panic_callback = |_: &[u32]| -> Result<(), String> { panic!("consumer panic") };
    assert_eq!(
        model
            .infer(&[0], generation(), Some(&mut panic_callback))
            .unwrap_err(),
        "token callback panicked"
    );
}
#[test]
fn settings_validation_eos_and_empty_generation() {
    let f = Fixture::new(&[]);
    let mut model = f.model("{}");
    let mut settings = generation();
    settings.temperature = 0.75;
    settings.seed = 17;
    assert_eq!(model.infer(&[0], settings, None).unwrap().tokens, [4, 4, 4]);
    let observer = unsafe { libloading::Library::new(&f.library) }.unwrap();
    unsafe {
        assert_eq!(
            observer
                .get::<unsafe extern "C" fn() -> f32>(b"test_temperature\0")
                .unwrap()(),
            0.75
        );
        assert_eq!(
            observer
                .get::<unsafe extern "C" fn() -> u64>(b"test_seed\0")
                .unwrap()(),
            17
        );
    }
    for temperature in [-1., f32::NAN, f32::INFINITY] {
        settings.temperature = temperature;
        assert!(model
            .infer(&[0], settings, None)
            .unwrap_err()
            .contains("temperature"));
    }
    assert!(model
        .infer(&[9], generation(), None)
        .unwrap_err()
        .contains("vocabulary"));
    settings = generation();
    settings.max_new_tokens = 33;
    assert!(model
        .infer(&[0], settings, None)
        .unwrap_err()
        .contains("context"));
    settings.max_new_tokens = 0;
    assert!(model.infer(&[0], settings, None).unwrap().tokens.is_empty());
    let mut eos = f.model("{\"case\":\"eos\"}");
    let out = eos.infer(&[0], generation(), None).unwrap();
    assert_eq!(out.reason, DONE_EOS);
    assert_eq!(out.tokens, [5]);
    assert!(
        unsafe { Model::load(&f.library, b"{\"case\":\"build_fail\"}") }
            .err()
            .unwrap()
            .contains("deliberate build failure")
    );
}
#[test]
fn generic_runtime_streaming_and_buffering_display_identical_text() {
    let f = Fixture::new(&[]);
    f.tokenizer();
    let tokenizer = tokenizers::Tokenizer::from_file(f.dir.join("tokenizer.json")).unwrap();
    let mut model = f.model("{}");
    let mut streamed = vec![];
    llm::generate(
        &mut model,
        &tokenizer,
        &[0],
        generation(),
        true,
        &mut streamed,
    )
    .unwrap();
    let mut buffered = vec![];
    llm::generate(
        &mut model,
        &tokenizer,
        &[0],
        generation(),
        false,
        &mut buffered,
    )
    .unwrap();
    assert_eq!(streamed, buffered);
    assert_eq!(
        String::from_utf8(buffered).unwrap(),
        "prompt hello world hello"
    );
}
#[test]
fn cli_loads_external_llm_library_and_forwards_sampling_options() {
    let f = Fixture::new(&[]);
    f.tokenizer();
    for streaming in [false, true] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_puppygrad"));
        command
            .arg("llm")
            .arg(&f.library)
            .arg("--model-dir")
            .arg(&f.dir)
            .args([
                "--prompt",
                "prompt",
                "--max-new-tokens",
                "2",
                "--temperature",
                "0.75",
                "--seed",
                "17",
            ]);
        if streaming {
            command.arg("--stream");
        }
        let result = command.output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            String::from_utf8(result.stdout).unwrap(),
            "prompt bar bar\n"
        );
    }
    let result = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .args(["llm", "missing.so", "--temperature", "-1"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("temperature must be finite"));
}

#[test]
fn pup_provider_samples_deterministically_and_reuses_owned_state() {
    let f = Fixture::new(&[]);
    let source = f.dir.join("model.pup");
    std::fs::write(&source, "scores = weight(\"scores\")\noutput scores\n").unwrap();
    std::fs::write(
        f.dir.join("config.json"),
        r#"{"vocab_size":6,"n_positions":32,"eos_token_id":5}"#,
    )
    .unwrap();
    let bytes = [-100.0f32, 0., 0., 0., 0., -100.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let view =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![6], &bytes).unwrap();
    std::fs::write(
        f.dir.join("model.safetensors"),
        safetensors::tensor::serialize([("scores", view)], None).unwrap(),
    )
    .unwrap();
    let config =
        serde_json::to_vec(&serde_json::json!({"source":source,"model_dir":f.dir,"device":"cpu","cpu_target":"native"}))
            .unwrap();
    let mut model = unsafe { Model::from_api(puppygrad::models::pup_llm::API, &config) }.unwrap();
    let greedy = model.infer(&[0], generation(), None).unwrap().tokens;
    assert_eq!(greedy, [1, 1, 1]);
    let mut settings = generation();
    settings.max_new_tokens = 12;
    settings.temperature = 1.;
    settings.seed = 17;
    let first = model.infer(&[0], settings, None).unwrap().tokens;
    assert!(
        first.iter().any(|&id| id != 1),
        "temperature must enable sampling"
    );
    let mut seen = vec![];
    let mut callback = |ids: &[u32]| {
        seen.extend_from_slice(ids);
        Ok(())
    };
    let second = model
        .infer(&[0], settings, Some(&mut callback))
        .unwrap()
        .tokens;
    assert_eq!(first, second);
    assert_eq!(first, seen);
    // Exercise CLI target forwarding through the runtime and benchmark worker.
    f.tokenizer();
    for prefix in [&["run"][..], &["llm", "run"][..]] {
        let result = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
            .args(prefix)
            .arg(&source)
            .arg("--model-dir")
            .arg(&f.dir)
            .args([
                "--cpu-target",
                "native",
                "--prompt",
                "prompt",
                "--max-new-tokens",
                "1",
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(String::from_utf8_lossy(&result.stderr).contains("CPU target native"));
    }
    let output = f.dir.join("native-benchmark");
    let result = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .args(["llm", "benchmark"])
        .arg(&source)
        .arg("--model-dir")
        .arg(&f.dir)
        .arg("--output-dir")
        .arg(&output)
        .args([
            "--cpu-target",
            "native",
            "--prompt",
            "prompt",
            "--max-threads",
            "1",
            "--runs",
            "1",
            "--warmups",
            "1",
            "--tokens",
            "1",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let trial: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("threads-01.json")).unwrap()).unwrap();
    assert_eq!(trial["cpu_build"]["cpu_target"], "native");
    assert!(std::fs::read_to_string(output.join("threads-01.log"))
        .unwrap()
        .contains("CPU target native"));
}

#[test]
fn byte_fragment_streaming_matches_buffered_text_including_final_partial_utf8() {
    let f = Fixture::new(&[]);
    let tokenizer=tokenizers::Tokenizer::from_bytes(r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"model":{"type":"WordLevel","vocab":{"prompt":0,"â":1,"Ĥ":2,"¬":3,"x":4,"[UNK]":5},"unk_token":"[UNK]"}}"#.as_bytes()).unwrap();
    let mut model = f.model("{\"case\":\"unicode\"}");
    for (count, expected) in [(2, "prompt�"), (3, "prompt€")] {
        let mut settings = generation();
        settings.max_new_tokens = count;
        let mut streamed = vec![];
        llm::generate(&mut model, &tokenizer, &[0], settings, true, &mut streamed).unwrap();
        let mut buffered = vec![];
        llm::generate(&mut model, &tokenizer, &[0], settings, false, &mut buffered).unwrap();
        assert_eq!(streamed, buffered);
        assert_eq!(String::from_utf8(buffered).unwrap(), expected);
    }
}

#[test]
fn benchmark_cli_sweeps_isolated_workers_and_writes_reusable_results() {
    let f = Fixture::new(&["-DTEST_EXPECT_THREADS"]);
    f.tokenizer();
    let output = f.dir.join("bench-results");
    let max_threads = std::thread::available_parallelism().unwrap().get().min(2);
    let result = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .args(["llm", "benchmark"])
        .arg(&f.library)
        .arg("--model-dir")
        .arg(&f.dir)
        .arg("--output-dir")
        .arg(&output)
        .args([
            "--prompt",
            "prompt",
            "--max-new-tokens",
            "3",
            "--warmups",
            "2",
            "--runs",
            "4",
            "--max-threads",
        ])
        .arg(max_threads.to_string())
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let stdout = String::from_utf8(result.stdout).unwrap();
    assert!(stdout.contains("Median seconds / 3 tokens"));
    let rows: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(output.join("results.json")).unwrap()).unwrap();
    assert_eq!(rows.len(), max_threads);
    let mut pids = std::collections::HashSet::new();
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(row["threads"], index + 1);
        assert!(pids.insert(row["worker_pid"].as_u64().unwrap()));
        assert_eq!(row["warmups"], 2);
        assert_eq!(row["output_token_ids"], serde_json::json!([1, 2, 1]));
        let mut samples: Vec<f64> = row["seconds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert_eq!(samples.len(), 4);
        samples.sort_by(f64::total_cmp);
        let median = (samples[1] + samples[2]) / 2.;
        assert!((row["median_seconds"].as_f64().unwrap() - median).abs() < 1e-12);
        assert!((row["tokens_per_second"].as_f64().unwrap() - 3. / median).abs() < 1e-5);
        assert!(output
            .join(format!("threads-{:02}.log", index + 1))
            .is_file());
    }
    assert_eq!(rows[0]["speedup_vs_one_thread"], 1.0);
    assert_eq!(
        std::fs::read_to_string(output.join("results.csv"))
            .unwrap()
            .lines()
            .count(),
        max_threads + 1
    );
    let markdown = std::fs::read_to_string(output.join("results.md")).unwrap();
    assert!(stdout.starts_with(&markdown));
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("metadata.json")).unwrap()).unwrap();
    assert_eq!(meta["status"], "complete");
    assert_eq!(
        meta["thread_trial_order"].as_array().unwrap().len(),
        max_threads
    );
    let original = std::fs::read(output.join("results.json")).unwrap();
    let retry = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .args(["llm", "benchmark", "--max-threads", "1", "--output-dir"])
        .arg(&output)
        .output()
        .unwrap();
    assert!(!retry.status.success());
    assert!(String::from_utf8_lossy(&retry.stderr).contains("cannot create new results directory"));
    assert_eq!(
        std::fs::read(output.join("results.json")).unwrap(),
        original
    );
}

#[test]
fn benchmark_cli_rejects_invalid_workloads_and_reports_worker_failure() {
    let f = Fixture::new(&[]);
    f.tokenizer();
    for flag in ["--max-threads", "--runs", "--warmups", "--max-new-tokens"] {
        let output = f.dir.join("invalid");
        let result = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
            .args(["llm", "benchmark", flag, "0", "--output-dir"])
            .arg(&output)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("must all be positive"));
        assert!(!output.exists());
    }
    let eos = f.dir.join("eos.so");
    std::fs::copy(&f.library, &eos).unwrap();
    let output = f.dir.join("failed");
    let result = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .args(["llm", "benchmark"])
        .arg(&eos)
        .arg("--model-dir")
        .arg(&f.dir)
        .arg("--output-dir")
        .arg(&output)
        .args(["--prompt", "prompt", "--max-threads", "1", "--tokens", "3"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("threads-01.log"));
    assert!(std::fs::read_to_string(output.join("threads-01.log"))
        .unwrap()
        .contains("EOS shortened benchmark workload"));
    assert!(!output.join("results.csv").exists());
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("metadata.json")).unwrap()).unwrap();
    assert_eq!(meta["status"], "failed");
}

#[test]
fn llm_run_subcommand_and_top_level_run_remain_compatible() {
    let f = Fixture::new(&[]);
    f.tokenizer();
    for prefix in [&["run"][..], &["llm", "run"][..]] {
        let result = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
            .args(prefix)
            .arg(&f.library)
            .arg("--model-dir")
            .arg(&f.dir)
            .args(["--prompt", "prompt", "--max-new-tokens", "2"])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            String::from_utf8(result.stdout).unwrap(),
            "prompt hello world\n"
        );
    }
}

#[test]
fn chat_runtime_hides_template_and_streaming_matches_buffered_output() {
    let f = Fixture::new(&[]);
    f.tokenizer();
    std::fs::write(f.dir.join("config.json"), r#"{"model_type":"qwen3"}"#).unwrap();
    for stream in [false, true] {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_puppygrad"));
        cmd.arg("llm")
            .arg(&f.library)
            .arg("--model-dir")
            .arg(&f.dir)
            .args(["--prompt", "prompt", "--max-new-tokens", "2"]);
        if stream {
            cmd.arg("--stream");
        }
        let output = cmd.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(String::from_utf8(output.stdout).unwrap(), "hello world\n");
    }
}
