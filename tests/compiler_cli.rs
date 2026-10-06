use std::process::Command;

#[test]
fn checks_pup_and_prints_primitive_graph() {
    let output = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .args(["check", "examples/matmul.pup", "--dump-pops"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("checked examples/matmul.pup"));
    assert!(stdout.contains("unique Pops"));
    assert!(stdout.contains("Reduce("));
    assert!(stdout.contains("Some([2, 4])"));
    assert!(!stdout.contains("Matmul("));
}

#[test]
fn invalid_source_reports_path_line_and_reason() {
    let output = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .args(["check", "tests/data/compiler/invalid-shape.pup"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("invalid-shape.pup:2:"), "{stderr}");
    assert!(
        stderr.contains("RESHAPE must preserve element count"),
        "{stderr}"
    );
}

#[test]
fn checks_bundled_library_and_user_functions() {
    let output = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .args(["check", "examples/linear.pup", "--dump-pops"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Some([2, 4])"));
    assert!(stdout.contains("Add("));
    assert!(!stdout.contains("Call("));
}

#[test]
fn run_rejects_unimplemented_devices_before_loading_assets() {
    let output = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .args([
            "run",
            "examples/llm.pup",
            "--device",
            "opencl:0",
            "--model-dir",
            "missing-assets",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("use --device cpu"));
}

struct EmitFixture(std::path::PathBuf);
impl EmitFixture {
    fn new() -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("puppygrad-emit-{}-{stamp}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        Self(dir)
    }
    fn emit(&self) -> Command {
        let mut c = Command::new(env!("CARGO_BIN_EXE_puppygrad"));
        c.current_dir(&self.0)
            .env("PATH", "")
            .args(["emit", "--raw"]);
        c
    }
}
impl Drop for EmitFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn emit_writes_only_c_without_a_compiler_or_checkpoint() {
    let f = EmitFixture::new();
    std::fs::write(
        f.0.join("model.pup"),
        include_str!("../examples/matmul.pup"),
    )
    .unwrap();
    let result = f.emit().arg("model.pup").output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(result.stderr.is_empty());
    let code = String::from_utf8(result.stdout).unwrap();
    assert!(code.starts_with("// Emitted from"));
    assert!(code.contains("int pup_run(const void **inputs,void **outputs,size_t threads)"));
    let result = f
        .emit()
        .args(["model.pup", "-o", "model.c"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(result.stdout.is_empty());
    assert_eq!(std::fs::read_to_string(f.0.join("model.c")).unwrap(), code);
    let result = f.emit().args(["model.pup", "-o", "-"]).output().unwrap();
    assert!(result.status.success());
    assert_eq!(result.stdout, code.as_bytes());
    let mut files = std::fs::read_dir(&f.0)
        .unwrap()
        .map(|p| p.unwrap().file_name())
        .collect::<Vec<_>>();
    files.sort();
    assert_eq!(files, ["model.c", "model.pup"]);
}

#[test]
fn pretty_emit_formats_c_and_reports_formatter_failures_without_clobbering_output() {
    let f = EmitFixture::new();
    std::fs::write(
        f.0.join("model.pup"),
        include_str!("../examples/matmul.pup"),
    )
    .unwrap();
    // User formatting configuration must not affect generated output.
    std::fs::write(f.0.join(".clang-format"), "IndentWidth: 17\n").unwrap();
    let emit = || {
        let mut c = Command::new(env!("CARGO_BIN_EXE_puppygrad"));
        c.current_dir(&f.0)
            .args(["emit", "model.pup", "-o", "model.c"]);
        c
    };
    let result = emit().output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(result.stdout.is_empty());
    let code = std::fs::read_to_string(f.0.join("model.c")).unwrap();
    assert!(code.contains("int pup_run(const void **inputs, void **outputs, size_t threads) {"));
    assert!(code.contains("    return err;\n}"));
    let stdout = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .current_dir(&f.0)
        .args(["emit", "model.pup"])
        .output()
        .unwrap();
    assert!(stdout.status.success());
    assert_eq!(stdout.stdout, code.as_bytes());

    // Run the formatted source through a plain C caller to check semantics.
    std::fs::write(
        f.0.join("main.c"),
        r#"
#include "model.c"
int main(void) {
    float a[] = {1, 2, 3, 4, 5, 6};
    float b[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float out[8];
    float expected[] = {38, 44, 50, 56, 83, 98, 113, 128};
    const void *inputs[] = {a, b};
    void *outputs[] = {out};
    if (pup_run(inputs, outputs, 2)) return 1;
    for (int i = 0; i < 8; ++i) if (out[i] != expected[i]) return 2;
    return 0;
}
"#,
    )
    .unwrap();
    let compiled = Command::new("cc")
        .current_dir(&f.0)
        .args(["-std=c11", "-pthread", "main.c", "-lm", "-o", "model"])
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    assert!(Command::new(f.0.join("model")).status().unwrap().success());

    let missing = emit().env("PATH", "").output().unwrap();
    assert!(!missing.status.success());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("install clang-format or use --raw"));
    assert!(missing.stdout.is_empty());
    assert_eq!(std::fs::read_to_string(f.0.join("model.c")).unwrap(), code);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let formatter = f.0.join("clang-format");
        std::fs::write(&formatter, "#!/bin/sh\necho formatter-broke >&2\nexit 1\n").unwrap();
        std::fs::set_permissions(&formatter, std::fs::Permissions::from_mode(0o755)).unwrap();
        let failed = emit().env("PATH", &f.0).output().unwrap();
        assert!(!failed.status.success());
        assert!(String::from_utf8_lossy(&failed.stderr).contains("formatter-broke"));
        assert!(failed.stdout.is_empty());
        assert_eq!(std::fs::read_to_string(f.0.join("model.c")).unwrap(), code);
    }
}

#[test]
fn emit_llm_uses_metadata_bindings_matching_runtime_without_a_tokenizer() {
    use puppygrad::models::pup_llm::Checkpoint;
    use safetensors::tensor::{serialize, Dtype, TensorView};
    let f = EmitFixture::new();
    let dir = f.0.join("models/gpt2");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        r#"{"n_layer":0,"n_head":2,"n_embd":4,"layer_norm_epsilon":0.00001}"#,
    )
    .unwrap();
    // NaN payloads cannot support valid inference, but values are irrelevant to emission.
    let data = vec![f32::NAN; 32]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let views = [
        (
            "transformer.wte.weight",
            TensorView::new(Dtype::F32, vec![6, 4], &data[..96]).unwrap(),
        ),
        (
            "transformer.wpe.weight",
            TensorView::new(Dtype::F32, vec![8, 4], &data).unwrap(),
        ),
        (
            "transformer.ln_f.weight",
            TensorView::new(Dtype::F32, vec![4], &data[..16]).unwrap(),
        ),
        (
            "transformer.ln_f.bias",
            TensorView::new(Dtype::F32, vec![4], &data[..16]).unwrap(),
        ),
        (
            "transformer.h.0.attn.bias",
            TensorView::new(Dtype::F32, vec![1], &data[..4]).unwrap(),
        ),
    ];
    std::fs::write(
        dir.join("model.safetensors"),
        serialize(views, None).unwrap(),
    )
    .unwrap();
    std::fs::write(f.0.join("llm.pup"), include_str!("../examples/llm.pup")).unwrap();
    let result = f
        .emit()
        .args(["./llm.pup", "--sequence-length", "3", "-o", "llm.c"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(result.stdout.is_empty());
    let code = std::fs::read_to_string(f.0.join("llm.c")).unwrap();
    assert!(code.contains("inputs[0]: I32, 3 elements, \"tokens\" [3]"));
    assert!(code.contains("\"wte.weight\" [6, 4]"));
    assert!(!f.0.join(".cache").exists());
    assert!(!dir.join("tokenizer.json").exists());
    let metadata = Checkpoint::metadata_context(&dir, 3).unwrap();
    let mut loaded = Checkpoint::load(&dir).unwrap();
    loaded.bind_tokens(&[0, 0, 0]).unwrap();
    assert_eq!(metadata.tensors.len(), loaded.context.tensors.len());
    for (name, spec) in metadata.tensors {
        let runtime = &loaded.context.tensors[&name];
        assert_eq!(
            (spec.slot, spec.dtype, spec.shape),
            (runtime.slot, runtime.dtype, runtime.shape.clone())
        );
    }
    // Header size/payload consistency is checked without accessing tensor values.
    let path = dir.join("model.safetensors");
    let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.set_len(file.metadata().unwrap().len() - 1).unwrap();
    let result = f
        .emit()
        .args(["llm.pup", "--model-dir", "models/gpt2"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert!(String::from_utf8_lossy(&result.stderr).contains("payload size does not match"));
}

#[test]
fn emit_errors_preserve_existing_output_and_source() {
    let f = EmitFixture::new();
    std::fs::write(
        f.0.join("model.pup"),
        include_str!("data/compiler/invalid-shape.pup"),
    )
    .unwrap();
    std::fs::write(f.0.join("model.c"), "keep existing output").unwrap();
    let result = f
        .emit()
        .args(["model.pup", "-o", "model.c"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("model.pup:2:"));
    assert_eq!(
        std::fs::read_to_string(f.0.join("model.c")).unwrap(),
        "keep existing output"
    );
    let source = "output cast(1, f32)\n";
    std::fs::write(f.0.join("model.pup"), source).unwrap();
    let result = f
        .emit()
        .args(["model.pup", "-o", "model.pup"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert_eq!(
        std::fs::read_to_string(f.0.join("model.pup")).unwrap(),
        source
    );
    let result = f
        .emit()
        .args(["model.pup", "--sequence-length", "0"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr)
        .contains("sequence length must be greater than zero"));
}

#[test]
fn emit_cuda_requires_no_cuda_installation() {
    let f = EmitFixture::new();
    std::fs::write(
        f.0.join("model.pup"),
        include_str!("../examples/matmul.pup"),
    )
    .unwrap();
    let result = f
        .emit()
        .args(["model.pup", "--backend", "cuda"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let code = String::from_utf8(result.stdout).unwrap();
    assert!(code.contains("__global__ void kernel"));
    assert!(!code.contains("pup_pool"));
    let result = f
        .emit()
        .args(["model.pup", "--backend", "cuda", "--profile"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("CUDA profiling"));
}
