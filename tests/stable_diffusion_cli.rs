use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[test]
fn stable_diffusion_rejects_empty_prompt_before_loading_assets() {
    let dir = make_temp_dir("empty-prompt");
    let out = dir.join("out.png");

    let output = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .args([
            "stable-diffusion",
            "--backend",
            "rust",
            "--model-dir",
            dir.to_str().unwrap(),
            "--prompt",
            " \t ",
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    fs::remove_dir_all(&dir).ok();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("prompt must not be empty"), "{stderr}");
    assert!(!stderr.contains("model_index.json"), "{stderr}");
}

#[test]
fn stable_diffusion_rejects_bad_dimensions_before_loading_assets() {
    let dir = make_temp_dir("bad-dimensions");
    let out = dir.join("out.png");

    let output = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .args([
            "stable-diffusion",
            "--backend",
            "rust",
            "--model-dir",
            dir.to_str().unwrap(),
            "--prompt",
            "hello",
            "--width",
            "510",
            "--height",
            "512",
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    fs::remove_dir_all(&dir).ok();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("multiples of 8"), "{stderr}");
    assert!(!stderr.contains("model_index.json"), "{stderr}");
}

#[test]
fn stable_diffusion_rust_backend_does_not_execute_python_backend() {
    let dir = make_temp_dir("rust-no-python");
    let out = dir.join("out.png");

    let output = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .args([
            "stable-diffusion",
            "--backend",
            "rust",
            "--python",
            "/definitely/missing/python-for-sd-test",
            "--model-dir",
            dir.to_str().unwrap(),
            "--prompt",
            "hello",
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    fs::remove_dir_all(&dir).ok();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("missing required files"), "{stderr}");
    assert!(stderr.contains("model_index.json"), "{stderr}");
    assert!(!stderr.contains("failed to start"), "{stderr}");
    assert!(!stderr.contains("python-diffusers backend"), "{stderr}");
}

#[test]
fn stable_diffusion_native_png_smoke_if_model_exists() {
    if std::env::var_os("PUPPYGRAD_SD_RUN_NATIVE_SMOKE").is_none() {
        return;
    }
    let Some(model_dir) = smoke_model_dir() else {
        return;
    };
    let out = std::env::temp_dir().join(format!(
        "puppygrad-sd-native-smoke-{}.png",
        std::process::id()
    ));

    let output = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .args([
            "stable-diffusion",
            "--backend",
            "rust",
            "--model-dir",
            model_dir.to_str().unwrap(),
            "--prompt",
            "hello",
            "--out",
            out.to_str().unwrap(),
            "--steps",
            "1",
            "--width",
            "16",
            "--height",
            "16",
            "--seed",
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
    assert_eq!(png_dimensions(&out).unwrap(), (16, 16));
    fs::remove_file(&out).ok();
}

#[test]
fn stable_diffusion_python_png_smoke_if_enabled() {
    if std::env::var_os("PUPPYGRAD_SD_RUN_PYTHON_SMOKE").is_none() {
        return;
    }
    let Some(model_dir) = smoke_model_dir() else {
        return;
    };
    let out = std::env::temp_dir().join(format!(
        "puppygrad-sd-python-smoke-{}.png",
        std::process::id()
    ));

    let output = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .args([
            "stable-diffusion",
            "--backend",
            "python-diffusers",
            "--model-dir",
            model_dir.to_str().unwrap(),
            "--prompt",
            "hello",
            "--out",
            out.to_str().unwrap(),
            "--steps",
            "1",
            "--width",
            "16",
            "--height",
            "16",
            "--seed",
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
    assert_eq!(png_dimensions(&out).unwrap(), (16, 16));
    fs::remove_file(&out).ok();
}

fn make_temp_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("puppygrad-sd-cli-{label}-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn smoke_model_dir() -> Option<PathBuf> {
    std::env::var_os("PUPPYGRAD_SD_SMOKE_MODEL_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            let path = PathBuf::from("/tmp/puppygrad-sd-narsil");
            path.is_dir().then_some(path)
        })
}

fn png_dimensions(path: &Path) -> Option<(u32, u32)> {
    let bytes = fs::read(path).ok()?;
    if bytes.len() < 24 || !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return None;
    }
    let width = u32::from_be_bytes(bytes[16..20].try_into().ok()?);
    let height = u32::from_be_bytes(bytes[20..24].try_into().ok()?);
    Some((width, height))
}
