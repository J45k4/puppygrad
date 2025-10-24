use std::fs;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn puppygrad_bin() -> &'static str {
    env!("CARGO_BIN_EXE_puppygrad")
}

#[test]
fn emit_c_from_inline_program() {
    let output = Command::new(puppygrad_bin())
        .arg("emit")
        .arg("c")
        .arg("fn add(a, b) { return a + b; }")
        .output()
        .expect("failed to run puppygrad binary");

    assert!(
        output.status.success(),
        "process exited with status {} stderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("stdout was not valid UTF-8");
    assert!(
        stdout.contains("double add"),
        "unexpected output: {}",
        stdout
    );
}

#[test]
fn emit_c_from_file() {
    let dir = std::env::temp_dir();
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time went backwards")
        .as_millis();
    let file_path = dir.join(format!("puppygrad_test_{}.pg", unique));

    fs::write(&file_path, "fn mul(a, b) { return a * b; }\n")
        .expect("failed to write temporary program file");

    let output = Command::new(puppygrad_bin())
        .arg("emit")
        .arg("c")
        .arg(file_path.as_os_str())
        .output()
        .expect("failed to run puppygrad binary");

    let _ = fs::remove_file(&file_path);

    assert!(
        output.status.success(),
        "process exited with status {} stderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("stdout was not valid UTF-8");
    assert!(
        stdout.contains("double mul"),
        "unexpected output: {}",
        stdout
    );
}

#[test]
fn run_inline_program() {
    let output = Command::new(puppygrad_bin())
        .arg("run")
        .arg("fn main() { return 0; }")
        .output()
        .expect("failed to run puppygrad binary");

    assert!(
        output.status.success(),
        "process exited with status {} stderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn run_example_program() {
    let output = Command::new(puppygrad_bin())
        .arg("run")
        .arg("examples/simple.grad")
        .output()
        .expect("failed to run puppygrad binary");

    assert!(
        output.status.success(),
        "process exited with status {} stderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("stdout was not valid UTF-8");
    assert_eq!(
        stdout.trim(),
        "sum 5",
        "unexpected program output: {}",
        stdout
    );
}
