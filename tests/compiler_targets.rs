use puppygrad::compiler::{
    cpu::{self, BuildOptions, CpuTarget, Tensor},
    source,
};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "pup-targets-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn floats(size: usize, scale: f32) -> Tensor {
    Tensor::F32(
        (0..size)
            .map(|i| ((i * 17 % 29) as f32 - 14.) * scale)
            .collect::<Vec<_>>()
            .into(),
    )
}

#[test]
fn targets_preserve_results_and_have_separate_validated_cache_entries() {
    let dir = Temp::new();
    let program = source::parse(
        r#"
a=reshape(param(0,f32,35),[5,7])
b=reshape(param(1,f32,63),[7,9])
bias=param(2,f32,9)
bytes=param(3,u8,8)
y=matmul(a,b)+bias
loss=reduce(y*y,add,2)
output y,grad(loss,a),grad(loss,b),cast(cast(bytes,f32)+1.0,u8)
"#,
    )
    .unwrap();
    let inputs = [
        floats(35, 0.02),
        floats(63, 0.03),
        floats(9, 0.01),
        Tensor::U8(vec![0, 1, 127, 128, 200, 254, 255, 7].into()),
    ];
    let mut reference: Option<Vec<Tensor>> = None;
    let mut paths = std::collections::HashSet::new();
    for target in [CpuTarget::Generic, CpuTarget::Native, CpuTarget::Avx2] {
        if target.validate().is_err() {
            continue;
        }
        let options = BuildOptions { cpu_target: target };
        let exe = cpu::compile_profiled_with_options(&program, &dir.0, &options).unwrap();
        assert!(!exe.cache_hit);
        assert!(paths.insert(exe.source_path.clone()));
        assert_eq!(exe.build_info.cpu_target, target);
        assert!(exe.build_info.flags.contains(&"-ffp-contract=off".into()));
        assert_eq!(
            exe.build_info.native_fingerprint.is_some(),
            target == CpuTarget::Native
        );
        for threads in [1, 2] {
            let outputs = exe.run_profiled(&inputs, threads).unwrap().outputs;
            if let Some(expected) = &reference {
                for (a, b) in outputs.iter().zip(expected) {
                    match (a, b) {
                        (Tensor::F32(a), Tensor::F32(b)) => assert_eq!(
                            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                        ),
                        (Tensor::U8(a), Tensor::U8(b)) => assert_eq!(a, b),
                        _ => panic!("unexpected output type"),
                    }
                }
            } else {
                reference = Some(outputs);
            }
        }
        let hit = cpu::compile_profiled_with_options(&program, &dir.0, &options).unwrap();
        assert!(hit.cache_hit);
        assert_eq!(exe.source_path, hit.source_path);
        let manifest = exe.source_path.with_extension("json");
        let saved: cpu::BuildInfo = serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
        assert_eq!(saved, exe.build_info);
        assert_eq!(
            hit.profile_metadata.as_ref().unwrap()["build"],
            serde_json::to_value(saved).unwrap()
        );
        // Existing code + library cannot qualify as a hit with a wrong manifest.
        fs::write(&manifest, b"{}").unwrap();
        let rebuilt = cpu::compile_profiled_with_options(&program, &dir.0, &options).unwrap();
        assert!(!rebuilt.cache_hit);
        assert_eq!(
            rebuilt.run_with_threads(&inputs, 1).unwrap()[0]
                .f32()
                .unwrap(),
            reference.as_ref().unwrap()[0].f32().unwrap()
        );
    }
}

#[cfg(unix)]
#[test]
fn cache_tracks_compiler_driver_identity() {
    // Each child has its own PATH; do not mutate the test runner's environment.
    if let Some(cache) = std::env::var_os("PUP_TARGET_CACHE_CHILD") {
        let program = source::parse("x=param(0,f32,2)\noutput x*x\n").unwrap();
        let exe = cpu::compile(&program.graph, program.root, Path::new(&cache)).unwrap();
        assert!(!exe.cache_hit);
        assert_eq!(
            exe.run_with_threads(&[Tensor::F32(vec![2., 3.].into())], 1)
                .unwrap()[0]
                .f32()
                .unwrap(),
            &[4., 9.]
        );
        assert!(
            cpu::compile(&program.graph, program.root, Path::new(&cache))
                .unwrap()
                .cache_hit
        );
        fs::write(
            Path::new(&cache).join(std::env::var("PUP_TARGET_RESULT_NAME").unwrap()),
            serde_json::to_vec(&exe.build_info).unwrap(),
        )
        .unwrap();
        return;
    }
    let dir = Temp::new();
    let info = cpu::BuildInfo::resolve(&BuildOptions::default()).unwrap();
    let original_path = std::env::var_os("PATH").unwrap();
    for name in ["driver-a", "driver-b"] {
        let bin = dir.0.join(name);
        fs::create_dir(&bin).unwrap();
        std::os::unix::fs::symlink(&info.compiler_path, bin.join("cc")).unwrap();
        let paths =
            std::env::join_paths(std::iter::once(bin).chain(std::env::split_paths(&original_path)))
                .unwrap();
        let result = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "cache_tracks_compiler_driver_identity",
                "--nocapture",
            ])
            .env("PATH", paths)
            .env("PUP_TARGET_CACHE_CHILD", dir.0.join("cache"))
            .env("PUP_TARGET_RESULT_NAME", name)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
    }
    let cache = dir.0.join("cache");
    assert_eq!(
        fs::read_dir(&cache)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|p| p.path().extension().is_some_and(|s| s == "so"))
            .count(),
        2
    );
    let a: cpu::BuildInfo =
        serde_json::from_slice(&fs::read(cache.join("driver-a")).unwrap()).unwrap();
    let b: cpu::BuildInfo =
        serde_json::from_slice(&fs::read(cache.join("driver-b")).unwrap()).unwrap();
    assert_ne!(a.compiler_path, b.compiler_path);
}

#[test]
fn cli_exposes_targets_and_rejects_invalid_choices() {
    for prefix in [
        &["train"][..],
        &["llm"][..],
        &["run"][..],
        &["llm", "benchmark"][..],
    ] {
        let help = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
            .args(prefix)
            .arg("--help")
            .output()
            .unwrap();
        assert!(help.status.success());
        let text = String::from_utf8_lossy(&help.stdout);
        assert!(
            text.contains("--cpu-target") && text.contains("avx2"),
            "{text}"
        );
        let bad = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
            .args(prefix)
            .args(["missing.pup", "--cpu-target", "imaginary"])
            .output()
            .unwrap();
        assert!(!bad.status.success());
        assert!(String::from_utf8_lossy(&bad.stderr).contains("invalid value"));
    }
}
