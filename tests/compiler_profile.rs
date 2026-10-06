use puppygrad::compiler::{cpu, source};
use std::{
    ffi::{c_void, CStr},
    fs,
    process::Command,
};

type Run =
    unsafe extern "C" fn(*const *const c_void, *const *mut c_void, usize, *mut u64, usize) -> i32;

#[test]
fn profiled_abi_matches_plain_execution_and_handles_errors() {
    let source = "a = reshape(param(0, f32, 6), [2, 3])\nb = reshape(param(1, f32, 6), [3, 2])\ny = matmul(a,b)\ni = param(2, i32, 1)\noutput load(index(y,i))\n";
    let program = source::parse(source).unwrap();
    let (code, _) = cpu::emit_profiled(&program).unwrap();
    let (plain, _) = cpu::emit(&program.graph, program.root).unwrap();
    assert!(!plain.contains("pup_clock_ns"));
    assert!(!plain.contains("pup_run_profiled"));
    let dir = std::env::temp_dir().join(format!(
        "puppygrad-profile-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&dir).unwrap();
    fs::write(dir.join("profile.c"), code).unwrap();
    let result = Command::new("cc")
        .args(["-std=c11", "-O2", "-pthread", "-shared", "-fPIC"])
        .arg(dir.join("profile.c"))
        .args(["-lm", "-o"])
        .arg(dir.join("profile.so"))
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    unsafe {
        let library = libloading::Library::new(dir.join("profile.so")).unwrap();
        let metadata_fn = library
            .get::<unsafe extern "C" fn() -> *const std::ffi::c_char>(b"pup_profile_metadata\0")
            .unwrap();
        let metadata: serde_json::Value =
            serde_json::from_slice(CStr::from_ptr(metadata_fn()).to_bytes()).unwrap();
        assert_eq!(
            metadata["reachable_pops"].as_u64().unwrap() as usize,
            program.graph.toposort(program.root).unwrap().len() - 1 // SINK is not executed.
        );
        let run = library.get::<Run>(b"pup_run_profiled\0").unwrap();
        let n = metadata["stats_length"].as_u64().unwrap() as usize;
        let mut stats = vec![0u64; n];
        let a = [1f32, 2., 3., 4., 5., 6.];
        let b = [1f32, 2., 3., 4., 5., 6.];
        let mut index = [1i32];
        let inputs = [
            a.as_ptr().cast(),
            b.as_ptr().cast(),
            index.as_mut_ptr().cast(),
        ];
        let mut out = [0f32; 2];
        let outputs = [out.as_mut_ptr().cast()];
        assert_eq!(
            run(
                inputs.as_ptr(),
                outputs.as_ptr(),
                1,
                stats.as_mut_ptr(),
                n - 1
            ),
            4
        );
        assert_eq!(
            run(
                inputs.as_ptr(),
                outputs.as_ptr(),
                1,
                std::ptr::null_mut(),
                n
            ),
            4
        );
        for threads in [1, 2] {
            assert_eq!(
                run(
                    inputs.as_ptr(),
                    outputs.as_ptr(),
                    threads,
                    stats.as_mut_ptr(),
                    n
                ),
                0
            );
            assert_eq!(out, [49., 64.]);
            assert!(stats[0] > 0);
            let mut kernel_total = 0;
            for kernel in metadata["kernels"].as_array().unwrap() {
                let offset = kernel["stats_offset"].as_u64().unwrap() as usize;
                assert_eq!(stats[offset], 1); // counters reset each invocation
                assert!(stats[offset + 1] >= stats[offset + 2] + stats[offset + 3]);
                kernel_total += stats[offset + 1];
                if kernel["op"] == "Matmul" {
                    assert_eq!(kernel["packed_bytes"], 0); // direct operand reads, no full packing buffers
                    assert_eq!(kernel["matmul_flops"], 24);
                    assert_eq!(kernel["bindings"][0]["name"], "y");
                }
            }
            assert!(stats[0] >= kernel_total + stats[1] + stats[2] + stats[3]);
        }
        std::ptr::write(index.as_mut_ptr(), 2);
        assert_eq!(
            run(inputs.as_ptr(), outputs.as_ptr(), 2, stats.as_mut_ptr(), n),
            2
        );
        assert_eq!(stats[2], 0); // failed before output copy
        std::ptr::write(index.as_mut_ptr(), 0);
        assert_eq!(
            run(inputs.as_ptr(), outputs.as_ptr(), 2, stats.as_mut_ptr(), n),
            0
        );
        assert_eq!(out, [22., 28.]);
        assert_eq!(
            run(inputs.as_ptr(), outputs.as_ptr(), 0, stats.as_mut_ptr(), n),
            3
        );
    }
    fs::remove_dir_all(dir).unwrap();
}
