use puppygrad::compiler::{cpu, device::Device, hip, source};
use std::path::Path;

#[test]
fn emits_hip_without_runtime_and_parses_devices() {
    for text in [
        include_str!("../examples/matmul.pup"),
        include_str!("../examples/linear.pup"),
    ] {
        let p = source::parse(text).unwrap();
        let (code, contractions) = hip::emit(&p.graph, p.root).unwrap();
        assert!(contractions > 0);
        assert!(code.contains("extern \"C\" __global__ void kernel"));
        assert!(code.contains("#include <hip/hip_runtime.h>"));
        assert!(!code.contains("__shfl_down_sync"));
    }
    assert_eq!(hip::device_index("hip").unwrap(), 0);
    assert_eq!(hip::device_index("hip:2").unwrap(), 2);
    for device in ["hip:-1", "hip:abc", "hip:", "cuda:0", "opencl"] {
        assert!(hip::device_index(device).is_err());
    }
    assert!(Device::parse("hip:0")
        .unwrap()
        .validate_target(cpu::CpuTarget::Native)
        .is_err());
}

fn programs() -> Vec<String> {
    vec![
        include_str!("../examples/matmul.pup").into(),
        include_str!("../examples/linear.pup").into(),
        "x = reshape(param(0, f32, 165), [5, 33])\noutput rms_norm(x, 1.0, 0.000001)\noutput softmax(x)".into(),
        "x = reshape(param(0, f32, 195), [3, 65])\noutput reduce(x, add, 1)\noutput reduce(x, max, 1)".into(),
        "a = reshape(param(0, f32, 96), [1, 96])\nb = reshape(param(1, f32, 672), [7, 96])\noutput matmul(a, permute(b,[1,0]))".into(),
        "a = reshape(param(0, f32, 561), [17, 33])\nb = reshape(param(1, f32, 627), [33, 19])\noutput matmul(a,b)".into(),
        "x = reshape(param(0, f32, 25), [1,1,5,5])\nw = reshape(param(1, f32, 9), [1,1,3,3])\noutput conv2d(x,w,0.0,1,1)".into(),
        "x = param(0, i32, 3)\noutput x + 1\noutput -x\noutput reduce(x,mul,0)".into(),
        "cache = state(\"cache\", f32, [8,2])\ni = param(0, i32, 2)\nx = reshape(param(1,f32,4),[2,2])\nwrite = store(index(cache,i),x)\noutput after(cache,write)".into(),
        "x = param(0, f32, 33)\nw = param(1, f32, 33)\nloss = reduce((x*w)*(x*w),add,1)\noutput loss\noutput grad(loss,w)".into(),
        "x = reshape(param(0, f32, 6), [3,2])\ni = param(1,i32,2)\noutput load(index(x,i))".into(),
        "a=reshape(param(0,f32,2275),[35,65])\nb=reshape(param(1,f32,4355),[65,67])\noutput silu(matmul(a,b)+0.25)".into(),
        "a=reshape(param(0,f32,2275),[35,65])\nb=reshape(param(1,f32,4355),[67,65])\noutput matmul(flip(a,[false,true]),permute(b,[1,0]))".into(),
        "a=reshape(param(0,f32,4690),[2,35,67])\nb=reshape(param(1,f32,9380),[2,67,70])\noutput batched_matmul(a,b)".into(),
        "a=reshape(param(0,f32,2275),[35,65])\nb=reshape(param(1,f32,4355),[65,67])\noutput matmul(pad(a,[0,1],[35,67]),pad(b,[1,0],[67,67]))".into(),
    ]
}

#[test]
fn hip_reductions_use_explicit_32_lane_shuffles() {
    let p = source::parse(&programs()[2]).unwrap();
    let (code, _) = hip::emit(&p.graph, p.root).unwrap();
    assert!(code.contains("return __shfl_down(x, delta, 32)"));
    assert!(code.contains("return __shfl(x, lane, 32)"));
    assert!(code.contains("// fused row reduction"));
    assert!(code.contains("pup_shfl_down(row_sum"));
    assert!(!code.contains("0xffffffff"));
}

#[test]
#[ignore = "requires HIPRTC, but no GPU"]
fn hiprtc_compiles_examples_and_schedules_for_wave32_and_wave64() {
    for architecture in ["gfx1100", "gfx90a"] {
        for text in programs() {
            let p = source::parse(&text).unwrap();
            let (code, _) = hip::emit(&p.graph, p.root).unwrap();
            let object = hip::compile_source(&code, architecture).unwrap();
            assert!(object.starts_with(b"\x7fELF"));
            assert!(object.len() > 64);
        }
    }
}

fn compare(text: &str, inputs: &[cpu::Tensor]) {
    let p = source::parse(text).unwrap();
    let reference = cpu::compile(&p.graph, p.root, Path::new(".cache/pup/hip-tests/cpu")).unwrap();
    let runtime = hip::Runtime::new(0).unwrap();
    let gpu = hip::compile_with_runtime(
        &p.graph,
        p.root,
        Path::new(".cache/pup/hip-tests/gpu"),
        &runtime,
    )
    .unwrap();
    let expected = reference.run(inputs).unwrap();
    for graph in [true, false] {
        runtime.set_graph_replay(graph);
        for _ in 0..2 {
            let actual = gpu.run(inputs).unwrap();
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(&expected) {
                match (actual, expected) {
                    (cpu::Tensor::F32(a), cpu::Tensor::F32(b)) => {
                        assert_eq!(a.len(), b.len());
                        for (a, b) in a.iter().zip(b.iter()) {
                            assert!(
                                a == b
                                    || (a.is_nan() && b.is_nan())
                                    || (a.is_finite()
                                        && b.is_finite()
                                        && (a - b).abs() <= 2e-5 * (1. + b.abs())),
                                "{a} != {b}"
                            );
                        }
                    }
                    (cpu::Tensor::I32(a), cpu::Tensor::I32(b)) => assert_eq!(a, b),
                    (cpu::Tensor::U8(a), cpu::Tensor::U8(b))
                    | (cpu::Tensor::Bool(a), cpu::Tensor::Bool(b)) => assert_eq!(a, b),
                    _ => panic!("dtype mismatch"),
                }
            }
        }
    }
}
#[test]
#[ignore = "requires AMD GPU and HIPRTC"]
fn hip_matches_cpu_operations_autodiff_contractions_and_convolution() {
    let f = |n| {
        cpu::Tensor::F32(
            (0..n)
                .map(|i| (i as f32 - 3.) / 32.)
                .collect::<Vec<_>>()
                .into(),
        )
    };
    let p = programs();
    compare(&p[0], &[f(6), f(12)]);
    compare(&p[1], &[f(6), f(12), f(4)]);
    compare(&p[2], &[f(165)]);
    compare(&p[3], &[f(195)]);
    compare(&p[4], &[f(96), f(672)]);
    compare(&p[5], &[f(561), f(627)]);
    compare(&p[6], &[f(25), f(9)]);
    compare(
        &p[7],
        &[cpu::Tensor::I32(vec![i32::MIN, i32::MAX, 2].into())],
    );
    compare(&p[9], &[f(33), f(33)]);
}

#[test]
#[ignore = "requires AMD GPU and HIPRTC"]
fn hip_register_tiles_match_cpu_for_layouts_tails_and_epilogues() {
    let f = |n| {
        cpu::Tensor::F32(
            (0..n)
                .map(|i| (i % 29) as f32 / 32. - 0.5)
                .collect::<Vec<_>>()
                .into(),
        )
    };
    let p = programs();
    compare(&p[11], &[f(2275), f(4355)]);
    compare(&p[12], &[f(2275), f(4355)]);
    compare(&p[13], &[f(4690), f(9380)]);
    compare(&p[14], &[f(2275), f(4355)]);
}

#[test]
#[ignore = "requires AMD GPU and HIPRTC"]
fn hip_wide_softmax_matches_cpu_with_nan_infinity_and_signed_zero() {
    for width in [4097, 8192, 40960] {
        let mut values = (0..5 * width)
            .map(|i| (i % 29) as f32 / 8. - 1.)
            .collect::<Vec<_>>();
        values[width + width / 2] = f32::NAN;
        values[2 * width + width - 1] = f32::INFINITY;
        values[3 * width..4 * width].fill(f32::NEG_INFINITY);
        for (i, v) in values[4 * width..].iter_mut().enumerate() {
            *v = if i % 2 == 0 { 0.0 } else { -0.0 };
        }
        compare(
            &format!(
                "x=reshape(param(0,f32,{}),[5,{width}])\noutput softmax(x)",
                5 * width
            ),
            &[cpu::Tensor::F32(values.into())],
        );
    }
}

fn mnist_context() -> source::Context {
    use puppygrad::compiler::pop::DType;
    let mut context = source::Context::default();
    for (slot, (name, shape)) in [
        ("images", vec![128, 28, 28]),
        ("labels", vec![128]),
        ("w1", vec![784, 64]),
        ("b1", vec![64]),
        ("w2", vec![64, 10]),
        ("b2", vec![10]),
        ("learning_rate", vec![]),
        ("valid", vec![128]),
    ]
    .into_iter()
    .enumerate()
    {
        context.tensors.insert(
            name.into(),
            source::TensorSpec {
                slot,
                shape,
                dtype: if slot < 2 { DType::U8 } else { DType::F32 },
            },
        );
    }
    context
}
#[test]
fn hip_emits_mnist_and_autodiff_with_named_external_inputs() {
    for text in [
        include_str!("../examples/mnist.pup"),
        include_str!("../examples/mnist_autodiff.pup"),
    ] {
        let p = source::parse_with_context(text, &mnist_context()).unwrap();
        let (code, contractions) = hip::emit(&p.graph, p.root).unwrap();
        assert!(contractions >= 5);
        assert!(code.contains("__shared__ float"));
        let plan = hip::memory_plan(&p.graph, p.root).unwrap();
        assert_eq!(plan.parameters.len(), 8);
        assert_eq!(plan.output_buffers.len(), 6);
    }
}
#[test]
#[ignore = "requires HIPRTC, but no GPU"]
fn hiprtc_compiles_mnist_manual_and_autodiff_examples() {
    for text in [
        include_str!("../examples/mnist.pup"),
        include_str!("../examples/mnist_autodiff.pup"),
    ] {
        let p = source::parse_with_context(text, &mnist_context()).unwrap();
        let (code, _) = hip::emit(&p.graph, p.root).unwrap();
        for arch in ["gfx1100", "gfx90a"] {
            assert!(hip::compile_source(&code, arch)
                .unwrap()
                .starts_with(b"\x7fELF"));
        }
    }
}
#[test]
#[ignore = "requires AMD GPU and HIPRTC"]
fn hip_bounds_failures_recover_and_weight_copy_on_write_refreshes() {
    let p = source::parse(&programs()[10]).unwrap();
    let runtime = hip::Runtime::new(0).unwrap();
    let e = hip::compile_with_runtime(
        &p.graph,
        p.root,
        Path::new(".cache/pup/hip-tests/bounds"),
        &runtime,
    )
    .unwrap();
    let mut inputs = vec![
        cpu::Tensor::F32(vec![1., 2., 3., 4., 5., 6.].into()),
        cpu::Tensor::I32(vec![0, 2].into()),
    ];
    assert!(e.run(&[]).is_err());
    assert_eq!(e.run(&inputs).unwrap()[0].f32().unwrap(), &[1., 2., 5., 6.]);
    for bad in [-1, 3] {
        inputs[1] = cpu::Tensor::I32(vec![bad, 0].into());
        assert!(e
            .run(&inputs)
            .unwrap_err()
            .to_string()
            .contains("out of bounds"));
    }
    inputs[1] = cpu::Tensor::I32(vec![0, 2].into());
    let old = e.run(&inputs).unwrap();
    let before = runtime.residency_stats();
    e.run(&inputs).unwrap();
    assert_eq!(runtime.residency_stats().allocations, before.allocations);
    assert_eq!(
        runtime.residency_stats().input_uploads,
        before.input_uploads
    );
    if let cpu::Tensor::F32(x) = &mut inputs[0] {
        std::sync::Arc::make_mut(x)[0] = 99.;
    }
    assert_eq!(
        e.run(&inputs).unwrap()[0].f32().unwrap(),
        &[99., 2., 5., 6.]
    );
    assert_eq!(old[0].f32().unwrap(), &[1., 2., 5., 6.]);
    assert_eq!(
        runtime.residency_stats().input_uploads,
        before.input_uploads + 1
    );
    let cached = hip::compile_with_runtime(
        &p.graph,
        p.root,
        Path::new(".cache/pup/hip-tests/bounds"),
        &runtime,
    )
    .unwrap();
    assert!(cached.cache_hit);
}

#[test]
#[ignore = "requires AMD GPU and HIPRTC"]
fn hip_kernel_index_adopts_files_and_rebuilds_changed_or_missing_binaries() {
    let cache =
        std::env::temp_dir().join(format!("puppygrad-hip-kernel-index-{}", std::process::id()));
    std::fs::create_dir_all(&cache).unwrap();
    let db_path = cache.join("puppygrad.db");
    let _database_scope = puppygrad::database::use_path(&db_path).unwrap();
    let program = source::parse("x = param(0, f32, 3)\noutput x * 2").unwrap();
    let runtime = hip::Runtime::new(0).unwrap();
    let first = hip::compile_with_runtime(&program.graph, program.root, &cache, &runtime).unwrap();
    let binary_path = first.source_path.with_extension("hsaco");
    let expected = std::fs::read(&binary_path).unwrap();
    // A pre-index cache remains usable and is registered without compilation.
    rusqlite::Connection::open(&db_path)
        .unwrap()
        .execute("DELETE FROM kernel_modules", [])
        .unwrap();
    let load = || {
        let executable =
            hip::compile_with_runtime(&program.graph, program.root, &cache, &runtime).unwrap();
        assert_eq!(
            executable
                .run(&[cpu::Tensor::F32(vec![1., 2., 3.].into())])
                .unwrap()[0]
                .f32()
                .unwrap(),
            &[2., 4., 6.]
        );
        executable
    };
    assert!(load().cache_hit);
    assert!(load().cache_hit);
    let counts = || {
        let db = rusqlite::Connection::open(&db_path).unwrap();
        db.query_row(
            "SELECT load_count,hit_count,compile_count FROM kernel_modules",
            [],
            |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, i64>(2)?,
                ))
            },
        )
        .unwrap()
    };
    assert_eq!(counts(), (2, 2, 0));
    let mut changed = expected.clone();
    *changed.last_mut().unwrap() ^= 1; // Same length, different SHA-256.
    std::fs::write(&binary_path, changed).unwrap();
    assert!(!load().cache_hit);
    assert_eq!(std::fs::read(&binary_path).unwrap(), expected);
    assert_eq!(counts(), (3, 2, 1));
    std::fs::remove_file(&binary_path).unwrap();
    assert!(!load().cache_hit);
    assert_eq!(counts(), (4, 2, 2));
    let db = rusqlite::Connection::open(&db_path).unwrap();
    let (backend, architecture, source_hash, binary_hash, source_path): (String, String, String, String, String) = db.query_row("SELECT backend,architecture,source_sha256,binary_sha256,source_path FROM kernel_modules", [], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))).unwrap();
    assert_eq!(backend, "hip");
    assert!(!architecture.is_empty());
    assert_eq!((source_hash.len(), binary_hash.len()), (64, 64));
    assert_eq!(cache.join(source_path), first.source_path);
    // Unsupported bookkeeping must leave the valid file cache usable.
    db.execute_batch("PRAGMA user_version=5").unwrap();
    assert!(load().cache_hit);
    assert_eq!(
        db.query_row("PRAGMA user_version", [], |r| r.get::<_, i32>(0))
            .unwrap(),
        5
    );
    drop(db);
    drop(first);
    std::fs::remove_dir_all(cache).unwrap();
}

#[test]
#[ignore = "requires HIPRTC, but no GPU"]
fn hiprtc_compiles_image_ddim_example_stage() {
    let (program, _) =
        puppygrad::models::pup_image::stages::step(include_str!("../examples/image.pup"), 8, 8)
            .unwrap();
    let (code, _) = hip::emit(&program.graph, program.root).unwrap();
    for arch in ["gfx1100", "gfx90a"] {
        assert!(hip::compile_source(&code, arch)
            .unwrap()
            .starts_with(b"\x7fELF"));
    }
}

#[test]
#[ignore = "requires AMD GPU and HIPRTC"]
fn hip_mnist_manual_and_autodiff_steps_match_cpu() {
    let f = |x: Vec<f32>| cpu::Tensor::F32(x.into());
    let mut images = vec![0u8; 128 * 784];
    for (i, pixel) in images[..5 * 784].iter_mut().enumerate() {
        *pixel = (i % 251) as u8;
    }
    let mut labels = vec![0u8; 128];
    labels[..5].copy_from_slice(&[1, 3, 5, 7, 9]);
    let mut valid = vec![0.; 128];
    valid[..5].fill(1.);
    let inputs = vec![
        cpu::Tensor::U8(images.into()),
        cpu::Tensor::U8(labels.into()),
        f(vec![0.002; 784 * 64]),
        f(vec![0.1; 64]),
        f((0..640).map(|i| (i % 19) as f32 * 0.01 - 0.09).collect()),
        f(vec![0.; 10]),
        f(vec![0.1]),
        f(valid),
    ];
    for text in [
        include_str!("../examples/mnist.pup"),
        include_str!("../examples/mnist_autodiff.pup"),
    ] {
        let p = source::parse_with_context(text, &mnist_context()).unwrap();
        let cpu = cpu::compile(
            &p.graph,
            p.root,
            Path::new(".cache/pup/hip-tests/mnist-cpu"),
        )
        .unwrap();
        let gpu =
            hip::compile(&p.graph, p.root, Path::new(".cache/pup/hip-tests/mnist"), 0).unwrap();
        let expected = cpu.run(&inputs).unwrap();
        let actual = gpu.run(&inputs).unwrap();
        assert_eq!(expected.len(), actual.len());
        for (actual, expected) in actual.iter().zip(&expected) {
            for (a, b) in actual.f32().unwrap().iter().zip(expected.f32().unwrap()) {
                assert!((a - b).abs() <= 2e-5 * (1. + b.abs()), "{a} != {b}");
            }
        }
    }
}
