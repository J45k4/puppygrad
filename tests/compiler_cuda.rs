use puppygrad::compiler::{
    cpu::{self, Tensor},
    cuda, source,
};
use std::path::Path;
fn floats(x: &[f32]) -> Tensor {
    Tensor::F32(x.to_vec().into())
}
fn program(text: &str) -> source::Program {
    source::parse(text).unwrap()
}
fn compare(text: &str, inputs: &[Tensor]) {
    let p = program(text);
    let cpu = cpu::compile(&p.graph, p.root, Path::new(".cache/pup/cuda-tests/cpu")).unwrap();
    let gpu = cuda::compile(&p.graph, p.root, Path::new(".cache/pup/cuda-tests/gpu"), 0).unwrap();
    let expected = cpu.run(inputs).unwrap();
    for _ in 0..2 {
        let actual = gpu.run(inputs).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(&expected) {
            match (a, b) {
                (Tensor::F32(a), Tensor::F32(b)) => {
                    for (a, b) in a.iter().zip(b.iter()) {
                        assert!((a - b).abs() <= 2e-5 * (1. + b.abs()), "{a} != {b}: {text}");
                    }
                }
                (Tensor::I32(a), Tensor::I32(b)) => assert_eq!(a, b),
                (Tensor::Bool(a), Tensor::Bool(b)) | (Tensor::U8(a), Tensor::U8(b)) => {
                    assert_eq!(a, b)
                }
                _ => panic!("dtype mismatch"),
            }
        }
    }
}
#[test]
fn emits_cuda_without_driver_or_toolkit() {
    let p = program(include_str!("../examples/matmul.pup"));
    let (code, gemms) = cuda::emit(&p.graph, p.root).unwrap();
    assert_eq!(gemms, 1);
    assert!(code.contains("__global__ void kernel"));
    assert!(code.contains("blockIdx.x"));
    assert!(!code.contains("pup_pool"));
    assert!(!code.contains("malloc"));
    assert_eq!(cuda::device_index("cuda:2").unwrap(), 2);
    assert!(cuda::device_index("cuda:-1").is_err());
    assert!(cuda::device_index("cuda:abc").is_err());
}
#[test]
#[ignore = "requires NVIDIA GPU and NVRTC; run explicitly with --ignored"]
fn gpu_matches_cpu_operations_views_and_contractions() {
    compare("a = reshape(param(0, f32, 6), [2, 3])\nb = reshape(param(1, f32, 6), [3, 2])\noutput matmul(a, b) - 2.0",&[floats(&[1.,2.,3.,4.,5.,6.]),floats(&[1.,2.,3.,4.,5.,6.])]);
    compare("a = reshape(param(0, f32, 12), [2, 2, 3])\nb = permute(a, [0, 2, 1])\noutput batched_matmul(a, b)",&[floats(&(1..=12).map(|x|x as f32).collect::<Vec<_>>())]);
    compare(
        "x = param(0, f32, 7)\noutput sin(x) + sqrt(x * x + 1.0) + log2(exp2(x))",
        &[floats(&[-2., -1., 0., 0.5, 1., 1.5, 2.])],
    );
    compare(
        "x = reshape(param(0, f32, 6), [2, 3])\noutput layer_norm(x, 1.0, 0.0, 0.00001)",
        &[floats(&[1., 2., 4., -1., 3., 7.])],
    );
    compare(
        "x = param(0, f32, 257)\noutput x + 0.5",
        &[floats(&(0..257).map(|x| x as f32 / 8.).collect::<Vec<_>>())],
    );
    compare(
        "x = param(0, i32, 3)\noutput cast(x < 0, u8)\noutput x + 1",
        &[Tensor::I32(vec![-1, 0, i32::MAX].into())],
    );
    compare("a = reshape(param(0, f32, 0), [2, 0])\nb = reshape(param(1, f32, 0), [0, 3])\noutput matmul(a, b)",&[floats(&[]),floats(&[])]);
    compare("x = param(0, f32, 0)\noutput x", &[floats(&[])]);
    compare("output cast(3.0, f32)\noutput cast(3.0, f32)", &[]);
    compare("x = reshape(param(0, f32, 4), [2, 2])\np = pad(x, [1, 1], [4, 4])\nf = flip(p, [true, false])\ns = shrink(f, [1, 1], [2, 2])\noutput expand(exp2(log2(s)), [2])", &[floats(&[1.,2.,3.,4.])]);
    compare("x = reshape(param(0, i32, 4), [2, 2])\noutput reduce(x, add, 1)\noutput reduce(x, mul, 1)\noutput reduce(x, max, 1)",&[Tensor::I32(vec![i32::MAX,2,1,3].into())]);
    // Exercise convolution's padded window indexing with no broadcast-product allocation.
    compare("x = reshape(param(0, f32, 25), [1, 1, 5, 5])\nw = reshape(param(1, f32, 9), [1, 1, 3, 3])\noutput conv2d(x, w, 0.0, 1, 1)",&[floats(&(0..25).map(|x|x as f32/8.).collect::<Vec<_>>()),floats(&[1.;9])]);
}
#[test]
#[ignore = "requires NVIDIA GPU and NVRTC; run explicitly with --ignored"]
fn gpu_checks_bounds_parameters_refresh_and_ptx_cache() {
    let p=program("x = reshape(param(0, f32, 6), [3, 2])\ni = param(1, i32, 2)\ny = load(index(x, i))\noutput reduce(permute(y, [1, 0]), add, 1)");
    let cache = Path::new(".cache/pup/cuda-tests/bounds");
    let exe = cuda::compile(&p.graph, p.root, cache, 0).unwrap();
    let x = floats(&[1., 2., 3., 4., 5., 6.]);
    assert!(exe
        .run(&[])
        .unwrap_err()
        .to_string()
        .contains("missing input"));
    assert!(exe
        .run(&[floats(&[1.]), Tensor::I32(vec![0, 1].into())])
        .is_err());
    assert!(exe.run(&[x.clone(), floats(&[0., 1.])]).is_err());
    for ids in [vec![-1, 0], vec![3, 0]] {
        assert!(exe
            .run(&[x.clone(), Tensor::I32(ids.into())])
            .unwrap_err()
            .to_string()
            .contains("out of bounds"));
    }
    let out = exe
        .run(&[x.clone(), Tensor::I32(vec![2, 0].into())])
        .unwrap();
    assert_eq!(out[0].f32().unwrap(), &[11., 3.]);
    let out = exe
        .run(&[
            floats(&[2., 4., 6., 8., 10., 12.]),
            Tensor::I32(vec![1, 2].into()),
        ])
        .unwrap();
    assert_eq!(out[0].f32().unwrap(), &[14., 22.]);
    let cached = cuda::compile(&p.graph, p.root, cache, 0).unwrap();
    assert!(cached.cache_hit);
    assert_eq!(
        cached.run(&[x, Tensor::I32(vec![2, 0].into())]).unwrap()[0]
            .f32()
            .unwrap(),
        &[11., 3.]
    );
}

#[test]
#[ignore = "requires NVIDIA GPU and NVRTC; run explicitly with --ignored"]
fn gpu_contractions_preserve_tails_and_updated_weights() {
    let p=program("a = reshape(param(0, f32, 2489), [19, 131])\nb = reshape(param(1, f32, 8777), [67, 131])\noutput matmul(a, permute(b, [1, 0])) + 0.125");
    let gpu = cuda::compile(
        &p.graph,
        p.root,
        Path::new(".cache/pup/cuda-tests/tails"),
        0,
    )
    .unwrap();
    let cpu = cpu::compile(&p.graph, p.root, Path::new(".cache/pup/cuda-tests/cpu")).unwrap();
    assert_eq!(gpu.gemm_count, 1);
    let a = floats(
        &(0..2489)
            .map(|i| ((i % 17) as f32 - 8.) / 8.)
            .collect::<Vec<_>>(),
    );
    for scale in [1., -2.] {
        let b = floats(
            &(0..8777)
                .map(|i| scale * ((i % 13) as f32 - 6.) / 16.)
                .collect::<Vec<_>>(),
        );
        let inputs = [a.clone(), b];
        let expected = cpu.run(&inputs).unwrap();
        let actual = gpu.run(&inputs).unwrap();
        assert_eq!(actual[0].f32().unwrap(), expected[0].f32().unwrap());
    }
}

#[test]
#[ignore = "requires NVIDIA GPU and NVRTC; run explicitly with --ignored"]
fn gpu_retains_inputs_across_shapes_and_refreshes_copy_on_write() {
    use std::sync::Arc;
    let runtime = cuda::Runtime::new(0).unwrap();
    runtime.set_graph_replay(true);
    let cache = Path::new(".cache/pup/cuda-tests/residency");
    let small = program(
        "x = param(0, f32, 2)\nw = param(1, f32, 4)\noutput x + load(index(w, cast(1, i32)))",
    );
    let large = program(
        "x = param(0, f32, 5)\nw = param(1, f32, 4)\noutput x + load(index(w, cast(1, i32)))",
    );
    let a = cuda::compile_with_runtime(&small.graph, small.root, cache, &runtime).unwrap();
    let b = cuda::compile_with_runtime(&large.graph, large.root, cache, &runtime).unwrap();
    let mut weights = floats(&[10., 20., 30., 40.]);
    let first_inputs = [floats(&[1., 2.]), weights.clone()];
    let first = a.run(&first_inputs).unwrap();
    assert_eq!(first[0].f32().unwrap(), &[21., 22.]);
    let cold = runtime.residency_stats();
    assert_eq!(cold.input_uploads, 2);
    assert_eq!(cold.input_uploaded_bytes, 24);
    a.run(&first_inputs).unwrap();
    let warm = runtime.residency_stats();
    assert_eq!(runtime.execution_stats().graph_builds, 1);
    assert_eq!(runtime.execution_stats().graph_launches, 2);
    assert_eq!(runtime.execution_stats().direct_kernel_launches, 0);
    assert_eq!(warm.allocations, cold.allocations);
    assert_eq!(warm.input_uploads, cold.input_uploads);
    let large_inputs = [floats(&[1., 2., 3., 4., 5.]), weights.clone()];
    assert_eq!(
        b.run(&large_inputs).unwrap()[0].f32().unwrap(),
        &[21., 22., 23., 24., 25.]
    );
    let grown = runtime.residency_stats();
    assert_eq!(grown.input_uploaded_bytes - warm.input_uploaded_bytes, 20);
    assert_eq!(grown.input_uploads - warm.input_uploads, 1);
    a.run(&first_inputs).unwrap();
    let shrunk = runtime.residency_stats();
    assert_eq!(
        runtime.execution_stats().graph_builds,
        3,
        "small executable must rebuild after shared buffers grow"
    );
    assert_eq!(shrunk.allocations, grown.allocations);
    assert_eq!(shrunk.input_uploaded_bytes - grown.input_uploaded_bytes, 8);
    // Retaining the host Arc makes replacement/copy-on-write updates observable.
    if let Tensor::F32(ref mut data) = weights {
        Arc::make_mut(data)[1] = -5.;
    }
    let changed = a.run(&[first_inputs[0].clone(), weights]).unwrap();
    assert_eq!(changed[0].f32().unwrap(), &[-4., -3.]);
    assert_eq!(
        runtime.execution_stats().graph_builds,
        3,
        "new input contents reuse recorded addresses"
    );
    assert_eq!(
        runtime.residency_stats().input_uploaded_bytes - shrunk.input_uploaded_bytes,
        16
    );
    assert_eq!(
        first[0].f32().unwrap(),
        &[21., 22.],
        "host outputs own their data"
    );
    // Each executable keeps the runtime alive; dropping an external owner is safe.
    drop(runtime);
    assert_eq!(a.run(&first_inputs).unwrap()[0].f32().unwrap(), &[21., 22.]);
    drop(b);
    drop(a);
}

#[test]
fn emits_generic_tiling_and_bounded_pointwise_fusion() {
    let p = program("a = reshape(param(0, f32, 2489), [19, 131])\nb = reshape(param(1, f32, 8777), [67, 131])\noutput matmul(a, permute(b, [1, 0]))");
    let (code, gemms) = cuda::emit(&p.graph, p.root).unwrap();
    assert_eq!(gemms, 1);
    assert!(code.contains("__shared__ float sa[16*32],sb[32*16]"));
    assert!(code.contains("__syncthreads()"));
    let p = program("x = param(0, f32, 257)\noutput sin(x * x + 1.0) * 0.5 + 2.0");
    let (code, _) = cuda::emit(&p.graph, p.root).unwrap();
    assert_eq!(code.matches("__global__ void kernel").count(), 1);
    assert!(code.contains("sinf"));
}

#[test]
fn fuses_contraction_epilogues_and_literal_ranges() {
    let p = program("a=reshape(param(0,f32,2489),[19,131])\nb=reshape(param(1,f32,8777),[67,131])\noutput silu(matmul(a,permute(b,[1,0]))+param(2,f32,67))*0.5");
    let (code, gemms) = cuda::emit(&p.graph, p.root).unwrap();
    assert_eq!(gemms, 1);
    assert_eq!(code.matches("__global__ void kernel").count(), 1);
    assert!(code.contains("exp2f"));
    let p = program("x=param(0,f32,129)\nr=cast(arange(129),f32)\noutput x+r\noutput x-r");
    let (code, _) = cuda::emit(&p.graph, p.root).unwrap();
    assert_eq!(code.matches("__global__ void kernel").count(), 3);
    // The shared non-scalar cast stays materialized; its integer range and
    // shared scalar elements are indexing expressions rather than kernels.
    assert!(!code.contains("i>=0 && i<1"));
    let p = program("a=reshape(param(0,f32,64),[2,32])\nb=reshape(param(1,f32,96),[3,32])\ny=matmul(a,permute(b,[1,0]))\noutput y\noutput y+1.0");
    let (code, _) = cuda::emit(&p.graph, p.root).unwrap();
    assert_eq!(code.matches("__global__ void kernel").count(), 2);
    let p = program("s=state(\"epilogue\",f32,[1,32])\ny=matmul(s,reshape(param(0,f32,32),[32,1]))\nw=store(index(s,cast(stack(0),i32)),reshape(param(1,f32,32),[1,32]))\nb=after(reshape(param(2,f32,1),[1,1]),w)\noutput y+b");
    let (code, _) = cuda::emit(&p.graph, p.root).unwrap();
    assert!(
        code.contains("=acc;"),
        "must retain the pre-store contraction"
    );
}

#[test]
#[ignore = "requires NVIDIA GPU and NVRTC"]
fn gpu_contraction_epilogues_and_literal_views_match_cpu() {
    for (m, k, n) in [(2, 3, 2), (3, 33, 9), (19, 131, 67)] {
        let p = format!("a=reshape(param(0,f32,{}),[{m},{k}])\nb=reshape(param(1,f32,{}),[{n},{k}])\nc=reshape(param(3,f32,{}),[{n},{k}])\nx=matmul(a,permute(b,[1,0]))\nu=matmul(a,permute(c,[1,0]))+0.5\noutput silu(x+param(2,f32,{n}))*u",m*k,n*k,n*k);
        compare(
            &p,
            &[
                floats(
                    &(0..m * k)
                        .map(|i| (i % 13) as f32 / 32. - 0.2)
                        .collect::<Vec<_>>(),
                ),
                floats(
                    &(0..n * k)
                        .map(|i| (i % 7) as f32 / 16. - 0.1)
                        .collect::<Vec<_>>(),
                ),
                floats(&(0..n).map(|i| i as f32 / 16.).collect::<Vec<_>>()),
                floats(
                    &(0..n * k)
                        .map(|i| (i % 11) as f32 / 32. - 0.1)
                        .collect::<Vec<_>>(),
                ),
            ],
        );
    }
    let mut text = String::from(
        "a=reshape(param(0,f32,6),[2,3])\nb=reshape(param(1,f32,6),[3,2])\nx0=matmul(a,b)+0.25\n",
    );
    for i in 0..12 {
        text.push_str(&format!("x{}=x{i}+x{i}\n", i + 1));
    }
    text.push_str("output x12");
    let p = program(&text);
    let (code, _) = cuda::emit(&p.graph, p.root).unwrap();
    assert!(
        code.len() < 40000,
        "repeated internal uses must not explode emitted expressions"
    );
    compare(
        &text,
        &[
            floats(&[1., 2., 3., 4., 5., 6.]),
            floats(&[1., 2., 3., 4., 5., 6.]),
        ],
    );
    compare("r=reshape(cast(arange(129),f32),[3,43])\nx=permute(r,[1,0])\noutput x+param(0,f32,3)\noutput flip(r,[false,true])", &[floats(&[0.5,-1.,2.])]);
    compare("r=cast(stack(9,6,3,0,-3),i32)\noutput r\noutput cast(stack(1,4,2),i32)\noutput cast(stack(2147483647,2147483648),i32)", &[]);
}

#[test]
fn fuses_private_stack_store_without_losing_overlap_snapshots() {
    let p=program("x=reshape(param(0,f32,6),[2,3])\ns=state(\"stack\",f32,[2,6])\npayload=reshape(permute(stack(x+1.0,x-1.0),[1,0,2]),[2,6])\nw=store(index(s,cast(arange(2),i32)),payload)\noutput after(s,w)");
    let (code, _) = cuda::emit(&p.graph, p.root).unwrap();
    assert_eq!(code.matches("__global__ void kernel").count(), 2, "{code}");
    let p=program("s=state(\"stack\",f32,[2,3])\npayload=reshape(stack(shrink(s,[0,0],[1,3]),shrink(s,[1,0],[1,3])),[2,3])\nw=store(index(s,cast(stack(1,0),i32)),payload)\noutput after(s,w)");
    let (code, _) = cuda::emit(&p.graph, p.root).unwrap();
    assert!(
        code.contains("i>=0 && i<3"),
        "overlapping stack stays materialized"
    );
}

#[test]
#[ignore = "requires NVIDIA GPU and NVRTC"]
fn gpu_fused_stores_and_epilogue_state_ordering() {
    compare("x=reshape(param(0,f32,6),[2,3])\ns=state(\"stack\",f32,[2,6])\npayload=reshape(permute(stack(x+1.0,x-1.0),[1,0,2]),[2,6])\nw=store(index(s,cast(arange(2),i32)),payload)\noutput after(s,w)", &[floats(&[1.,2.,3.,4.,5.,6.])]);
    compare("x=reshape(param(0,f32,6),[2,3])\ns=state(\"stack\",f32,[2,3])\nw=store(index(s,cast(arange(2),i32)),x)\na=after(s,w)\npayload=reshape(stack(shrink(a,[0,0],[1,3]),shrink(a,[1,0],[1,3])),[2,3])\nc=store(index(a,cast(stack(1,0),i32)),payload)\noutput after(a,c)", &[floats(&[1.,2.,3.,4.,5.,6.])]);
    // The contraction reads the old state even though its eventual bias is
    // explicitly dependent on a write. Check against known answers rather
    // than another backend's fusion planner.
    let p=program("s=state(\"epilogue\",f32,[1,32])\ny=matmul(s,reshape(param(0,f32,32),[32,1]))\nw=store(index(s,cast(stack(0),i32)),reshape(param(1,f32,32),[1,32]))\nb=after(reshape(param(2,f32,1),[1,1]),w)\noutput y+b");
    let gpu = cuda::compile(
        &p.graph,
        p.root,
        Path::new(".cache/pup/cuda-tests/epilogue-state"),
        0,
    )
    .unwrap();
    let inputs = [floats(&[1.; 32]), floats(&[2.; 32]), floats(&[3.])];
    assert_eq!(gpu.run(&inputs).unwrap()[0].f32().unwrap(), &[3.]);
    assert_eq!(gpu.run(&inputs).unwrap()[0].f32().unwrap(), &[67.]);
}

#[test]
fn fuses_primitive_row_normalization_and_keeps_shared_reductions() {
    for expression in ["rms_norm(x, param(1, f32, 129), 0.000001)", "softmax(x)"] {
        let p = program(&format!(
            "x = reshape(param(0, f32, 1161), [9,129])\noutput {expression}"
        ));
        let (code, _) = cuda::emit(&p.graph, p.root).unwrap();
        assert_eq!(code.matches("__global__ void kernel").count(), 1, "{code}");
        assert!(code.contains("fused row reduction"));
    }
    let p = program("x = reshape(param(0, f32, 1161), [9,129])\noutput reduce(permute(x*x,[1,0]),add,1)\noutput rms_norm(x, param(1,f32,129),0.000001)");
    let (code, _) = cuda::emit(&p.graph, p.root).unwrap();
    assert!(!code.contains("fused row reduction"));
    assert!(code.contains("__shfl_down_sync"));
    let p = program("s=state(\"row\",f32,[1,33])\nx=s\nw=store(index(s,cast(stack(0),i32)),reshape(cast(arange(33),f32),[1,33]))\nweight=after(param(0,f32,33),w)\noutput rms_norm(x,weight,0.000001)");
    let (code, _) = cuda::emit(&p.graph, p.root).unwrap();
    assert!(
        !code.contains("fused row reduction"),
        "fusion cannot delay statistics across a store"
    );
}

#[test]
#[ignore = "requires NVIDIA GPU and NVRTC"]
fn gpu_parallel_reductions_and_row_fusions_match_cpu_with_tails() {
    for (rows, width) in [(1, 32), (3, 33), (9, 129), (2, 1025)] {
        let input = floats(
            &(0..rows * width)
                .map(|i| ((i * 7 % 31) as f32 - 15.) / 32.)
                .collect::<Vec<_>>(),
        );
        let header = format!(
            "x = reshape(param(0,f32,{}),[{rows},{width}])\n",
            rows * width
        );
        for expression in [
            "softmax(x)",
            "rms_norm(x,1.25,0.000001)",
            "layer_norm(x,1.25,-0.5,0.000001)",
        ] {
            compare(&(header.clone() + "output " + expression), &[input.clone()]);
        }
        compare(
            &(header + "output reduce(permute(x,[1,0]),add,1)\noutput reduce(x,add,2)"),
            &[input],
        );
    }
    let p = program("x = param(0,f32,65)\noutput reduce(x,max,1)");
    let cpu = cpu::compile(&p.graph, p.root, Path::new(".cache/pup/cuda-tests/cpu")).unwrap();
    let gpu = cuda::compile(&p.graph, p.root, Path::new(".cache/pup/cuda-tests/gpu"), 0).unwrap();
    for nan_at in [None, Some(0), Some(31), Some(63), Some(64)] {
        let mut input = vec![-0.0; 65];
        input[0] = 100.;
        input[64] = 0.;
        if let Some(i) = nan_at {
            input[i] = f32::NAN;
        }
        let input = [floats(&input)];
        let expected = cpu.run(&input).unwrap()[0].f32().unwrap()[0];
        let actual = gpu.run(&input).unwrap()[0].f32().unwrap()[0];
        assert!(
            if expected.is_nan() {
                actual.is_nan()
            } else {
                actual.to_bits() == expected.to_bits()
            },
            "{nan_at:?}: {actual} != {expected}"
        );
    }
    compare(
        "x=param(0,f32,97)\noutput reduce(x,mul,1)",
        &[floats(
            &(0..97)
                .map(|i| 1. + (i % 3) as f32 / 1024.)
                .collect::<Vec<_>>(),
        )],
    );
}

#[test]
#[ignore = "requires NVIDIA GPU and NVRTC; run explicitly with --ignored"]
fn gpu_tiled_batched_and_noncontiguous_views_match_cpu() {
    let a = floats(
        &(0..1330)
            .map(|i| (i % 17) as f32 / 16. - 0.5)
            .collect::<Vec<_>>(),
    );
    let b = floats(
        &(0..1190)
            .map(|i| (i % 13) as f32 / 16. - 0.25)
            .collect::<Vec<_>>(),
    );
    compare("a = reshape(param(0, f32, 1330), [2, 19, 35])\nb = reshape(param(1, f32, 1190), [2, 35, 17])\noutput batched_matmul(a, b)", &[a.clone(), b.clone()]);
    compare("a = reshape(param(0, f32, 1330), [2, 19, 35])\nb = reshape(param(1, f32, 1190), [2, 17, 35])\noutput batched_matmul(flip(a, [false, false, true]), permute(b, [0, 2, 1]))", &[a, b]);
    let a = floats(
        &(0..19 * 35)
            .map(|i| (i % 17) as f32 / 16. - 0.5)
            .collect::<Vec<_>>(),
    );
    let b = floats(
        &(0..35 * 17)
            .map(|i| (i % 13) as f32 / 16. - 0.25)
            .collect::<Vec<_>>(),
    );
    compare("a = reshape(param(0, f32, 665), [19, 35])\nb = reshape(param(1, f32, 595), [35, 17])\noutput matmul(a * 0.5 + 0.125, b * 2.0) + 0.25", &[a.clone(), b.clone()]);
    compare("a = reshape(param(0, f32, 665), [19, 35])\nb = reshape(param(1, f32, 595), [35, 17])\nap = pad(a, [0, 1], [19, 37])\nbp = pad(b, [1, 0], [37, 17])\noutput matmul(ap, bp)", &[a,b]);
    compare("x = reshape(param(0, f32, 196), [1, 4, 7, 7])\nw = reshape(param(1, f32, 612), [17, 4, 3, 3])\noutput conv2d(x, w, 0.0, 1, 1)", &[
        floats(&(0..196).map(|i| (i % 17) as f32 / 16. - 0.5).collect::<Vec<_>>()),
        floats(&(0..612).map(|i| (i % 13) as f32 / 16. - 0.25).collect::<Vec<_>>()),
    ]);
}

#[test]
#[ignore = "requires NVIDIA GPU and NVRTC; run explicitly with --ignored"]
fn gpu_warp_contractions_preserve_small_batches_and_reduction_tails() {
    compare("a = reshape(param(0, f32, 0), [0, 35])\nb = reshape(param(1, f32, 665), [35, 19])\noutput matmul(a, b)", &[floats(&[]),floats(&vec![0.;665])]);
    for rows in [1, 3, 15] {
        let a = floats(
            &(0..rows * 35)
                .map(|i| (i % 17) as f32 / 16. - 0.5)
                .collect::<Vec<_>>(),
        );
        let b = floats(
            &(0..19 * 35)
                .map(|i| (i % 23) as f32 / 32. - 0.25)
                .collect::<Vec<_>>(),
        );
        compare(&format!("a = reshape(param(0, f32, {}), [{rows}, 35])\nb = reshape(param(1, f32, 665), [19, 35])\noutput matmul(a, permute(b, [1, 0]))",rows*35), &[a,b]);
    }
    let a = floats(
        &(0..2 * 3 * 35)
            .map(|i| (i % 19) as f32 / 16.)
            .collect::<Vec<_>>(),
    );
    let b = floats(
        &(0..2 * 19 * 35)
            .map(|i| (i % 17) as f32 / 32.)
            .collect::<Vec<_>>(),
    );
    compare("a = reshape(param(0, f32, 210), [2, 3, 35])\nb = reshape(param(1, f32, 1330), [2, 19, 35])\noutput batched_matmul(a, permute(b, [0, 2, 1]))", &[a,b]);
}
