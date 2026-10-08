//! Native BF16 storage, F32 arithmetic/accumulation, and shared backend parity.
use puppygrad::compiler::{
    cpu::{self, Tensor},
    cuda, gpu, hip,
    pop::DType,
    source,
};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "puppygrad-bf16-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn bf16(values: &[f32]) -> Tensor {
    Tensor::BF16(
        values
            .iter()
            .map(|&x| half::bf16::from_f32(x).to_bits())
            .collect::<Vec<_>>()
            .into(),
    )
}
fn compare(a: &[Tensor], b: &[Tensor]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        match (a, b) {
            (Tensor::BF16(a), Tensor::BF16(b)) => assert_eq!(a, b),
            (Tensor::F32(a), Tensor::F32(b)) => {
                assert_eq!(a.len(), b.len());
                for (&a, &b) in a.iter().zip(b.iter()) {
                    if b.is_nan() || b.is_infinite() || b == 0. {
                        assert_eq!(a.to_bits(), b.to_bits());
                    } else {
                        assert!((a - b).abs() <= 5e-5 * b.abs().max(1.), "{a} != {b}");
                    }
                }
            }
            _ => panic!("different tensor types: {a:?} {b:?}"),
        }
    }
}
fn cases() -> Vec<(String, Vec<Tensor>, Vec<Tensor>)> {
    let bits: Vec<_> = (0..=u16::MAX).collect();
    let widened: Vec<_> = bits
        .iter()
        .map(|&x| half::bf16::from_bits(x).to_f32())
        .collect();
    let roundtrip = bf16(&widened);
    let floats: Vec<_> = (0..4096u32)
        .map(|x| f32::from_bits(x.wrapping_mul(0x9e3779b9)))
        .chain([
            f32::from_bits(0x3f808000),
            f32::from_bits(0x3f818000),
            -0.,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7f800001),
        ])
        .collect();
    let narrowed = bf16(&floats);
    let expanded: Vec<_> = narrowed
        .bf16()
        .unwrap()
        .iter()
        .map(|&x| half::bf16::from_bits(x).to_f32())
        .collect();
    let raw = vec![0x7f81, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0xffc1];
    let mut cases=vec![
        ("x=param(0,bf16,65536)\noutput x\noutput cast(x,f32)\noutput cast(cast(x,f32),bf16)".into(), vec![Tensor::BF16(bits.clone().into())], vec![Tensor::BF16(bits.into()),Tensor::F32(widened.into()),roundtrip]),
        (format!("x=param(0,f32,{})\noutput cast(x,bf16)\noutput cast(cast(x,bf16),f32)",floats.len()),vec![Tensor::F32(floats.into())],vec![narrowed,Tensor::F32(expanded.into())]),
        ("x=reshape(param(0,bf16,6),[2,3])\ni=param(1,i32,2)\ny=load(index(x,i))\noutput y\noutput permute(x,[1,0])\noutput stack(y,y)\noutput pad(x,[0,1],[2,5])\noutput where(x < 0.0,x,x)".into(),vec![Tensor::BF16(raw.clone().into()),Tensor::I32(vec![1,0].into())],vec![
            Tensor::BF16(vec![raw[3],raw[4],raw[5],raw[0],raw[1],raw[2]].into()),
            Tensor::BF16(vec![raw[0],raw[3],raw[1],raw[4],raw[2],raw[5]].into()),
            Tensor::BF16([&raw[3..],&raw[..3],&raw[3..],&raw[..3]].concat().into()),
            Tensor::BF16(vec![0,raw[0],raw[1],raw[2],0,0,raw[3],raw[4],raw[5],0].into()),
            Tensor::BF16(raw.into()),
        ]),
        ("x=param(0,bf16,4)\ny=param(1,f32,4)\noutput x+y\noutput x*x\noutput reduce(x,add,1)\noutput sqrt(x*x)".into(),vec![bf16(&[1.,-2.,3.,-4.]),Tensor::F32(vec![0.1,0.2,0.3,0.4].into())],vec![Tensor::F32(vec![1.1,-1.8,3.3,-3.6].into()),Tensor::F32(vec![1.,4.,9.,16.].into()),Tensor::F32(vec![-2.].into()),Tensor::F32(vec![1.,2.,3.,4.].into())]),
    ];
    for (m, n, k) in [(1, 7, 96), (17, 19, 33), (35, 67, 65)] {
        for left_bf16 in [false, true] {
            let av: Vec<_> = (0..m * k).map(|i| (i % 11) as f32 / 8. - 0.5).collect();
            let bv: Vec<_> = (0..n * k).map(|i| (i % 13) as f32 / 16. - 0.375).collect();
            let expected: Vec<_> = (0..m * n)
                .map(|i| {
                    (0..k)
                        .map(|r| av[i / n * k + r] * bv[i % n * k + r])
                        .sum::<f32>()
                        + 0.25
                })
                .collect();
            cases.push((format!("a=reshape(param(0,{},{}),[{m},{k}])\nb=reshape(param(1,bf16,{}),[{n},{k}])\noutput matmul(a,permute(b,[1,0]))+0.25", if left_bf16 {"bf16"} else {"f32"},m*k,n*k),vec![if left_bf16 {bf16(&av)} else {Tensor::F32(av.into())},bf16(&bv)],vec![Tensor::F32(expected.into())]));
        }
    }
    cases
}
fn run_cases(backend: Option<gpu::Backend>) {
    let f = Fixture::new();
    for (text, inputs, expected) in cases() {
        let p = source::parse(&text).unwrap();
        let plan = hip::memory_plan(&p.graph, p.root).unwrap();
        for (&slot, param) in &plan.parameters {
            if let Tensor::BF16(x) = &inputs[slot] {
                assert_eq!(param.dtype, DType::BF16);
                assert_eq!(param.bytes, x.len() * 2);
            }
        }
        let actual = match backend {
            None => cpu::compile(&p.graph, p.root, &f.0.join("cpu"))
                .unwrap()
                .run(&inputs)
                .unwrap(),
            Some(backend) => {
                let runtime = gpu::Runtime::new(backend, 0).unwrap();
                let e =
                    gpu::compile_with_runtime(&p.graph, p.root, &f.0.join(backend.tag()), &runtime)
                        .unwrap();
                let out = e.run(&inputs).unwrap();
                assert_eq!(
                    e.residency_stats().input_uploaded_bytes,
                    plan.input_bytes() as u64
                );
                let uploads = e.residency_stats().input_uploads;
                compare(&e.run(&inputs).unwrap(), &expected);
                assert_eq!(e.residency_stats().input_uploads, uploads);
                runtime.set_graph_replay(false);
                compare(&e.run(&inputs).unwrap(), &expected);
                out
            }
        };
        compare(&actual, &expected);
    }
}
#[test]
fn bf16_cpu_storage_casts_views_and_fused_mixed_matmul() {
    run_cases(None);
}
#[test]
#[ignore = "requires HIP GPU and HIPRTC"]
fn bf16_hip_storage_casts_views_and_fused_mixed_matmul() {
    run_cases(Some(gpu::Backend::Hip));
}
#[test]
#[ignore = "requires CUDA GPU and NVRTC"]
fn bf16_cuda_storage_casts_views_and_fused_mixed_matmul() {
    run_cases(Some(gpu::Backend::Cuda));
}
#[test]
fn bf16_emission_retains_native_buffers_and_fused_contractions() {
    for (text, _, _) in cases() {
        let p = source::parse(&text).unwrap();
        for (code, count) in [
            cuda::emit(&p.graph, p.root).unwrap(),
            hip::emit(&p.graph, p.root).unwrap(),
        ] {
            assert!(code.contains("unsigned short *"));
            if text.contains("matmul") {
                assert_eq!(count, 1);
                assert!(code.contains("pup_bf16_load("));
            }
        }
    }
}
#[test]
#[ignore = "requires HIPRTC, no GPU"]
fn bf16_hiprtc_wave32_and_wave64_compile() {
    for architecture in ["gfx1201", "gfx1100", "gfx90a"] {
        for (text, _, _) in cases() {
            let p = source::parse(&text).unwrap();
            let (code, _) = hip::emit(&p.graph, p.root).unwrap();
            assert!(hip::compile_source(&code, architecture)
                .unwrap()
                .starts_with(b"\x7fELF"));
        }
    }
}

fn retained_state(backend: Option<gpu::Backend>) {
    let f = Fixture::new();
    let p = source::parse("cache=state(\"cache\",bf16,[4,2])\ni=param(0,i32,2)\nx=reshape(param(1,bf16,4),[2,2])\nw=store(index(cache,i),x)\noutput after(cache,w)").unwrap();
    let runtime = backend.map(|b| gpu::Runtime::new(b, 0).unwrap());
    let device = runtime
        .as_ref()
        .map(|r| gpu::compile_with_runtime(&p.graph, p.root, &f.0.join("gpu"), r).unwrap());
    let host_runtime = cpu::Runtime::default();
    let host = if runtime.is_none() {
        let mut executable = cpu::compile(&p.graph, p.root, &f.0.join("cpu")).unwrap();
        executable.share_runtime(&host_runtime);
        Some(executable)
    } else {
        None
    };
    let run = |inputs: &[Tensor]| {
        if let Some(e) = &device {
            e.run(inputs).unwrap()
        } else {
            host.as_ref().unwrap().run(inputs).unwrap()
        }
    };
    let first = [
        Tensor::I32(vec![1, 3].into()),
        Tensor::BF16(vec![0x7f81, 0x8000, 0x3f80, 0xbf80].into()),
    ];
    compare(
        &run(&first),
        &[Tensor::BF16(
            vec![0, 0, 0x7f81, 0x8000, 0, 0, 0x3f80, 0xbf80].into(),
        )],
    );
    let second = [Tensor::I32(vec![0, 2].into()), bf16(&[2., 3., 4., 5.])];
    compare(
        &run(&second),
        &[Tensor::BF16(
            vec![
                0x4000, 0x4040, 0x7f81, 0x8000, 0x4080, 0x40a0, 0x3f80, 0xbf80,
            ]
            .into(),
        )],
    );
    if let Some(e) = &device {
        assert_eq!(e.residency_stats().input_uploaded_bytes, 32);
    }
    if let Some(r) = &runtime {
        r.reset_state().unwrap();
    } else {
        host_runtime.reset_state().unwrap();
    }
    compare(
        &run(&second),
        &[Tensor::BF16(
            vec![0x4000, 0x4040, 0, 0, 0x4080, 0x40a0, 0, 0].into(),
        )],
    );
}
#[test]
fn bf16_cpu_retained_state_and_mutated_input() {
    retained_state(None);
}
#[test]
#[ignore = "requires HIP GPU and HIPRTC"]
fn bf16_hip_retained_state_and_mutated_input() {
    retained_state(Some(gpu::Backend::Hip));
}
#[test]
#[ignore = "requires CUDA GPU and NVRTC"]
fn bf16_cuda_retained_state_and_mutated_input() {
    retained_state(Some(gpu::Backend::Cuda));
}
