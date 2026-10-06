use puppygrad::compiler::{
    cpu::{self, BuildOptions, CpuTarget, Tensor},
    source,
};
use puppygrad::models::stable_diffusion::{self as sd, Conv2dOptions, SdTensor};
use std::path::Path;

fn floats(data: Vec<f32>) -> Tensor {
    Tensor::F32(data.into())
}
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!((a - b).abs() < 2e-5, "index {i}: {a} != {b}");
    }
}

#[test]
fn packed_batched_matmul_handles_tails_targets_and_updated_weights() {
    for (batch, m, n, k) in [(2, 19, 67, 131), (2, 19, 67, 2051), (2, 259, 67, 1025)] {
        let text=format!("a=reshape(param(0,f32,{}),[{batch},{m},{k}])\nb=permute(reshape(param(1,f32,{}),[{batch},{n},{k}]),[0,2,1])\noutput batched_matmul(a,b)+0.125\n",batch*m*k,batch*n*k);
        let p = source::parse(&text).unwrap();
        let a = (0..batch * m * k)
            .map(|i| ((i % 23) as f32 - 11.) / 16.)
            .collect::<Vec<_>>();
        let original = (0..batch * n * k)
            .map(|i| ((i % 17) as f32 - 8.) / 32.)
            .collect::<Vec<_>>();
        for cpu_target in [CpuTarget::Generic, CpuTarget::Native] {
            let exe = cpu::compile_profiled_with_options(
                &p,
                Path::new(".cache/pup/tests"),
                &BuildOptions { cpu_target },
            )
            .unwrap();
            let meta = exe.profile_metadata.as_ref().unwrap();
            assert!(meta["packed_workspace_bytes"].as_u64().unwrap() <= 512 * 1024);
            assert!(meta["kernels"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v["packed_bytes"].as_u64().unwrap() > 0));
            let plain = cpu::compile_with_options(
                &p.graph,
                p.root,
                Path::new(".cache/pup/tests"),
                &BuildOptions { cpu_target },
            )
            .unwrap();
            for scale in [1., -2.] {
                let b = original.iter().map(|x| x * scale).collect::<Vec<_>>();
                let expected = (0..batch)
                    .flat_map(|batch| {
                        let a = &a;
                        let b = &b;
                        (0..m).flat_map(move |i| {
                            (0..n).map(move |j| {
                                (0..k).fold(0., |acc, r| {
                                    acc + a[(batch * m + i) * k + r] * b[(batch * n + j) * k + r]
                                }) + 0.125
                            })
                        })
                    })
                    .collect::<Vec<_>>();
                for threads in [1, 3, 8] {
                    let out = exe
                        .run_profiled(&[floats(a.clone()), floats(b.clone())], threads)
                        .unwrap();
                    assert_eq!(out.outputs[0].f32().unwrap(), &expected);
                    let actual = plain
                        .run_with_threads(&[floats(a.clone()), floats(b.clone())], threads)
                        .unwrap();
                    assert_eq!(actual[0].f32().unwrap(), &expected);
                    for kernel in meta["kernels"].as_array().unwrap() {
                        let offset = kernel["stats_offset"].as_u64().unwrap() as usize;
                        assert_eq!(out.counters[offset], 1);
                        assert!(
                            out.counters[offset + 1]
                                >= out.counters[offset + 2] + out.counters[offset + 3]
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn packed_convolution_preserves_strides_padding_and_odd_channels() {
    let x = SdTensor::new(
        [2, 16, 9, 11],
        (0..3168).map(|i| (i as f32 * 0.17).sin()).collect(),
    )
    .unwrap();
    let w = SdTensor::new(
        [67, 16, 3, 3],
        (0..9648).map(|i| (i as f32 * 0.13).cos() * 0.1).collect(),
    )
    .unwrap();
    let bias = (0..67).map(|i| i as f32 * 0.001).collect::<Vec<_>>();
    for stride in [1, 2] {
        let p=source::parse(&format!("x=reshape(param(0,f32,3168),[2,16,9,11])\nw=reshape(param(1,f32,9648),[67,16,3,3])\nb=param(2,f32,67)\noutput conv2d(x,w,b,{stride},1)\n")).unwrap();
        let exe = cpu::compile_profiled_with_options(
            &p,
            Path::new(".cache/pup/tests"),
            &BuildOptions {
                cpu_target: CpuTarget::Native,
            },
        )
        .unwrap();
        let expected =
            sd::conv2d_nchw(&x, &w, Some(&bias), Conv2dOptions { stride, padding: 1 }).unwrap();
        for threads in [1, 3] {
            let out = exe
                .run_with_threads(
                    &[
                        floats(x.data().to_vec()),
                        floats(w.data().to_vec()),
                        floats(bias.clone()),
                    ],
                    threads,
                )
                .unwrap();
            close(out[0].f32().unwrap(), expected.data());
        }
    }
}

#[test]
fn parallel_elementwise_padding_and_reduction_preserve_uneven_chunks() {
    let rows = 32771;
    let text=format!("x=reshape(param(0,f32,{}),[{rows},4])\ny=reduce(permute(x,[1,0]),add,1)\nz=reshape(exp2(y),[{rows},1])+x*0.5\np=pad(x,[1,1],[{},6])\noutput y,z,p\n",rows*4,rows+2);
    let p = source::parse(&text).unwrap();
    let exe = cpu::compile_profiled(&p, Path::new(".cache/pup/tests")).unwrap();
    let x = (0..rows * 4)
        .map(|i| ((i % 19) as f32 - 9.) / 64.)
        .collect::<Vec<_>>();
    let expected = x
        .chunks_exact(4)
        .map(|r| r.iter().fold(0., |a, b| a + b))
        .collect::<Vec<_>>();
    for threads in [1, 3, 8] {
        let actual = exe.run_with_threads(&[floats(x.clone())], threads).unwrap();
        assert_eq!(actual[0].f32().unwrap(), &expected);
        let z = expected
            .iter()
            .enumerate()
            .flat_map(|(i, &v)| x[i * 4..i * 4 + 4].iter().map(move |x| v.exp2() + x * 0.5))
            .collect::<Vec<_>>();
        close(actual[1].f32().unwrap(), &z);
        let padded = actual[2].f32().unwrap();
        for y in 0..rows + 2 {
            for col in 0..6 {
                let value = if y > 0 && y <= rows && (1..5).contains(&col) {
                    x[(y - 1) * 4 + col - 1]
                } else {
                    0.
                };
                assert_eq!(padded[y * 6 + col], value, "{y},{col},threads={threads}");
            }
        }
    }
}
