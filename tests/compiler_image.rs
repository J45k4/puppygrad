use puppygrad::compiler::{
    cpu::{self, Tensor},
    source,
};
use puppygrad::models::stable_diffusion::{self as sd, Conv2dOptions, SdTensor};
use std::path::Path;
fn floats(v: Vec<f32>) -> Tensor {
    Tensor::F32(v.into())
}
fn close(actual: &[f32], expected: &[f32], epsilon: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!((a - b).abs() <= epsilon, "index {i}: {a} != {b}");
    }
}
#[test]
fn convolution_strided_windows_match_reference_without_im2col_storage() {
    let x = SdTensor::new(
        [2, 3, 7, 9],
        (0..378).map(|i| (i as f32 * 0.17).sin()).collect(),
    )
    .unwrap();
    let w = SdTensor::new(
        [5, 3, 3, 2],
        (0..90).map(|i| (i as f32 * 0.13).cos() * 0.1).collect(),
    )
    .unwrap();
    let bias = vec![0.1, -0.1, 0.0, 0.2, -0.2];
    for stride in [1, 2] {
        for padding in [0, 1] {
            let p=source::parse(&format!("x = reshape(param(0,f32,378),[2,3,7,9])\nw = reshape(param(1,f32,90),[5,3,3,2])\nb = param(2,f32,5)\ny = conv2d(x,w,b,{stride},{padding})\noutput y\n")).unwrap();
            let exe = cpu::compile_profiled(&p, Path::new(".cache/pup/tests")).unwrap();
            assert_eq!(
                exe.gemm_count, 1,
                "convolution must lower through tiled GEMM"
            );
            let expected =
                sd::conv2d_nchw(&x, &w, Some(&bias), Conv2dOptions { stride, padding }).unwrap();
            for threads in [1, 3] {
                let actual = exe
                    .run_with_threads(
                        &[
                            floats(x.data().to_vec()),
                            floats(w.data().to_vec()),
                            floats(bias.clone()),
                        ],
                        threads,
                    )
                    .unwrap();
                close(actual[0].f32().unwrap(), expected.data(), 2e-5);
            }
            let metadata = exe.profile_metadata.as_ref().unwrap();
            let allocations = metadata["memory_plan"]["allocations"].as_array().unwrap();
            let patch_bytes =
                expected.shape()[0] * expected.shape()[2] * expected.shape()[3] * 3 * 3 * 2 * 4;
            assert!(
                allocations
                    .iter()
                    .all(|a| a["bytes"].as_u64().unwrap() < patch_bytes as u64),
                "patch matrix was materialized: {metadata}"
            );
        }
    }
}
#[test]
fn window_outputs_and_nearest_upsampling_preserve_coordinates() {
    let p=source::parse("x=reshape(param(0,f32,6),[1,1,2,3])\na=window(x,[2,2],[1,1])\nb=upsample2d(x)\nc=concat_channels(x,x+10.0)\noutput a,b,c\n").unwrap();
    let exe = cpu::compile(&p.graph, p.root, Path::new(".cache/pup/tests")).unwrap();
    let y = exe.run(&[floats(vec![1., 2., 3., 4., 5., 6.])]).unwrap();
    close(y[0].f32().unwrap(), &[1., 2., 4., 5., 2., 3., 5., 6.], 0.0);
    close(
        y[1].f32().unwrap(),
        &[
            1., 1., 2., 2., 3., 3., 1., 1., 2., 2., 3., 3., 4., 4., 5., 5., 6., 6., 4., 4., 5., 5.,
            6., 6.,
        ],
        0.0,
    );
    close(
        y[2].f32().unwrap(),
        &[1., 2., 3., 4., 5., 6., 11., 12., 13., 14., 15., 16.],
        0.0,
    );
}
#[test]
fn group_norm_and_sine_gradient_match_reference() {
    let p=source::parse("x=reshape(param(0,f32,16),[1,4,2,2])\nw=param(1,f32,4)\nb=param(2,f32,4)\ny=group_norm(x,w,b,2,0.00001)\ns=param(3,f32,3)\ng=grad(reduce(sin(s),add,1),s)\noutput y,g\n").unwrap();
    let exe = cpu::compile(&p.graph, p.root, Path::new(".cache/pup/tests")).unwrap();
    let x = (0..16).map(|i| i as f32 * 0.13).collect::<Vec<_>>();
    let w = vec![1., 0.5, -1., 2.];
    let b = vec![0., 1., 0.2, -0.1];
    let s = vec![-1.3, 0.2, 2.7];
    let expected = sd::group_norm_nchw(
        &SdTensor::new([1, 4, 2, 2], x.clone()).unwrap(),
        2,
        &w,
        &b,
        1e-5,
    )
    .unwrap();
    let y = exe
        .run(&[floats(x), floats(w), floats(b), floats(s.clone())])
        .unwrap();
    close(y[0].f32().unwrap(), expected.data(), 1e-5);
    close(
        y[1].f32().unwrap(),
        &s.iter().map(|x| x.cos()).collect::<Vec<_>>(),
        1e-6,
    );
}

#[test]
fn compiled_guidance_and_ddim_match_reference_scheduler() {
    use puppygrad::models::pup_image::stages;
    let template = std::fs::read_to_string("examples/image.pup").unwrap();
    let (p, mut weights) = stages::step(&template, 2, 3).unwrap();
    let exe = cpu::compile(&p.graph, p.root, Path::new(".cache/pup/tests")).unwrap();
    let sample = sd::deterministic_normal_tensor([1, 4, 2, 3], 42).unwrap();
    let uncond = sd::deterministic_normal_tensor([1, 4, 2, 3], 43).unwrap();
    let cond = sd::deterministic_normal_tensor([1, 4, 2, 3], 44).unwrap();
    let mut scheduler = sd::DdimScheduler::new(sd::DdimSchedulerConfig {
        class_name: "DDIMScheduler".into(),
        num_train_timesteps: 10,
        beta_start: 0.0001,
        beta_end: 0.02,
        beta_schedule: "linear".into(),
        clip_sample: false,
        prediction_type: "epsilon".into(),
        timestep_spacing: None,
        steps_offset: None,
    })
    .unwrap();
    scheduler.set_timesteps(5).unwrap();
    for scale in [0.0, 1.0, 7.5] {
        let expected = scheduler
            .step(
                &sd::classifier_free_guidance(&uncond, &cond, scale).unwrap(),
                8,
                &sample,
            )
            .unwrap();
        for (name, tensor) in [("sample", &sample), ("uncond", &uncond), ("cond", &cond)] {
            weights.set(name, floats(tensor.data().to_vec()));
        }
        for (name, value) in [
            ("scale", scale),
            ("alpha", scheduler.alphas_cumprod[8]),
            ("previous_alpha", scheduler.alphas_cumprod[6]),
        ] {
            weights.set(name, floats(vec![value]));
        }
        let actual = exe.run(&weights.inputs).unwrap();
        close(actual[0].f32().unwrap(), expected.data(), 2e-6);
    }
}
