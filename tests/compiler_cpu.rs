use puppygrad::compiler::{
    cpu::{compile, emit, Tensor},
    source,
};
use std::path::Path;
fn execute(text: &str, inputs: Vec<Tensor>) -> Vec<Tensor> {
    let p = source::parse(text).unwrap();
    compile(&p.graph, p.root, Path::new(".cache/pup/tests"))
        .unwrap()
        .run(&inputs)
        .unwrap()
}
fn floats(v: &[f32]) -> Tensor {
    Tensor::F32(v.to_vec().into())
}
#[test]
fn compiled_matmul_and_nested_expressions() {
    let source="a = reshape(param(0, f32, 6), [2, 3])\nb = reshape(param(1, f32, 6), [3, 2])\ny = matmul(a, b) + 2.0 * -1.0\noutput y";
    let p = source::parse(source).unwrap();
    assert_eq!(emit(&p.graph, p.root).unwrap().1, 1);
    let outputs = execute(
        source,
        vec![
            floats(&[1., 2., 3., 4., 5., 6.]),
            floats(&[1., 2., 3., 4., 5., 6.]),
        ],
    );
    assert_eq!(outputs[0].f32().unwrap(), &[20., 26., 47., 62.]);
}
#[test]
fn compiled_batched_strided_matmul() {
    let source="a = reshape(param(0, f32, 12), [2, 2, 3])\nb = permute(reshape(param(1, f32, 12), [2, 2, 3]), [0, 2, 1])\noutput batched_matmul(a, b)";
    let outputs = execute(
        source,
        vec![
            floats(&(1..=12).map(|x| x as f32).collect::<Vec<_>>()),
            floats(&(1..=12).map(|x| x as f32).collect::<Vec<_>>()),
        ],
    );
    assert_eq!(
        outputs[0].f32().unwrap(),
        &[14., 32., 32., 77., 194., 266., 266., 365.]
    );
}
#[test]
fn compiled_views_reduction_and_gather_bounds() {
    let text="x = reshape(param(0, f32, 6), [3, 2])\ni = param(1, i32, 2)\ny = load(index(x, i))\noutput reduce(permute(y, [1, 0]), add, 1)";
    let p = source::parse(text).unwrap();
    let exe = compile(&p.graph, p.root, Path::new(".cache/pup/tests")).unwrap();
    let input = floats(&[1., 2., 3., 4., 5., 6.]);
    let result = exe
        .run(&[input.clone(), Tensor::I32(vec![2, 0].into())])
        .unwrap();
    assert_eq!(result[0].f32().unwrap(), &[11., 3.]);
    assert!(exe
        .run(&[input.clone(), Tensor::I32(vec![-1, 0].into())])
        .unwrap_err()
        .to_string()
        .contains("out of bounds"));
    assert!(exe
        .run(&[input, Tensor::I32(vec![3, 0].into())])
        .unwrap_err()
        .to_string()
        .contains("out of bounds"));
}
#[test]
fn bounded_loop_carries_values_without_leaking_locals() {
    let result=execute("x = cast(1.0, f32)\nfor layer in range(3):\n    delta = layer + 1\n    x = x + delta\noutput x",vec![]);
    assert_eq!(result[0].f32().unwrap(), &[7.]);
    assert!(
        source::parse("x = 1\nfor layer in range(3):\n    delta = layer\noutput delta").is_err()
    );
    assert!(source::parse("x = 1\nfor layer in range(4097):\n    x = x + 1\noutput x").is_err());
}

#[test]
fn compiled_gpt2_matches_reference_for_multiple_prefixes() {
    use puppygrad::models::pup_llm::Checkpoint;
    use puppygrad::{
        compiler::{
            pop::{DType, Scalar},
            source::{Context, TensorSpec},
        },
        models::gpt2::{Gpt2BlockWeights, Gpt2Config, Gpt2Model, Gpt2Weights},
    };
    let cfg = Gpt2Config::new(13, 8, 8, 2, 2);
    let mut seed = 17u32;
    let mut values = |len: usize| {
        (0..len)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 16) as f32 / 65536. - 0.5) * 0.3
            })
            .collect::<Vec<_>>()
    };
    let mut blocks = vec![];
    for _ in 0..cfg.n_layer {
        blocks.push(Gpt2BlockWeights {
            ln_1_g: vec![1.; 8],
            ln_1_b: values(8),
            c_attn_w: values(8 * 24),
            c_attn_b: values(24),
            c_proj_w: values(64),
            c_proj_b: values(8),
            ln_2_g: vec![1.; 8],
            ln_2_b: values(8),
            c_fc_w: values(8 * 32),
            c_fc_b: values(32),
            c_proj_mlp_w: values(32 * 8),
            c_proj_mlp_b: values(8),
        });
    }
    let weights = Gpt2Weights {
        wte: values(13 * 8),
        wpe: values(8 * 8),
        blocks,
        ln_f_g: vec![1.; 8],
        ln_f_b: values(8),
    };
    let mut ckpt = Checkpoint {
        context: Context::default(),
        inputs: vec![Tensor::I32(vec![].into())],
        model_type: Some("gpt2".into()),
    };
    for (name, n) in [("n_layer", 2), ("n_head", 2), ("n_embd", 8)] {
        ckpt.context.constants.insert(name.into(), Scalar::Int(n));
    }
    ckpt.context
        .constants
        .insert("layer_norm_epsilon".into(), Scalar::float(1e-5));
    let mut bind = |name: String, shape: Vec<usize>, data: &[f32]| {
        ckpt.context.tensors.insert(
            name,
            TensorSpec {
                slot: ckpt.inputs.len(),
                dtype: DType::F32,
                shape,
            },
        );
        ckpt.inputs.push(floats(data));
    };
    bind("wte.weight".into(), vec![13, 8], &weights.wte);
    bind("wpe.weight".into(), vec![8, 8], &weights.wpe);
    bind("ln_f.weight".into(), vec![8], &weights.ln_f_g);
    bind("ln_f.bias".into(), vec![8], &weights.ln_f_b);
    for (layer, w) in weights.blocks.iter().enumerate() {
        for (key, shape, data) in [
            ("ln_1.weight", vec![8], &w.ln_1_g),
            ("ln_1.bias", vec![8], &w.ln_1_b),
            ("attn.c_attn.weight", vec![8, 24], &w.c_attn_w),
            ("attn.c_attn.bias", vec![24], &w.c_attn_b),
            ("attn.c_proj.weight", vec![8, 8], &w.c_proj_w),
            ("attn.c_proj.bias", vec![8], &w.c_proj_b),
            ("ln_2.weight", vec![8], &w.ln_2_g),
            ("ln_2.bias", vec![8], &w.ln_2_b),
            ("mlp.c_fc.weight", vec![8, 32], &w.c_fc_w),
            ("mlp.c_fc.bias", vec![32], &w.c_fc_b),
            ("mlp.c_proj.weight", vec![32, 8], &w.c_proj_mlp_w),
            ("mlp.c_proj.bias", vec![8], &w.c_proj_mlp_b),
        ] {
            bind(format!("h.{layer}.{key}"), shape, data);
        }
    }
    let reference = Gpt2Model::new(cfg, weights).unwrap();
    for ids in [&[1usize][..], &[1, 4, 7][..], &[2, 3, 5, 8][..]] {
        ckpt.bind_tokens(ids).unwrap();
        let p =
            source::parse_with_context(include_str!("../examples/llm.pup"), &ckpt.context).unwrap();
        let exe = compile(&p.graph, p.root, Path::new(".cache/pup/tests")).unwrap();
        assert_eq!(exe.gemm_count, 13);
        let out = exe.run(&ckpt.inputs).unwrap();
        let expected = reference.forward(ids).unwrap();
        for (a, b) in out[0]
            .f32()
            .unwrap()
            .iter()
            .zip(expected.last_logits().unwrap())
        {
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
        }
    }
}

#[test]
fn empty_contractions_and_parameter_validation() {
    let p=source::parse("a = reshape(param(0, f32, 0), [2, 0])\nb = reshape(param(1, f32, 0), [0, 3])\noutput matmul(a, b)").unwrap();
    let exe = compile(&p.graph, p.root, Path::new(".cache/pup/tests")).unwrap();
    assert_eq!(
        exe.run(&[floats(&[]), floats(&[])]).unwrap()[0]
            .f32()
            .unwrap(),
        &[0.; 6]
    );
    assert!(exe
        .run(&[])
        .unwrap_err()
        .to_string()
        .contains("missing input slot"));
    assert!(exe
        .run(&[floats(&[1.]), floats(&[])])
        .unwrap_err()
        .to_string()
        .contains("expected F32[0]"));
}

#[test]
fn casts_and_weak_constants_obey_concrete_dtypes() {
    let outputs = execute(
        "a = param(0, f32, 3)\noutput cast(a, bool), a + 16777217.0",
        vec![floats(&[-2., 0., 1.])],
    );
    let Tensor::Bool(flags) = &outputs[0] else {
        panic!("bool output")
    };
    assert_eq!(flags.as_ref(), &[1, 0, 1]);
    assert_eq!(
        outputs[1].f32().unwrap(),
        &[16777214., 16777216., 16777216.]
    );
    let outputs = execute(
        "a = param(0, i32, 1)\noutput a + 1",
        vec![Tensor::I32(vec![i32::MAX].into())],
    );
    let Tensor::I32(values) = &outputs[0] else {
        panic!("int output")
    };
    assert_eq!(values.as_ref(), &[i32::MIN]);
}

#[test]
fn oversized_view_output_is_rejected_before_allocation() {
    let p = source::parse("x = expand(cast(1.0, f32), [600000000])\noutput x").unwrap();
    assert!(emit(&p.graph, p.root)
        .unwrap_err()
        .to_string()
        .contains("workspace exceeds"));
}

#[test]
fn movement_ops_compose_through_generated_indices() {
    let outputs=execute("x = reshape(param(0, f32, 4), [2, 2])\np = pad(x, [1, 1], [4, 4])\nf = flip(p, [true, false])\ns = shrink(f, [1, 1], [2, 2])\noutput expand(exp2(log2(s)), [2])",vec![floats(&[1.,2.,3.,4.])]);
    for (actual, expected) in outputs[0]
        .f32()
        .unwrap()
        .iter()
        .zip([3., 4., 1., 2., 3., 4., 1., 2.])
    {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn shared_product_is_not_eliminated_by_contraction_fusion() {
    let text="a = reshape(param(0, f32, 4), [2, 1, 2])\nb = reshape(permute(reshape(param(1, f32, 4), [2, 2]), [1, 0]), [1, 2, 2])\np = a * b\noutput reduce(permute(p, [2, 0, 1]), add, 1), p";
    let p = source::parse(text).unwrap();
    assert_eq!(emit(&p.graph, p.root).unwrap().1, 0);
    let outputs = execute(
        text,
        vec![floats(&[1., 2., 3., 4.]), floats(&[5., 6., 7., 8.])],
    );
    assert_eq!(outputs[0].f32().unwrap(), &[19., 22., 43., 50.]);
    assert_eq!(
        outputs[1].f32().unwrap(),
        &[5., 14., 6., 16., 15., 28., 18., 32.]
    );
}

#[test]
fn explicit_thread_counts_preserve_contractions_and_reject_zero() {
    let p=source::parse("a = reshape(param(0, f32, 4096), [64, 64])\nb = reshape(param(1, f32, 4096), [64, 64])\noutput matmul(a, b)").unwrap();
    let exe = compile(&p.graph, p.root, Path::new(".cache/pup/tests")).unwrap();
    let input = floats(&(0..4096).map(|i| (i % 7) as f32 - 3.).collect::<Vec<_>>());
    let inputs = [input.clone(), input];
    let serial = exe.run_with_threads(&inputs, 1).unwrap();
    let parallel = exe.run_with_threads(&inputs, 2).unwrap();
    assert_eq!(serial[0].f32().unwrap(), parallel[0].f32().unwrap());
    assert!(exe
        .run_with_threads(&inputs, 0)
        .unwrap_err()
        .to_string()
        .contains("threads must be greater than zero"));
}

#[test]
fn emitted_model_runs_from_c_without_host_compute_callbacks() {
    use std::process::Command;
    let p = source::parse("a = reshape(param(0, f32, 63), [7, 9])\nb = permute(reshape(param(1, f32, 630), [70, 9]), [1, 0])\ny = matmul(a, b)\ni = param(2, i32, 1)\nz = load(index(param(3, f32, 2), i))\noutput y, z").unwrap();
    let (c, contractions) = emit(&p.graph, p.root).unwrap();
    assert_eq!(contractions, 1);
    let dir = tempfile_dir_for_c();
    std::fs::write(dir.join("model.c"), c).unwrap();
    std::fs::write(
        dir.join("main.c"),
        r#"
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <errno.h>
#include <pthread.h>
int pup_run(const void **inputs, void **outputs, size_t threads);
/* Exercise cleanup when creating the second worker fails. */
static int fail_after=-1;
int __real_pthread_create(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
int __wrap_pthread_create(pthread_t *t, const pthread_attr_t *a, void *(*f)(void *), void *p) {
    if(fail_after==0) return EAGAIN;
    if(fail_after>0) fail_after--;
    return __real_pthread_create(t,a,f,p);
}
int main(void) {
    float a[63], b[630], c[490], values[]={3,7}, z;
    int32_t ix=1;
    for(size_t i=0;i<63;i++) a[i]=(float)(i%7)-3;
    for(size_t i=0;i<630;i++) b[i]=(float)(i%11)-5;
    const void *inputs[]={a,b,&ix,values};
    void *outputs[]={c,&z};
    if(pup_run(inputs,outputs,0)!=3) return 1;
    fail_after=1;
    if(pup_run(inputs,outputs,3)!=3) return 2;
    fail_after=-1;
    size_t threads[]={1,2,3,8};
    for(size_t trial=0;trial<12;trial++) {
        ix=1;
        if(pup_run(inputs,outputs,threads[trial%4]) || z!=7) return 3;
        for(size_t i=0;i<7;i++) for(size_t j=0;j<70;j++) {
            float expected=0;
            for(size_t r=0;r<9;r++) expected+=a[i*9+r]*b[j*9+r];
            if(c[i*70+j]!=expected) return 4;
        }
        ix=-1;
        if(pup_run(inputs,outputs,threads[trial%4])!=2) return 5;
    }
    puts("standalone C: strided matmul, tile tails, threads, error cleanup OK");
    return 0;
}
"#,
    )
    .unwrap();
    let result = Command::new("cc")
        .args([
            "-std=c11",
            "-O2",
            "-fno-math-errno",
            "-fwrapv",
            "-ffp-contract=off",
            "-pthread",
        ])
        .arg(dir.join("model.c"))
        .arg(dir.join("main.c"))
        .args(["-lm", "-Wl,--wrap=pthread_create", "-o"])
        .arg(dir.join("standalone"))
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let result = Command::new(dir.join("standalone")).output().unwrap();
    assert!(
        result.status.success(),
        "C process returned {:?}: {}",
        result.status.code(),
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("standalone C:"));
    std::fs::remove_dir_all(dir).unwrap();
}

fn tempfile_dir_for_c() -> std::path::PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "puppygrad-standalone-c-{}-{stamp}",
        std::process::id()
    ));
    std::fs::create_dir(&dir).unwrap();
    dir
}

#[test]
fn empty_matrix_outputs_do_not_pack_or_launch_workers() {
    for (m, n, k) in [(0, 3, 2), (2, 0, 3), (0, 0, 0)] {
        let text=format!("a = reshape(param(0, f32, {}), [{m}, {k}])\nb = reshape(param(1, f32, {}), [{k}, {n}])\noutput matmul(a, b)", m*k, k*n);
        let out = execute(
            &text,
            vec![floats(&vec![1.; m * k]), floats(&vec![2.; k * n])],
        );
        assert!(out[0].is_empty());
    }
}

#[test]
fn byte_inputs_and_outputs_preserve_full_unsigned_range() {
    let text = "pixels = param(0,u8,4)\nx = cast(pixels,f32) / 255.0\noutput x, pixels\n";
    let outputs = execute(text, vec![Tensor::U8(vec![0, 1, 128, 255].into())]);
    assert_eq!(outputs[0].f32().unwrap(), &[0., 1. / 255., 128. / 255., 1.]);
    let Tensor::U8(bytes) = &outputs[1] else {
        panic!("expected u8 output")
    };
    assert_eq!(&**bytes, &[0, 1, 128, 255]);
}
