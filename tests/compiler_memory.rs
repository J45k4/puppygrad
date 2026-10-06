use puppygrad::compiler::{
    cpu::{self, Tensor},
    source,
};
use serde_json::Value;
use std::path::Path;

fn floats(x: &[f32]) -> Tensor {
    Tensor::F32(x.to_vec().into())
}
fn check_plan(metadata: &Value) {
    let arena = metadata["workspace_bytes"].as_u64().unwrap();
    let allocations = metadata["memory_plan"]["allocations"].as_array().unwrap();
    for (i, a) in allocations.iter().enumerate() {
        let field = |v: &Value, key: &str| v[key].as_u64().unwrap();
        assert!(field(a, "offset") + field(a, "bytes") <= arena);
        assert_eq!(field(a, "offset") % 8, 0);
        for b in &allocations[..i] {
            let live = field(a, "first_kernel") <= field(b, "last_kernel")
                && field(b, "first_kernel") <= field(a, "last_kernel");
            let overlap = field(a, "offset") < field(b, "offset") + field(b, "bytes")
                && field(b, "offset") < field(a, "offset") + field(a, "bytes");
            assert!(
                !live || !overlap || field(a, "bytes") == 0 || field(b, "bytes") == 0,
                "overlapping live buffers: {a} {b}"
            );
        }
    }
}

#[test]
fn reuses_dead_storage_but_keeps_view_bases_and_duplicate_outputs_alive() {
    let text = r#"
a = param(0, f32, 8)
saved = exp2(a)
view = permute(reshape(saved, [2, 4]), [1, 0])
x = a
for step in range(6):
    e = exp2(x)
    x = log2(e + e)
output view, x, view
"#;
    let program = source::parse(text).unwrap();
    let exe = cpu::compile_profiled(&program, Path::new(".cache/pup/tests")).unwrap();
    let metadata = exe.profile_metadata.as_ref().unwrap();
    check_plan(metadata);
    let total: u64 = metadata["memory_plan"]["allocations"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["bytes"].as_u64().unwrap())
        .sum();
    assert!(metadata["workspace_bytes"].as_u64().unwrap() < total);
    let inputs = [floats(&[0., 1., 2., 3., 0., 1., 2., 3.])];
    for threads in [1, 3] {
        let out = exe.run_profiled(&inputs, threads).unwrap().outputs;
        assert_eq!(out[0].f32().unwrap(), &[1., 1., 2., 2., 4., 4., 8., 8.]);
        assert_eq!(out[2].f32().unwrap(), out[0].f32().unwrap());
        assert_eq!(out[1].f32().unwrap(), &[6., 7., 8., 9., 6., 7., 8., 9.]);
    }
}

#[test]
fn fused_arithmetic_preserves_rounding_and_byte_wraps() {
    let text="x = param(0,f32,2)\nu = param(1,u8,2)\na = (x + 1.0) - x\nb = (u + cast(1,u8)) * cast(2,u8)\noutput a, b\n";
    let p = source::parse(text).unwrap();
    let exe = cpu::compile(&p.graph, p.root, Path::new(".cache/pup/tests")).unwrap();
    let out = exe
        .run_with_threads(
            &[floats(&[16777216., 1.]), Tensor::U8(vec![255, 127].into())],
            1,
        )
        .unwrap();
    assert_eq!(out[0].f32().unwrap(), &[0., 1.]);
    let Tensor::U8(bytes) = &out[1] else {
        panic!("u8 output")
    };
    assert_eq!(&**bytes, &[0, 0]);
}

#[test]
fn shared_contraction_and_epilogue_dependencies_remain_correct() {
    let text = r#"
a = reshape(param(0,f32,6), [2,3])
b = reshape(param(1,f32,6), [3,2])
c = matmul(a,b)
y = c + reduce(c,add,1)
output y, c
"#;
    let p = source::parse(text).unwrap();
    let exe = cpu::compile_profiled(&p, Path::new(".cache/pup/tests")).unwrap();
    check_plan(exe.profile_metadata.as_ref().unwrap());
    let out = exe
        .run_with_threads(
            &[
                floats(&[1., 2., 3., 4., 5., 6.]),
                floats(&[1., 2., 3., 4., 5., 6.]),
            ],
            2,
        )
        .unwrap();
    assert_eq!(out[1].f32().unwrap(), &[22., 28., 49., 64.]);
    assert_eq!(out[0].f32().unwrap(), &[93., 120., 120., 156.]);
}

#[test]
fn competing_reduction_epilogues_do_not_eliminate_each_others_inputs() {
    let text = "a = reshape(param(0,f32,4), [2,2])\nb = reshape(param(1,f32,4), [2,2])\nc = matmul(a,b)\nd = matmul(b,a)\ny = c + d\nz = reduce(a,add,1) + reduce(b,add,1)\noutput y,z\n";
    let p = source::parse(text).unwrap();
    let exe = cpu::compile_profiled(&p, Path::new(".cache/pup/tests")).unwrap();
    check_plan(exe.profile_metadata.as_ref().unwrap());
    let out = exe
        .run_with_threads(&[floats(&[1., 2., 3., 4.]), floats(&[5., 6., 7., 8.])], 2)
        .unwrap();
    assert_eq!(out[0].f32().unwrap(), &[42., 56., 74., 96.]);
    assert_eq!(out[1].f32().unwrap(), &[16., 20.]);
}

#[test]
fn workspace_limit_includes_caller_outputs_after_planning() {
    let text="x = reshape(param(0,u8,600000000),[2,300000000])\ny = reduce(cast(x,f32),add,1)\noutput reshape(y,[1,300000000])\n";
    let p = source::parse(text).unwrap();
    let error = cpu::emit(&p.graph, p.root).unwrap_err().to_string();
    assert!(
        error.contains("workspace exceeds 2 GiB including outputs"),
        "{error}"
    );
}

#[test]
fn contraction_coordinates_preserve_sliced_flipped_and_fused_operands() {
    let text = r#"
raw = reshape(param(0,u8,99), [9,11])
a = cast(flip(shrink(raw, [1,1], [7,9]), [true,false]), f32) / 255.0
b = flip(permute(reshape(param(1,f32,630), [70,9]), [1,0]), [true,false])
output matmul(a,b)
"#;
    let p = source::parse(text).unwrap();
    let exe = cpu::compile(&p.graph, p.root, Path::new(".cache/pup/tests")).unwrap();
    let raw = (0..99).map(|i| (i * 3 % 256) as u8).collect::<Vec<_>>();
    let b = (0..630).map(|i| (i % 17) as f32 - 8.).collect::<Vec<_>>();
    let expected = (0..7)
        .flat_map(|i| {
            let raw = &raw;
            let b = &b;
            (0..70).map(move |j| {
                (0..9).fold(0., |acc, k| {
                    acc + (raw[(7 - i) * 11 + k + 1] as f32 / 255.) * b[j * 9 + 8 - k]
                })
            })
        })
        .collect::<Vec<_>>();
    for threads in [1, 3] {
        let out = exe
            .run_with_threads(&[Tensor::U8(raw.clone().into()), floats(&b)], threads)
            .unwrap();
        assert_eq!(out[0].f32().unwrap(), expected);
    }
}
