use puppygrad::compiler::{
    cpu::{self, Executable, Tensor},
    pop::DType,
    source::{self, Context, TensorSpec},
};
use std::path::Path;
fn floats(x: &[f32]) -> Tensor {
    Tensor::F32(x.to_vec().into())
}
fn compile(text: &str) -> Executable {
    let p = source::parse(text).unwrap();
    cpu::compile_profiled(&p, Path::new(".cache/pup/tests")).unwrap()
}
fn close(a: f32, b: f32, tol: f32) {
    assert!(
        (a - b).abs() <= tol * (1. + b.abs()),
        "actual {a}, expected {b}"
    );
}
fn finite_differences(exe: &Executable, values: &[Vec<f32>]) {
    let mut inputs: Vec<_> = values.iter().map(|v| floats(v)).collect();
    let outputs = exe.run_with_threads(&inputs, 1).unwrap();
    for (slot, values) in values.iter().enumerate() {
        let analytic = outputs[slot + 1].f32().unwrap();
        assert_eq!(analytic.len(), values.len());
        for i in 0..values.len() {
            let mut plus = values.clone();
            plus[i] += 0.001;
            let mut minus = values.clone();
            minus[i] -= 0.001;
            inputs[slot] = floats(&plus);
            let hi = exe.run_with_threads(&inputs, 1).unwrap()[0].f32().unwrap()[0];
            inputs[slot] = floats(&minus);
            let lo = exe.run_with_threads(&inputs, 1).unwrap()[0].f32().unwrap()[0];
            inputs[slot] = floats(values);
            close(analytic[i], (hi - lo) / 0.002, 0.005);
        }
    }
}
#[test]
fn shared_paths_and_broadcast_gradients_sum_to_original_shapes() {
    let exe=compile("x=reshape(param(0,f32,6),[2,3])\nb=param(1,f32,3)\ns=reshape(param(2,f32,1),[])\ny=x*b+s\nloss=reduce(y*y,add,2)\noutput loss,grad(loss,x),grad(loss,b),grad(loss,s)\n");
    finite_differences(
        &exe,
        &[
            vec![0.1, 0.2, -0.3, 0.4, 0.5, -0.6],
            vec![0.4, -0.2, 0.8],
            vec![0.3],
        ],
    );
}
#[test]
fn unary_arithmetic_derivatives_match_finite_differences() {
    let exe=compile("x=param(0,f32,5)\ny=exp2(x)+log2(x)+sqrt(x)+x/(x+1.0)-x+x*x\nloss=reduce(y,add,1)\noutput loss,grad(loss,x)\n");
    finite_differences(&exe, &[vec![0.5, 0.8, 1.1, 1.4, 1.8]]);
}
#[test]
fn movement_and_stack_pullbacks_match_finite_differences() {
    let exe = compile(
        r#"
x=reshape(param(0,f32,6),[2,3])
p=pad(x,[1,1],[4,5])
f=flip(p,[true,false])
s=shrink(f,[1,0],[2,4])
t=permute(s,[1,0])
a=stack(t,t*2.0)
b=expand(a,[2])
loss=reduce(b*b,add,4)
output loss,grad(loss,x)
"#,
    );
    finite_differences(&exe, &[vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]);
}
#[test]
fn max_ties_relu_zero_and_where_have_explicit_subgradients() {
    let exe=compile("x=param(0,f32,3)\na=reduce(max(x,0.0),add,1)\nb=reduce(x,max,1)\nc=reduce(where(x<0.0,x*x,x),add,1)\noutput grad(a,x),grad(b,x),grad(c,x)\n");
    let out = exe.run_with_threads(&[floats(&[-1., 0., 2.])], 1).unwrap();
    assert_eq!(out[0].f32().unwrap(), &[0., 0., 1.]);
    assert_eq!(out[1].f32().unwrap(), &[0., 0., 1.]);
    assert_eq!(out[2].f32().unwrap(), &[-2., 1., 1.]);
    let out = exe.run_with_threads(&[floats(&[2., 2., 1.])], 1).unwrap();
    assert_eq!(out[1].f32().unwrap(), &[0.5, 0.5, 0.]);
    let exe=compile("a=param(0,f32,1)\nb=param(1,f32,1)\nloss=reduce(max(a,b),add,1)\noutput grad(loss,a),grad(loss,b)\n");
    let out = exe
        .run_with_threads(&[floats(&[2.]), floats(&[2.])], 1)
        .unwrap();
    assert_eq!(out[0].f32().unwrap(), &[0.]);
    assert_eq!(out[1].f32().unwrap(), &[1.]);
}
#[test]
fn product_reduction_handles_zero_factors() {
    let exe = compile("x=param(0,f32,3)\nloss=reduce(x,mul,1)\noutput grad(loss,x)\n");
    for (x, expected) in [
        ([2., 3., 4.], [12., 8., 6.]),
        ([2., 0., 4.], [0., 8., 0.]),
        ([0., 0., 4.], [0., 0., 0.]),
    ] {
        let out = exe.run_with_threads(&[floats(&x)], 1).unwrap();
        assert_eq!(out[0].f32().unwrap(), &expected);
    }
}
#[test]
fn matmul_pullbacks_remain_matrix_contractions() {
    let exe=compile("a=reshape(param(0,f32,6),[2,3])\nb=reshape(param(1,f32,6),[3,2])\nc=matmul(a,b)\nloss=reduce(c*c,add,2)\noutput loss,grad(loss,a),grad(loss,b)\n");
    assert_eq!(exe.gemm_count, 3);
    finite_differences(
        &exe,
        &[
            vec![0.1, 0.2, 0.3, -0.4, 0.5, 0.6],
            vec![0.2, -0.3, 0.4, 0.5, 0.6, 0.7],
        ],
    );
    let exe=compile("a=reshape(param(0,f32,12),[2,2,3])\nb=reshape(param(1,f32,12),[2,3,2])\nc=batched_matmul(a,b)\nloss=reduce(c*c,add,3)\noutput loss,grad(loss,a),grad(loss,b)\n");
    assert_eq!(exe.gemm_count, 3);
    finite_differences(
        &exe,
        &[
            (0..12).map(|i| i as f32 * 0.03 - 0.1).collect(),
            (0..12).map(|i| i as f32 * 0.02 + 0.1).collect(),
        ],
    );
}
#[test]
fn shared_contraction_product_also_receives_other_path_contributions() {
    let exe = compile(
        r#"
a=reshape(param(0,f32,4),[2,1,2])
b=reshape(param(1,f32,4),[1,2,2])
p=a*b
c=reduce(permute(p,[2,0,1]),add,1)
loss=reduce(c*c,add,2)+reduce(p,add,3)
output loss,grad(loss,a),grad(loss,b)
"#,
    );
    finite_differences(&exe, &[vec![0.1, 0.2, 0.3, 0.4], vec![0.5, 0.6, 0.7, 0.8]]);
}
#[test]
fn disconnected_and_discrete_paths_produce_zeros_and_empty_shapes_work() {
    let exe =
        compile("x=param(0,f32,3)\ny=reduce(cast(cast(x,i32),f32),add,1)\noutput grad(y,x)\n");
    let out = exe
        .run_with_threads(&[floats(&[1.2, 2.3, 3.4])], 1)
        .unwrap();
    assert_eq!(out[0].f32().unwrap(), &[0., 0., 0.]);
    let exe = compile("x=param(0,f32,0)\ny=reduce(x,add,1)\noutput grad(y,x)\n");
    assert!(exe.run_with_threads(&[floats(&[])], 1).unwrap()[0].is_empty());
    let exe = compile("x=param(0,f32,3)\ny=cast(2.0,f32)\noutput grad(y,x)\n");
    assert_eq!(
        exe.run_with_threads(&[floats(&[1., 2., 3.])], 1).unwrap()[0]
            .f32()
            .unwrap(),
        &[0., 0., 0.]
    );
}
#[test]
fn invalid_targets_and_unsupported_requested_paths_are_errors() {
    for (text,message) in [
        ("x=param(0,f32,2)\noutput grad(x,x)\n","scalar f32 loss"),
        ("x=param(0,i32,1)\ny=reduce(cast(x,f32),add,1)\noutput grad(y,x)\n","f32 differentiation targets"),
        ("x=param(0,f32,3)\ni=param(1,i32,1)\ny=reduce(load(index(x,i)),add,1)\noutput grad(y,x)\n","scatter-add"),
        ("x=cast(1.0,f32)\noutput grad(x)\n","expects")
    ] { assert!(source::parse(text).err().unwrap().to_string().contains(message)); }
    // Unsupported gather derivative is irrelevant when only another branch is requested.
    let exe=compile("x=param(0,f32,3)\ni=param(1,i32,1)\nw=param(2,f32,scalar)\ny=reduce(load(index(x,i)),add,1)+w*w\noutput grad(y,w)\n");
    assert_eq!(
        exe.run_with_threads(
            &[
                floats(&[1., 2., 3.]),
                Tensor::I32(vec![1].into()),
                floats(&[3.])
            ],
            1
        )
        .unwrap()[0]
            .f32()
            .unwrap(),
        &[6.]
    );
}
#[test]
fn repeated_requests_share_nodes_and_scalar_second_derivatives_work() {
    let p = source::parse(
        "x=param(0,f32,scalar)\ny=x*x*x\na=grad(y,x)\nb=grad(y,x)\noutput a,b,grad(a,x)\n",
    )
    .unwrap();
    assert_eq!(
        p.bindings.iter().find(|b| b.name == "a").unwrap().value,
        p.bindings.iter().find(|b| b.name == "b").unwrap().value
    );
    let exe = cpu::compile(&p.graph, p.root, Path::new(".cache/pup/tests")).unwrap();
    let out = exe.run_with_threads(&[floats(&[2.])], 1).unwrap();
    assert_eq!(out[0].f32().unwrap(), &[12.]);
    assert_eq!(out[2].f32().unwrap(), &[12.]);
}
fn mnist(text: &str) -> Executable {
    let mut c = Context::default();
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
        c.tensors.insert(
            name.into(),
            TensorSpec {
                slot,
                dtype: if slot < 2 { DType::U8 } else { DType::F32 },
                shape,
            },
        );
    }
    let p = source::parse_with_context(text, &c).unwrap();
    cpu::compile_profiled(&p, Path::new(".cache/pup/tests")).unwrap()
}
#[test]
fn mnist_autodiff_matches_handwritten_updates_full_and_padded_batches() {
    let manual = mnist(include_str!("../examples/mnist.pup"));
    let automatic = mnist(include_str!("../examples/mnist_autodiff.pup"));
    assert_eq!(automatic.gemm_count, 5);
    for valid_rows in [128, 96] {
        let inputs = [
            Tensor::U8(
                (0..128 * 784)
                    .map(|i| (i * 17) as u8)
                    .collect::<Vec<_>>()
                    .into(),
            ),
            Tensor::U8((0..128).map(|i| (i % 10) as u8).collect::<Vec<_>>().into()),
            floats(
                &(0..784 * 64)
                    .map(|i| ((i % 31) as f32 - 15.) * 0.002)
                    .collect::<Vec<_>>(),
            ),
            floats(&[0.1; 64]),
            floats(
                &(0..64 * 10)
                    .map(|i| ((i % 19) as f32 - 9.) * 0.02)
                    .collect::<Vec<_>>(),
            ),
            floats(&[0.; 10]),
            floats(&[0.1]),
            floats(
                &(0..128)
                    .map(|i| if i < valid_rows { 1. } else { 0. })
                    .collect::<Vec<_>>(),
            ),
        ];
        let expected = manual.run_with_threads(&inputs, 1).unwrap();
        for threads in [1, 2] {
            let actual = automatic.run_with_threads(&inputs, threads).unwrap();
            for (a, b) in actual.iter().zip(&expected) {
                for (&a, &b) in a.f32().unwrap().iter().zip(b.f32().unwrap()) {
                    close(a, b, 1e-5);
                }
            }
        }
    }
}

#[test]
fn contraction_shortcut_preserves_requested_intermediate_gradients() {
    use puppygrad::compiler::{
        autodiff,
        pop::{Arg, Op},
    };
    let text="a=reshape(param(0,f32,4),[2,1,2])\nb=reshape(param(1,f32,4),[1,2,2])\np=a*b\nq=permute(p,[2,0,1])\nc=reduce(q,add,1)\nloss=reduce(c,add,2)\noutput loss\n";
    let mut program = source::parse(text).unwrap();
    let value = |name: &str| {
        program
            .bindings
            .iter()
            .find(|b| b.name == name)
            .unwrap()
            .value
    };
    let (loss, a, p, q) = (value("loss"), value("a"), value("p"), value("q"));
    let gradients = autodiff::gradients(&mut program.graph, loss, &[a, p, q]).unwrap();
    let root = program
        .graph
        .apply(Op::Sink, &gradients, Arg::None)
        .unwrap();
    let exe = cpu::compile(&program.graph, root, Path::new(".cache/pup/tests")).unwrap();
    let out = exe
        .run_with_threads(&[floats(&[1., 2., 3., 4.]), floats(&[5., 6., 7., 8.])], 1)
        .unwrap();
    assert_eq!(out[0].f32().unwrap(), &[12., 14., 12., 14.]);
    assert_eq!(out[1].f32().unwrap(), &[1.; 8]);
    assert_eq!(out[2].f32().unwrap(), &[1.; 8]);
}

#[test]
fn contraction_input_views_remain_differentiable_boundaries() {
    let exe = compile(
        r#"
x=reshape(param(0,f32,4),[2,2])
y=reshape(param(1,f32,6),[2,3])
a=reshape(x,[2,1,2])
bt=permute(y,[1,0])
b=reshape(bt,[1,3,2])
c=reduce(permute(a*b,[2,0,1]),add,1)
loss=reduce(c,add,2)
output grad(loss,x),grad(loss,a),grad(loss,y),grad(loss,bt),grad(loss,b)
"#,
    );
    let out = exe
        .run_with_threads(
            &[
                floats(&[1., 2., 3., 4.]),
                floats(&[5., 6., 7., 8., 9., 10.]),
            ],
            1,
        )
        .unwrap();
    assert_eq!(out[0].f32().unwrap(), &[18., 27., 18., 27.]);
    assert_eq!(out[1].f32().unwrap(), &[18., 27., 18., 27.]);
    assert_eq!(out[2].f32().unwrap(), &[4., 4., 4., 6., 6., 6.]);
    assert_eq!(out[3].f32().unwrap(), &[4., 6., 4., 6., 4., 6.]);
    assert_eq!(out[4].f32().unwrap(), &[4., 6., 4., 6., 4., 6.]);
}

#[test]
fn batched_contraction_pullback_sums_broadcast_batches() {
    let exe = compile(
        r#"
a=reshape(param(0,f32,4),[1,2,2])
b=reshape(param(1,f32,12),[3,2,2])
x=reshape(a,[1,2,1,2])
y=reshape(permute(b,[0,2,1]),[3,1,2,2])
c=reduce(permute(x*y,[3,0,1,2]),add,1)
loss=reduce(c*c,add,3)
output loss,grad(loss,a),grad(loss,b)
"#,
    );
    finite_differences(
        &exe,
        &[
            vec![0.1, 0.2, 0.3, -0.4],
            vec![
                0.2, -0.1, 0.4, 0.6, -0.3, 0.2, 0.1, 0.5, 0.7, 0.1, -0.2, 0.3,
            ],
        ],
    );
}
