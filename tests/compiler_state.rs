use puppygrad::compiler::{
    cpu::{self, Tensor},
    cuda, source,
};
use std::path::Path;

const COUNTER: &str = "s = state(\"counter\", i32, [1])\nold = load(index(s, cast(0, i32)))\nw = store(index(s, cast(0, i32)), old + 1)\noutput load(index(after(s, w), cast(0, i32)))";
fn ints(t: &Tensor) -> &[i32] {
    match t {
        Tensor::I32(x) => x,
        _ => panic!("expected i32"),
    }
}
#[test]
fn retained_counter_reset_and_independent_runtime() {
    let p = source::parse(COUNTER).unwrap();
    let mut a = cpu::compile(&p.graph, p.root, Path::new(".cache/pup/state-tests")).unwrap();
    let runtime = cpu::Runtime::default();
    a.share_runtime(&runtime);
    for i in 1..=4 {
        assert_eq!(ints(&a.run(&[]).unwrap()[0]), &[i]);
    }
    let b = cpu::compile(&p.graph, p.root, Path::new(".cache/pup/state-tests")).unwrap();
    assert_eq!(ints(&b.run(&[]).unwrap()[0]), &[1]);
    runtime.reset_state().unwrap();
    assert_eq!(ints(&a.run(&[]).unwrap()[0]), &[1]);
}
const ROWS: &str = "x = reshape(param(0, f32, 6), [3, 2])\ns = state(\"rows\", f32, [4, 2])\nw = store(index(s, cast(stack(1, 1, 2), i32)), x)\na = after(s, w)\nc = store(index(a, cast(stack(1, 2), i32)), shrink(a, [0, 0], [2, 2]))\noutput after(a, c)";
#[test]
fn ordered_duplicate_indices_and_overlapping_copy_snapshot() {
    let p = source::parse(ROWS).unwrap();
    let exe = cpu::compile(&p.graph, p.root, Path::new(".cache/pup/state-tests")).unwrap();
    let out = exe
        .run(&[Tensor::F32(vec![1., 2., 3., 4., 5., 6.].into())])
        .unwrap();
    assert_eq!(out[0].f32().unwrap(), &[0., 0., 0., 0., 3., 4., 0., 0.]);
}
#[test]
fn state_writes_have_identity_and_require_explicit_dependencies() {
    assert!(source::parse(
        "s = state(\"x\", i32, [1])\noutput store(index(s,cast(0,i32)),cast(1,i32))"
    )
    .is_err());
    assert!(source::parse(
        "s = state(\"x\", i32, [1])\nw = store(index(s, cast(0, i32)), cast(1, i32))\noutput s"
    )
    .err()
    .unwrap()
    .to_string()
    .contains("AFTER"));
    assert!(source::parse(
        "s = param(0, i32, 1)\nw = store(index(s, cast(0, i32)), cast(1, i32))\noutput after(s,w)"
    )
    .is_err());
    let p = source::parse("def write(s):\n    w = store(index(s, cast(0, i32)), cast(1, i32))\n    return after(s,w)\ns = state(\"x\", i32, [1])\na = write(s)\nb = write(s)\noutput a, b").unwrap();
    assert_eq!(
        p.graph
            .toposort(p.root)
            .unwrap()
            .iter()
            .filter(|&&v| p.graph.node(v).unwrap().op() == puppygrad::compiler::pop::Op::Store)
            .count(),
        2
    );
}
#[test]
fn bounds_checked_writes_and_reset_after_error() {
    let p = source::parse("i = param(0, i32, 1)\ns = state(\"x\", i32, [2])\nw = store(index(s, i), cast(stack(7),i32))\noutput after(s,w)").unwrap();
    let exe = cpu::compile(&p.graph, p.root, Path::new(".cache/pup/state-tests")).unwrap();
    assert!(exe.run(&[Tensor::I32(vec![2].into())]).is_err());
    assert_eq!(
        ints(&exe.run(&[Tensor::I32(vec![1].into())]).unwrap()[0]),
        &[0, 7]
    );
}
#[test]
#[ignore = "requires NVIDIA GPU and NVRTC"]
fn cuda_state_is_retained_and_reset_without_host_uploads() {
    let p = source::parse(COUNTER).unwrap();
    let runtime = cuda::Runtime::new(0).unwrap();
    runtime.set_graph_replay(true);
    let exe = cuda::compile_with_runtime(
        &p.graph,
        p.root,
        Path::new(".cache/pup/state-tests/cuda"),
        &runtime,
    )
    .unwrap();
    for i in 1..=4 {
        assert_eq!(ints(&exe.run(&[]).unwrap()[0]), &[i]);
    }
    let before = runtime.residency_stats();
    assert_eq!(before.state_bytes, 4);
    assert_eq!(before.input_uploads, 0);
    runtime.reset_state().unwrap();
    assert_eq!(ints(&exe.run(&[]).unwrap()[0]), &[1]);
    let after = runtime.residency_stats();
    assert_eq!(before.allocations, after.allocations);
    assert_eq!(runtime.execution_stats().graph_builds, 1);
    assert_eq!(runtime.execution_stats().graph_launches, 5);
    runtime.set_graph_replay(false);
    assert_eq!(ints(&exe.run(&[]).unwrap()[0]), &[2]);
    assert_eq!(
        runtime.execution_stats().direct_kernel_launches,
        exe.kernel_count() as u64
    );
    runtime.set_graph_replay(true);
    assert_eq!(ints(&exe.run(&[]).unwrap()[0]), &[3]);
    assert_eq!(runtime.execution_stats().graph_builds, 1);
    let p = source::parse(ROWS).unwrap();
    // Independent runtime isolates different state declarations.
    let exe = cuda::compile(
        &p.graph,
        p.root,
        Path::new(".cache/pup/state-tests/cuda"),
        0,
    )
    .unwrap();
    let out = exe
        .run(&[Tensor::F32(vec![1., 2., 3., 4., 5., 6.].into())])
        .unwrap();
    assert_eq!(out[0].f32().unwrap(), &[0., 0., 0., 0., 3., 4., 0., 0.]);
    let p=source::parse("s = state(\"x\", f32, [3])\nw = store(index(s,cast(arange(3),i32)),cast(stack(1,2,3),f32))\na = after(s,w)\nc = store(index(a,cast(stack(1,2,0),i32)),reshape(a + 1.0,[3]))\noutput after(a,c)").unwrap();
    let exe = cuda::compile(
        &p.graph,
        p.root,
        Path::new(".cache/pup/state-tests/cuda"),
        0,
    )
    .unwrap();
    assert_eq!(exe.run(&[]).unwrap()[0].f32().unwrap(), &[4., 2., 3.]);
}

#[test]
fn store_payload_view_materializes_before_overlapping_update() {
    let text="s = state(\"x\", f32, [3])\nw = store(index(s, cast(arange(3), i32)), cast(stack(1,2,3), f32))\na = after(s,w)\npayload = reshape(permute(reshape(a + 1.0, [1,3]), [1,0]), [3])\nc = store(index(a, cast(stack(1,2,0), i32)), payload)\noutput after(a,c)";
    let p = source::parse(text).unwrap();
    let exe = cpu::compile(&p.graph, p.root, Path::new(".cache/pup/state-tests")).unwrap();
    assert_eq!(exe.run(&[]).unwrap()[0].f32().unwrap(), &[4., 2., 3.]);
}

#[test]
fn state_declared_before_caller_parameter_has_an_independent_slot() {
    let p=source::parse("s = state(\"sum\", i32, [1])\nx = param(0, i32, 1)\nw = store(index(s,cast(stack(0),i32)), s + x)\noutput after(s,w)").unwrap();
    assert_eq!(p.states[0].slot, 1);
    let exe = cpu::compile(&p.graph, p.root, Path::new(".cache/pup/state-tests")).unwrap();
    assert_eq!(
        ints(&exe.run(&[Tensor::I32(vec![3].into())]).unwrap()[0]),
        &[3]
    );
    assert_eq!(
        ints(&exe.run(&[Tensor::I32(vec![2].into())]).unwrap()[0]),
        &[5]
    );
}

#[test]
fn retained_state_survives_input_shape_specializations() {
    use puppygrad::compiler::{
        pop::DType,
        source::{Context, TensorSpec},
    };
    let text="x = input(\"x\")\ns = state(\"rows\", i32, [4])\np = state(\"position\", i32, [1])\nstart = load(index(p,cast(0,i32)))\nw = store(index(s, arange(dim(x,0)) + start), x)\nadvance = store(index(p,cast(0,i32)), start + dim(x,0))\noutput after(s,w,advance)";
    let runtime = cpu::Runtime::default();
    for (values, expected) in [(vec![2, 3], vec![2, 3, 0, 0]), (vec![4], vec![2, 3, 4, 0])] {
        let mut context = Context::default();
        context.tensors.insert(
            "x".into(),
            TensorSpec {
                slot: 0,
                dtype: DType::I32,
                shape: vec![values.len()],
            },
        );
        let p = source::parse_with_context(text, &context).unwrap();
        let mut exe = cpu::compile(&p.graph, p.root, Path::new(".cache/pup/state-tests")).unwrap();
        exe.share_runtime(&runtime);
        assert_eq!(
            ints(&exe.run(&[Tensor::I32(values.into())]).unwrap()[0]),
            expected
        );
    }
}
