use super::pop::{Arg, DType, Graph, Op, ReduceOp, Result, Scalar, Value};
use super::rewrite::{rewrite, simplify_views};
use super::source;
use super::spec::{verify, Spec, TINYGRAD_REVISION};
use super::tensor::{input, matmul};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct Corpus {
    revision: String,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    nodes: Vec<Instruction>,
    expected: Vec<Expected>,
    failed_at: Option<usize>,
    program: Option<bool>,
    kernel_graph: Option<bool>,
}
#[derive(Deserialize)]
struct Instruction {
    op: Op,
    src: Vec<usize>,
    arg: Arg,
}
#[derive(Deserialize)]
struct Expected {
    dtype: DType,
    shape: Option<Vec<usize>>,
}

#[test]
fn matches_pinned_tinygrad_spec_fixtures() {
    let corpus: Corpus =
        serde_json::from_str(include_str!("../../tests/data/compiler/tinygrad-spec.json")).unwrap();
    assert_eq!(corpus.revision, TINYGRAD_REVISION);
    for case in corpus.cases {
        let mut graph = Graph::new();
        let mut nodes = vec![];
        let mut failure = None;
        for (i, ins) in case.nodes.into_iter().enumerate() {
            let src = ins.src.iter().map(|&j| nodes[j]).collect::<Vec<_>>();
            match graph.apply(ins.op, &src, ins.arg) {
                Ok(value) => {
                    assert_ne!(
                        case.failed_at,
                        Some(i),
                        "{} should reject node {i}",
                        case.name
                    );
                    let node = graph.node(value).unwrap();
                    assert_eq!(
                        node.dtype(),
                        case.expected[i].dtype,
                        "{} node {i} dtype",
                        case.name
                    );
                    assert_eq!(
                        node.shape(),
                        case.expected[i].shape.as_deref(),
                        "{} node {i} shape",
                        case.name
                    );
                    nodes.push(value);
                }
                Err(_) => {
                    failure = Some(i);
                    break;
                }
            }
        }
        assert_eq!(failure, case.failed_at, "{} construction", case.name);
        if failure.is_none() {
            let root = *nodes.last().unwrap();
            assert!(verify(&graph, root, Spec::Tensor).is_ok());
            assert_eq!(
                verify(&graph, root, Spec::Program).is_ok(),
                case.program.unwrap(),
                "{} program",
                case.name
            );
            assert_eq!(
                verify(&graph, root, Spec::KernelGraph).is_ok(),
                case.kernel_graph.unwrap(),
                "{} kernel graph",
                case.name
            );
        }
    }
}

#[test]
fn interning_preserves_float_bits_and_rejects_foreign_sources() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.param(0, DType::F32, Some(6))?;
    assert_eq!(a, graph.param(0, DType::F32, Some(6))?);
    let product = graph.apply(Op::Mul, &[a, a], Arg::None)?;
    assert_eq!(product, graph.apply(Op::Mul, &[a, a], Arg::None)?);
    assert_eq!(graph.len(), 2);
    assert!(graph.param(0, DType::I32, Some(6)).is_err());
    let positive = graph.constant(Scalar::float(0.0))?;
    let negative = graph.constant(Scalar::float(-0.0))?;
    assert_ne!(positive, negative);
    assert_eq!(negative, graph.constant(Scalar::float(-0.0))?);
    let nan = graph.constant(Scalar::Float(0x7ff8000000000001))?;
    assert_eq!(nan, graph.constant(Scalar::Float(0x7ff8000000000001))?);
    assert_ne!(nan, graph.constant(Scalar::Float(0x7ff8000000000002))?);
    let mut other = Graph::new();
    let foreign = other.param(0, DType::F32, Some(6))?;
    assert!(graph.apply(Op::Add, &[a, foreign], Arg::None).is_err());
    assert!(graph.toposort(foreign).is_err());
    Ok(())
}

#[test]
fn construction_checks_arity_arguments_and_static_shape_limits() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.param(0, DType::F32, Some(6))?;
    assert!(graph.apply(Op::Add, &[a], Arg::None).is_err());
    assert!(graph.apply(Op::Neg, &[a], Arg::Axes(vec![])).is_err());
    assert!(graph.apply(Op::Reshape, &[a], Arg::None).is_err());
    assert!(graph.param(1, DType::WeakInt, None).is_err());
    let huge = graph.shape_value(&[i64::MAX as usize, 3])?;
    assert!(graph.apply(Op::Expand, &[a, huge], Arg::None).is_err());
    let symbolic = graph.param(2, DType::I32, None)?;
    assert!(graph.apply(Op::Reshape, &[a, symbolic], Arg::None).is_err());
    Ok(())
}

#[test]
fn matmul_decomposition_has_correct_indexing_without_product_allocation() -> Result<()> {
    for (m, k, n) in [(2, 3, 4), (1, 4, 3), (3, 1, 2), (2, 0, 3)] {
        let mut graph = Graph::new();
        let a = input(&mut graph, 0, DType::F32, &[m, k])?;
        let b = input(&mut graph, 1, DType::F32, &[k, n])?;
        let c = matmul(&mut graph, a, b)?;
        assert_eq!(graph.node(c)?.shape(), Some([m, n].as_slice()));
        assert!(graph.len() < 30); // graph size is independent of tensor element count
        let av: Vec<f64> = (0..m * k).map(|i| i as f64 - 2.0).collect();
        let bv: Vec<f64> = (0..k * n).map(|i| 3.0 - i as f64).collect();
        let inputs = HashMap::from([(0, av.clone()), (1, bv.clone())]);
        for i in 0..m {
            for j in 0..n {
                let expected: f64 = (0..k).map(|p| av[i * k + p] * bv[p * n + j]).sum();
                assert_eq!(element(&graph, c, &[i, j], &inputs), expected);
            }
        }
        verify(&graph, c, Spec::Tensor)?;
    }
    Ok(())
}

#[test]
fn source_language_builds_pops_and_preserves_source_metadata() -> Result<()> {
    let program = source::parse(include_str!("../../examples/matmul.pup"))?;
    let graph = &program.graph;
    let result = graph.node(program.root)?.src()[0];
    assert_eq!(graph.node(result)?.shape(), Some([2, 4].as_slice()));
    assert_eq!(graph.node(result)?.op(), Op::Reduce);
    let inputs = HashMap::from([
        (0, vec![1., 2., 3., 4., 5., 6.]),
        (1, (1..=12).map(f64::from).collect()),
    ]);
    let actual = (0..2)
        .flat_map(|i| (0..4).map(move |j| (i, j)))
        .map(|(i, j)| element(graph, result, &[i, j], &inputs))
        .collect::<Vec<_>>();
    assert_eq!(actual, vec![38., 44., 50., 56., 83., 98., 113., 128.]);
    let p = source::parse("x = const(2)\ny = const(2)\noutput x, y\n")?;
    assert_eq!(p.bindings[0].value, p.bindings[1].value);
    assert_eq!(p.bindings[1].name, "y");
    assert_eq!(p.bindings[1].line, 2);
    Ok(())
}

#[test]
fn source_errors_are_localized_and_unsupported_ops_are_not_guessed() {
    for (source, message) in [
        ("x = add(missing, missing)\noutput x", "1: unknown value"),
        ("x = const(1)\nx = const(2)\noutput x", "2: binding"),
        (
            "x = unknown_op(a, b)\noutput x",
            "1: unknown or unimplemented Pop",
        ),
        (
            "x = const(1)\noutput x\ny = const(2)",
            "3: bindings must precede",
        ),
        ("x = const(1)", "1: expected at least one output"),
        ("x = reshape(a, [2, 3)\noutput x", "1: unclosed list"),
        (
            "x = const(999999999999999999999)\noutput x",
            "1: invalid or out-of-range integer",
        ),
    ] {
        let error = source::parse(source).err().expect("source should fail");
        assert!(error.to_string().starts_with(message), "{source}: {error}");
    }
}

#[test]
fn source_functions_expand_with_local_scope_and_shared_results() -> Result<()> {
    let program = source::parse(
        "def outer(x):\n    y = twice(x)\n    return y\n\n\
         def twice(x):\n    y = add(x, x)\n    return y\n\n\
         x = param(0, f32, scalar)\nother = param(1, f32, scalar)\n\
         y = const(42)\na = outer(x)\nb = outer(x)\nc = outer(other)\noutput a, b, c, y",
    )?;
    let graph = &program.graph;
    let roots = graph.node(program.root)?.src();
    assert_eq!(roots[0], roots[1]);
    assert_ne!(roots[0], roots[2]);
    assert_eq!(graph.node(roots[0])?.op(), Op::Add);
    assert_eq!(graph.node(roots[3])?.arg(), &Arg::Scalar(Scalar::Int(42)));
    assert_eq!(program.bindings.len(), 6); // function locals don't leak into the module
    Ok(())
}

#[test]
fn bundled_linear_is_source_composition_with_correct_weights_and_bias() -> Result<()> {
    let program = source::parse(include_str!("../../examples/linear.pup"))?;
    let graph = &program.graph;
    let result = graph.node(program.root)?.src()[0];
    assert_eq!(graph.node(result)?.shape(), Some([2, 4].as_slice()));
    let inputs = HashMap::from([
        (0, vec![1., 2., 3., 4., 5., 6.]),
        (1, (1..=12).map(f64::from).collect()),
        (2, vec![1., 2., 3., 4.]),
    ]);
    let actual = (0..2)
        .flat_map(|i| (0..4).map(move |j| (i, j)))
        .map(|(i, j)| element(graph, result, &[i, j], &inputs))
        .collect::<Vec<_>>();
    assert_eq!(actual, vec![15., 34., 53., 72., 33., 79., 125., 171.]);
    verify(graph, program.root, Spec::Tensor)?;
    Ok(())
}

#[test]
fn library_matmul_specializes_shapes_and_handles_empty_contractions() -> Result<()> {
    for (m, k, n) in [(2, 3, 4), (1, 4, 3), (3, 1, 2), (2, 0, 3)] {
        let source = format!("import nn\na0 = param(0, f32, {})\nb0 = param(1, f32, {})\na = reshape(a0, [{m}, {k}])\nb = reshape(b0, [{k}, {n}])\ny = nn.matmul(a, b)\noutput y", m*k, k*n);
        let program = source::parse(&source)?;
        let graph = &program.graph;
        let result = graph.node(program.root)?.src()[0];
        assert_eq!(graph.node(result)?.shape(), Some([m, n].as_slice()));
        let inputs = HashMap::from([
            (0, (0..m * k).map(|i| i as f64 - 2.).collect::<Vec<_>>()),
            (1, (0..k * n).map(|i| 3. - i as f64).collect::<Vec<_>>()),
        ]);
        for i in 0..m {
            for j in 0..n {
                let expected: f64 = (0..k)
                    .map(|p| inputs[&0][i * k + p] * inputs[&1][p * n + j])
                    .sum();
                assert_eq!(element(graph, result, &[i, j], &inputs), expected);
            }
        }
    }
    // Zero-sized output must not conceal unequal contraction dimensions.
    let invalid = "import nn\na0 = param(0, f32, 6)\nb0 = param(1, f32, 0)\na = reshape(a0, [2, 3])\nb = reshape(b0, [4, 0])\ny = nn.matmul(a, b)\noutput y";
    let error = source::parse(invalid).err().unwrap().to_string();
    assert!(error.contains("6: in nn.matmul"), "{error}");
    assert!(error.contains("assert_eq failed: 3 != 4"), "{error}");
    Ok(())
}

#[test]
fn function_errors_reject_recursion_captures_and_invalid_declarations() {
    for (source, message) in [
        ("def f(x):\n    y = neg(x)\na = const(1)\noutput a", "function requires a final return"),
        ("def f(x):\n  return x\na = const(1)\noutput a", "four spaces"),
        ("def f(x, x):\n    return x\na = const(1)\noutput a", "unique identifiers"),
        ("def add(x):\n    return x\na = const(1)\noutput a", "primitive name collision"),
        ("def f(x):\n    x = neg(x)\n    return x\na = const(1)\noutput a", "binding \"x\" already exists"),
        ("def f(x):\n    return x\n    y = neg(x)\na = const(1)\noutput a", "return must be the final"),
        ("def f(x):\n    return x\na = const(1)\nb = f(a, a)\noutput b", "expects 1 arguments, got 2"),
        ("def f(x):\n    y = add(x, global)\n    return y\nglobal = const(1)\nb = f(global)\noutput b", "unknown value \"global\""),
        ("def f(x):\n    y = g(x)\n    return y\ndef g(x):\n    y = f(x)\n    return y\na = const(1)\nb = f(a)\noutput b", "recursive function expansion"),
        ("import other\na = const(1)\noutput a", "unknown library"),
    ] {
        let error = source::parse(source).err().expect("source should fail").to_string();
        assert!(error.contains(message), "{source}: {error}");
    }
    let error = source::parse(
        "def f(x):\n    y = neg(missing)\n    return y\na = const(1)\nb = f(a)\noutput b",
    )
    .err()
    .unwrap()
    .to_string();
    assert!(
        error.contains("5: in f (definition line 1): body line 2:"),
        "{error}"
    );
}

#[test]
fn rewrites_preserve_sharing_original_nodes_and_output_contract() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.param(0, DType::F32, Some(6))?;
    let identity = graph.reshape(a, &[6])?;
    let sum = graph.apply(Op::Add, &[identity, identity], Arg::None)?;
    let before = graph.dump(sum)?;
    let simplified = simplify_views(&mut graph, sum)?;
    assert_eq!(graph.node(simplified)?.src(), &[a, a]);
    assert_eq!(graph.dump(sum)?, before);
    let mut visits = HashMap::new();
    rewrite(&mut graph, sum, |_, value| {
        *visits.entry(value).or_insert(0) += 1;
        Ok(None)
    })?;
    assert!(visits.values().all(|&count| count == 1));
    let scalar = graph.constant(Scalar::float(0.0))?;
    assert!(rewrite(&mut graph, a, |_, _| Ok(Some(scalar))).is_err());
    let mut other = Graph::new();
    let foreign = other.param(0, DType::F32, Some(6))?;
    assert!(rewrite(&mut graph, a, |_, _| Ok(Some(foreign))).is_err());
    let zero_add = graph.apply(Op::Add, &[scalar, scalar], Arg::None)?;
    assert_eq!(simplify_views(&mut graph, zero_add)?, zero_add);
    Ok(())
}

#[test]
fn deep_graphs_traverse_without_recursion_and_ignore_unreachable_nodes() -> Result<()> {
    let mut graph = Graph::new();
    let a = graph.param(0, DType::F32, None)?;
    graph.param(1, DType::F32, None)?;
    let mut root = a;
    for _ in 0..20_000 {
        root = graph.apply(Op::Neg, &[root], Arg::None)?;
    }
    let order = graph.toposort(root)?;
    assert_eq!(order.len(), 20_001);
    assert_eq!(order[0], a);
    assert_eq!(order.last(), Some(&root));
    let mut other = Graph::new();
    let other_a = other.param(0, DType::F32, None)?;
    assert_eq!(graph.dump(a)?, other.dump(other_a)?);
    Ok(())
}

// Independent test-only scalar evaluator. It reads graph indices rather than
// materializing the broadcast product, and compares to a conventional dot loop.
fn element(
    graph: &Graph,
    value: Value,
    coords: &[usize],
    inputs: &HashMap<usize, Vec<f64>>,
) -> f64 {
    let node = graph.node(value).unwrap();
    let shape = node.shape().unwrap();
    let src = node.src();
    let at = |v, c: &[usize]| element(graph, v, c, inputs);
    match (node.op(), node.arg()) {
        (Op::Param, Arg::Param(p)) => inputs[&p.slot][flat_index(coords, shape)],
        (Op::Reshape, _) => {
            let mut index = flat_index(coords, shape);
            let source_shape = graph.node(src[0]).unwrap().shape().unwrap();
            let mut original = vec![0; source_shape.len()];
            for axis in (0..source_shape.len()).rev() {
                original[axis] = index % source_shape[axis];
                index /= source_shape[axis];
            }
            at(src[0], &original)
        }
        (Op::Permute, Arg::Axes(axes)) => {
            let mut original = vec![0; axes.len()];
            for (i, &axis) in axes.iter().enumerate() {
                original[axis] = coords[i];
            }
            at(src[0], &original)
        }
        (Op::Mul | Op::Add, _) => {
            let values = src.iter().map(|&v| {
                let source_shape = graph.node(v).unwrap().shape().unwrap();
                let original = coords[coords.len() - source_shape.len()..]
                    .iter()
                    .zip(source_shape)
                    .map(|(&i, &s)| if s == 1 { 0 } else { i })
                    .collect::<Vec<_>>();
                at(v, &original)
            });
            if node.op() == Op::Mul {
                values.product()
            } else {
                values.sum()
            }
        }
        (
            Op::Reduce,
            Arg::Reduce {
                op: ReduceOp::Add,
                num_axes: 1,
            },
        ) => {
            let mut original = [vec![0], coords.to_vec()].concat();
            (0..graph.node(src[0]).unwrap().shape().unwrap()[0])
                .map(|k| {
                    original[0] = k;
                    at(src[0], &original)
                })
                .sum()
        }
        _ => panic!("unsupported test operation {:?}", node.op()),
    }
}
fn flat_index(coords: &[usize], shape: &[usize]) -> usize {
    coords
        .iter()
        .zip(shape)
        .fold(0, |index, (&coord, &dim)| index * dim + coord)
}

#[test]
fn prelude_is_implicit_and_legacy_names_share_expansion() -> Result<()> {
    let direct = source::parse("a = reshape(param(0, f32, 6), [2, 3])\nb = reshape(param(1, f32, 12), [3, 4])\nx = matmul(a, b)\ny = nn.matmul(a, b)\noutput x, y")?;
    let outputs = direct.graph.node(direct.root)?.src();
    assert_eq!(outputs[0], outputs[1]);
    assert_eq!(
        direct.graph.node(outputs[0])?.shape(),
        Some([2, 4].as_slice())
    );
    let legacy = source::parse("import nn\na = reshape(param(0, f32, 6), [2, 3])\nb = reshape(param(1, f32, 12), [3, 4])\nx = nn.matmul(a, b)\ny = nn.matmul(a, b)\noutput x, y")?;
    assert_eq!(
        direct.graph.dump(direct.root)?,
        legacy.graph.dump(legacy.root)?
    );
    let collision = source::parse("def matmul(a, b):\n    return a\nx = cast(1, f32)\noutput x")
        .err()
        .unwrap();
    assert!(collision
        .to_string()
        .contains("provided by the standard library"));
    Ok(())
}
