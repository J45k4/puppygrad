use puppygrad::{
    compiler::{
        cuda,
        pop::{DType, Scalar},
        source::{self, Context, TensorSpec},
    },
    runtime::llm_capacity,
};

const RETAINED: &str = "x = input(\"tokens\")\ns = state(\"history\", i32, [config(\"buffer_capacity\")])\nw = store(index(s, arange(dim(x, 0))), x)\noutput after(s, w)";
fn context() -> Context {
    let mut c = Context::default();
    c.constants.insert("n_positions".into(), Scalar::Int(1024));
    c.constants
        .insert("buffer_capacity".into(), Scalar::Int(512));
    c.tensors.insert(
        "tokens".into(),
        TensorSpec {
            slot: 0,
            dtype: DType::I32,
            shape: vec![1],
        },
    );
    c
}

#[test]
fn context_search_respects_budget_reserve_and_model_limit_without_gpu() {
    let c = context();
    let target = llm_capacity::request_plan(RETAINED, &c, 257, 257, Some(8), true).unwrap();
    let reserve = 100;
    let report = llm_capacity::determine(
        RETAINED,
        &c,
        target.total_bytes() + reserve,
        reserve,
        Some(8),
        false,
    )
    .unwrap();
    assert!(report.retained);
    assert_eq!(report.prefill_chunk_tokens, Some(8));
    assert!(report.max_context_tokens >= 257 && report.max_context_tokens < 1024);
    assert!(report.planned_device_bytes <= report.budget_bytes);
    let next = llm_capacity::request_plan(
        RETAINED,
        &c,
        report.max_context_tokens + 1,
        report.max_context_tokens + 1,
        Some(8),
        true,
    )
    .unwrap();
    assert!(next.total_bytes() > report.budget_bytes);
    let more =
        llm_capacity::determine(RETAINED, &c, target.total_bytes() * 2, 0, Some(8), false).unwrap();
    assert!(more.max_context_tokens > report.max_context_tokens);
    let full = llm_capacity::determine(RETAINED, &c, 1_000_000, 0, None, false).unwrap();
    assert_eq!(full.max_context_tokens, 1024);
    assert_eq!(full.limiting_reason, "model position limit");
    assert!(llm_capacity::determine(RETAINED, &c, 0, 0, None, false)
        .unwrap_err()
        .to_string()
        .contains("minimum context"));
    assert!(llm_capacity::determine(RETAINED, &c, 1, 2, None, false)
        .unwrap_err()
        .to_string()
        .contains("reserve"));
}

#[test]
fn chunks_require_retention_and_fixed_state_layout() {
    let c = context();
    let pure = "output input(\"tokens\") + 1";
    assert!(
        llm_capacity::determine(pure, &c, 1_000_000, 0, Some(8), false)
            .unwrap_err()
            .to_string()
            .contains("retained state")
    );
    let report = llm_capacity::determine(pure, &c, 1_000_000, 0, None, false).unwrap();
    assert!(!report.retained);
    assert_eq!(report.prefill_chunk_tokens, None);
    assert!(llm_capacity::determine(RETAINED, &c, 1_000_000, 0, Some(0), false).is_err());
    let changes = RETAINED.replace("config(\"buffer_capacity\")", "dim(x, 0)");
    assert!(
        llm_capacity::request_plan(&changes, &c, 32, 7, Some(4), true)
            .unwrap_err()
            .to_string()
            .contains("state layout")
    );
    let renames = RETAINED
        .replace("s = state", "length = dim(x, 0)\ns = state")
        .replace("\"history\"", "f\"history.{length}\"");
    assert!(
        llm_capacity::request_plan(&renames, &c, 32, 7, Some(4), true)
            .unwrap_err()
            .to_string()
            .contains("state layout")
    );
    let single = llm_capacity::determine(RETAINED, &c, 1_000_000, 0, None, true).unwrap();
    assert_eq!(single.prefill_chunk_tokens, None);
}

#[test]
fn merged_plan_counts_shared_weights_once_and_keeps_largest_buffers() {
    let build = |n| {
        let p = source::parse(&format!("x = param(0, f32, {n})\ns = state(\"s\", f32, [1])\nw = store(index(s, cast(0, i32)), reduce(x, add, 1))\noutput after(s, w)")).unwrap();
        cuda::memory_plan(&p.graph, p.root).unwrap()
    };
    let mut large = build(17);
    let small = build(3);
    let expected = large.total_bytes();
    large.merge(&small).unwrap();
    assert_eq!(large.total_bytes(), expected);
    assert_eq!(large.input_bytes(), 17 * 4);
    assert_eq!(large.state_bytes(), 4);
    let mut wrong_dtype = small.clone();
    wrong_dtype
        .parameters
        .values_mut()
        .find(|p| p.state)
        .unwrap()
        .dtype = DType::I32;
    assert!(large.merge(&wrong_dtype).is_err());
}

#[test]
#[ignore = "requires NVIDIA GPU and NVRTC"]
fn device_plan_matches_actual_allocations_and_checks_live_memory() {
    use puppygrad::compiler::cpu::Tensor;
    let p = source::parse("x = param(0, f32, 17)\ns = state(\"s\", f32, [1])\nw = store(index(s, cast(0, i32)), reduce(x, add, 1))\noutput after(s, w)").unwrap();
    let plan = cuda::memory_plan(&p.graph, p.root).unwrap();
    let rt = cuda::Runtime::new(0).unwrap();
    let free = rt.memory_info().unwrap();
    assert!(free.free_bytes > 0 && free.free_bytes <= free.total_bytes);
    rt.check_memory(&plan, 0).unwrap();
    assert!(rt.check_memory(&plan, free.total_bytes).is_err());
    rt.set_memory_limit(Some(plan.total_bytes() - 1)).unwrap();
    assert!(rt
        .check_memory(&plan, 0)
        .unwrap_err()
        .to_string()
        .contains("--max-memory"));
    rt.set_memory_limit(Some(plan.total_bytes())).unwrap();
    let e = cuda::compile_with_runtime(
        &p.graph,
        p.root,
        std::path::Path::new(".cache/pup/capacity-tests"),
        &rt,
    )
    .unwrap();
    e.run(&[Tensor::F32(vec![1.; 17].into())]).unwrap();
    assert_eq!(plan.total_bytes(), rt.residency_stats().resident_bytes);
    rt.check_memory(&plan, 0).unwrap();
    assert!(rt.set_memory_limit(Some(plan.total_bytes() - 1)).is_err());
    // Bypass the planner to prove allocations themselves enforce the cap.
    let large = source::parse("output param(0, f32, 1024) + 1").unwrap();
    let grow = cuda::compile_with_runtime(
        &large.graph,
        large.root,
        std::path::Path::new(".cache/pup/capacity-tests"),
        &rt,
    )
    .unwrap();
    let error = grow.run(&[Tensor::F32(vec![1.; 1024].into())]).unwrap_err();
    assert!(error.to_string().contains("--max-memory"));
    assert!(rt.residency_stats().resident_bytes <= plan.total_bytes());
    rt.set_memory_limit(None).unwrap();
    let output = grow.run(&[Tensor::F32(vec![1.; 1024].into())]).unwrap();
    assert_eq!(output[0].f32().unwrap(), &[2.; 1024]);
}
