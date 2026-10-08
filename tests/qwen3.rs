//! Small deterministic checkpoints exercise the same .pup source as Qwen3-0.6B.
//! The independent scalar reference deliberately uses head_dim != hidden/heads.
use puppygrad::{
    compiler::{cpu, gpu, hip, source},
    models::pup_llm::Checkpoint,
};
use safetensors::tensor::{serialize, Dtype, TensorView};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
const D: usize = 8;
const HEADS: usize = 4;
const HD: usize = 4;
const KV: usize = 2;
const H: usize = 12;
const VOCAB: usize = 7;
const LAYERS: usize = 2;
const EPS: f32 = 1e-6;
struct Fixture {
    dir: PathBuf,
    weights: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
}
impl Fixture {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!(
            "puppygrad-qwen3-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut weights = BTreeMap::new();
        let mut add = |name: String, shape: Vec<usize>, norm: bool| {
            let seed = name.bytes().map(usize::from).sum::<usize>();
            let values = (0..shape.iter().product())
                .map(|i| {
                    if norm {
                        1. + (i % 3) as f32 / 32.
                    } else {
                        ((i * 13 + seed) % 23) as f32 / 32. - 11. / 32.
                    }
                })
                .collect::<Vec<_>>();
            weights.insert(name, (shape, values));
        };
        add("model.embed_tokens.weight".into(), vec![VOCAB, D], false);
        add("model.norm.weight".into(), vec![D], true);
        for i in 0..LAYERS {
            for (tail, shape, norm) in [
                ("input_layernorm.weight", vec![D], true),
                ("post_attention_layernorm.weight", vec![D], true),
                ("self_attn.q_proj.weight", vec![HEADS * HD, D], false),
                ("self_attn.k_proj.weight", vec![KV * HD, D], false),
                ("self_attn.v_proj.weight", vec![KV * HD, D], false),
                ("self_attn.o_proj.weight", vec![D, HEADS * HD], false),
                ("self_attn.q_norm.weight", vec![HD], true),
                ("self_attn.k_norm.weight", vec![HD], true),
                ("mlp.gate_proj.weight", vec![H, D], false),
                ("mlp.up_proj.weight", vec![H, D], false),
                ("mlp.down_proj.weight", vec![D, H], false),
            ] {
                add(format!("model.layers.{i}.{tail}"), shape, norm);
            }
        }
        // A duplicate serialized head must not allocate another tied input.
        weights.insert(
            "lm_head.weight".into(),
            weights["model.embed_tokens.weight"].clone(),
        );
        let bytes = weights
            .iter()
            .map(|(name, (_, values))| {
                (
                    name,
                    values
                        .iter()
                        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let views = weights.iter().map(|(name, (shape, _))| {
            (
                name.as_str(),
                TensorView::new(Dtype::BF16, shape.clone(), &bytes[name]).unwrap(),
            )
        });
        std::fs::write(
            dir.join("model.safetensors"),
            serialize(views, None).unwrap(),
        )
        .unwrap();
        std::fs::write(dir.join("config.json"),serde_json::to_vec(&serde_json::json!({"model_type":"qwen3","hidden_size":D,"intermediate_size":H,"num_hidden_layers":LAYERS,"num_attention_heads":HEADS,"num_key_value_heads":KV,"head_dim":HD,"rms_norm_eps":EPS,"rope_theta":1000000.,"max_position_embeddings":32,"vocab_size":VOCAB,"eos_token_id":6,"tie_word_embeddings":true})).unwrap()).unwrap();
        Self { dir, weights }
    }
    fn w(&self, name: &str) -> &[f32] {
        &self.weights[name].1
    }
    fn reference(&self, tokens: &[usize]) -> Vec<f32> {
        let mut x = tokens
            .iter()
            .flat_map(|t| {
                self.w("model.embed_tokens.weight")[t * D..(t + 1) * D]
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        for layer in 0..LAYERS {
            let w = |tail: &str| self.w(&format!("model.layers.{layer}.{tail}"));
            let norm = normalize(&x, D, w("input_layernorm.weight"));
            let mut q = normalize(
                &linear(&norm, D, w("self_attn.q_proj.weight")),
                HD,
                w("self_attn.q_norm.weight"),
            );
            let mut k = normalize(
                &linear(&norm, D, w("self_attn.k_proj.weight")),
                HD,
                w("self_attn.k_norm.weight"),
            );
            let v = linear(&norm, D, w("self_attn.v_proj.weight"));
            rotate(&mut q, tokens.len(), HEADS);
            rotate(&mut k, tokens.len(), KV);
            let mut attended = vec![0.; tokens.len() * HEADS * HD];
            for t in 0..tokens.len() {
                for head in 0..HEADS {
                    let kvhead = head / (HEADS / KV);
                    let qi = (t * HEADS + head) * HD;
                    let scores = (0..=t)
                        .map(|j| {
                            (0..HD)
                                .map(|d| q[qi + d] * k[(j * KV + kvhead) * HD + d])
                                .sum::<f32>()
                                / (HD as f32).sqrt()
                        })
                        .collect::<Vec<_>>();
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let probs = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
                    let sum = probs.iter().sum::<f32>();
                    for d in 0..HD {
                        attended[qi + d] = (0..=t)
                            .map(|j| probs[j] / sum * v[(j * KV + kvhead) * HD + d])
                            .sum();
                    }
                }
            }
            let projected = linear(&attended, HEADS * HD, w("self_attn.o_proj.weight"));
            for (a, b) in x.iter_mut().zip(projected) {
                *a += b;
            }
            let norm = normalize(&x, D, w("post_attention_layernorm.weight"));
            let gate = linear(&norm, D, w("mlp.gate_proj.weight"));
            let up = linear(&norm, D, w("mlp.up_proj.weight"));
            let activated = gate
                .into_iter()
                .zip(up)
                .map(|(g, u)| g / (1. + (-g).exp()) * u)
                .collect::<Vec<_>>();
            let down = linear(&activated, H, w("mlp.down_proj.weight"));
            for (a, b) in x.iter_mut().zip(down) {
                *a += b;
            }
        }
        let last = normalize(&x[x.len() - D..], D, self.w("model.norm.weight"));
        linear(&last, D, self.w("model.embed_tokens.weight"))
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}
fn normalize(x: &[f32], width: usize, w: &[f32]) -> Vec<f32> {
    x.chunks_exact(width)
        .flat_map(|row| {
            let scale = (row.iter().map(|v| v * v).sum::<f32>() / width as f32 + EPS).sqrt();
            row.iter().zip(w).map(move |(x, w)| x / scale * w)
        })
        .collect()
}
fn linear(x: &[f32], width: usize, w: &[f32]) -> Vec<f32> {
    x.chunks_exact(width)
        .flat_map(|row| {
            w.chunks_exact(width)
                .map(move |column| row.iter().zip(column).map(|(a, b)| a * b).sum::<f32>())
        })
        .collect()
}
fn rotate(x: &mut [f32], tokens: usize, heads: usize) {
    for t in 0..tokens {
        for h in 0..heads {
            for d in 0..HD / 2 {
                let angle = t as f32 / 1000000_f32.powf((2 * d) as f32 / HD as f32);
                let (s, c) = angle.sin_cos();
                let i = (t * heads + h) * HD + d;
                let a = x[i];
                let b = x[i + HD / 2];
                x[i] = a * c - b * s;
                x[i + HD / 2] = b * c + a * s;
            }
        }
    }
}
fn assert_logits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }
}
#[test]
fn qwen3_bf16_tied_bindings_and_cpu_match_independent_reference() {
    let f = Fixture::new();
    let mut checkpoint = Checkpoint::load(&f.dir).unwrap();
    assert_eq!(checkpoint.model_type.as_deref(), Some("qwen3"));
    let tokens = [1, 3, 2];
    checkpoint.bind_tokens(&tokens).unwrap();
    let metadata = Checkpoint::metadata_context(&f.dir, tokens.len()).unwrap();
    assert_eq!(metadata.constants, checkpoint.context.constants);
    assert_eq!(metadata.tensors.len(), checkpoint.context.tensors.len());
    for (key, spec) in &metadata.tensors {
        let loaded = &checkpoint.context.tensors[key];
        assert_eq!(spec.slot, loaded.slot);
        assert_eq!(spec.dtype, loaded.dtype);
        assert_eq!(spec.shape, loaded.shape);
    }
    assert_eq!(
        metadata.tensors["lm_head.weight"].slot,
        metadata.tensors["model.embed_tokens.weight"].slot
    );
    assert_eq!(
        metadata.constants["n_positions"],
        puppygrad::compiler::pop::Scalar::Int(32)
    );
    assert_eq!(checkpoint.inputs.len(), f.weights.len()); // token slot replaces skipped head
    let program =
        source::parse_with_context(include_str!("../examples/qwen3.pup"), &checkpoint.context)
            .unwrap();
    let exe = cpu::compile(&program.graph, program.root, &f.dir.join("cpu")).unwrap();
    let actual = exe.run(&checkpoint.inputs).unwrap();
    assert_logits(actual[0].f32().unwrap(), &f.reference(&tokens));
}

#[test]
fn qwen3_provider_exposes_context_eos_and_generates_through_generic_ffi() {
    use puppygrad::runtime::llm_ffi::{Generation, Model};
    let f = Fixture::new();
    let config = serde_json::to_vec(&serde_json::json!({
        "source": PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/qwen3.pup"),
        "model_dir": f.dir, "device": "cpu", "threads": 1
    }))
    .unwrap();
    let mut model = unsafe { Model::from_api(puppygrad::models::pup_llm::API, &config) }.unwrap();
    assert_eq!(model.info.context_length, 32);
    assert_eq!(model.info.eos_token, 6);
    assert_eq!(model.info.vocab_size, VOCAB as u32);
    let expected = f.reference(&[1, 3, 2]);
    let choice = expected
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32;
    let output = model
        .infer(
            &[1, 3, 2],
            Generation {
                max_new_tokens: 1,
                temperature: 0.,
                seed: 42,
                reserved: 0,
            },
            None,
        )
        .unwrap();
    assert_eq!(output.tokens, [choice]);
    let mut reference_config: serde_json::Value = serde_json::from_slice(&config).unwrap();
    reference_config["verify_reference"] = serde_json::json!(true);
    let error = unsafe {
        Model::from_api(
            puppygrad::models::pup_llm::API,
            &serde_json::to_vec(&reference_config).unwrap(),
        )
    }
    .err()
    .unwrap();
    assert!(error.contains("GPT-2 only"), "{error}");
}
#[test]
#[ignore = "requires NVIDIA GPU and NVRTC; run explicitly with --ignored"]
fn qwen3_cuda_matches_independent_reference_across_prefixes() {
    qwen3_gpu_reference(gpu::Backend::Cuda);
}
fn qwen3_gpu_reference(backend: gpu::Backend) {
    let f = Fixture::new();
    let mut checkpoint = Checkpoint::load(&f.dir).unwrap();
    let runtime = gpu::Runtime::new(backend, 0).unwrap();
    for tokens in [&[1usize][..], &[1, 3, 2][..], &[1, 3, 2, 4][..]] {
        checkpoint.bind_tokens(tokens).unwrap();
        let p =
            source::parse_with_context(include_str!("../examples/qwen3.pup"), &checkpoint.context)
                .unwrap();
        let exe = gpu::compile_with_runtime(&p.graph, p.root, &f.dir.join(backend.tag()), &runtime)
            .unwrap();
        let actual = exe.run(&checkpoint.inputs).unwrap();
        assert_logits(actual[0].f32().unwrap(), &f.reference(tokens));
        let before = exe.residency_stats();
        exe.run(&checkpoint.inputs).unwrap();
        let after = exe.residency_stats();
        assert_eq!(before.allocations, after.allocations);
        assert_eq!(before.input_uploaded_bytes, after.input_uploaded_bytes);
    }
}

fn cached_chunks(f: &Fixture, backend: Option<gpu::Backend>) {
    use puppygrad::compiler::pop::Scalar;
    let mut checkpoint = Checkpoint::load(&f.dir).unwrap();
    checkpoint
        .context
        .constants
        .insert("buffer_capacity".into(), Scalar::Int(8));
    let cpu_runtime = cpu::Runtime::default();
    let gpu_runtime = backend.map(|backend| gpu::Runtime::new(backend, 0).unwrap());
    for chunks in [
        vec![vec![1, 3], vec![2], vec![4]],
        vec![vec![2], vec![1, 4], vec![3]],
    ] {
        cpu_runtime.reset_state().unwrap();
        if let Some(runtime) = &gpu_runtime {
            runtime.reset_state().unwrap();
        }
        let mut prefix = Vec::new();
        for chunk in chunks {
            prefix.extend_from_slice(&chunk);
            checkpoint.bind_tokens(&chunk).unwrap();
            let p = source::parse_with_context(
                include_str!("../examples/qwen3_cached.pup"),
                &checkpoint.context,
            )
            .unwrap();
            let actual = if let Some(runtime) = &gpu_runtime {
                gpu::compile_with_runtime(
                    &p.graph,
                    p.root,
                    &f.dir.join(runtime.backend().tag()),
                    runtime,
                )
                .unwrap()
                .run(&checkpoint.inputs)
                .unwrap()
            } else {
                let mut e = cpu::compile(&p.graph, p.root, &f.dir.join("cpu")).unwrap();
                e.share_runtime(&cpu_runtime);
                e.run(&checkpoint.inputs).unwrap()
            };
            assert_logits(actual[0].f32().unwrap(), &f.reference(&prefix));
        }
    }
}
#[test]
fn cached_qwen3_cpu_matches_full_prefix_with_chunks_and_reset() {
    cached_chunks(&Fixture::new(), None);
}
#[test]
#[ignore = "requires NVIDIA GPU and NVRTC"]
fn cached_qwen3_cuda_matches_full_prefix_with_chunks_and_reset() {
    cached_chunks(&Fixture::new(), Some(gpu::Backend::Cuda));
}
fn cached_provider(backend: Option<gpu::Backend>, request_sized: bool) {
    use puppygrad::runtime::llm_ffi::{Generation, Model};
    let f = Fixture::new();
    if backend.is_some() {
        let config_path = f.dir.join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
        config["max_position_embeddings"] = serde_json::json!(8192);
        std::fs::write(config_path, serde_json::to_vec(&config).unwrap()).unwrap();
    }
    let mut config = serde_json::json!({"source":PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/qwen3_cached.pup"),"model_dir":f.dir,"device":backend.map_or("cpu",gpu::Backend::tag),"threads":1,"prefill_chunk":2,"context_budget_mib":1,"context_reserve_mib":0});
    if request_sized {
        config["context_request"] = serde_json::json!({"capacity":16,"prompt_tokens":7});
        if backend.is_some() {
            config["max_memory"] = serde_json::json!(1024 * 1024);
        }
    }
    let encoded = serde_json::to_vec(&config).unwrap();
    let mut model = unsafe { Model::from_api(puppygrad::models::pup_llm::API, &encoded) }.unwrap();
    if request_sized {
        assert_eq!(model.info.context_length, 16);
    } else if backend.is_some() {
        assert!(model.info.context_length >= 8 && model.info.context_length < 8192);
        let context = Checkpoint::metadata_context(&f.dir, 1).unwrap();
        let expected = puppygrad::runtime::llm_capacity::determine_for_backend(
            backend.unwrap(),
            include_str!("../examples/qwen3_cached.pup"),
            &context,
            1024 * 1024,
            0,
            Some(2),
            false,
        )
        .unwrap();
        assert_eq!(
            model.info.context_length,
            expected.max_context_tokens as u64
        );
        let oversized = vec![1; model.info.context_length as usize + 1];
        let error = model
            .infer(
                &oversized,
                Generation {
                    max_new_tokens: 1,
                    temperature: 0.,
                    seed: 42,
                    reserved: 0,
                },
                None,
            )
            .unwrap_err();
        assert!(error.contains("context"), "{error}");
    }
    for prompt in [
        vec![1usize, 3, 2, 4, 1, 2, 3],
        vec![4, 1],
        vec![1, 3, 2, 4, 1, 2, 3],
    ] {
        let mut history = prompt.clone();
        let mut expected = vec![];
        for _ in 0..4 {
            let row = f.reference(&history);
            let next = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            expected.push(next as u32);
            history.push(next);
            if next == 6 {
                break;
            }
        }
        let mut streamed = vec![];
        let mut callback = |tokens: &[u32]| {
            streamed.extend_from_slice(tokens);
            Ok(())
        };
        let output = model
            .infer(
                &prompt.iter().map(|&t| t as u32).collect::<Vec<_>>(),
                Generation {
                    max_new_tokens: 4,
                    temperature: 0.,
                    seed: 42,
                    reserved: 0,
                },
                Some(&mut callback),
            )
            .unwrap();
        assert_eq!(output.tokens, expected);
        assert_eq!(streamed, expected);
    }
    if request_sized && backend.is_some() {
        drop(model);
        config["max_memory"] = serde_json::json!(1);
        let encoded = serde_json::to_vec(&config).unwrap();
        let error = unsafe { Model::from_api(puppygrad::models::pup_llm::API, &encoded) }
            .err()
            .unwrap();
        assert!(
            error.contains("requested context") && error.contains("budget is 1"),
            "{error}"
        );
    }
}

#[test]
fn cached_qwen3_uses_chunked_ffi_and_resets_between_inferences() {
    cached_provider(None, false);
}
#[test]
fn cached_qwen3_request_context_preserves_sampling_streaming_and_reset() {
    cached_provider(None, true);
}
#[test]
fn requested_context_validates_lengths_and_keeps_model_position_limit() {
    use puppygrad::runtime::llm_ffi::Model;
    let f = Fixture::new();
    for (capacity, prompt_tokens, minimum_capacity, valid) in [
        (0, 1, None, false),
        (16, 0, None, false),
        (16, 17, None, false),
        (64, 33, None, false),
        (64, 7, None, true),
        (64, 7, Some(33), false),
        (64, 7, Some(6), false),
        (16, 7, Some(17), false),
        (64, 7, Some(32), true),
    ] {
        let config = serde_json::to_vec(&serde_json::json!({
            "source":PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/qwen3_cached.pup"),
            "model_dir":f.dir,"device":"cpu",
            "context_request":{"capacity":capacity,"prompt_tokens":prompt_tokens,"minimum_capacity":minimum_capacity}
        }))
        .unwrap();
        let result = unsafe { Model::from_api(puppygrad::models::pup_llm::API, &config) };
        assert_eq!(
            result.is_ok(),
            valid,
            "capacity={capacity}, prompt={prompt_tokens}"
        );
        if let Ok(model) = result {
            assert_eq!(model.info.context_length, 32);
        }
    }
}
#[test]
#[ignore = "requires NVIDIA GPU and NVRTC"]
fn cached_qwen3_cuda_request_context_reuses_planned_prefill_tail_and_decode() {
    cached_provider(Some(gpu::Backend::Cuda), true);
}

#[test]
#[ignore = "requires NVIDIA GPU and NVRTC; uses a tiny checkpoint and bounded device buffers"]
fn cached_qwen3_cuda_shrinks_preferred_context_to_request_under_memory_pressure() {
    use puppygrad::runtime::{
        llm_capacity,
        llm_ffi::{Generation, Model},
    };
    let f = Fixture::new();
    let text = include_str!("../examples/qwen3_cached.pup");
    let context = Checkpoint::metadata_context(&f.dir, 1).unwrap();
    let budget = llm_capacity::request_plan(text, &context, 10, 7, Some(2), true)
        .unwrap()
        .total_bytes();
    let larger = llm_capacity::request_plan(text, &context, 32, 7, Some(2), true)
        .unwrap()
        .total_bytes();
    assert!(larger > budget && budget < 1024 * 1024);
    let config = serde_json::to_vec(&serde_json::json!({
        "source":PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/qwen3_cached.pup"),
        "model_dir":f.dir, "device":"cuda:0",
        "context_reserve_mib":0, "max_memory":budget,
        "context_request":{"capacity":32,"prompt_tokens":7,"minimum_capacity":10}
    }))
    .unwrap();
    let mut model = unsafe { Model::from_api(puppygrad::models::pup_llm::API, &config) }.unwrap();
    assert_eq!(model.info.context_length, 10);
    let prompt = [0, 1, 2, 3, 4, 5, 0];
    let mut history = prompt.iter().map(|&t| t as usize).collect::<Vec<_>>();
    let mut expected = vec![];
    for _ in 0..4 {
        let logits = f.reference(&history);
        let token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        expected.push(token);
        history.push(token as usize);
        if token == model.info.eos_token {
            break;
        }
    }
    for _ in 0..2 {
        let output = model
            .infer(
                &prompt,
                Generation {
                    max_new_tokens: 4,
                    temperature: 0.,
                    seed: 42,
                    reserved: 0,
                },
                None,
            )
            .unwrap();
        assert_eq!(output.tokens, expected);
    }
}
#[test]
#[ignore = "requires NVIDIA GPU and NVRTC"]
fn cached_qwen3_ffi_automatically_limits_context_and_chunks_prefill() {
    cached_provider(Some(gpu::Backend::Cuda), false);
}

#[test]
#[ignore = "requires AMD GPU and HIPRTC"]
fn qwen3_hip_matches_independent_reference_across_prefixes() {
    qwen3_gpu_reference(gpu::Backend::Hip);
}
#[test]
#[ignore = "requires AMD GPU and HIPRTC"]
fn cached_qwen3_hip_matches_full_prefix_with_chunks_and_reset() {
    cached_chunks(&Fixture::new(), Some(gpu::Backend::Hip));
}
#[test]
#[ignore = "requires AMD GPU and HIPRTC"]
fn cached_qwen3_hip_ffi_limits_context_chunks_prefill_and_resets() {
    cached_provider(Some(gpu::Backend::Hip), false);
}

#[test]
fn checkpoint_loader_rejects_truncated_payload_and_invalid_offsets() {
    let f = Fixture::new();
    let path = f.dir.join("model.safetensors");
    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
    assert!(Checkpoint::load(&f.dir)
        .err()
        .unwrap()
        .to_string()
        .contains("payload size"));
    assert!(Checkpoint::metadata_context(&f.dir, 1).is_err());
    let header_size = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_size]).unwrap();
    header["model.norm.weight"]["data_offsets"] = serde_json::json!([0, 1]);
    let new_header = serde_json::to_vec(&header).unwrap();
    let mut bad = (new_header.len() as u64).to_le_bytes().to_vec();
    bad.extend(new_header);
    bad.extend(&bytes[8 + header_size..]);
    std::fs::write(&path, bad).unwrap();
    assert!(Checkpoint::load(&f.dir).is_err());
    assert!(Checkpoint::metadata_context(&f.dir, 1).is_err());
}
#[test]
#[ignore = "requires HIPRTC, but no GPU"]
fn hiprtc_compiles_qwen3_full_cached_prefill_and_decode() {
    let f = Fixture::new();
    let mut checkpoint = Checkpoint::load(&f.dir).unwrap();
    for template in [
        include_str!("../examples/qwen3.pup"),
        include_str!("../examples/qwen3_cached.pup"),
    ] {
        for tokens in [&[1usize][..], &[1, 3, 2][..]] {
            checkpoint.bind_tokens(tokens).unwrap();
            let p = source::parse_with_context(template, &checkpoint.context).unwrap();
            let (code, _) = hip::emit(&p.graph, p.root).unwrap();
            for arch in ["gfx1100", "gfx90a"] {
                assert!(hip::compile_source(&code, arch)
                    .unwrap()
                    .starts_with(b"\x7fELF"));
            }
        }
    }
}

fn growing_cached_provider(backend: Option<gpu::Backend>, stop_for_memory: bool) {
    use puppygrad::runtime::llm_ffi::{Generation, Model, DONE_CONTEXT, DONE_LIMIT, DONE_MEMORY};
    let f = Fixture::new();
    // Disable EOS to exercise multiple capacity boundaries with deterministic sampling.
    let path = f.dir.join("config.json");
    let mut checkpoint: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    checkpoint.as_object_mut().unwrap().remove("eos_token_id");
    std::fs::write(&path, serde_json::to_vec(&checkpoint).unwrap()).unwrap();
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/qwen3_cached.pup");
    let mut config = serde_json::json!({"source":source,"model_dir":f.dir,"device":backend.map_or("cpu",gpu::Backend::tag),"threads":1,"prefill_chunk":1,"context_reserve_mib":0,"grow_context":true,"context_request":{"capacity":4,"prompt_tokens":1,"minimum_capacity":1},"cache_dir":f.dir.join("compiled")});
    if let Some(backend) = backend {
        let context = Checkpoint::metadata_context(&f.dir, 1).unwrap();
        let initial = puppygrad::runtime::llm_capacity::request_plan_for_backend(
            backend,
            include_str!("../examples/qwen3_cached.pup"),
            &context,
            4,
            1,
            Some(1),
            true,
        )
        .unwrap();
        let limit = if stop_for_memory {
            initial.total_bytes()
        } else {
            1024 * 1024
        };
        assert!(limit <= 1024 * 1024);
        config["max_memory"] = serde_json::json!(limit);
    }
    let mut model = unsafe {
        Model::from_api(
            puppygrad::models::pup_llm::API,
            &serde_json::to_vec(&config).unwrap(),
        )
    }
    .unwrap();
    assert_eq!(
        model.info.context_length, 32,
        "logical context must not be limited to the initial bucket"
    );
    let settings = Generation {
        max_new_tokens: 12,
        temperature: 0.,
        reserved: 0,
        seed: 42,
    };
    let mut expected = vec![];
    let mut history = vec![1];
    for _ in 0..12 {
        let row = f.reference(&history);
        let next = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        history.push(next);
        expected.push(next as u32);
    }
    let output = model.infer(&[1], settings, None).unwrap();
    if stop_for_memory {
        assert_eq!(output.reason, DONE_MEMORY);
        assert_eq!(output.tokens, expected[..4]);
        return;
    }
    assert_eq!(output.reason, DONE_LIMIT);
    assert_eq!(
        output.tokens, expected,
        "KV data must survive capacity growth"
    );
    let again = model.infer(&[1], settings, None).unwrap();
    assert_eq!(
        again.tokens, expected,
        "later requests reset state without losing the grown capacity"
    );
    // Host callback failures must stop the provider itself, rather than merely hiding output.
    let mut count = 0;
    let mut cancel = |ids: &[u32]| {
        count += ids.len();
        if count == 3 {
            Err("stop now".into())
        } else {
            Ok(())
        }
    };
    assert_eq!(
        model.infer(&[1], settings, Some(&mut cancel)).unwrap_err(),
        "stop now"
    );
    assert_eq!(count, 3);
    let full = model
        .infer(
            &[1],
            Generation {
                max_new_tokens: 32,
                ..settings
            },
            None,
        )
        .unwrap();
    assert_eq!(full.tokens.len(), 32);
    assert_eq!(full.reason, DONE_CONTEXT);
}

#[test]
fn cached_qwen3_cpu_grows_context_preserves_kv_and_stops_on_callback_failure() {
    growing_cached_provider(None, false);
}
#[test]
#[ignore = "requires CUDA; synthetic model buffers capped at 1 MiB"]
fn cached_qwen3_cuda_grows_context_preserves_kv_and_stops_on_callback_failure() {
    growing_cached_provider(Some(gpu::Backend::Cuda), false);
}
#[test]
#[ignore = "requires CUDA; synthetic model buffers capped below 1 MiB"]
fn cached_qwen3_cuda_returns_partial_output_when_context_cannot_grow() {
    growing_cached_provider(Some(gpu::Backend::Cuda), true);
}

fn reusable_prefill_provider(backend: Option<gpu::Backend>, tight_memory: bool) {
    use puppygrad::runtime::llm_ffi::{Generation, Model};
    let f = Fixture::new();
    let path = f.dir.join("config.json");
    let mut checkpoint: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    checkpoint.as_object_mut().unwrap().remove("eos_token_id");
    std::fs::write(&path, serde_json::to_vec(&checkpoint).unwrap()).unwrap();
    let mut config = serde_json::json!({"source":PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/qwen3_cached.pup"),"model_dir":f.dir,"device":backend.map_or("cpu",gpu::Backend::tag),"threads":1,"context_reserve_mib":0,"grow_context":true,"context_request":{"capacity":32,"prompt_tokens":1,"minimum_capacity":1},"cache_dir":f.dir.join("compiled")});
    if backend.is_some() {
        let limit = if tight_memory {
            let context = Checkpoint::metadata_context(&f.dir, 1).unwrap();
            puppygrad::runtime::llm_capacity::request_plan_for_backend(
                backend.unwrap(),
                include_str!("../examples/qwen3_cached.pup"),
                &context,
                32,
                1,
                Some(1),
                true,
            )
            .unwrap()
            .total_bytes()
        } else {
            1024 * 1024
        };
        assert!(limit <= 1024 * 1024);
        config["max_memory"] = serde_json::json!(limit);
    }
    let settings = Generation {
        max_new_tokens: 3,
        temperature: 0.,
        reserved: 0,
        seed: 42,
    };
    let mut exact_elapsed = None;
    if backend.is_some() && !tight_memory {
        let mut exact = config.clone();
        exact["grow_context"] = serde_json::json!(false);
        let mut baseline = unsafe {
            Model::from_api(
                puppygrad::models::pup_llm::API,
                &serde_json::to_vec(&exact).unwrap(),
            )
        }
        .unwrap();
        for length in [1, 8] {
            baseline.infer(&vec![1; length], settings, None).unwrap();
        }
        let prompt = (0..17)
            .map(|i| ((i * 3 + 1) % VOCAB) as u32)
            .collect::<Vec<_>>();
        let started = std::time::Instant::now();
        baseline.infer(&prompt, settings, None).unwrap();
        exact_elapsed = Some(started.elapsed());
        drop(baseline);
    }
    let mut model = unsafe {
        Model::from_api(
            puppygrad::models::pup_llm::API,
            &serde_json::to_vec(&config).unwrap(),
        )
    }
    .unwrap();
    for length in [1, 8, 13, 16, 17, 21] {
        let prompt = (0..length).map(|i| (i * 3 + 1) % VOCAB).collect::<Vec<_>>();
        let mut history = prompt.clone();
        let mut expected = vec![];
        for _ in 0..3 {
            let row = f.reference(&history);
            let next = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            history.push(next);
            expected.push(next as u32);
        }
        let input = prompt.iter().map(|&id| id as u32).collect::<Vec<_>>();
        let mut streamed = vec![];
        let mut callback = |ids: &[u32]| {
            streamed.extend_from_slice(ids);
            Ok(())
        };
        let started = std::time::Instant::now();
        let output = model
            .infer(
                &input,
                Generation {
                    max_new_tokens: 3,
                    temperature: 0.,
                    reserved: 0,
                    seed: 42,
                },
                Some(&mut callback),
            )
            .unwrap();
        let elapsed = started.elapsed();
        if length == 17 {
            if let Some(exact) = exact_elapsed {
                eprintln!("synthetic 17-token prompt after warmup: exact-shape {:.3} ms, reusable-shapes {:.3} ms", exact.as_secs_f64()*1000., elapsed.as_secs_f64()*1000.);
            }
        }
        assert_eq!(output.tokens, expected, "prompt length {length}");
        assert_eq!(streamed, expected);
    }
}
#[test]
fn cached_qwen3_cpu_reusable_prefill_preserves_positions_logits_and_streaming() {
    reusable_prefill_provider(None, false);
}
#[test]
#[ignore = "requires CUDA; synthetic model buffers capped at 1 MiB"]
fn cached_qwen3_cuda_reusable_prefill_preserves_positions_logits_and_streaming() {
    reusable_prefill_provider(Some(gpu::Backend::Cuda), false);
}

#[test]
#[ignore = "requires CUDA; synthetic model buffers capped below 1 MiB"]
fn cached_qwen3_cuda_reusable_prefill_falls_back_to_decode_under_memory_pressure() {
    reusable_prefill_provider(Some(gpu::Backend::Cuda), true);
}

#[test]
#[ignore = "requires HIP; synthetic model buffers capped at 1 MiB"]
fn cached_qwen3_hip_reusable_prefill_preserves_positions_logits_and_streaming() {
    reusable_prefill_provider(Some(gpu::Backend::Hip), false);
}

#[test]
#[ignore = "requires HIP; synthetic model buffers capped below 1 MiB"]
fn cached_qwen3_hip_reusable_prefill_falls_back_to_decode_under_memory_pressure() {
    reusable_prefill_provider(Some(gpu::Backend::Hip), true);
}
