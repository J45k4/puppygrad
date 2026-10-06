use puppygrad::{
    compiler::{cuda, pop::Scalar, source},
    models::pup_llm::Checkpoint,
};
use std::{
    collections::HashMap,
    io::{self, BufRead, Write},
    path::Path,
    time::Instant,
};
fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let model_dir = Path::new(
        args.get(1)
            .map(String::as_str)
            .unwrap_or("models/qwen3-0.6b"),
    );
    let source_path = Path::new(
        args.get(2)
            .map(String::as_str)
            .unwrap_or("examples/qwen3.pup"),
    );
    let cache_dir = Path::new(args.get(3).map(String::as_str).unwrap_or(".cache/pup/cuda"));
    if args.get(4).is_some_and(|mode| mode == "ffi") {
        run_ffi(model_dir, source_path);
        return;
    }
    if args.get(4).is_some_and(|mode| mode == "plan") {
        let capacity: usize = args[5].parse().unwrap();
        let sequence_length: usize = args[6].parse().unwrap();
        let started = Instant::now();
        let result = (|| -> Result<_, Box<dyn std::error::Error>> {
            let mut context = Checkpoint::metadata_context(model_dir, sequence_length)?;
            context
                .constants
                .insert("buffer_capacity".into(), Scalar::Int(capacity as i64));
            let text = std::fs::read_to_string(source_path)?;
            let p = source::parse_with_context(&text, &context)?;
            let (code, gemms) = cuda::emit(&p.graph, p.root)?;
            Ok((code.matches("__global__ void kernel").count(), gemms))
        })();
        let output = match result {
            Ok((kernels, gemms)) => serde_json::json!({"kernel_count":kernels,"gemm_count":gemms}),
            Err(e) => serde_json::json!({"error":e.to_string()}),
        };
        println!(
            "{}",
            serde_json::json!({"plan_ms":started.elapsed().as_secs_f64()*1000.,"result":output})
        );
        return;
    }
    let mut checkpoint = Checkpoint::load(model_dir).unwrap();
    if let Some(capacity) = args.get(4) {
        let capacity: usize = capacity
            .parse()
            .expect("expected positive KV capacity or ffi/plan");
        assert!(capacity > 0);
        let limit = match checkpoint.context.constants["n_positions"] {
            Scalar::Int(n) => n as usize,
            _ => unreachable!(),
        };
        assert!(
            capacity <= limit,
            "KV capacity exceeds the checkpoint context limit"
        );
        checkpoint
            .context
            .constants
            .insert("buffer_capacity".into(), Scalar::Int(capacity as i64));
    }
    let source = std::fs::read_to_string(source_path).unwrap();
    let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let mut executables = HashMap::new();
    let runtime = cuda::Runtime::new(0).unwrap();
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let command: serde_json::Value = serde_json::from_str(&line).unwrap();
        if let Some(prompt) = command["prompt"].as_str() {
            let formatted=format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
            println!(
                "{}",
                serde_json::json!({"tokens":tokenizer.encode(formatted,true).unwrap().get_ids()})
            );
            io::stdout().flush().unwrap();
            continue;
        }
        if let Some(ids) = command["decode"].as_array() {
            let ids = ids
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect::<Vec<_>>();
            println!(
                "{}",
                serde_json::json!({"text":tokenizer.decode(&ids,true).unwrap()})
            );
            io::stdout().flush().unwrap();
            continue;
        }
        let tokens = command["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        if !command["retain"].as_bool().unwrap_or(false) {
            runtime.reset_state().unwrap();
        }
        checkpoint.bind_tokens(&tokens).unwrap();
        let n = tokens.len();
        let prepare_started = Instant::now();
        let mut compile_ms = 0.;
        if !executables.contains_key(&n) {
            let p = source::parse_with_context(&source, &checkpoint.context).unwrap();
            executables.insert(
                n,
                cuda::compile_with_runtime(&p.graph, p.root, cache_dir, &runtime).unwrap(),
            );
            compile_ms = prepare_started.elapsed().as_secs_f64() * 1000.;
        }
        let exe = &executables[&n];
        if let Some(enabled) = command["graph_replay"].as_bool() {
            runtime.set_graph_replay(enabled);
        }
        let before = exe.residency_stats();
        let execution_before = exe.execution_stats();
        let started = Instant::now();
        let output = exe.run(&checkpoint.inputs).unwrap();
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.;
        let values = output[0].f32().unwrap();
        let argmax = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        if let Some(path) = command["save"].as_str() {
            std::fs::write(
                path,
                values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        }
        let stats = exe.residency_stats();
        let execution = exe.execution_stats();
        let profile = serde_json::json!({"run_graph_builds":execution.graph_builds-execution_before.graph_builds,"run_graph_launches":execution.graph_launches-execution_before.graph_launches,"run_direct_kernel_launches":execution.direct_kernel_launches-execution_before.direct_kernel_launches,"row_fusion_count":exe.row_fusion_count(),"parallel_reduction_count":exe.parallel_reduction_count(),"kernel_count":exe.kernel_count(),"run_allocations":stats.allocations-before.allocations,"run_input_uploads":stats.input_uploads-before.input_uploads,"run_input_uploaded_bytes":stats.input_uploaded_bytes-before.input_uploaded_bytes,"allocations":stats.allocations,"input_uploads":stats.input_uploads,"input_uploaded_bytes":stats.input_uploaded_bytes,"resident_bytes":stats.resident_bytes,"input_bytes":stats.input_bytes,"state_bytes":stats.state_bytes,"arena_bytes":stats.arena_bytes});
        println!(
            "{}",
            serde_json::json!({"elapsed_ms":elapsed_ms,"compile_ms":compile_ms,"argmax":argmax,"cache_hit":exe.cache_hit,"workspace_bytes":exe.workspace_bytes(),"source_path":exe.source_path,"profile":profile})
        );
        io::stdout().flush().unwrap();
    }
}

fn run_ffi(model_dir: &Path, source_path: &Path) {
    use puppygrad::runtime::llm_ffi::{Generation, Model};
    let config = serde_json::to_vec(&serde_json::json!({
        "model_dir": model_dir, "source": source_path, "device": "cuda:0"
    }))
    .unwrap();
    let mut model = unsafe { Model::from_api(puppygrad::models::pup_llm::API, &config) }.unwrap();
    for line in io::stdin().lock().lines() {
        let command: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        if command["info"].as_bool() == Some(true) {
            println!("{}", serde_json::json!({"context_length":model.info.context_length,"vocab_size":model.info.vocab_size,"eos_token":model.info.eos_token}));
            io::stdout().flush().unwrap();
            continue;
        }
        let tokens = command["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let started = Instant::now();
        let mut previous = 0.;
        let mut latencies = Vec::new();
        let mut on_tokens = |tokens: &[u32]| {
            assert_eq!(tokens.len(), 1);
            let elapsed = started.elapsed().as_secs_f64() * 1000.;
            latencies.push(elapsed - previous);
            previous = elapsed;
            Ok(())
        };
        let output = model
            .infer(
                &tokens,
                Generation {
                    max_new_tokens: command["count"].as_u64().unwrap_or(8),
                    temperature: 0.,
                    seed: 42,
                    reserved: 0,
                },
                Some(&mut on_tokens),
            )
            .unwrap();
        println!(
            "{}",
            serde_json::json!({"elapsed_ms":started.elapsed().as_secs_f64()*1000.,"token_ms":latencies,"tokens":output.tokens})
        );
        io::stdout().flush().unwrap();
    }
}
