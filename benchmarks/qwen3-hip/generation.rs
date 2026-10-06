//! Persistent worker measuring the production LLM FFI, including host sampling.
use puppygrad::runtime::llm_ffi::{Generation, Model};
use std::io::{self, BufRead, Write};
use std::time::Instant;

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let config = serde_json::to_vec(&serde_json::json!({
        "model_dir": args[1], "source": args[2], "device": "hip:0"
    }))
    .unwrap();
    let mut model = unsafe { Model::from_api(puppygrad::models::pup_llm::API, &config) }.unwrap();
    let tokenizer =
        tokenizers::Tokenizer::from_file(format!("{}/tokenizer.json", args[1])).unwrap();
    for line in io::stdin().lock().lines() {
        let command: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        let result = if command["info"].as_bool() == Some(true) {
            serde_json::json!({"context_length":model.info.context_length,"vocab_size":model.info.vocab_size,"eos_token":model.info.eos_token})
        } else if let Some(prompt) = command["prompt"].as_str() {
            let formatted = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
            serde_json::json!({"tokens":tokenizer.encode(formatted,true).unwrap().get_ids()})
        } else {
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
            match model.infer(
                &tokens,
                Generation {
                    max_new_tokens: command["count"].as_u64().unwrap_or(32),
                    temperature: 0.,
                    seed: 42,
                    reserved: 0,
                },
                Some(&mut on_tokens),
            ) {
                Ok(output) => {
                    serde_json::json!({"elapsed_ms":started.elapsed().as_secs_f64()*1000.,"token_ms":latencies,"tokens":output.tokens})
                }
                Err(error) => serde_json::json!({"error":error.to_string()}),
            }
        };
        println!("{result}");
        io::stdout().flush().unwrap();
    }
}
