//! LLM application runtime: CLI configuration, tokenization, and text display.
//! Model construction, generation, sampling, and resources live behind the FFI API.
use super::llm_ffi::{Generation, Model, Result};
use std::{
    io::{self, Write},
    path::{Path, PathBuf},
    time::Instant,
};
#[derive(Debug)]
pub struct Options {
    pub program: PathBuf,
    pub model_dir: PathBuf,
    pub device: String,
    pub threads: Option<usize>,
    pub cpu_target: crate::compiler::cpu::CpuTarget,
    pub prompt: String,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub seed: u64,
    pub stream: bool,
    pub verify_reference: bool,
}
pub fn run(options: Options) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let generation = Generation {
        max_new_tokens: options.max_new_tokens as u64,
        temperature: options.temperature,
        reserved: 0,
        seed: options.seed,
    };
    generation.validate()?;
    let mut model = load_model(
        &options.program,
        &options.model_dir,
        &options.device,
        options.threads,
        options.verify_reference,
        options.cpu_target,
    )?;
    let (tokenizer, input) = encode_prompt(&model, &options.model_dir, &options.prompt)?;
    let started = Instant::now();
    let stdout = io::stdout();
    let mut writer = stdout.lock();
    let echo_prompt = format_prompt(&options.model_dir, &options.prompt)? == options.prompt;
    let output = generate_displayed(
        &mut model,
        &tokenizer,
        &input,
        generation,
        options.stream,
        echo_prompt,
        &mut writer,
    )?;
    writeln!(writer)?;
    writer.flush()?;
    eprintln!(
        "generated {} tokens in {:.3}s; LLM ABI v1, {}",
        output.len(),
        started.elapsed().as_secs_f64(),
        if options.stream {
            "streaming callbacks"
        } else {
            "buffered output"
        }
    );
    Ok(())
}
/// Shared output path: streaming emits decoded increments; buffered execution
/// registers no on_tokens callback and queries retained output after completion.
pub fn generate(
    model: &mut Model,
    tokenizer: &tokenizers::Tokenizer,
    input: &[u32],
    generation: Generation,
    stream: bool,
    writer: &mut dyn Write,
) -> Result<Vec<u32>> {
    generate_displayed(model, tokenizer, input, generation, stream, true, writer)
}
fn generate_displayed(
    model: &mut Model,
    tokenizer: &tokenizers::Tokenizer,
    input: &[u32],
    generation: Generation,
    stream: bool,
    echo_prompt: bool,
    writer: &mut dyn Write,
) -> Result<Vec<u32>> {
    let mut decoded = String::new();
    let mut decoder = tokenizer.decode_stream(true);
    if stream && echo_prompt {
        for &token in input {
            if let Some(chunk) = decoder.step(token).map_err(|e| e.to_string())? {
                writer
                    .write_all(chunk.as_bytes())
                    .map_err(|e| e.to_string())?;
                decoded.push_str(&chunk);
            }
        }
        writer.flush().map_err(|e| e.to_string())?;
    }
    let output = if stream {
        let mut on_tokens = |tokens: &[u32]| -> Result<()> {
            for &token in tokens {
                if tokenizer.id_to_token(token).is_none() {
                    return Err("model generated an ID absent from the tokenizer".into());
                }
                if let Some(chunk) = decoder.step(token).map_err(|e| e.to_string())? {
                    writer
                        .write_all(chunk.as_bytes())
                        .map_err(|e| e.to_string())?;
                    decoded.push_str(&chunk);
                }
            }
            writer.flush().map_err(|e| e.to_string())
        };
        model.infer(input, generation, Some(&mut on_tokens))?
    } else {
        model.infer(input, generation, None)?
    };
    if output
        .tokens
        .iter()
        .any(|&id| tokenizer.id_to_token(id).is_none())
    {
        return Err("model generated an ID absent from the tokenizer".into());
    }
    let mut all = if echo_prompt { input.to_vec() } else { vec![] };
    all.extend_from_slice(&output.tokens);
    let complete = tokenizer.decode(&all, true).map_err(|e| e.to_string())?;
    // Flush a final incomplete byte sequence as the tokenizer would in buffered mode.
    let tail = complete
        .strip_prefix(&decoded)
        .ok_or("streaming decoder changed already emitted text")?;
    writer
        .write_all(tail.as_bytes())
        .map_err(|e| e.to_string())?;
    writer.flush().map_err(|e| e.to_string())?;
    Ok(output.tokens)
}

/// Open either a compiled provider or the built-in .pup adapter.
pub(super) fn load_model(
    program: &Path,
    model_dir: &Path,
    device: &str,
    threads: Option<usize>,
    verify_reference: bool,
    cpu_target: crate::compiler::cpu::CpuTarget,
) -> std::result::Result<Model, Box<dyn std::error::Error>> {
    if threads == Some(0) {
        return Err("threads must be greater than zero".into());
    }
    let config = serde_json::to_vec(
        &serde_json::json!({"source":program,"model_dir":model_dir,"device":device,"verify_reference":verify_reference,"threads":threads,"cpu_target":cpu_target}),
    )?;
    let model = if program.extension().is_some_and(|e| e == "pup") {
        unsafe { Model::from_api(crate::models::pup_llm::API, &config) }?
    } else {
        if cpu_target != crate::compiler::cpu::CpuTarget::Generic {
            return Err(
                "--cpu-target applies to .pup compilation; shared libraries are already compiled"
                    .into(),
            );
        }
        if verify_reference {
            return Err(
                "--verify-reference is currently supported by the .pup GPT-2 adapter only".into(),
            );
        }
        let library = program.canonicalize()?;
        unsafe { Model::load(&library, &config) }?
    };
    Ok(model)
}

pub(super) fn encode_prompt(
    model: &Model,
    model_dir: &Path,
    prompt: &str,
) -> std::result::Result<(tokenizers::Tokenizer, Vec<u32>), Box<dyn std::error::Error>> {
    let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| e.to_string())?;
    let formatted = format_prompt(model_dir, prompt)?;
    let input = tokenizer
        .encode(formatted, true)
        .map_err(|e| e.to_string())?
        .get_ids()
        .to_vec();
    if tokenizer
        .get_vocab(true)
        .values()
        .any(|&id| id >= model.info.vocab_size)
    {
        return Err("tokenizer contains IDs outside the model vocabulary".into());
    }
    Ok((tokenizer, input))
}

/// The CLI currently accepts one user message. Qwen3 uses its documented
/// non-thinking generation prefix; the model ABI continues to receive token IDs.
fn format_prompt(
    model_dir: &Path,
    prompt: &str,
) -> std::result::Result<String, Box<dyn std::error::Error>> {
    let config = model_dir.join("config.json");
    if config.is_file() {
        let config: serde_json::Value = serde_json::from_slice(&std::fs::read(config)?)?;
        if config.get("model_type").and_then(|v| v.as_str()) == Some("qwen3") {
            return Ok(format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        }
    }
    Ok(prompt.to_owned())
}

#[cfg(test)]
mod prompt_tests {
    use super::*;
    #[test]
    fn qwen3_non_thinking_template_matches_official_single_user_prefix() {
        let dir =
            std::env::temp_dir().join(format!("puppygrad-qwen3-template-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), r#"{"model_type":"qwen3"}"#).unwrap();
        assert_eq!(
            format_prompt(&dir, "Hello").unwrap(),
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        std::fs::write(dir.join("config.json"), r#"{"model_type":"gpt2"}"#).unwrap();
        assert_eq!(format_prompt(&dir, "Hello").unwrap(), "Hello");
        std::fs::remove_file(dir.join("config.json")).unwrap();
        assert_eq!(format_prompt(&dir, "Hello").unwrap(), "Hello");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
