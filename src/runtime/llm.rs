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
    pub max_memory: Option<super::memory_limit::MemoryLimit>,
}
pub fn run(options: Options) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let generation = Generation {
        max_new_tokens: options.max_new_tokens as u64,
        temperature: options.temperature,
        reserved: 0,
        seed: options.seed,
    };
    generation.validate()?;
    let mut model = load_model_with_policy(
        &options.program,
        &options.model_dir,
        &options.device,
        options.threads,
        options.verify_reference,
        options.cpu_target,
        LoadPolicy {
            max_memory: options.max_memory.map(|m| m.0),
            ..Default::default()
        },
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
pub(super) fn generate_displayed(
    model: &mut Model,
    tokenizer: &tokenizers::Tokenizer,
    input: &[u32],
    generation: Generation,
    stream: bool,
    echo_prompt: bool,
    writer: &mut dyn Write,
) -> Result<Vec<u32>> {
    Ok(generate_output_displayed(
        model,
        tokenizer,
        input,
        generation,
        stream,
        echo_prompt,
        writer,
    )?
    .tokens)
}
pub(super) fn generate_output_displayed(
    model: &mut Model,
    tokenizer: &tokenizers::Tokenizer,
    input: &[u32],
    generation: Generation,
    stream: bool,
    echo_prompt: bool,
    writer: &mut dyn Write,
) -> Result<super::llm_ffi::Output> {
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
    Ok(output)
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
    load_model_with_policy(
        program,
        model_dir,
        device,
        threads,
        verify_reference,
        cpu_target,
        LoadPolicy::default(),
    )
}

#[derive(Default)]
pub(super) struct LoadPolicy<'a> {
    pub grow_context: bool,
    pub cache_dir: Option<&'a Path>,
    pub context_request: Option<super::llm_capacity::ContextRequest>,
    pub max_memory: Option<usize>,
}

pub(super) fn load_model_with_policy(
    program: &Path,
    model_dir: &Path,
    device: &str,
    threads: Option<usize>,
    verify_reference: bool,
    cpu_target: crate::compiler::cpu::CpuTarget,
    policy: LoadPolicy<'_>,
) -> std::result::Result<Model, Box<dyn std::error::Error>> {
    if threads == Some(0) {
        return Err("threads must be greater than zero".into());
    }
    let mut config = serde_json::json!({"source":program,"model_dir":model_dir,"device":device,"verify_reference":verify_reference,"threads":threads,"cpu_target":cpu_target});
    if policy.max_memory.is_some() && !program.extension().is_some_and(|e| e == "pup") {
        return Err("--max-memory currently requires a GPU .pup provider".into());
    }
    config["grow_context"] = serde_json::json!(policy.grow_context);
    if let Some(limit) = policy.max_memory {
        config["max_memory"] = serde_json::to_value(limit)?;
    }
    if let Some(cache) = policy.cache_dir {
        config["cache_dir"] = serde_json::to_value(cache)?;
    }
    if let Some(request) = policy.context_request {
        config["context_request"] = serde_json::to_value(request)?;
    }
    let config = serde_json::to_vec(&config)?;
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
    let (tokenizer, input) = tokenize_prompt(model_dir, prompt)?;
    validate_tokenizer(model, &tokenizer)?;
    Ok((tokenizer, input))
}

pub(super) fn tokenize_prompt(
    model_dir: &Path,
    prompt: &str,
) -> std::result::Result<(tokenizers::Tokenizer, Vec<u32>), Box<dyn std::error::Error>> {
    let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| e.to_string())?;
    let input = tokenize_with(&tokenizer, model_dir, prompt)?;
    Ok((tokenizer, input))
}

pub(super) fn tokenize_with(
    tokenizer: &tokenizers::Tokenizer,
    model_dir: &Path,
    prompt: &str,
) -> std::result::Result<Vec<u32>, Box<dyn std::error::Error>> {
    let formatted = format_prompt(model_dir, prompt)?;
    let input = tokenizer
        .encode(formatted, true)
        .map_err(|e| e.to_string())?
        .get_ids()
        .to_vec();
    Ok(input)
}

pub(super) fn tokenize_conversation_with(
    tokenizer: &tokenizers::Tokenizer,
    model_dir: &Path,
    turns: &[super::conversation::Turn],
    prompt: &str,
    older_messages: usize,
    feedback: Option<&str>,
) -> std::result::Result<Vec<u32>, Box<dyn std::error::Error>> {
    let formatted = format_chat_prompt(
        is_qwen3(model_dir)?,
        turns,
        prompt,
        older_messages,
        feedback,
    );
    Ok(tokenizer
        .encode(formatted, true)
        .map_err(|e| e.to_string())?
        .get_ids()
        .to_vec())
}

fn format_chat_prompt(
    qwen3: bool,
    turns: &[super::conversation::Turn],
    prompt: &str,
    older_messages: usize,
    feedback: Option<&str>,
) -> String {
    let system = (older_messages > 0 || feedback.is_some()).then(|| format!(
        "You are a helpful assistant. The app saves the conversation to a file. Earlier messages outside your context: {older_messages}.\n\nTo read earlier messages, your ENTIRE response must be FETCH_OLDER N, with no explanation, quotes or other text. N is a positive integer from 1 to 1024. Example response: FETCH_OLDER 2\n\nWhen asked about an earlier detail that is absent from the visible conversation, fetch earlier messages before answering. Never pretend that you fetched them, and never invent a missing detail. The app will insert the retrieved user/assistant turns before the recent messages and ask the same question again. If a fetch is refused, request fewer messages or explain that the detail is unavailable. If no earlier messages remain, answer from the visible conversation."
    ));
    // Put the result beside the pending question so the model sees that the
    // previous control request has already been handled, rather than repeating it.
    let continued_prompt = feedback.map(|feedback| format!("{prompt}\n\n[Application FETCH_OLDER result: {feedback} Earlier messages still outside context: {older_messages}. Continue answering the question above using the visible conversation. Do not repeat a successful fetch. Request additional messages only if the required detail is still absent.]"));
    format_conversation(
        qwen3,
        turns,
        continued_prompt.as_deref().unwrap_or(prompt),
        system.as_deref(),
    )
}

pub(super) fn validate_tokenizer(
    model: &Model,
    tokenizer: &tokenizers::Tokenizer,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    if tokenizer
        .get_vocab(true)
        .values()
        .any(|&id| id >= model.info.vocab_size)
    {
        return Err("tokenizer contains IDs outside the model vocabulary".into());
    }
    Ok(())
}

/// The CLI currently accepts one user message. Qwen3 uses its documented
/// non-thinking generation prefix; the model ABI continues to receive token IDs.
fn format_prompt(
    model_dir: &Path,
    prompt: &str,
) -> std::result::Result<String, Box<dyn std::error::Error>> {
    Ok(format_conversation(is_qwen3(model_dir)?, &[], prompt, None))
}

fn is_qwen3(model_dir: &Path) -> std::result::Result<bool, Box<dyn std::error::Error>> {
    let config = model_dir.join("config.json");
    if config.is_file() {
        let config: serde_json::Value = serde_json::from_slice(&std::fs::read(config)?)?;
        return Ok(config.get("model_type").and_then(|v| v.as_str()) == Some("qwen3"));
    }
    Ok(false)
}

fn format_conversation(
    qwen3: bool,
    turns: &[super::conversation::Turn],
    prompt: &str,
    system: Option<&str>,
) -> String {
    let mut text = String::new();
    if qwen3 {
        if let Some(system) = system {
            text.push_str(&format!("<|im_start|>system\n{system}<|im_end|>\n"));
        }
        for turn in turns {
            text.push_str(&format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>\n",
                turn.user, turn.assistant
            ));
        }
        text.push_str(&format!(
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ));
    } else {
        if let Some(system) = system {
            text.push_str(&format!("System: {system}\n\n"));
        }
        for turn in turns {
            text.push_str(&format!(
                "User: {}\nAssistant: {}\n\n",
                turn.user, turn.assistant
            ));
        }
        if turns.is_empty() && system.is_none() {
            text.push_str(prompt);
        } else {
            text.push_str(&format!("User: {prompt}\nAssistant:"));
        }
    }
    text
}

#[cfg(test)]
mod prompt_tests {
    use super::*;
    #[test]
    fn archive_control_instructions_only_appear_when_older_messages_exist() {
        let turns = [super::super::conversation::Turn {
            user: "My name is puppy".into(),
            assistant: "Hello puppy".into(),
        }];
        let ordinary = format_chat_prompt(true, &turns, "What is my name", 0, None);
        assert_eq!(
            ordinary,
            format_conversation(true, &turns, "What is my name", None)
        );
        assert!(!ordinary.contains("FETCH_OLDER"));
        let archived = format_chat_prompt(true, &turns, "What is my name", 2, None);
        assert!(archived.starts_with("<|im_start|>system\n"));
        assert!(archived.contains("Earlier messages outside your context: 2"));
        assert!(archived.contains("FETCH_OLDER N"));
        let fetched = format_chat_prompt(
            true,
            &turns,
            "What is my name",
            0,
            Some("Added 2 older messages."),
        );
        assert!(fetched.starts_with("<|im_start|>system\n"));
        assert!(fetched.contains("[Application FETCH_OLDER result: Added 2 older messages."));
    }

    #[test]
    fn conversation_roles_and_fetch_feedback_precede_the_current_question() {
        let turns = [super::super::conversation::Turn {
            user: "My name is Teppo".into(),
            assistant: "Hello Teppo".into(),
        }];
        let formatted = format_conversation(
            true,
            &turns,
            "What is my name?",
            Some("FETCH_OLDER result: Added 2 messages."),
        );
        assert_eq!(formatted, "<|im_start|>system\nFETCH_OLDER result: Added 2 messages.<|im_end|>\n<|im_start|>user\nMy name is Teppo<|im_end|>\n<|im_start|>assistant\nHello Teppo<|im_end|>\n<|im_start|>user\nWhat is my name?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        assert_eq!(
            format_conversation(false, &turns, "What is my name?", None),
            "User: My name is Teppo\nAssistant: Hello Teppo\n\nUser: What is my name?\nAssistant:"
        );
    }
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
