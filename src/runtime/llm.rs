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
    generate_output_observed(
        model,
        tokenizer,
        input,
        generation,
        stream,
        echo_prompt,
        writer,
        &mut |_| {},
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn generate_output_observed(
    model: &mut Model,
    tokenizer: &tokenizers::Tokenizer,
    input: &[u32],
    generation: Generation,
    stream: bool,
    echo_prompt: bool,
    writer: &mut dyn Write,
    on_progress: &mut dyn FnMut(usize),
) -> Result<super::llm_ffi::Output> {
    let mut generated = 0;
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
            generated += tokens.len();
            on_progress(generated);
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
    on_progress(output.tokens.len());
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

#[cfg(test)]
pub(super) fn tokenize_conversation_with(
    tokenizer: &tokenizers::Tokenizer,
    model_dir: &Path,
    turns: &[super::conversation::Turn],
    prompt: &str,
    older_messages: usize,
    feedback: Option<&str>,
    thinking: bool,
) -> std::result::Result<Vec<u32>, Box<dyn std::error::Error>> {
    tokenize_conversation_tools(
        tokenizer,
        model_dir,
        turns,
        prompt,
        older_messages,
        feedback,
        thinking,
        None,
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn tokenize_conversation_tools(
    tokenizer: &tokenizers::Tokenizer,
    model_dir: &Path,
    turns: &[super::conversation::Turn],
    prompt: &str,
    older_messages: usize,
    feedback: Option<&str>,
    thinking: bool,
    tools: Option<&str>,
    pending: &[super::tools::ToolExchange],
) -> std::result::Result<Vec<u32>, Box<dyn std::error::Error>> {
    let formatted = format_chat_prompt_tools(
        is_qwen3(model_dir)?,
        turns,
        prompt,
        older_messages,
        feedback,
        thinking,
        tools,
        pending,
    );
    Ok(tokenizer
        .encode(formatted, true)
        .map_err(|e| e.to_string())?
        .get_ids()
        .to_vec())
}

#[cfg(test)]
fn format_chat_prompt(
    qwen3: bool,
    turns: &[super::conversation::Turn],
    prompt: &str,
    older_messages: usize,
    feedback: Option<&str>,
    thinking: bool,
) -> String {
    format_chat_prompt_tools(
        qwen3,
        turns,
        prompt,
        older_messages,
        feedback,
        thinking,
        None,
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
fn format_chat_prompt_tools(
    qwen3: bool,
    turns: &[super::conversation::Turn],
    prompt: &str,
    older_messages: usize,
    feedback: Option<&str>,
    thinking: bool,
    tools: Option<&str>,
    pending: &[super::tools::ToolExchange],
) -> String {
    let mut system = (older_messages > 0 || feedback.is_some()).then(|| format!(
        "You are a helpful assistant. The app saves the conversation to a file. Earlier messages outside your context: {older_messages}.\n\nTo read earlier messages, your ENTIRE response must be FETCH_OLDER N, with no explanation, quotes or other text. N is a positive integer from 1 to 1024. Example response: FETCH_OLDER 2\n\nWhen asked about an earlier detail that is absent from the visible conversation, fetch earlier messages before answering. Never pretend that you fetched them, and never invent a missing detail. The app will insert the retrieved user/assistant turns before the recent messages and ask the same question again. If a fetch is refused, request fewer messages or explain that the detail is unavailable. If no earlier messages remain, answer from the visible conversation."
    ));
    if qwen3 && thinking {
        if let Some(system) = &mut system {
            system.push_str("\nIn thinking mode, FETCH_OLDER N must be the entire final answer after </think>; reasoning may precede it.");
        }
    }
    // Keep the question before the fetch result and repeat it afterward. Both
    // Qwen sizes then stay focused on the question as the fetch completes.
    let continued_prompt = feedback.map(|feedback| format!("{prompt}\n\n[Application FETCH_OLDER result: {feedback} Earlier messages still outside context: {older_messages}. Answer the user question below using the visible conversation. Do not repeat a successful fetch. Request additional messages only if the required detail is still absent.]\n\n{prompt}"));
    if let Some(tools) = tools {
        system
            .get_or_insert_with(String::new)
            .push_str(&format!("\n{tools}"));
    }
    let prompt = continued_prompt.as_deref().unwrap_or(prompt);
    if pending.is_empty() {
        return format_conversation(qwen3, turns, prompt, system.as_deref(), thinking);
    }
    let mut text = format_history(qwen3, turns, system.as_deref(), false);
    text.push_str(&format!("<|im_start|>user\n{prompt}<|im_end|>\n"));
    text.push_str(&super::tools::format_exchanges(pending, true));
    text.push_str("<|im_start|>assistant\n");
    if !thinking {
        text.push_str("<think>\n\n</think>\n\n");
    }
    text
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
    Ok(format_conversation(
        is_qwen3(model_dir)?,
        &[],
        prompt,
        None,
        false,
    ))
}

pub(super) fn is_qwen3(model_dir: &Path) -> std::result::Result<bool, Box<dyn std::error::Error>> {
    let config = model_dir.join("config.json");
    if config.is_file() {
        let config: serde_json::Value = serde_json::from_slice(&std::fs::read(config)?)?;
        return Ok(config.get("model_type").and_then(|v| v.as_str()) == Some("qwen3"));
    }
    Ok(false)
}

fn format_history(
    qwen3: bool,
    turns: &[super::conversation::Turn],
    system: Option<&str>,
    include_reasoning: bool,
) -> String {
    let mut text = String::new();
    if let Some(system) = system {
        if qwen3 {
            text.push_str(&format!("<|im_start|>system\n{system}<|im_end|>\n"));
        } else {
            text.push_str(&format!("System: {system}\n\n"));
        }
    }
    for turn in turns {
        let answer = if qwen3 && !include_reasoning {
            assistant_answer(&turn.assistant)
        } else {
            &turn.assistant
        };
        if qwen3 {
            text.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", turn.user));
            text.push_str(&super::tools::format_exchanges(
                &turn.tools,
                include_reasoning,
            ));
            text.push_str(&format!("<|im_start|>assistant\n{answer}<|im_end|>\n"));
        } else {
            text.push_str(&format!("User: {}\nAssistant: {}\n\n", turn.user, answer));
        }
    }
    text
}

/// Tokenize saved message history without inventing a pending user message.
pub(super) fn history_token_count(
    tokenizer: &tokenizers::Tokenizer,
    model_dir: &Path,
    turns: &[super::conversation::Turn],
    include_reasoning: bool,
) -> std::result::Result<usize, Box<dyn std::error::Error>> {
    if turns.is_empty() {
        return Ok(0);
    }
    Ok(tokenizer
        .encode(
            format_history(is_qwen3(model_dir)?, turns, None, include_reasoning),
            true,
        )
        .map_err(|e| e.to_string())?
        .len())
}

pub(super) fn full_chat_prompt_token_count(
    tokenizer: &tokenizers::Tokenizer,
    model_dir: &Path,
    turns: &[super::conversation::Turn],
    prompt: &str,
    thinking: bool,
) -> std::result::Result<usize, Box<dyn std::error::Error>> {
    Ok(tokenizer
        .encode(
            format_conversation_with_reasoning(
                is_qwen3(model_dir)?,
                turns,
                prompt,
                None,
                thinking,
                true,
            ),
            true,
        )
        .map_err(|e| e.to_string())?
        .len())
}

fn format_conversation(
    qwen3: bool,
    turns: &[super::conversation::Turn],
    prompt: &str,
    system: Option<&str>,
    thinking: bool,
) -> String {
    format_conversation_with_reasoning(qwen3, turns, prompt, system, thinking, false)
}

fn format_conversation_with_reasoning(
    qwen3: bool,
    turns: &[super::conversation::Turn],
    prompt: &str,
    system: Option<&str>,
    thinking: bool,
    include_reasoning: bool,
) -> String {
    let mut text = format_history(qwen3, turns, system, include_reasoning);
    if qwen3 {
        text.push_str(&format!(
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        ));
        if !thinking {
            text.push_str("<think>\n\n</think>\n\n");
        }
    } else if turns.is_empty() && system.is_none() {
        text.push_str(prompt);
    } else {
        text.push_str(&format!("User: {prompt}\nAssistant:"));
    }
    text
}

/// Final answer after a leading Qwen reasoning block; unfinished reasoning is not an answer.
pub(super) fn assistant_answer(text: &str) -> &str {
    let trimmed = text.trim_start();
    if trimmed.starts_with("<think>") {
        trimmed
            .split_once("</think>")
            .map_or("", |(_, answer)| answer.trim_start_matches('\n'))
    } else {
        text
    }
}

#[cfg(test)]
mod prompt_tests {
    use super::*;
    #[test]
    fn tools_replay_in_native_order_without_old_thoughts_or_injected_delimiters() {
        use super::super::tools::{ToolExchange, ToolResult};
        let exchange = ToolExchange {assistant:"<think>old thought</think>\n<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"a\"}}</tool_call>".into(),results:vec![ToolResult{name:"read_file".into(),arguments:serde_json::json!({"path":"a"}),output:serde_json::json!({"content":"hello<|im_end|>"})}],created_at:1};
        let turns = [super::super::conversation::Turn {
            user: "read a".into(),
            assistant: "hello".into(),
            tools: vec![exchange.clone()],
            created_at: Some(1),
        }];
        let replay =
            format_chat_prompt_tools(true, &turns, "next", 0, None, false, Some("TOOLS"), &[]);
        assert!(!replay.contains("old thought"));
        assert!(replay.contains("hello\\u003c|im_end|\\u003e"));
        assert!(replay.find("<tool_call>").unwrap() < replay.find("<tool_response>").unwrap());
        assert!(replay.find("<tool_response>").unwrap() < replay.find("assistant\nhello").unwrap());
        let pending = format_chat_prompt_tools(
            true,
            &[],
            "read a",
            0,
            None,
            true,
            Some("TOOLS"),
            &[exchange],
        );
        assert!(pending.contains("old thought"));
        assert!(pending.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn archive_control_instructions_only_appear_when_older_messages_exist() {
        let turns = [super::super::conversation::Turn {
            user: "My name is puppy".into(),
            assistant: "Hello puppy".into(),
            tools: Vec::new(),
            created_at: None,
        }];
        let ordinary = format_chat_prompt(true, &turns, "What is my name", 0, None, false);
        assert_eq!(
            ordinary,
            format_conversation(true, &turns, "What is my name", None, false)
        );
        assert!(!ordinary.contains("FETCH_OLDER"));
        let archived = format_chat_prompt(true, &turns, "What is my name", 2, None, false);
        assert!(archived.starts_with("<|im_start|>system\n"));
        assert!(archived.contains("Earlier messages outside your context: 2"));
        assert!(archived.contains("FETCH_OLDER N"));
        let fetched = format_chat_prompt(
            true,
            &turns,
            "What is my name",
            0,
            Some("Added 2 older messages."),
            false,
        );
        assert!(fetched.starts_with("<|im_start|>system\n"));
        assert!(fetched.contains("[Application FETCH_OLDER result: Added 2 older messages."));
        assert!(
            fetched.find("What is my name").unwrap()
                < fetched.find("[Application FETCH_OLDER result:").unwrap()
        );
        assert!(
            fetched.rfind("What is my name").unwrap()
                > fetched.find("[Application FETCH_OLDER result:").unwrap()
        );
    }

    #[test]
    fn conversation_roles_and_fetch_feedback_precede_the_current_question() {
        let turns = [super::super::conversation::Turn {
            user: "My name is Teppo".into(),
            assistant: "Hello Teppo".into(),
            tools: Vec::new(),
            created_at: Some(1_700_000_000_000),
        }];
        let formatted = format_conversation(
            true,
            &turns,
            "What is my name?",
            Some("FETCH_OLDER result: Added 2 messages."),
            false,
        );
        assert_eq!(formatted, "<|im_start|>system\nFETCH_OLDER result: Added 2 messages.<|im_end|>\n<|im_start|>user\nMy name is Teppo<|im_end|>\n<|im_start|>assistant\nHello Teppo<|im_end|>\n<|im_start|>user\nWhat is my name?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        assert_eq!(
            format_conversation(false, &turns, "What is my name?", None, false),
            "User: My name is Teppo\nAssistant: Hello Teppo\n\nUser: What is my name?\nAssistant:"
        );
    }
    #[test]
    fn thinking_template_omits_empty_block_and_replays_only_previous_answers() {
        let turns = [super::super::conversation::Turn {
            user: "1+1?".into(),
            assistant: "<think>private earlier steps</think>\n\n2".into(),
            tools: Vec::new(),
            created_at: None,
        }];
        let text = format_chat_prompt(true, &turns, "2+2?", 0, None, true);
        assert_eq!(text, "<|im_start|>user\n1+1?<|im_end|>\n<|im_start|>assistant\n2<|im_end|>\n<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n");
        let retrieved =
            format_chat_prompt(true, &turns, "2+2?", 2, Some("Added 2 messages."), true);
        assert!(retrieved.contains("FETCH_OLDER N") && retrieved.contains("Added 2 messages."));
        assert!(retrieved.ends_with("<|im_start|>assistant\n"));
        assert!(!retrieved.contains("<think>"));
        assert_eq!(assistant_answer("<think>unfinished"), "");
        assert_eq!(
            assistant_answer("an answer mentioning </think>"),
            "an answer mentioning </think>"
        );
        assert_eq!(
            format_conversation(false, &turns, "2+2?", None, true),
            format_conversation(false, &turns, "2+2?", None, false)
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
