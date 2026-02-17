use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor as CandleTensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::qwen2;
use clap::{Parser, Subcommand};
use hf_hub::{api::sync::Api, Repo, RepoType};
use puppygrad::engine::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(name = "puppygrad")]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run Qwen2/Qwen2.5 causal LM on CPU.
    Qwen {
        /// Local model directory containing config.json/tokenizer.json/(model*.safetensors)
        #[arg(long)]
        model_dir: Option<PathBuf>,

        /// Hugging Face model id (used if --model-dir not provided)
        #[arg(long, default_value = "Qwen/Qwen2.5-0.5B-Instruct")]
        model_id: String,

        /// HF revision/tag (when downloading from hub)
        #[arg(long, default_value = "main")]
        revision: String,

        /// Download model files into --model-dir when missing.
        #[arg(long)]
        download: bool,

        /// Prompt text
        #[arg(long)]
        prompt: String,

        /// Max new tokens to generate
        #[arg(long, default_value_t = 128)]
        max_new_tokens: usize,

        /// Temperature (<= 0 => greedy)
        #[arg(long, default_value_t = 0.8)]
        temperature: f64,

        /// Top-p nucleus sampling (optional)
        #[arg(long)]
        top_p: Option<f64>,

        /// Top-k sampling (optional)
        #[arg(long)]
        top_k: Option<usize>,

        /// RNG seed
        #[arg(long, default_value_t = 299792458)]
        seed: u64,

        /// Repeat penalty (1.0 = disabled)
        #[arg(long, default_value_t = 1.1)]
        repeat_penalty: f32,

        /// How many last tokens are considered for repeat penalty
        #[arg(long, default_value_t = 128)]
        repeat_last_n: usize,

        /// Force dtype for weights (f16/bf16/f32). Default f16.
        #[arg(long)]
        dtype: Option<String>,

        /// If set, wrap prompt in a simple instruct-style template.
        #[arg(long)]
        instruct: bool,
    },

    /// Train y = 2x + 3 with scalar parameters using the in-house autograd engine.
    DemoLinear {
        /// Number of SGD steps.
        #[arg(long, default_value_t = 300)]
        steps: usize,

        /// SGD learning rate.
        #[arg(long, default_value_t = 0.1)]
        lr: f32,

        /// RNG seed for parameter initialization.
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Print progress every N steps.
        #[arg(long, default_value_t = 25)]
        log_every: usize,
    },

    /// Quick matrix multiply + backward sanity check.
    MatmulCheck,
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: std::collections::HashMap<String, String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Qwen {
            model_dir,
            model_id,
            revision,
            download,
            prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            seed,
            repeat_penalty,
            repeat_last_n,
            dtype,
            instruct,
        } => run_qwen(RunQwenArgs {
            model_dir,
            model_id,
            revision,
            download,
            prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            seed,
            repeat_penalty,
            repeat_last_n,
            dtype,
            instruct,
        }),
        Command::DemoLinear {
            steps,
            lr,
            seed,
            log_every,
        } => run_demo_linear(steps, lr, seed, log_every),
        Command::MatmulCheck => run_matmul_check(),
    }
}

struct RunQwenArgs {
    model_dir: Option<PathBuf>,
    model_id: String,
    revision: String,
    download: bool,
    prompt: String,
    max_new_tokens: usize,
    temperature: f64,
    top_p: Option<f64>,
    top_k: Option<usize>,
    seed: u64,
    repeat_penalty: f32,
    repeat_last_n: usize,
    dtype: Option<String>,
    instruct: bool,
}

fn run_qwen(args: RunQwenArgs) -> Result<()> {
    let device = Device::Cpu;
    let dtype = parse_dtype(args.dtype.as_deref())?;

    let (model_dir, config_path, tokenizer_path, safetensors_paths) =
        if let Some(dir) = args.model_dir {
            if args.download {
                maybe_download_model_to_dir(&args.model_id, &args.revision, &dir)?;
            }
            let config = dir.join("config.json");
            let tokenizer = dir.join("tokenizer.json");
            if !config.exists() {
                bail!(
                    "config.json not found in {} (use --download to fetch model files)",
                    dir.display()
                );
            }
            if !tokenizer.exists() {
                bail!(
                    "tokenizer.json not found in {} (use --download to fetch model files)",
                    dir.display()
                );
            }
            let sts = find_safetensors_files(&dir).with_context(|| {
                format!(
                    "no safetensors found in {} (use --download to fetch model files)",
                    dir.display()
                )
            })?;
            (dir, config, tokenizer, sts)
        } else {
            let api = Api::new().context("hf-hub Api::new failed")?;
            let repo = api.repo(Repo::with_revision(
                args.model_id.clone(),
                RepoType::Model,
                args.revision,
            ));
            println!("Downloading model from HF: {}", args.model_id);

            let tokenizer = repo
                .get("tokenizer.json")
                .context("failed to download tokenizer.json")?;
            let config = repo
                .get("config.json")
                .context("failed to download config.json")?;

            let safetensors_paths = if let Ok(p) = repo.get("model.safetensors.index.json") {
                hub_load_safetensors_shards(&repo, &p)?
            } else {
                vec![repo
                    .get("model.safetensors")
                    .context("failed to download model.safetensors")?]
            };

            let dir = config
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("."));
            (dir, config, tokenizer, safetensors_paths)
        };

    let cfg: qwen2::Config = serde_json::from_slice(&fs::read(&config_path)?)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Tokenizer load failed: {e}"))?;

    let prompt_text = if args.instruct {
        format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\n{}<|im_end|>\n\
             <|im_start|>assistant\n",
            args.prompt
        )
    } else {
        args.prompt.clone()
    };

    let mut tokens: Vec<u32> = tokenizer
        .encode(prompt_text.as_str(), true)
        .map_err(|e| anyhow::anyhow!("Tokenizer encode failed: {e}"))?
        .get_ids()
        .to_vec();

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensors_paths, dtype, &device)? };
    let mut model = qwen2::ModelForCausalLM::new(&cfg, vb)?;

    let sampling = if args.temperature <= 0.0 {
        Sampling::ArgMax
    } else {
        match (args.top_k, args.top_p) {
            (None, None) => Sampling::All {
                temperature: args.temperature,
            },
            (Some(k), None) => Sampling::TopK {
                k,
                temperature: args.temperature,
            },
            (None, Some(p)) => Sampling::TopP {
                p,
                temperature: args.temperature,
            },
            (Some(k), Some(p)) => Sampling::TopKThenTopP {
                k,
                p,
                temperature: args.temperature,
            },
        }
    };
    let mut logits_processor = LogitsProcessor::from_sampling(args.seed, sampling);
    model.clear_kv_cache();

    print!("{}", args.prompt);
    std::io::stdout().flush()?;

    let mut last_printed = String::new();
    let mut index_pos: usize = 0;
    let start = std::time::Instant::now();
    for step in 0..args.max_new_tokens {
        let (context_size, seqlen_offset) = if step == 0 {
            (tokens.len(), 0usize)
        } else {
            (1usize, index_pos)
        };

        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = CandleTensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, seqlen_offset)?;
        let logits = logits.squeeze(0)?.squeeze(0)?;

        let logits = if (args.repeat_penalty - 1.0).abs() < f32::EPSILON {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &tokens[start_at..],
            )?
        };

        index_pos += context_size;
        let next = logits_processor.sample(&logits)? as u32;
        tokens.push(next);

        let decoded = tokenizer
            .decode(&tokens, true)
            .map_err(|e| anyhow::anyhow!("Tokenizer decode failed: {e}"))?;
        if decoded.starts_with(&last_printed) {
            let delta = &decoded[last_printed.len()..];
            print!("{delta}");
        } else {
            print!("{decoded}");
        }
        std::io::stdout().flush()?;
        last_printed = decoded;
    }

    let dt = start.elapsed().as_secs_f64();
    eprintln!(
        "\n\nDone. {:.2} tok/s (rough, includes decode+printing)",
        (args.max_new_tokens as f64) / dt
    );
    eprintln!("Model dir: {}", model_dir.display());
    Ok(())
}

fn maybe_download_model_to_dir(model_id: &str, revision: &str, dir: &Path) -> Result<()> {
    let config = dir.join("config.json");
    let tokenizer = dir.join("tokenizer.json");
    let has_weights = find_safetensors_files(dir).is_ok();
    if config.exists() && tokenizer.exists() && has_weights {
        return Ok(());
    }

    fs::create_dir_all(dir)
        .with_context(|| format!("failed creating model dir {}", dir.display()))?;
    println!(
        "Downloading {model_id}@{revision} into {} ...",
        dir.display()
    );

    let api = Api::new().context("hf-hub Api::new failed")?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    let config_src = repo
        .get("config.json")
        .context("failed to download config.json")?;
    copy_file_to(&config_src, &dir.join("config.json"))?;

    let tokenizer_src = repo
        .get("tokenizer.json")
        .context("failed to download tokenizer.json")?;
    copy_file_to(&tokenizer_src, &dir.join("tokenizer.json"))?;

    if let Ok(index_src) = repo.get("model.safetensors.index.json") {
        copy_file_to(&index_src, &dir.join("model.safetensors.index.json"))?;
        let data = fs::read(&index_src)?;
        let idx: SafetensorsIndex = serde_json::from_slice(&data)?;
        let mut uniq = HashSet::new();
        let mut shard_names: Vec<String> = idx
            .weight_map
            .values()
            .filter(|v| uniq.insert(v.to_string()))
            .cloned()
            .collect();
        shard_names.sort();
        for name in shard_names {
            let src = repo
                .get(&name)
                .with_context(|| format!("failed downloading shard {name}"))?;
            copy_file_to(&src, &dir.join(&name))?;
        }
    } else {
        let model_src = repo
            .get("model.safetensors")
            .context("failed to download model.safetensors")?;
        copy_file_to(&model_src, &dir.join("model.safetensors"))?;
    }

    Ok(())
}

fn copy_file_to(src: &Path, dst: &Path) -> Result<()> {
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed creating directory {}", parent.display()))?;
    }
    fs::copy(src, dst).with_context(|| {
        format!(
            "failed copying {} to {}",
            src.display(),
            dst.as_os_str().to_string_lossy()
        )
    })?;
    Ok(())
}

fn parse_dtype(dtype: Option<&str>) -> Result<DType> {
    match dtype {
        Some("f16") => Ok(DType::F16),
        Some("bf16") => Ok(DType::BF16),
        Some("f32") => Ok(DType::F32),
        Some(other) => bail!("Unsupported dtype: {other} (use f16|bf16|f32)"),
        None => Ok(DType::F16),
    }
}

fn find_safetensors_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let data = fs::read(&index)?;
        let idx: SafetensorsIndex = serde_json::from_slice(&data)?;
        let mut uniq = HashSet::new();
        let mut files: Vec<PathBuf> = idx
            .weight_map
            .values()
            .filter(|v| uniq.insert(v.to_string()))
            .map(|f| dir.join(f))
            .collect();
        files.sort();
        for file in &files {
            if !file.exists() {
                bail!("Missing shard file referenced by index: {}", file.display());
            }
        }
        return Ok(files);
    }

    bail!(
        "No model.safetensors or model.safetensors.index.json found in {}",
        dir.display()
    )
}

fn hub_load_safetensors_shards(
    repo: &hf_hub::api::sync::ApiRepo,
    index_path: &Path,
) -> Result<Vec<PathBuf>> {
    let data = fs::read(index_path)?;
    let idx: SafetensorsIndex = serde_json::from_slice(&data)?;
    let mut uniq = HashSet::new();
    let mut shard_names: Vec<String> = idx
        .weight_map
        .values()
        .filter(|v| uniq.insert(v.to_string()))
        .cloned()
        .collect();
    shard_names.sort();

    let mut out = Vec::with_capacity(shard_names.len());
    for name in shard_names {
        out.push(
            repo.get(&name)
                .with_context(|| format!("failed downloading shard {name}"))?,
        );
    }
    Ok(out)
}

fn run_demo_linear(steps: usize, lr: f32, seed: u64, log_every: usize) -> Result<()> {
    let x = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], vec![4], false)?;
    let y = Tensor::from_vec(vec![1.0, 3.0, 5.0, 7.0], vec![4], false)?;

    let mut rng = StdRng::seed_from_u64(seed);
    let w = Tensor::scalar(rng.random_range(-0.5f32..0.5f32), true);
    let b = Tensor::scalar(rng.random_range(-0.5f32..0.5f32), true);

    let initial_loss = mse(&x, &y, &w, &b)?.item()?;
    println!(
        "init: w={:.5} b={:.5} loss={:.6}",
        w.item()?,
        b.item()?,
        initial_loss
    );

    let log_every = log_every.max(1);
    for step in 0..steps {
        w.zero_grad();
        b.zero_grad();

        let loss = mse(&x, &y, &w, &b)?;
        loss.backward()?;

        w.step(lr)?;
        b.step(lr)?;

        if (step + 1) % log_every == 0 || step + 1 == steps {
            println!(
                "step {:>4}: loss={:.6} w={:.5} b={:.5}",
                step + 1,
                loss.item()?,
                w.item()?,
                b.item()?
            );
        }
    }

    let final_loss = mse(&x, &y, &w, &b)?.item()?;
    println!(
        "done: w={:.5} b={:.5} loss={:.6}",
        w.item()?,
        b.item()?,
        final_loss
    );
    Ok(())
}

fn run_matmul_check() -> Result<()> {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true)?;
    let b = Tensor::from_vec(vec![5.0, 6.0], vec![2, 1], true)?;
    let out = a.matmul(&b)?.mean()?;
    out.backward()?;

    println!("out={:.5}", out.item()?);
    println!("grad(a)={:?}", a.grad().unwrap_or_default());
    println!("grad(b)={:?}", b.grad().unwrap_or_default());
    Ok(())
}

fn mse(x: &Tensor, y: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    let pred = x.mul(w)?.add(b)?;
    let diff = pred.sub(y)?;
    diff.mul(&diff)?.mean()
}
