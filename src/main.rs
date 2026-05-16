use clap::{Parser, Subcommand};
use puppygrad::engine::Tensor;
use puppygrad::models::gpt2::{
    default_gpt2_small_dir, download_gpt2_small_assets, download_huggingface_gpt2_assets,
    Gpt2Runtime,
};
use std::io::Write;
use std::path::PathBuf;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser, Debug)]
#[command(name = "puppygrad")]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run GPT-2 small through puppygrad's native reference model.
    Gpt2 {
        /// Local directory containing config.json, tokenizer.json, and model.safetensors.
        #[arg(long)]
        model_dir: Option<PathBuf>,

        /// Hugging Face model id used with --download.
        #[arg(long, default_value = "gpt2")]
        model_id: String,

        /// Hugging Face revision used with --download.
        #[arg(long, default_value = "main")]
        revision: String,

        /// Download missing model assets into --model-dir before running.
        #[arg(long)]
        download: bool,

        /// Prompt text.
        #[arg(long)]
        prompt: String,

        /// Max new tokens to generate.
        #[arg(long, default_value_t = 32)]
        max_new_tokens: usize,
    },

    /// Placeholder for the future in-house Qwen runtime.
    Qwen {
        /// Local model directory reserved for future native weight loading.
        #[arg(long)]
        model_dir: Option<PathBuf>,

        /// Model id reserved for future metadata/download tooling.
        #[arg(long, default_value = "Qwen/Qwen2.5-0.5B-Instruct")]
        model_id: String,

        /// Revision reserved for future metadata/download tooling.
        #[arg(long, default_value = "main")]
        revision: String,

        /// Reserved flag for future native model asset handling.
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

        /// Reserved dtype selector for the future native runtime.
        #[arg(long)]
        dtype: Option<String>,

        /// Reserved flag for future prompt templating.
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

        /// Print progress every N steps.
        #[arg(long, default_value_t = 25)]
        log_every: usize,
    },

    /// Quick matrix multiply + backward sanity check.
    MatmulCheck,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Gpt2 {
            model_dir,
            model_id,
            revision,
            download,
            prompt,
            max_new_tokens,
        } => run_gpt2(RunGpt2Args {
            model_dir,
            model_id,
            revision,
            download,
            prompt,
            max_new_tokens,
        }),
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
            log_every,
        } => run_demo_linear(steps, lr, log_every),
        Command::MatmulCheck => run_matmul_check(),
    }
}

struct RunGpt2Args {
    model_dir: Option<PathBuf>,
    model_id: String,
    revision: String,
    download: bool,
    prompt: String,
    max_new_tokens: usize,
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

fn run_gpt2(args: RunGpt2Args) -> Result<()> {
    let model_dir = args.model_dir.unwrap_or_else(default_gpt2_small_dir);
    if args.download {
        eprintln!(
            "downloading missing GPT-2 assets into {}",
            model_dir.display()
        );
        if args.model_id == "gpt2" && args.revision == "main" {
            download_gpt2_small_assets(&model_dir)?;
        } else {
            download_huggingface_gpt2_assets(&args.model_id, &args.revision, &model_dir)?;
        }
    }

    eprintln!("loading GPT-2 from {}", model_dir.display());
    let runtime = Gpt2Runtime::from_dir(&model_dir)?;
    let mut stdout = std::io::stdout().lock();
    runtime.stream_greedy_text(&args.prompt, args.max_new_tokens, |text| {
        write!(stdout, "{text}")?;
        stdout.flush()?;
        Ok::<(), Box<dyn std::error::Error>>(())
    })?;
    writeln!(stdout)?;
    Ok(())
}

fn run_qwen(args: RunQwenArgs) -> Result<()> {
    println!("qwen runtime is not implemented yet.");
    println!(
        "The external runtime was removed so transformer work can target puppygrad's native engine."
    );
    println!("requested model: {}@{}", args.model_id, args.revision);
    if let Some(model_dir) = args.model_dir {
        println!("model dir: {}", model_dir.display());
    }
    println!("prompt: {}", args.prompt);
    println!(
        "generation args: max_new_tokens={} temperature={} top_k={:?} top_p={:?} seed={} repeat_penalty={} repeat_last_n={} dtype={:?} instruct={} download={}",
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        args.seed,
        args.repeat_penalty,
        args.repeat_last_n,
        args.dtype,
        args.instruct,
        args.download
    );
    Ok(())
}

fn run_demo_linear(steps: usize, lr: f32, log_every: usize) -> Result<()> {
    let x = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], vec![4], false)?;
    let y = Tensor::from_vec(vec![1.0, 3.0, 5.0, 7.0], vec![4], false)?;

    let w = Tensor::scalar(-0.25, true);
    let b = Tensor::scalar(0.5, true);

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
    Ok(diff.mul(&diff)?.mean()?)
}
