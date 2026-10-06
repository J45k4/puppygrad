//! Generic supervised training over named file tensors and an ordinary .pup buffer program.
//! The host owns data/state; forward, loss, gradients and optimizer are entirely compiled C.
use super::{
    data::{self, LoadedTensor, Result},
    profile::Profile,
};
use crate::compiler::{
    cpu::{self, Executable, Tensor},
    pop::{DType, Op},
    source::{self, Context, TensorSpec},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(clap::Args, Debug)]
pub struct Options {
    pub source: PathBuf,
    /// Buffer/dataset contract; defaults to SOURCE with .train.json extension.
    #[arg(long)]
    pub config: Option<PathBuf>,
    #[arg(long, default_value_t = 5)]
    pub epochs: usize,
    #[arg(long, default_value_t = 1)]
    pub threads: usize,
    /// CPU instruction target for generated C.
    #[arg(long, value_enum, default_value_t = cpu::CpuTarget::Generic)]
    pub cpu_target: cpu::CpuTarget,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    #[arg(long, default_value_t = 0.1)]
    pub learning_rate: f32,
    #[arg(long, alias = "output")]
    pub output_dir: Option<PathBuf>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    path: PathBuf,
    #[serde(default)]
    format: Option<data::Format>,
    #[serde(default)]
    csv: Option<data::csv::Schema>,
    #[serde(default)]
    tensor: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    md5: Option<String>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Dataset {
    train: File,
    test: File,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Initializer {
    Zeros,
    Normal { stddev: f32 },
    Constant { value: f32 },
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct State {
    input: String,
    output: String,
    shape: Vec<usize>,
    init: Initializer,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Classification {
    prediction: String,
    labels: String,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Contract {
    version: u32,
    batch_size: usize,
    datasets: BTreeMap<String, Dataset>,
    state: Vec<State>,
    learning_rate: String,
    valid: String,
    loss: String,
    #[serde(default)]
    classification: Option<Classification>,
}
struct BoundContract {
    context: Context,
    outputs: Vec<usize>,
    loss: usize,
    prediction: Option<usize>,
    classes: usize,
}

fn checksum(path: &Path, expected: &str) -> Result<()> {
    let out = Command::new("md5sum")
        .arg(path)
        .output()
        .map_err(|e| format!("archive verification needs md5sum: {e}"))?;
    if !out.status.success()
        || String::from_utf8_lossy(&out.stdout)
            .split_whitespace()
            .next()
            != Some(expected)
    {
        return Err(format!("{}: checksum mismatch", path.display()).into());
    }
    Ok(())
}
fn load_file(spec: &mut File, base: &Path) -> Result<LoadedTensor> {
    if !spec.path.is_absolute() {
        spec.path = base.join(&spec.path);
    }
    if !spec.path.exists() {
        let url = spec
            .url
            .as_ref()
            .ok_or_else(|| format!("missing tensor file {}", spec.path.display()))?;
        let md5 = spec
            .md5
            .as_ref()
            .ok_or("download needs an expected checksum")?;
        eprintln!("Downloading {url}");
        let bytes = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()?
            .get(url)
            .send()?
            .error_for_status()?
            .bytes()?;
        if let Some(parent) = spec.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let temp = spec
            .path
            .with_extension(format!("{}.part", std::process::id()));
        fs::write(&temp, &bytes)?;
        checksum(&temp, md5)?;
        fs::rename(temp, &spec.path)?;
    }
    if let Some(md5) = &spec.md5 {
        checksum(&spec.path, md5)?;
    }
    spec.path = spec.path.canonicalize()?;
    let mut tensors = data::load(
        &spec.path,
        &data::LoadOptions {
            format: spec.format,
            csv: spec.csv.clone(),
        },
    )?;
    let name = match &spec.tensor {
        Some(s) => s.clone(),
        None if tensors.len() == 1 => tensors.keys().next().unwrap().clone(),
        None => return Err("file has multiple tensors; select one by name in the contract".into()),
    };
    tensors
        .remove(&name)
        .ok_or_else(|| format!("tensor {name:?} not found in {}", spec.path.display()).into())
}
fn load_datasets(
    config: &mut Contract,
    base: &Path,
) -> Result<(
    BTreeMap<String, LoadedTensor>,
    BTreeMap<String, LoadedTensor>,
)> {
    if config.datasets.is_empty() {
        return Err("training contract has no dataset tensors".into());
    }
    let mut train = BTreeMap::new();
    let mut test = BTreeMap::new();
    for (name, dataset) in &mut config.datasets {
        let a = load_file(&mut dataset.train, base)?;
        let b = load_file(&mut dataset.test, base)?;
        a.rows()?;
        b.rows()?;
        if a.dtype != b.dtype || a.shape[1..] != b.shape[1..] {
            return Err(format!("{name}: train/test tensor schemas differ").into());
        }
        a.dtype.compiler()?;
        train.insert(name.clone(), a);
        test.insert(name.clone(), b);
    }
    for split in [&train, &test] {
        let rows = split.values().next().unwrap().rows()?;
        if rows == 0 || split.values().any(|t| t.rows().ok() != Some(rows)) {
            return Err("all tensors in a split must have the same nonzero sample count".into());
        }
    }
    Ok((train, test))
}
fn context(config: &Contract, data: &BTreeMap<String, LoadedTensor>) -> Result<Context> {
    if config.version != 1 || config.batch_size == 0 {
        return Err("expected training contract version 1 and a positive batch_size".into());
    }
    let mut ctx = Context::default();
    let mut slot = 0;
    let mut add = |name: &str, dtype: DType, shape: Vec<usize>| -> Result<()> {
        if name.is_empty() || ctx.tensors.contains_key(name) {
            return Err(format!("duplicate or empty input name {name:?}").into());
        }
        data::elements(&shape)?;
        ctx.tensors
            .insert(name.into(), TensorSpec { slot, dtype, shape });
        slot += 1;
        Ok(())
    };
    for (name, tensor) in data {
        let mut shape = tensor.shape.clone();
        shape[0] = config.batch_size;
        add(name, tensor.dtype.compiler()?, shape)?;
    }
    if config.state.is_empty() {
        return Err("training contract has no state buffers".into());
    }
    for state in &config.state {
        match state.init {
            Initializer::Normal { stddev } if !stddev.is_finite() || stddev <= 0. => {
                return Err("initializer stddev must be positive and finite".into())
            }
            Initializer::Constant { value } if !value.is_finite() => {
                return Err("initializer value must be finite".into())
            }
            _ => {}
        }
        add(&state.input, DType::F32, state.shape.clone())?;
    }
    add(&config.learning_rate, DType::F32, vec![])?;
    add(&config.valid, DType::F32, vec![config.batch_size])?;
    Ok(ctx)
}
fn bind_outputs(
    config: &Contract,
    program: &source::Program,
    context: Context,
    data: &BTreeMap<String, LoadedTensor>,
) -> Result<BoundContract> {
    let roots = if program.graph.node(program.root)?.op() == Op::Sink {
        program.graph.node(program.root)?.src().to_vec()
    } else {
        vec![program.root]
    };
    let output = |name: &str| -> Result<usize> {
        let binding = program
            .bindings
            .iter()
            .rev()
            .find(|b| b.name == name)
            .ok_or_else(|| format!("no output binding named {name:?}"))?;
        roots
            .iter()
            .position(|&v| v == binding.value)
            .ok_or_else(|| format!("{name:?} is not exported by output").into())
    };
    let mut outputs = Vec::new();
    let mut used = HashSet::new();
    for state in &config.state {
        let i = output(&state.output)?;
        let node = program.graph.node(roots[i])?;
        if node.dtype() != DType::F32
            || node.shape() != Some(state.shape.as_slice())
            || !used.insert(i)
        {
            return Err(format!("invalid updated-state output {:?}", state.output).into());
        }
        outputs.push(i);
    }
    let loss = output(&config.loss)?;
    let node = program.graph.node(roots[loss])?;
    if node.dtype() != DType::F32 || data::elements(node.shape().unwrap())? != 1 {
        return Err("loss output must contain one f32 mean over valid rows".into());
    }
    let (prediction, classes) = if let Some(metric) = &config.classification {
        let i = output(&metric.prediction)?;
        let node = program.graph.node(roots[i])?;
        let shape = node.shape().unwrap();
        if node.dtype() != DType::F32
            || shape.len() != 2
            || shape[0] != config.batch_size
            || shape[1] < 2
        {
            return Err("classification predictions must be f32[batch,classes]".into());
        }
        let labels = data
            .get(&metric.labels)
            .ok_or("classification label tensor is not a dataset input")?;
        for row in 0..labels.rows()? {
            if labels.class_label(row)? >= shape[1] {
                return Err("classification label outside prediction class range".into());
            }
        }
        (Some(i), shape[1])
    } else {
        (None, 0)
    };
    Ok(BoundContract {
        context,
        outputs,
        loss,
        prediction,
        classes,
    })
}
// Small deterministic host RNG, used only for initialization and batch shuffling.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9e3779b97f4a7c15).max(1))
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn uniform(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f32 {
        ((-2.0 * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos())
            as f32
    }
    fn shuffle(&mut self, values: &mut [usize]) {
        for i in (1..values.len()).rev() {
            values.swap(i, (self.next() % (i + 1) as u64) as usize);
        }
    }
}
fn floats(v: Vec<f32>) -> Tensor {
    Tensor::F32(v.into())
}
fn initial_state(config: &Contract, seed: u64) -> Result<Vec<Tensor>> {
    let mut rng = Rng::new(seed);
    config
        .state
        .iter()
        .map(|s| {
            let n = data::elements(&s.shape)?;
            Ok(floats(
                (0..n)
                    .map(|_| match s.init {
                        Initializer::Zeros => 0.,
                        Initializer::Constant { value } => value,
                        Initializer::Normal { stddev } => rng.normal() * stddev,
                    })
                    .collect(),
            ))
        })
        .collect()
}
fn batch(
    data: &BTreeMap<String, LoadedTensor>,
    rows: &[usize],
    state: &[Tensor],
    config: &Contract,
    lr: f32,
) -> Result<Vec<Tensor>> {
    let mut inputs = data
        .values()
        .map(|t| t.batch(rows, config.batch_size))
        .collect::<Result<Vec<_>>>()?;
    inputs.extend_from_slice(state);
    inputs.push(floats(vec![lr]));
    let mut valid = vec![0.; config.batch_size];
    valid[..rows.len()].fill(1.);
    inputs.push(floats(valid));
    Ok(inputs)
}
fn score(
    outputs: &[Tensor],
    data: &BTreeMap<String, LoadedTensor>,
    rows: &[usize],
    config: &Contract,
    bound: &BoundContract,
    confusion: &mut [Vec<usize>],
) -> Result<usize> {
    let Some(index) = bound.prediction else {
        return Ok(0);
    };
    let scores = outputs[index].f32()?;
    if scores.iter().any(|x| !x.is_finite()) {
        return Err("non-finite prediction".into());
    }
    let labels = &data[&config.classification.as_ref().unwrap().labels];
    let mut correct = 0;
    for (row, &sample) in rows.iter().enumerate() {
        let values = &scores[row * bound.classes..(row + 1) * bound.classes];
        let predicted = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .unwrap()
            .0;
        let label = labels.class_label(sample)?;
        if label >= bound.classes {
            return Err("classification label outside prediction class range".into());
        }
        correct += usize::from(label == predicted);
        if !confusion.is_empty() {
            confusion[label][predicted] += 1;
        }
    }
    Ok(correct)
}
fn evaluate(
    exe: &Executable,
    data: &BTreeMap<String, LoadedTensor>,
    state: &[Tensor],
    config: &Contract,
    bound: &BoundContract,
    threads: usize,
) -> Result<Value> {
    let count = data.values().next().unwrap().rows()?;
    let indices: Vec<_> = (0..count).collect();
    let mut loss = 0.;
    let mut correct = 0;
    let mut confusion = vec![vec![0; bound.classes]; bound.classes];
    for rows in indices.chunks(config.batch_size) {
        let out = exe.run_with_threads(&batch(data, rows, state, config, 0.)?, threads)?;
        let value = out[bound.loss].f32()?[0];
        if !value.is_finite() {
            return Err("non-finite loss".into());
        }
        loss += value as f64 * rows.len() as f64;
        correct += score(&out, data, rows, config, bound, &mut confusion)?;
    }
    Ok(
        json!({"loss":loss/count as f64,"examples":count,"accuracy":bound.prediction.map(|_|correct as f64/count as f64),"confusion_matrix":confusion}),
    )
}
fn write_json(path: &Path, value: &Value) -> Result<()> {
    fs::write(path, serde_json::to_string_pretty(value)? + "\n")?;
    Ok(())
}
fn save_state(path: &Path, state: &[Tensor], config: &Contract) -> Result<()> {
    use safetensors::tensor::{serialize, Dtype, TensorView};
    let bytes = state
        .iter()
        .map(|t| {
            Ok(t.f32()?
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>>>()?;
    let views = config
        .state
        .iter()
        .zip(&bytes)
        .map(|(s, data)| {
            Ok((
                s.input.as_str(),
                TensorView::new(Dtype::F32, s.shape.clone(), data)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    fs::write(path, serialize(views, None)?)?;
    Ok(())
}
pub fn run(options: Options) -> Result<()> {
    if options.epochs == 0
        || options.threads == 0
        || !options.learning_rate.is_finite()
        || options.learning_rate <= 0.
    {
        return Err("epochs, threads and learning-rate must be positive".into());
    }
    options.cpu_target.validate()?;
    let config_path = options
        .config
        .clone()
        .unwrap_or_else(|| options.source.with_extension("train.json"))
        .canonicalize()?;
    let mut config: Contract = serde_json::from_slice(&fs::read(&config_path)?)?;
    let load_start = Instant::now();
    let (train, test) = load_datasets(&mut config, config_path.parent().unwrap())?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    let source_text = fs::read_to_string(&options.source)?;
    let start = Instant::now();
    let ctx = context(&config, &train)?;
    let program = source::parse_with_context(&source_text, &ctx)?;
    let bound = bind_outputs(&config, &program, ctx, &train)?;
    // Validate held-out labels before starting a training run as well.
    if let Some(metric) = &config.classification {
        for row in 0..test[&metric.labels].rows()? {
            if test[&metric.labels].class_label(row)? >= bound.classes {
                return Err("test label outside prediction class range".into());
            }
        }
    }
    let exe = cpu::compile_profiled_with_options(
        &program,
        Path::new(".cache/pup/cpu"),
        &cpu::BuildOptions {
            cpu_target: options.cpu_target,
        },
    )?;
    let build_seconds = start.elapsed().as_secs_f64();
    let metadata = exe.profile_metadata.as_ref().unwrap();
    let output = options.output_dir.clone().unwrap_or_else(|| {
        PathBuf::from(format!(
            ".cache/train/run-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    });
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&output)?;
    let mut state = initial_state(&config, options.seed)?;
    let mut rng = Rng::new(options.seed);
    let count = train.values().next().unwrap().rows()?;
    println!(
        "{count} training / {} test rows; {} C kernels; {} workspace bytes",
        test.values().next().unwrap().rows()?,
        metadata["kernels"].as_array().unwrap().len(),
        metadata["workspace_bytes"]
    );
    let initial = evaluate(&exe, &test, &state, &config, &bound, options.threads)?;
    println!(
        "Initial test: loss {:.4}, accuracy {:?}",
        initial["loss"].as_f64().unwrap(),
        initial["accuracy"].as_f64().map(|x| x * 100.)
    );
    let warm_rows: Vec<_> = (0..count.min(config.batch_size)).collect();
    let warm = batch(&train, &warm_rows, &state, &config, options.learning_rate)?;
    for _ in 0..3 {
        exe.run_profiled(&warm, options.threads)?;
    }
    let mut profile = Profile::new(metadata);
    let mut history = Vec::new();
    let mut indices: Vec<_> = (0..count).collect();
    for epoch in 1..=options.epochs {
        let start = Instant::now();
        rng.shuffle(&mut indices);
        let mut loss = 0.;
        let mut correct = 0;
        for rows in indices.chunks(config.batch_size) {
            let inputs = batch(&train, rows, &state, &config, options.learning_rate)?;
            let call_start = Instant::now();
            let result = exe.run_profiled(&inputs, options.threads)?;
            profile.record(&result.counters, call_start.elapsed());
            let value = result.outputs[bound.loss].f32()?[0];
            if !value.is_finite() {
                return Err("non-finite training loss".into());
            }
            loss += value as f64 * rows.len() as f64;
            correct += score(&result.outputs, &train, rows, &config, &bound, &mut [])?;
            state = bound
                .outputs
                .iter()
                .map(|&i| result.outputs[i].clone())
                .collect();
        }
        let seconds = start.elapsed().as_secs_f64();
        let accuracy = bound.prediction.map(|_| correct as f64 / count as f64);
        let row = json!({"epoch":epoch,"train_loss":loss/count as f64,"train_accuracy":accuracy,"train_seconds":seconds,"examples_per_second":count as f64/seconds});
        println!(
            "Epoch {epoch}: loss {:.4}, accuracy {:?}, {:.2}s, {:.0} rows/s",
            loss / count as f64,
            accuracy.map(|x| x * 100.),
            seconds,
            count as f64 / seconds
        );
        history.push(row);
        write_json(&output.join("epochs.json"), &json!(history))?;
    }
    let final_test = evaluate(&exe, &test, &state, &config, &bound, options.threads)?;
    let metrics = profile.report();
    let mut build = serde_json::to_value(&exe.build_info)?;
    build["cache_hit"] = json!(exe.cache_hit);
    build["parse_compile_load_seconds"] = json!(build_seconds);
    build["c_source"] = json!(exe.source_path);
    build["c_source_bytes"] = json!(fs::metadata(&exe.source_path)?.len());
    let report = json!({"runtime":"generic Rust training host; generated C compute","contract":config,"source":options.source,"epochs":options.epochs,
        "learning_rate":options.learning_rate,"seed":options.seed,"rng":"xorshift64 + Box-Muller","threads":options.threads,"debug_build":cfg!(debug_assertions),
        "dataset_load_seconds":load_seconds,"input_bindings":bound.context.tensors.iter().map(|(name,spec)|json!({"name":name,"slot":spec.slot,"dtype":spec.dtype,"shape":spec.shape})).collect::<Vec<_>>(),
        "cpu":fs::read_to_string("/proc/cpuinfo").ok().and_then(|s|s.lines().find(|l|l.starts_with("model name")).map(str::to_owned)),
        "build":build,
        "initial_test":initial,"final_test":final_test,"history":history,"profile":metrics,
        "notes":["Training counters exclude dataset loading, compilation, evaluation and three warmups. Partial batches are zero-padded with a valid-row mask.",
        "Packing/compute are nested inside kernel elapsed; do not add them again.","Workspace is the generated tensor arena, excluding caller buffers, host data and thread stacks.",
        "Instrumentation overhead is included; host-call timing includes output allocation and FFI.","Evaluation runs the same training graph with learning_rate=0, discards updated state and includes backward work; no inference timing is reported."]});
    write_json(&output.join("metrics.json"), &report)?;
    write_json(&output.join("compiler-metadata.json"), metadata)?;
    fs::write(output.join("program.pup"), source_text)?;
    fs::copy(&exe.source_path, output.join("program.c"))?;
    write_json(
        &output.join("program.train.json"),
        &serde_json::to_value(&config)?,
    )?;
    save_state(&output.join("state.safetensors"), &state, &config)?;
    let mut csv =
        "kernel,op,bindings,calls,elapsed_ms,packing_ms,compute_ms,packed_bytes\n".to_owned();
    for k in metrics["kernels"].as_array().unwrap() {
        let names = k["bindings"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b["name"].as_str().unwrap())
            .collect::<Vec<_>>()
            .join(";");
        csv += &format!(
            "{},{},{},{},{},{},{},{}\n",
            k["id"],
            k["op"].as_str().unwrap(),
            names,
            k["calls"],
            k["elapsed_ms"],
            k["packing_ms"],
            k["compute_ms"],
            k["packed_bytes_total"]
        );
    }
    fs::write(output.join("kernels.csv"), csv)?;
    println!(
        "Final test: loss {:.4}, accuracy {:?}",
        final_test["loss"].as_f64().unwrap(),
        final_test["accuracy"].as_f64().map(|x| x * 100.)
    );
    println!(
        "Training-step median/p95: {:.3}/{:.3} ms",
        metrics["host_call_median_ms"].as_f64().unwrap(),
        metrics["host_call_p95_ms"].as_f64().unwrap()
    );
    println!("Results: {}", output.display());
    Ok(())
}
