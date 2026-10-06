//! Checkpoint binding and static stage specialization. Neural math lives in .pup.
use crate::compiler::{
    cpu::{self, Tensor},
    pop::DType,
    source::{self, Context, TensorSpec},
};
use crate::models::stable_diffusion::{AutoencoderKlConfig, ClipTextConfig, Unet2DConditionConfig};
use std::path::Path;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone)]
pub struct Weights {
    pub context: Context,
    pub inputs: Vec<Tensor>,
}
impl Weights {
    pub fn load(path: &Path, decoder_only: bool) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let store = safetensors::SafeTensors::deserialize(&bytes)?;
        let mut names = store.names();
        names.sort();
        let mut this = Self {
            context: Context::default(),
            inputs: vec![],
        };
        for name in names {
            if decoder_only
                && !name.starts_with("decoder.")
                && !name.starts_with("post_quant_conv.")
            {
                continue;
            }
            if name.ends_with("position_ids") {
                continue;
            }
            let tensor = store.tensor(name)?;
            let values = crate::models::safetensors::tensor_data_as_f32(name, &tensor)?;
            this.context.tensors.insert(
                name.to_string(),
                TensorSpec {
                    slot: this.inputs.len(),
                    dtype: DType::F32,
                    shape: tensor.shape().to_vec(),
                },
            );
            this.inputs.push(Tensor::F32(values.into()));
        }
        Ok(this)
    }
    pub fn bind(&mut self, name: &str, dtype: DType, shape: Vec<usize>) {
        let slot = if let Some(spec) = self.context.tensors.get(name) {
            spec.slot
        } else {
            self.inputs.push(Tensor::F32(vec![].into()));
            self.inputs.len() - 1
        };
        self.context
            .tensors
            .insert(name.into(), TensorSpec { slot, dtype, shape });
    }
    pub fn set(&mut self, name: &str, value: Tensor) {
        self.inputs[self.context.tensors[name].slot] = value;
    }
}
struct Builder<'a> {
    text: String,
    context: &'a Context,
    next: usize,
}
impl<'a> Builder<'a> {
    fn new(template: &str, context: &'a Context) -> Self {
        Self {
            text: template.into(),
            context,
            next: 0,
        }
    }
    fn emit(&mut self, expr: impl AsRef<str>) -> String {
        let name = format!("stage_v{}", self.next);
        self.next += 1;
        self.text += &format!("\n{name} = {}\n", expr.as_ref());
        name
    }
    fn has(&self, name: &str) -> bool {
        self.context.tensors.contains_key(name)
    }
    fn shape(&self, name: &str) -> Result<&[usize]> {
        self.context
            .tensors
            .get(name)
            .map(|s| s.shape.as_slice())
            .ok_or_else(|| format!("missing image checkpoint tensor {name}").into())
    }
    fn weight(&self, name: &str) -> String {
        format!("weight({name:?})")
    }
    fn bias(&self, prefix: &str) -> String {
        let name = format!("{prefix}.bias");
        if self.has(&name) {
            self.weight(&name)
        } else {
            "cast(0.0, f32)".into()
        }
    }
    fn linear(&mut self, x: &str, prefix: &str) -> String {
        self.emit(format!(
            "linear({x}, {}, {})",
            self.weight(&format!("{prefix}.weight")),
            self.bias(prefix)
        ))
    }
    fn conv(&mut self, x: &str, prefix: &str, stride: usize, padding: usize) -> String {
        self.emit(format!(
            "conv2d({x}, {}, {}, {stride}, {padding})",
            self.weight(&format!("{prefix}.weight")),
            self.bias(prefix)
        ))
    }
    fn norm(&mut self, x: &str, prefix: &str, groups: usize, eps: f32) -> String {
        self.emit(format!(
            "group_norm({x}, {}, {}, {groups}, {eps})",
            self.weight(&format!("{prefix}.weight")),
            self.bias(prefix)
        ))
    }
    fn ln(&mut self, x: &str, prefix: &str, eps: f32) -> String {
        self.emit(format!(
            "layer_norm({x}, {}, {}, {eps})",
            self.weight(&format!("{prefix}.weight")),
            self.bias(prefix)
        ))
    }
    fn fields(&self, prefix: &str, names: &[&str]) -> String {
        names
            .iter()
            .map(|n| self.weight(&format!("{prefix}.{n}")))
            .collect::<Vec<_>>()
            .join(", ")
    }
    fn finish(mut self, output: &str) -> Result<source::Program> {
        self.text += &format!("\noutput {output}\n");
        source::parse_with_context(&self.text, self.context).map_err(Into::into)
    }
    fn resnet(&mut self, x: &str, time: &str, prefix: &str, groups: usize, eps: f32) -> String {
        let fields = self.fields(
            prefix,
            &[
                "norm1.weight",
                "norm1.bias",
                "conv1.weight",
                "conv1.bias",
                "time_emb_proj.weight",
                "time_emb_proj.bias",
                "norm2.weight",
                "norm2.bias",
                "conv2.weight",
                "conv2.bias",
            ],
        );
        let h = self.emit(format!("sd_resnet({x}, {time}, {fields}, {groups}, {eps})"));
        let residual = if self.has(&format!("{prefix}.conv_shortcut.weight")) {
            self.conv(x, &format!("{prefix}.conv_shortcut"), 1, 0)
        } else {
            x.into()
        };
        self.emit(format!("{h} + {residual}"))
    }
    fn spatial(
        &mut self,
        x: &str,
        context: &str,
        prefix: &str,
        groups: usize,
        heads: usize,
    ) -> Result<String> {
        let n = self.norm(x, &format!("{prefix}.norm"), groups, 1e-6);
        let h = self.conv(&n, &format!("{prefix}.proj_in"), 1, 0);
        let mut h = self.emit(format!(
            "reshape(permute({h}, [0, 2, 3, 1]), [dim({h}, 2) * dim({h}, 3), dim({h}, 1)])"
        ));
        let mut block = 0;
        while self.has(&format!("{prefix}.transformer_blocks.{block}.norm1.weight")) {
            let p = format!("{prefix}.transformer_blocks.{block}");
            let fields = self.fields(
                &p,
                &[
                    "norm1.weight",
                    "norm1.bias",
                    "attn1.to_q.weight",
                    "attn1.to_k.weight",
                    "attn1.to_v.weight",
                    "attn1.to_out.0.weight",
                    "attn1.to_out.0.bias",
                    "norm2.weight",
                    "norm2.bias",
                    "attn2.to_q.weight",
                    "attn2.to_k.weight",
                    "attn2.to_v.weight",
                    "attn2.to_out.0.weight",
                    "attn2.to_out.0.bias",
                    "norm3.weight",
                    "norm3.bias",
                    "ff.net.0.proj.weight",
                    "ff.net.0.proj.bias",
                    "ff.net.2.weight",
                    "ff.net.2.bias",
                ],
            );
            h = self.emit(format!("sd_transformer({h}, {context}, {fields}, {heads})"));
            block += 1;
        }
        if block == 0 {
            return Err(format!("missing transformer blocks at {prefix}").into());
        }
        let h = self.emit(format!(
            "permute(reshape({h}, [1, dim({x}, 2), dim({x}, 3), dim({h}, 1)]), [0, 3, 1, 2])"
        ));
        let h = self.conv(&h, &format!("{prefix}.proj_out"), 1, 0);
        Ok(self.emit(format!("{x} + {h}")))
    }
    fn vae_resnet(&mut self, x: &str, prefix: &str, groups: usize) -> String {
        let fields = self.fields(
            prefix,
            &[
                "norm1.weight",
                "norm1.bias",
                "conv1.weight",
                "conv1.bias",
                "norm2.weight",
                "norm2.bias",
                "conv2.weight",
                "conv2.bias",
            ],
        );
        let h = self.emit(format!("sd_vae_resnet({x}, {fields}, {groups})"));
        let residual = if self.has(&format!("{prefix}.conv_shortcut.weight")) {
            self.conv(x, &format!("{prefix}.conv_shortcut"), 1, 0)
        } else {
            x.into()
        };
        self.emit(format!("{residual} + {h}"))
    }
    fn vae_projection(&mut self, x: &str, prefix: &str, old: &str, new: &str) -> Result<String> {
        let p = if self.has(&format!("{prefix}.{new}.weight")) {
            format!("{prefix}.{new}")
        } else {
            format!("{prefix}.{old}")
        };
        let shape = self.shape(&format!("{p}.weight"))?;
        let (oc, ic) = (shape[0], shape[1]);
        Ok(self.emit(format!(
            "linear({x}, reshape({}, [{oc}, {ic}]), {})",
            self.weight(&format!("{p}.weight")),
            self.bias(&p)
        )))
    }
}

pub fn clip(template: &str, w: &mut Weights, cfg: &ClipTextConfig) -> Result<source::Program> {
    w.bind("tokens", DType::I32, vec![cfg.max_position_embeddings]);
    let mut g = Builder::new(template, &w.context);
    let mut h=g.emit("load(index(weight(\"text_model.embeddings.token_embedding.weight\"), input(\"tokens\"))) + weight(\"text_model.embeddings.position_embedding.weight\")");
    for i in 0..cfg.num_hidden_layers {
        let p = format!("text_model.encoder.layers.{i}");
        let n = g.ln(&h, &format!("{p}.layer_norm1"), cfg.layer_norm_eps);
        let fields = g.fields(
            &format!("{p}.self_attn"),
            &[
                "q_proj.weight",
                "q_proj.bias",
                "k_proj.weight",
                "k_proj.bias",
                "v_proj.weight",
                "v_proj.bias",
                "out_proj.weight",
                "out_proj.bias",
            ],
        );
        let a = g.emit(format!(
            "sd_clip_attention({n}, {fields}, {})",
            cfg.num_attention_heads
        ));
        h = g.emit(format!("{h} + {a}"));
        let n = g.ln(&h, &format!("{p}.layer_norm2"), cfg.layer_norm_eps);
        let n = g.linear(&n, &format!("{p}.mlp.fc1"));
        let activation = if cfg.hidden_act == "gelu" {
            "gelu_erf"
        } else {
            "quick_gelu"
        };
        let n = g.emit(format!("{activation}({n})"));
        let n = g.linear(&n, &format!("{p}.mlp.fc2"));
        h = g.emit(format!("{h} + {n}"));
    }
    let h = g.ln(&h, "text_model.final_layer_norm", cfg.layer_norm_eps);
    g.finish(&h)
}
fn per_block(value: &serde_json::Value, index: usize, default: usize) -> Result<usize> {
    let n = if value.is_null() {
        default
    } else if let Some(n) = value.as_u64() {
        n as usize
    } else {
        value
            .get(index)
            .and_then(serde_json::Value::as_u64)
            .ok_or("invalid per-block configuration")? as usize
    };
    if n == 0 {
        return Err("zero per-block configuration".into());
    }
    Ok(n)
}
pub fn unet(
    template: &str,
    w: &mut Weights,
    cfg: &Unet2DConditionConfig,
    h: usize,
    width: usize,
    tokens: usize,
) -> Result<source::Program> {
    let block_count = cfg.block_out_channels.len();
    if cfg.down_block_types.len() != block_count || cfg.up_block_types.len() != block_count {
        return Err("UNet block counts disagree".into());
    }
    if cfg
        .down_block_types
        .iter()
        .any(|s| !matches!(s.as_str(), "DownBlock2D" | "CrossAttnDownBlock2D"))
        || cfg
            .up_block_types
            .iter()
            .any(|s| !matches!(s.as_str(), "UpBlock2D" | "CrossAttnUpBlock2D"))
    {
        return Err("unsupported UNet block type".into());
    }
    w.bind("sample", DType::F32, vec![1, 4, h, width]);
    w.bind(
        "conditioning",
        DType::F32,
        vec![tokens, cfg.cross_attention_dim],
    );
    w.bind("timestep", DType::F32, vec![]);
    let mut g = Builder::new(template, &w.context);
    let groups = cfg.norm_num_groups.unwrap_or(32);
    let first = cfg.block_out_channels[0];
    let half = first / 2;
    if first % 2 != 0 || half == 0 {
        return Err("time embedding width must be positive and even".into());
    }
    let frequencies = g.emit(format!(
        "exp(cast(arange({half}), f32) * -9.210340371976184 / {half}) * input(\"timestep\")"
    ));
    let time=g.emit(format!("reshape(pad(cos({frequencies}), [0], [{}]) + pad(sin({frequencies}), [{half}], [{}]), [1, {first}])",first,first));
    let time = g.linear(&time, "time_embedding.linear_1");
    let time = g.emit(format!("silu({time})"));
    let time = g.linear(&time, "time_embedding.linear_2");
    let mut x = g.conv("input(\"sample\")", "conv_in", 1, 1);
    let mut residuals = vec![x.clone()];
    for block in 0..block_count {
        let layers = per_block(&cfg.layers_per_block, block, 2)?;
        for layer in 0..layers {
            let p = format!("down_blocks.{block}.resnets.{layer}");
            x = g.resnet(&x, &time, &p, groups, 1e-5);
            if cfg.down_block_types[block] == "CrossAttnDownBlock2D" {
                x = g.spatial(
                    &x,
                    "input(\"conditioning\")",
                    &format!("down_blocks.{block}.attentions.{layer}"),
                    groups,
                    per_block(&cfg.attention_head_dim, block, 8)?,
                )?;
            }
            residuals.push(x.clone());
        }
        if block + 1 < block_count {
            x = g.conv(
                &x,
                &format!("down_blocks.{block}.downsamplers.0.conv"),
                2,
                1,
            );
            residuals.push(x.clone());
        }
    }
    x = g.resnet(&x, &time, "mid_block.resnets.0", groups, 1e-5);
    x = g.spatial(
        &x,
        "input(\"conditioning\")",
        "mid_block.attentions.0",
        groups,
        per_block(&cfg.attention_head_dim, block_count - 1, 8)?,
    )?;
    x = g.resnet(&x, &time, "mid_block.resnets.1", groups, 1e-5);
    for block in 0..block_count {
        let reverse = block_count - 1 - block;
        let layers = per_block(&cfg.layers_per_block, reverse, 2)? + 1;
        for layer in 0..layers {
            let skip = residuals.pop().ok_or("UNet skip underflow")?;
            x = g.emit(format!("concat_channels({x}, {skip})"));
            x = g.resnet(
                &x,
                &time,
                &format!("up_blocks.{block}.resnets.{layer}"),
                groups,
                1e-5,
            );
            if cfg.up_block_types[block] == "CrossAttnUpBlock2D" {
                x = g.spatial(
                    &x,
                    "input(\"conditioning\")",
                    &format!("up_blocks.{block}.attentions.{layer}"),
                    groups,
                    per_block(&cfg.attention_head_dim, reverse, 8)?,
                )?;
            }
        }
        if block + 1 < block_count {
            x = g.emit(format!("upsample2d({x})"));
            x = g.conv(&x, &format!("up_blocks.{block}.upsamplers.0.conv"), 1, 1);
        }
    }
    if !residuals.is_empty() {
        return Err("unused UNet skips".into());
    }
    x = g.norm(&x, "conv_norm_out", groups, 1e-5);
    x = g.emit(format!("silu({x})"));
    x = g.conv(&x, "conv_out", 1, 1);
    g.finish(&x)
}
pub fn vae(
    template: &str,
    w: &mut Weights,
    cfg: &AutoencoderKlConfig,
    h: usize,
    width: usize,
    pixels: bool,
) -> Result<source::Program> {
    w.bind("sample", DType::F32, vec![1, cfg.latent_channels, h, width]);
    let groups = cfg.norm_num_groups.ok_or("missing VAE groups")?;
    let mut g = Builder::new(template, &w.context);
    let mut x = g.emit(format!("input(\"sample\") / {}", cfg.scaling_factor));
    if g.has("post_quant_conv.weight") {
        x = g.conv(&x, "post_quant_conv", 1, 0);
    }
    x = g.conv(&x, "decoder.conv_in", 1, 1);
    x = g.vae_resnet(&x, "decoder.mid_block.resnets.0", groups);
    let p = "decoder.mid_block.attentions.0";
    if g.has(&format!("{p}.group_norm.weight")) {
        let n = g.norm(&x, &format!("{p}.group_norm"), groups, 1e-6);
        let n = g.emit(format!(
            "reshape(permute({n}, [0, 2, 3, 1]), [dim({n}, 2)*dim({n}, 3), dim({n}, 1)])"
        ));
        let q = g.vae_projection(&n, p, "query", "to_q")?;
        let k = g.vae_projection(&n, p, "key", "to_k")?;
        let v = g.vae_projection(&n, p, "value", "to_v")?;
        let a = g.emit(format!("attention({q}, {k}, {v}, 1)"));
        let a = g.vae_projection(&a, p, "proj_attn", "to_out.0")?;
        let a = g.emit(format!(
            "permute(reshape({a}, [1, dim({x}, 2), dim({x}, 3), dim({a}, 1)]), [0, 3, 1, 2])"
        ));
        x = g.emit(format!("{x} + {a}"));
    }
    x = g.vae_resnet(&x, "decoder.mid_block.resnets.1", groups);
    for block in 0..cfg.up_block_types.len() {
        if cfg.up_block_types[block] != "UpDecoderBlock2D" {
            return Err("unsupported VAE up block".into());
        }
        for layer in 0..cfg.layers_per_block.unwrap_or(2) + 1 {
            x = g.vae_resnet(
                &x,
                &format!("decoder.up_blocks.{block}.resnets.{layer}"),
                groups,
            );
        }
        if block + 1 < cfg.up_block_types.len() {
            x = g.emit(format!("upsample2d({x})"));
            x = g.conv(
                &x,
                &format!("decoder.up_blocks.{block}.upsamplers.0.conv"),
                1,
                1,
            );
        }
    }
    x = g.norm(&x, "decoder.conv_norm_out", groups, 1e-6);
    x = g.emit(format!("silu({x})"));
    x = g.conv(&x, "decoder.conv_out", 1, 1);
    if pixels {
        x = g.emit(format!("sd_pixels({x})"));
    }
    g.finish(&x)
}
pub fn step(template: &str, h: usize, w: usize) -> Result<(source::Program, Weights)> {
    let mut weights = Weights {
        context: Context::default(),
        inputs: vec![],
    };
    for name in ["sample", "uncond", "cond"] {
        weights.bind(name, DType::F32, vec![1, 4, h, w]);
    }
    for name in ["scale", "alpha", "previous_alpha"] {
        weights.bind(name, DType::F32, vec![]);
    }
    let mut b = Builder::new(template, &weights.context);
    let x=b.emit("sd_ddim(input(\"sample\"), input(\"uncond\"), input(\"cond\"), input(\"scale\"), input(\"alpha\"), input(\"previous_alpha\"))");
    Ok((b.finish(&x)?, weights))
}
pub fn compile(
    program: &source::Program,
    options: &cpu::BuildOptions,
    label: &str,
) -> Result<cpu::Executable> {
    let exe = cpu::compile_profiled_with_options(program, Path::new(".cache/pup/cpu"), options)?;
    eprintln!(
        "image {label}: {} kernels, {} workspace bytes, CPU {}; C: {}",
        exe.profile_metadata.as_ref().unwrap()["kernels"]
            .as_array()
            .unwrap()
            .len(),
        exe.profile_metadata.as_ref().unwrap()["workspace_bytes"],
        options.cpu_target,
        exe.source_path.display()
    );
    Ok(exe)
}
