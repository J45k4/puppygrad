mod error;
mod kernels;
mod model;
mod weights;

pub use error::{Result, VitsError};
pub use kernels::{
    channel_layer_norm_in_place, conv1d, conv_transpose1d, elementwise_affine, flip_channels,
    gated_tanh_sigmoid, leaky_relu_in_place, residual_coupling_reverse, same_padding, Conv1dParams,
    ConvTranspose1dParams,
};
pub use model::{
    debug_synthesize_phoneme_ids, duration_path, expand_by_durations, infer_frame_count,
    log_durations_to_durations, DeterministicDurationPredictorWeights, DeterministicRng,
    VitsSynthesisScales,
};
pub use weights::{
    ConvWeights, DdsConvLayerWeights, DdsConvWeights, DenseTensor, DurationFlowWeights,
    GeneratorResBlockWeights, GeneratorWeights, LayerNormWeights, ResidualCouplingBlockWeights,
    ResidualCouplingFlowWeights, StochasticDurationPredictorWeights,
    TextEncoderAttentionLayerWeights, TextEncoderFfnLayerWeights, TextEncoderWeights,
    VitsWeightConfig, VitsWeights, WavenetWeights,
};
