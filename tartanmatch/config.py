"""Architecture hyper-parameters of the released TartanMatch model."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class TartanMatchConfig:
    """
    Architecture configuration. The defaults describe the released ``tartanmatch_v1`` checkpoint.

    Attributes:
        encoder_size: DINOv2 ViT size ("small", "base", or "large").
        patch_size: ViT patch size in pixels.
        info_sharing_dim: Width of the multi-view global-attention transformer.
        info_sharing_depth: Number of global-attention blocks.
        info_sharing_num_heads: Attention heads per global-attention block.
        info_sharing_intermediate_layers: Block indices whose outputs feed the DPT heads
            alongside the encoder output and the final block output.
        dpt_layer_dims: Channel widths of the four DPT pyramid levels.
        dpt_feature_dim: Channel width of the fused DPT feature.
        dpt_hidden_dims: Hidden widths of the DPT regression convolutions.
        flow_std: Flow normalization scale (in pixels) at ``flow_base_shape``; the head's
            raw output is multiplied by ``flow_std`` scaled to the inference resolution.
        flow_base_shape: (H, W) at which ``flow_std`` is defined.
        event_num_bins: Temporal bins of the event voxel grid input.
        inference_resolutions: Supported (H, W) model input sizes; the one with the closest
            aspect ratio to the inputs is used. Each must be divisible by ``patch_size``.
    """

    encoder_size: str = "large"
    patch_size: int = 14
    info_sharing_dim: int = 768
    info_sharing_depth: int = 12
    info_sharing_num_heads: int = 12
    info_sharing_intermediate_layers: Tuple[int, ...] = (5, 8)
    dpt_layer_dims: Tuple[int, int, int, int] = (96, 192, 384, 768)
    dpt_feature_dim: int = 256
    dpt_hidden_dims: Tuple[int, int] = (128, 128)
    flow_std: float = 25.0
    flow_base_shape: Tuple[int, int] = (224, 224)
    event_num_bins: int = 15
    inference_resolutions: Tuple[Tuple[int, int], ...] = ((420, 560),)
