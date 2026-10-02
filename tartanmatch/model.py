"""
TartanMatch: dense correspondence between any two of {RGB, depth, thermal, LiDAR, event} views.

Pipeline: modality-specific projection to a 3-channel image -> shared DINOv2 encoder ->
multi-view global-attention transformer -> DPT heads predicting flow and covisibility
for the source view.
"""

import os
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from tartanmatch.config import TartanMatchConfig
from tartanmatch.nn import DINOv2Encoder, DPTHead, MultiViewGlobalAttentionTransformer
from tartanmatch.preprocess import (
    depth_to_log_with_mask,
    events_to_voxel_grid,
    normalize_voxel_grid,
    resize_to,
    select_inference_resolution,
    unmap_dense_map,
    unmap_flow,
)

Modality = Literal["rgb", "depth", "thermal", "lidar", "event"]
MODALITIES: Tuple[Modality, ...] = ("rgb", "depth", "thermal", "lidar", "event")

# ImageNet statistics used by DINOv2; every modality is mapped into this space before the encoder.
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

DEFAULT_CHECKPOINT_FILENAME = "tartanmatch_v1.safetensors"

ArrayLike = Union[torch.Tensor, np.ndarray]


@dataclass
class TartanMatchOutput:
    """
    Dense correspondence prediction for the source view.

    Attributes:
        flow: (B, 2, H, W) float32. For source pixel (x, y), the matching target pixel is
            (x + flow[0], y + flow[1]), in target-image pixel units.
        covisibility: (B, H, W) float32 in [0, 1]; probability that the source pixel is
            visible in the target view.
    """

    flow: torch.Tensor
    covisibility: torch.Tensor


class TartanMatch(nn.Module):
    """Multimodal dense matching network."""

    def __init__(self, config: TartanMatchConfig = TartanMatchConfig()):
        super().__init__()
        self.config = config

        self.encoder = DINOv2Encoder(size=config.encoder_size, patch_size=config.patch_size)
        self.info_sharing = MultiViewGlobalAttentionTransformer(
            input_embed_dim=self.encoder.embed_dim,
            dim=config.info_sharing_dim,
            depth=config.info_sharing_depth,
            num_heads=config.info_sharing_num_heads,
            max_num_views=2,
            intermediate_layer_indices=list(config.info_sharing_intermediate_layers),
        )

        # Heads consume [encoder output, two intermediate transformer outputs, final transformer output]
        head_input_dims = [self.encoder.embed_dim] + [config.info_sharing_dim] * 3
        head_kwargs = dict(
            input_feature_dims=head_input_dims,
            layer_dims=list(config.dpt_layer_dims),
            feature_dim=config.dpt_feature_dim,
            hidden_dims=config.dpt_hidden_dims,
        )
        self.flow_head = DPTHead(output_dim=2, **head_kwargs)
        self.covisibility_head = DPTHead(output_dim=1, **head_kwargs)

        # Modality projections into the encoder's 3-channel input space
        self.depth_mask_conv = nn.Conv2d(2, 3, kernel_size=1)
        self.event_projection = nn.Conv2d(config.event_num_bins, 3, kernel_size=1)

        self.register_buffer("image_mean", torch.tensor(IMAGE_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(IMAGE_STD).view(1, 3, 1, 1), persistent=False)

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        device: Union[str, torch.device] = "cuda",
        config: TartanMatchConfig = TartanMatchConfig(),
    ) -> "TartanMatch":
        """
        Build the model and load released weights.

        Args:
            checkpoint: Local path to a ``.safetensors`` file, or a Hugging Face Hub repo id
                (e.g. ``"theairlabcmu/TartanMatch"``) from which ``tartanmatch_v1.safetensors`` is fetched.
            device: Device to place the model on.
            config: Architecture config matching the checkpoint.

        Returns:
            Model in eval mode on ``device``.
        """
        from safetensors.torch import load_file

        if not os.path.exists(checkpoint):
            from huggingface_hub import hf_hub_download

            checkpoint = hf_hub_download(repo_id=checkpoint, filename=DEFAULT_CHECKPOINT_FILENAME)

        model = cls(config)
        model.load_state_dict(load_file(checkpoint), strict=True)
        return model.to(device).eval()

    @property
    def device(self) -> torch.device:
        return self.image_mean.device

    # ------------------------------------------------------------- preprocessing
    def _project_to_encoder_input(self, x: torch.Tensor, modality: Modality) -> torch.Tensor:
        """
        Map one modality tensor (at model resolution) to a normalized 3-channel encoder input.

        Args:
            x: rgb/thermal (B, 3, H, W) in [0, 1]; depth/lidar (B, 1, H, W) metric depth;
                event (B, num_bins, H, W) voxel grid.
            modality: Which of the above ``x`` is.

        Returns:
            Tensor of shape (B, 3, H, W).
        """
        if modality in ("rgb", "thermal"):
            image = x
        elif modality in ("depth", "lidar"):
            image = self.depth_mask_conv(depth_to_log_with_mask(x))
        elif modality == "event":
            return self.event_projection(normalize_voxel_grid(x))
        else:
            raise ValueError(f"Unknown modality {modality!r}; expected one of {MODALITIES}.")
        return (image - self.image_mean) / self.image_std

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        source: torch.Tensor,
        source_modality: Modality,
        target: torch.Tensor,
        target_modality: Modality,
    ) -> TartanMatchOutput:
        """
        Predict flow and covisibility at the model's input resolution.

        Both inputs must already be at a supported inference resolution (see
        ``TartanMatchConfig.inference_resolutions``) and share the same shape.
        Use :meth:`predict` for arbitrary input sizes.

        Args:
            source: Source view tensor (see :meth:`_project_to_encoder_input` for per-modality format).
            source_modality: Modality of ``source``.
            target: Target view tensor.
            target_modality: Modality of ``target``.

        Returns:
            TartanMatchOutput at the input resolution.
        """
        if source.shape[-2:] != target.shape[-2:]:
            raise ValueError(f"Source and target must share a spatial shape, got {source.shape} and {target.shape}.")
        height, width = source.shape[-2:]

        # Shared encoder over both views in one batch
        encoder_input = torch.cat(
            [self._project_to_encoder_input(source, source_modality), self._project_to_encoder_input(target, target_modality)],
            dim=0,
        )
        source_features, target_features = self.encoder(encoder_input).chunk(2, dim=0)

        # Joint attention across views; keep the source-view branch for the heads
        final_features, intermediate_features = self.info_sharing([source_features, target_features])
        layered_features = [
            source_features.float().contiguous(),
            *[level[0].float().contiguous() for level in intermediate_features],
            final_features[0].float().contiguous(),
        ]

        # Heads run in full precision
        with torch.autocast(device_type=self.device.type, enabled=False):
            flow = self.flow_head(layered_features, (height, width))
            flow = flow * self._flow_scale(height, width)
            covisibility = torch.sigmoid(self.covisibility_head(layered_features, (height, width)))[:, 0]
        return TartanMatchOutput(flow=flow, covisibility=covisibility)

    def _flow_scale(self, height: int, width: int) -> torch.Tensor:
        """Per-axis factor turning the head's normalized output into pixels at (height, width)."""
        base_height, base_width = self.config.flow_base_shape
        scale = torch.tensor(
            [self.config.flow_std * width / base_width, self.config.flow_std * height / base_height],
            device=self.device,
        )
        return scale.view(1, 2, 1, 1)

    # ------------------------------------------------------------------ predict
    @torch.inference_mode()
    def predict(
        self,
        source: ArrayLike,
        source_modality: Modality,
        target: ArrayLike,
        target_modality: Modality,
        event_resolution: Optional[Tuple[int, int]] = None,
        mixed_precision: bool = True,
    ) -> TartanMatchOutput:
        """
        Predict dense correspondences between two views of arbitrary (possibly different) sizes.

        Accepted input formats per modality (a leading batch dimension is optional):
            rgb / thermal: (3, H, W) uint8 in [0, 255], or float in [0, 1]. Thermal is a 3-channel
                (grayscale replicated) image.
            depth / lidar: (1, H, W) float32 metric depth; invalid pixels are 0.
            event: (num_bins, H, W) float32 voxel grid, or raw events as a (N, 4) numpy array
                [timestamp, x, y, polarity] together with ``event_resolution=(H, W)``.

        Args:
            source: Source view.
            source_modality: Modality of ``source``.
            target: Target view.
            target_modality: Modality of ``target``.
            event_resolution: Sensor (H, W); required only for raw event arrays.
            mixed_precision: Run the backbone under float16 autocast on CUDA.

        Returns:
            TartanMatchOutput at the source view's original resolution; flow points into the
            target view's original pixel coordinates.
        """
        source_tensor = self._to_model_input(source, source_modality, event_resolution)
        target_tensor = self._to_model_input(target, target_modality, event_resolution)
        if source_tensor.shape[0] != target_tensor.shape[0]:
            raise ValueError("Source and target batch sizes differ.")
        source_hw, target_hw = tuple(source_tensor.shape[-2:]), tuple(target_tensor.shape[-2:])

        # Resize both views to the closest supported model resolution
        model_hw = select_inference_resolution(source_hw, target_hw, list(self.config.inference_resolutions))
        source_resized, target_resized = resize_to(source_tensor, model_hw), resize_to(target_tensor, model_hw)

        use_autocast = mixed_precision and self.device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_autocast):
            output = self.forward(source_resized, source_modality, target_resized, target_modality)

        # Map predictions back to the original resolutions
        flow = unmap_flow(output.flow, source_hw, target_hw)
        covisibility = unmap_dense_map(output.covisibility[:, None], source_hw)[:, 0]
        return TartanMatchOutput(flow=flow, covisibility=covisibility)

    def _to_model_input(
        self, data: ArrayLike, modality: Modality, event_resolution: Optional[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        Validate one view and convert it to a float32 (B, C, H, W) tensor on the model device.

        See :meth:`predict` for the accepted formats.
        """
        if modality not in MODALITIES:
            raise ValueError(f"Unknown modality {modality!r}; expected one of {MODALITIES}.")

        # Raw events -> voxel grid
        if modality == "event" and isinstance(data, np.ndarray) and data.ndim == 2:
            if event_resolution is None:
                raise ValueError("event_resolution=(H, W) is required when passing raw (N, 4) events.")
            data = events_to_voxel_grid(data, event_resolution[0], event_resolution[1], self.config.event_num_bins)

        tensor = torch.as_tensor(data)
        if tensor.dim() == 3:
            tensor = tensor[None]
        if tensor.dim() != 4:
            raise ValueError(f"{modality} input must be (C, H, W) or (B, C, H, W), got {tuple(tensor.shape)}.")

        expected_channels = {"rgb": 3, "thermal": 3, "depth": 1, "lidar": 1, "event": self.config.event_num_bins}[modality]
        if tensor.shape[1] != expected_channels:
            raise ValueError(f"{modality} input must have {expected_channels} channels, got {tensor.shape[1]}.")

        # Images: uint8 -> [0, 1]; everything else must already be float
        if modality in ("rgb", "thermal"):
            if tensor.dtype == torch.uint8:
                tensor = tensor.float() / 255.0
            elif tensor.is_floating_point():
                if tensor.min() < 0.0 or tensor.max() > 1.0:
                    raise ValueError(f"Float {modality} input must be in [0, 1].")
            else:
                raise ValueError(f"{modality} input must be uint8 or float, got {tensor.dtype}.")
        elif not tensor.is_floating_point():
            raise ValueError(f"{modality} input must be a float tensor, got {tensor.dtype}.")

        return tensor.to(device=self.device, dtype=torch.float32)
