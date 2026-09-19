"""
Resizing inputs to the model's inference resolution and mapping predictions back.

The model always runs at a fixed resolution. Flow predicted at that resolution is
mapped back to the caller's source / target image sizes, so the returned flow is in
source-image pixels pointing at target-image pixels.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F


def select_inference_resolution(
    source_hw: Tuple[int, int], target_hw: Tuple[int, int], candidates_hw: List[Tuple[int, int]]
) -> Tuple[int, int]:
    """
    Pick the candidate (H, W) whose aspect ratio is closest to the input pair's aspect ratios.

    Args:
        source_hw: Source image (H, W).
        target_hw: Target image (H, W).
        candidates_hw: Supported model resolutions as (H, W) tuples.

    Returns:
        The selected (H, W).
    """
    source_aspect = source_hw[0] / source_hw[1]
    target_aspect = target_hw[0] / target_hw[1]
    return min(
        candidates_hw,
        key=lambda hw: abs(hw[0] / hw[1] - source_aspect) + abs(hw[0] / hw[1] - target_aspect),
    )


def resize_to(images: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    """Antialiased bilinear resize of a float (B, C, H, W) tensor to ``size_hw``."""
    return F.interpolate(images, size=size_hw, mode="bilinear", align_corners=False, antialias=True)


def unmap_flow(
    flow: torch.Tensor, source_hw: Tuple[int, int], target_hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Map flow predicted at the model resolution back to the original image resolutions.

    The flow is upsampled to the source resolution (nearest, to keep values crisp) and
    rescaled so that source coordinates live in the source image and end points live in
    the target image, which may have a different size.

    Args:
        flow: Predicted flow (B, 2, h, w) at model resolution, in model-resolution pixels.
        source_hw: Original source image (H_s, W_s).
        target_hw: Original target image (H_t, W_t).

    Returns:
        Flow of shape (B, 2, H_s, W_s) in original pixel units.
    """
    _, _, model_height, model_width = flow.shape
    device = flow.device

    # Pixel-center coordinates at model resolution, bilinearly upsampled to source resolution
    model_coords = torch.stack(
        torch.meshgrid(
            torch.arange(model_width, device=device) + 0.5,
            torch.arange(model_height, device=device) + 0.5,
            indexing="xy",
        ),
        dim=0,
    )[None].float()
    source_coords = F.interpolate(model_coords, size=source_hw, mode="bilinear", align_corners=False)
    target_coords = F.interpolate(flow.float(), size=source_hw, mode="nearest") + source_coords

    # Scale each coordinate set into its own image's pixel units
    source_scale = torch.tensor([source_hw[1] / model_width, source_hw[0] / model_height], device=device)
    target_scale = torch.tensor([target_hw[1] / model_width, target_hw[0] / model_height], device=device)
    source_coords = source_coords * source_scale.view(1, 2, 1, 1)
    target_coords = target_coords * target_scale.view(1, 2, 1, 1)
    return (target_coords - source_coords).to(flow.dtype)


def unmap_dense_map(values: torch.Tensor, source_hw: Tuple[int, int]) -> torch.Tensor:
    """Nearest-neighbour upsample a per-source-pixel map (B, C, h, w) to the source resolution."""
    return F.interpolate(values, size=source_hw, mode="nearest")
