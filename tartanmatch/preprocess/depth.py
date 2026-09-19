"""
Depth / LiDAR preprocessing: metric depth -> (log depth, validity mask) two-channel input.
"""

import torch


def percentile_normalize(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    p_low: float = 0.025,
    p_high: float = 0.975,
) -> torch.Tensor:
    """
    Map ``values`` to [0, 1] using the p_low / p_high percentiles of the valid entries.

    Entries outside the percentile range and invalid entries become 0.

    Args:
        values: Tensor of any shape.
        valid_mask: Boolean tensor of the same shape marking usable entries.
        p_low: Lower percentile in [0, 1].
        p_high: Upper percentile in [0, 1].

    Returns:
        Normalized tensor with the same shape as ``values``.
    """
    valid_values = values[valid_mask]
    if valid_values.numel() == 0:
        return torch.zeros_like(values)

    sorted_values = torch.sort(valid_values)[0]
    last_index = sorted_values.numel() - 1
    low = sorted_values[int(p_low * last_index)]
    high = sorted_values[int(p_high * last_index)]
    if high <= low:
        return torch.zeros_like(values)

    normalized = (values - low) / (high - low)
    normalized[(normalized < 0) | (normalized > 1)] = 0.0
    normalized[~valid_mask] = 0.0
    return normalized


def depth_to_log_with_mask(depth: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of metric depth maps into the model's two-channel depth input.

    Channel 0 is percentile-normalized log1p(depth) over valid pixels; channel 1 is the
    validity mask (depth > 0 and finite). Normalization statistics are per sample.

    Args:
        depth: Metric depth of shape (B, 1, H, W). Zero, negative, or non-finite values are invalid.

    Returns:
        Tensor of shape (B, 2, H, W) in [0, 1].
    """
    if depth.dim() != 4 or depth.shape[1] != 1:
        raise ValueError(f"depth must be (B, 1, H, W), got {tuple(depth.shape)}.")

    outputs = []
    for sample in depth[:, 0]:
        valid_mask = (sample > 0) & torch.isfinite(sample)
        log_depth = torch.zeros_like(sample)
        log_depth[valid_mask] = torch.log1p(sample[valid_mask])
        normalized = percentile_normalize(log_depth, valid_mask)
        outputs.append(torch.stack([normalized, valid_mask.to(sample.dtype)], dim=0))
    return torch.stack(outputs, dim=0)
