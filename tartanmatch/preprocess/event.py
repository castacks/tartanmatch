"""
Event-camera preprocessing: raw events -> voxel grid, and voxel-grid standardization.
"""

import numpy as np
import torch


def events_to_voxel_grid(events: np.ndarray, height: int, width: int, num_bins: int = 15) -> np.ndarray:
    """
    Convert raw events to a voxel grid by trilinear splatting (E-RAFT convention).

    Each event's polarity is distributed over its two neighbouring temporal bins and,
    for non-integer coordinates (e.g. after rectification), its four neighbouring pixels.

    Args:
        events: (N, 4) array with columns [timestamp, x, y, polarity]; polarity in {-1, +1}.
        height: Sensor height in pixels.
        width: Sensor width in pixels.
        num_bins: Number of temporal bins.

    Returns:
        (num_bins, height, width) float32 voxel grid.
    """
    voxel = np.zeros((num_bins, height, width), dtype=np.float32)
    if events is None or len(events) == 0:
        return voxel
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"events must be (N, 4) [t, x, y, p], got {events.shape}.")

    timestamps = events[:, 0].astype(np.float64)
    x = events[:, 1].astype(np.float64)
    y = events[:, 2].astype(np.float64)
    polarity = events[:, 3].astype(np.float32)

    # Normalize timestamps to the continuous bin axis [0, num_bins - 1]
    t_min, t_max = timestamps.min(), timestamps.max()
    if t_max > t_min:
        t_norm = (timestamps - t_min) / (t_max - t_min) * (num_bins - 1)
    else:
        t_norm = np.zeros_like(timestamps)

    x0, y0, t0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64), np.floor(t_norm).astype(np.int64)
    wx, wy, wt = (x - x0).astype(np.float32), (y - y0).astype(np.float32), (t_norm - t0).astype(np.float32)

    # Splat every event onto its (up to) 8 neighbouring voxels
    for dx in (0, 1):
        for dy in (0, 1):
            for dt in (0, 1):
                xi, yi, ti = x0 + dx, y0 + dy, t0 + dt
                inside = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height) & (ti >= 0) & (ti < num_bins)
                weight = polarity * (wx if dx else 1.0 - wx) * (wy if dy else 1.0 - wy) * (wt if dt else 1.0 - wt)
                np.add.at(voxel, (ti[inside], yi[inside], xi[inside]), weight[inside])
    return voxel


def normalize_voxel_grid(voxel: torch.Tensor) -> torch.Tensor:
    """
    Standardize each sample's non-zero voxel entries to zero mean and unit variance.

    Zero entries (no event activity) stay exactly zero, so the sparsity pattern is
    preserved and only event magnitudes are rescaled.

    Args:
        voxel: Voxel grid of shape (B, num_bins, H, W).

    Returns:
        Normalized voxel grid with the same shape.
    """
    if voxel.dim() != 4:
        raise ValueError(f"voxel must be (B, num_bins, H, W), got {tuple(voxel.shape)}.")

    output = torch.zeros_like(voxel)
    for sample_index in range(voxel.shape[0]):
        sample = voxel[sample_index]
        nonzero_mask = sample != 0
        if not nonzero_mask.any():
            continue
        nonzero_values = sample[nonzero_mask]
        centered = nonzero_values - nonzero_values.mean()
        std = nonzero_values.std()
        output[sample_index][nonzero_mask] = centered / std if std > 0 else centered
    return output
