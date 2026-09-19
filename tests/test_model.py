"""Tests for TartanMatch: preprocessing helpers and end-to-end shapes for every modality."""

import os

import numpy as np
import pytest
import torch

from tartanmatch import MODALITIES, TartanMatch, TartanMatchConfig
from tartanmatch.preprocess import events_to_voxel_grid, unmap_flow


def test_unmap_flow_identity_at_model_resolution():
    """Flow at the model resolution maps to itself when input and model sizes coincide."""
    flow = torch.randn(1, 2, 420, 560)
    assert torch.allclose(unmap_flow(flow, (420, 560), (420, 560)), flow, atol=1e-3)


def test_unmap_flow_scales_end_points_into_target_size():
    """When the target is twice as wide, the matched target x-coordinates double."""
    model_hw, source_hw, target_hw = (10, 20), (10, 20), (10, 40)
    flow = torch.zeros(1, 2, *model_hw)
    flow[:, 0] = 5.0  # every source pixel matches the target pixel 5 px to its right at model resolution
    unmapped = unmap_flow(flow, source_hw, target_hw)

    source_x = torch.arange(model_hw[1]).float() + 0.5
    expected_target_x = 2.0 * (source_x + 5.0)
    assert torch.allclose(source_x + unmapped[0, 0, 0], expected_target_x, atol=1e-4)
    assert torch.allclose(unmapped[0, 1], torch.zeros(model_hw), atol=1e-4)


def test_events_to_voxel_grid_integer_coordinates():
    """Integer events splat only in time: total mass equals the polarity sum."""
    events = np.array([[0.0, 3, 4, 1.0], [1.0, 3, 4, -1.0], [2.0, 5, 5, 1.0]], dtype=np.float32)
    voxel = events_to_voxel_grid(events, height=8, width=8, num_bins=3)
    assert voxel.shape == (3, 8, 8)
    assert np.isclose(voxel.sum(), 1.0)
    assert np.isclose(voxel[:, 4, 3].sum(), 0.0)  # +1 and -1 at the same pixel cancel


@pytest.fixture(scope="module")
def random_model():
    torch.manual_seed(0)
    return TartanMatch(TartanMatchConfig()).eval()


@pytest.mark.parametrize("source_modality", MODALITIES)
def test_predict_shapes_all_modalities(random_model, source_modality):
    """Every modality runs end to end and returns full-resolution outputs (random weights)."""
    height, width = 96, 128
    inputs = {
        "rgb": torch.randint(0, 256, (3, height, width), dtype=torch.uint8),
        "thermal": torch.randint(0, 256, (3, height, width), dtype=torch.uint8),
        "depth": torch.rand(1, height, width) * 10,
        "lidar": (torch.rand(1, height, width) > 0.9).float() * 5,
        "event": torch.randn(15, height, width),
    }
    output = random_model.predict(inputs[source_modality], source_modality, inputs["rgb"], "rgb")
    assert output.flow.shape == (1, 2, height, width)
    assert output.covisibility.shape == (1, height, width)
    assert torch.isfinite(output.flow).all()
    assert (output.covisibility >= 0).all() and (output.covisibility <= 1).all()


def test_raw_events_require_resolution(random_model):
    events = np.zeros((10, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="event_resolution"):
        random_model.predict(events, "event", events, "event")
