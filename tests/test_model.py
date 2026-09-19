"""
Tests for TartanMatch.

``test_reference_equivalence`` compares against outputs recorded from the original training
code; it is skipped unless TARTANMATCH_CKPT and TARTANMATCH_REFERENCE_DIR are set.
"""

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


@pytest.mark.skipif(
    not (os.environ.get("TARTANMATCH_CKPT") and os.environ.get("TARTANMATCH_REFERENCE_DIR")),
    reason="Set TARTANMATCH_CKPT and TARTANMATCH_REFERENCE_DIR to run the equivalence tests.",
)
@pytest.mark.parametrize(
    "reference_file, mixed_precision, max_flow_p99_px, max_covis_p99",
    [
        # Pure float32 with TF32 off: bit-exact for non-image pairs; image pairs differ only by the
        # order of normalization and resizing (about 1e-3 px).
        ("reference_fp32_notf32.npz", False, 0.01, 1e-3),
        # Float16 autocast (the default): the transformer amplifies ~1e-6 input rounding differences
        # to ~1 px at the 99th percentile; the original shows the same spread under 1e-6 perturbations.
        ("reference.npz", True, 1.5, 0.05),
    ],
)
def test_reference_equivalence(reference_file, mixed_precision, max_flow_p99_px, max_covis_p99):
    """Released model reproduces the original training-code outputs for all 25 modality pairs."""
    torch.backends.cuda.matmul.allow_tf32 = mixed_precision
    torch.backends.cudnn.allow_tf32 = mixed_precision
    reference_dir = os.environ["TARTANMATCH_REFERENCE_DIR"]
    inputs = np.load(os.path.join(reference_dir, "inputs.npz"))
    reference = np.load(os.path.join(reference_dir, reference_file))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TartanMatch.from_pretrained(os.environ["TARTANMATCH_CKPT"], device=device)

    worst_flow_error, worst_covis_error = 0.0, 0.0
    for source_modality in MODALITIES:
        for target_modality in MODALITIES:
            output = model.predict(
                inputs[source_modality],
                source_modality,
                inputs[target_modality],
                target_modality,
                mixed_precision=mixed_precision,
            )
            flow = output.flow[0].permute(1, 2, 0).cpu().numpy()
            covisibility = output.covisibility[0].cpu().numpy()
            flow_error = np.linalg.norm(flow - reference[f"flow_{source_modality}_{target_modality}"], axis=-1)
            covis_error = np.abs(covisibility - reference[f"covis_{source_modality}_{target_modality}"])
            worst_flow_error = max(worst_flow_error, float(np.percentile(flow_error, 99)))
            worst_covis_error = max(worst_covis_error, float(np.percentile(covis_error, 99)))
    print(f"[{reference_file}] worst 99th-percentile error over 25 pairs: flow {worst_flow_error:.4f} px, covisibility {worst_covis_error:.4f}")
    assert worst_flow_error < max_flow_p99_px
    assert worst_covis_error < max_covis_p99
