#!/usr/bin/env python3
"""
Convert a UFM training checkpoint (PyTorch Lightning ``.ckpt``) into a TartanMatch release checkpoint.

The Lightning checkpoint stores the model under ``state_dict`` with a ``model.`` prefix
and the training repo's module names. This script strips the prefix, renames the head
modules to TartanMatch's names, drops constant buffers that TartanMatch derives from its
config, validates the result by strictly loading it into ``TartanMatch``, and writes a
``.safetensors`` file.

Usage:
    python scripts/convert_lightning_checkpoint.py <lightning.ckpt> <output.safetensors>
"""

import argparse
from typing import Dict

import torch
from safetensors.torch import save_file

from tartanmatch import TartanMatch, __version__

# (training-repo prefix, TartanMatch prefix)
KEY_PREFIX_RENAMES = (
    ("head1.0.0.", "flow_head.dpt."),
    ("head1.0.1.", "flow_head.regressor."),
    ("uncertainty_head.0.0.", "covisibility_head.dpt."),
    ("uncertainty_head.0.1.", "covisibility_head.regressor."),
)

# Constant buffers recomputed from TartanMatchConfig; not stored in the release file.
DROPPED_KEYS = {
    "head1.1.adaptors.flow.flow_mean",
    "head1.1.adaptors.flow.flow_std",
    "info_sharing.view_pos_table",
}


def rename_key(key: str) -> str:
    """Map one training-repo state-dict key to its TartanMatch name."""
    for old_prefix, new_prefix in KEY_PREFIX_RENAMES:
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix) :]
    return key


def convert_state_dict(lightning_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Strip the Lightning ``model.`` prefix, rename heads, and drop config-derived buffers.

    Args:
        lightning_state_dict: The ``state_dict`` entry of a Lightning checkpoint.

    Returns:
        State dict loadable into ``TartanMatch`` with ``strict=True``.
    """
    converted = {}
    for key, value in lightning_state_dict.items():
        if not key.startswith("model."):
            continue
        key = key[len("model.") :]
        if key in DROPPED_KEYS:
            continue
        converted[rename_key(key)] = value.contiguous()
    drop_duplicate_dpt_projections(converted)
    return converted


def drop_duplicate_dpt_projections(state_dict: Dict[str, torch.Tensor]) -> None:
    """
    Remove the training repo's duplicate registrations of the DPT level projections.

    The training code registers each level's 3x3 projection conv three times
    (``scratch.layer{i}_rn``, ``scratch.layer_rn.{i}`` and ``input_process.{i}.1``).
    TartanMatch keeps only ``input_process.{i}.1``; the copies are verified equal and dropped.

    Raises:
        ValueError: If a duplicate differs from the kept tensor.
    """
    for head in ("flow_head", "covisibility_head"):
        for level in range(4):
            kept_key = f"{head}.dpt.input_process.{level}.1.weight"
            for duplicate_key in (
                f"{head}.dpt.scratch.layer{level + 1}_rn.weight",
                f"{head}.dpt.scratch.layer_rn.{level}.weight",
            ):
                if not torch.equal(state_dict[duplicate_key], state_dict[kept_key]):
                    raise ValueError(f"{duplicate_key} differs from {kept_key}; refusing to drop it.")
                del state_dict[duplicate_key]


def check_dropped_buffers_match_config(model: TartanMatch, lightning_state_dict: Dict[str, torch.Tensor]) -> None:
    """
    Verify that the buffers dropped from the checkpoint equal what TartanMatch derives from its config.

    Raises:
        ValueError: If a dropped buffer differs from the model's own value.
    """
    model_height, model_width = model.config.inference_resolutions[0]
    flow_scale = model._flow_scale(model_height, model_width).cpu()
    base_height, base_width = model.config.flow_base_shape
    trained_std = lightning_state_dict["model.head1.1.adaptors.flow.flow_std"].cpu()
    trained_scale = trained_std * torch.tensor([model_width / base_width, model_height / base_height]).view(1, 2, 1, 1)
    expected = {
        "model.info_sharing.view_pos_table": model.info_sharing.view_pos_table.cpu(),
        "model.head1.1.adaptors.flow.flow_mean": torch.zeros(1, 2, 1, 1),
    }
    for key, model_value in expected.items():
        if not torch.allclose(lightning_state_dict[key].cpu().float(), model_value.float(), atol=1e-6):
            raise ValueError(f"Checkpoint buffer {key} does not match the value derived from TartanMatchConfig.")
    if not torch.allclose(trained_scale.float(), flow_scale.float(), atol=1e-6):
        raise ValueError("Checkpoint flow_std does not match TartanMatchConfig.flow_std.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lightning_ckpt", help="Path to the Lightning .ckpt file.")
    parser.add_argument("output", help="Output .safetensors path.")
    args = parser.parse_args()

    checkpoint = torch.load(args.lightning_ckpt, map_location="cpu", weights_only=False, mmap=True)
    state_dict = convert_state_dict(checkpoint["state_dict"])

    # Validate against the architecture before writing anything
    model = TartanMatch()
    model.load_state_dict(state_dict, strict=True)
    check_dropped_buffers_match_config(model, checkpoint["state_dict"])
    num_params = sum(tensor.numel() for tensor in state_dict.values())
    print(f"Converted {len(state_dict)} tensors ({num_params / 1e6:.1f}M parameters); strict load OK.")

    metadata = {
        "format": "pt",
        "tartanmatch_version": __version__,
        "source_epoch": str(checkpoint.get("epoch", "")),
        "source_global_step": str(checkpoint.get("global_step", "")),
    }
    save_file(state_dict, args.output, metadata=metadata)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
