#!/usr/bin/env python3
"""
Run TartanMatch on the bundled TartanAir-V2 example (five frames of one trajectory, each in a
different modality) and save, for every ordered modality pair, the predicted flow (colour
coded) and the target's RGB frame warped into the source frame, masked by covisibility.

Usage:
    python examples/demo.py --checkpoint checkpoints/tartanmatch_v1.safetensors
    python examples/demo.py --checkpoint castacks/tartanmatch --pairs rgb:event thermal:lidar
"""

import argparse
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from tartanmatch import MODALITIES, TartanMatch, TartanMatchOutput

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "oldbrickhouseday")
EVENT_SENSOR_HW = (640, 640)


def load_inputs(asset_dir: str) -> Dict[str, np.ndarray]:
    """
    Load one example input per modality in the format ``TartanMatch.predict`` accepts.

    Returns:
        rgb / thermal: (3, H, W) uint8; depth / lidar: (1, H, W) float32 metric depth;
        event: (N, 4) float32 raw events [t, x, y, polarity].
    """
    read_rgb = lambda name: cv2.cvtColor(cv2.imread(os.path.join(asset_dir, name)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    return {
        "rgb": read_rgb("rgb.png"),
        "thermal": read_rgb("thermal.png"),
        "depth": np.load(os.path.join(asset_dir, "depth.npy"))[None],
        "lidar": np.load(os.path.join(asset_dir, "lidar.npy"))[None],
        "event": np.load(os.path.join(asset_dir, "events.npy")),
    }


def load_frame_rgb(asset_dir: str, modality: str) -> np.ndarray:
    """RGB render (H, W, 3) uint8 of the frame that was captured in ``modality`` (visualisation only)."""
    name = "rgb.png" if modality == "rgb" else f"frame_rgb_{modality}.png"
    return cv2.cvtColor(cv2.imread(os.path.join(asset_dir, name)), cv2.COLOR_BGR2RGB)


def flow_to_color(flow: np.ndarray, max_magnitude: float) -> np.ndarray:
    """Colour-code a (H, W, 2) flow field: hue = direction, saturation = magnitude."""
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (angle / 2).astype(np.uint8)
    hsv[..., 1] = np.clip(magnitude / max_magnitude * 255, 0, 255).astype(np.uint8)
    hsv[..., 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def warp_target_into_source(target_rgb: np.ndarray, flow: np.ndarray, covisibility: np.ndarray) -> np.ndarray:
    """Sample the target image at (x + flow_x, y + flow_y); pixels predicted as not covisible are black."""
    height, width = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    warped = cv2.remap(target_rgb, grid_x + flow[..., 0], grid_y + flow[..., 1], interpolation=cv2.INTER_LINEAR)
    return (warped * (covisibility[..., None] > 0.5)).astype(np.uint8)


def parse_pairs(pair_args: List[str]) -> List[Tuple[str, str]]:
    """Parse ``source:target`` strings; default to every ordered pair."""
    if not pair_args:
        return [(source, target) for source in MODALITIES for target in MODALITIES if source != target]
    pairs = []
    for pair in pair_args:
        source, target = pair.split(":")
        if source not in MODALITIES or target not in MODALITIES:
            raise ValueError(f"Unknown modality in pair {pair!r}; expected two of {MODALITIES}.")
        pairs.append((source, target))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to tartanmatch_v1.safetensors or a Hub repo id.")
    parser.add_argument("--assets", default=ASSET_DIR, help="Directory with the example inputs.")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(ASSET_DIR)), "outputs"), help="Output directory.")
    parser.add_argument("--pairs", nargs="*", default=[], help="Ordered source:target pairs (default: all).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = TartanMatch.from_pretrained(args.checkpoint, device=args.device)
    inputs = load_inputs(args.assets)

    for source_modality, target_modality in parse_pairs(args.pairs):
        output: TartanMatchOutput = model.predict(
            inputs[source_modality],
            source_modality,
            inputs[target_modality],
            target_modality,
            event_resolution=EVENT_SENSOR_HW,
        )
        flow = output.flow[0].permute(1, 2, 0).cpu().numpy()
        covisibility = output.covisibility[0].cpu().numpy()

        flow_image = flow_to_color(flow, max_magnitude=max(np.percentile(np.linalg.norm(flow, axis=-1), 95), 1.0))
        warped = warp_target_into_source(load_frame_rgb(args.assets, target_modality), flow, covisibility)
        panel = np.concatenate([load_frame_rgb(args.assets, source_modality), flow_image, warped], axis=1)
        out_path = os.path.join(args.out, f"{source_modality}_to_{target_modality}.png")
        cv2.imwrite(out_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print(f"{source_modality:>7} -> {target_modality:<7} mean covisibility {covisibility.mean():.2f}  saved {out_path}")


if __name__ == "__main__":
    main()
