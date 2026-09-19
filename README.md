# TartanMatch: Towards Universal Dense Correspondence Across Modalities

Hyeokjoon Kwon, Jiting Cai, Ruogu Li, Kritan Bhandari, Geethika Hemkumar, Parv Maheshwari, Yuheng Qiu,
Yuchen Zhang, Sebastian Scherer, Wenshan Wang

Dense correspondence between any two views captured in **RGB, depth, thermal, LiDAR, or event** modalities.
Given a source view and a target view (same or different modality), TartanMatch predicts, for every source
pixel, the matching target pixel (optical flow) and the probability that the pixel is visible in the target
(covisibility).

TartanMatch is the multimodal successor of [UFM](https://uniflowmatch.github.io/): a shared DINOv2 encoder,
a multi-view global-attention transformer, and DPT heads. Each non-RGB modality is mapped into the encoder's
image space by a small learned projection, so a single network serves all 25 modality pairs.

## Installation

```bash
git clone https://github.com/castacks/TartanMatch.git
cd TartanMatch
pip install -e .            # core: torch, numpy, safetensors, huggingface_hub
pip install -e ".[demo]"    # adds opencv-python, matplotlib for the example script
```

The DINOv2 architecture definition is fetched once from `facebookresearch/dinov2` through `torch.hub`
(code only; no DINOv2 weights are downloaded because the TartanMatch checkpoint already contains the
fine-tuned encoder).

## Checkpoint

| Name | Description | Params |
|---|---|---|
| `tartanmatch_v1.safetensors` | All-modality model, ViT-L/14 encoder, 420x560 inference resolution | 441.5M |

Download it from the [GitHub release](https://github.com/castacks/TartanMatch/releases) into `checkpoints/`,
or pass a Hugging Face Hub repo id to `from_pretrained` once the weights are hosted there.

## Quick start

```python
import cv2, numpy as np
from tartanmatch import TartanMatch

model = TartanMatch.from_pretrained("checkpoints/tartanmatch_v1.safetensors", device="cuda")

asset = "examples/assets/oldbrickhouseday"
read_rgb = lambda name: cv2.cvtColor(cv2.imread(f"{asset}/{name}"), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
rgb = read_rgb("rgb.png")                      # (3, H, W) uint8
thermal = read_rgb("thermal.png")              # (3, H, W) uint8
lidar = np.load(f"{asset}/lidar.npy")[None]    # (1, H, W) float32 metric depth, 0 = no return
events = np.load(f"{asset}/events.npy")        # (N, 4) raw events [t, x, y, polarity]

out = model.predict(rgb, "rgb", events, "event", event_resolution=(640, 640))
out = model.predict(lidar, "lidar", thermal, "thermal")

flow = out.flow[0]                 # (2, H, W): target pixel = source pixel + flow
covisibility = out.covisibility[0] # (H, W) in [0, 1]
```

Any of the five modalities can be the source or the target. `predict` accepts arbitrary (and different)
source / target sizes; inputs are resized to the model resolution internally and the flow is returned in
the original pixel units of the source and target images.

### Input formats

| Modality | Format passed to `predict` |
|---|---|
| `rgb` | `(3, H, W)` uint8, or float in [0, 1] |
| `thermal` | `(3, H, W)` uint8 (grayscale thermal replicated or false-colour), or float in [0, 1] |
| `depth` | `(1, H, W)` float32 metric depth, 0 where invalid |
| `lidar` | `(1, H, W)` float32 LiDAR depth projected into the camera, 0 where there is no return |
| `event` | `(15, H, W)` float32 voxel grid, **or** raw events `(N, 4)` float `[timestamp, x, y, polarity]` with `event_resolution=(H, W)` |

A leading batch dimension is optional for every modality. Depth and LiDAR are normalized per sample
(percentile-scaled log depth), so any consistent metric unit works. Raw events are voxelized with
`tartanmatch.preprocess.events_to_voxel_grid` (trilinear splatting into 15 temporal bins).

## Examples

`examples/assets/oldbrickhouseday/` holds five consecutive frames of one TartanAir-V2 trajectory, each in a
different modality, so every pair of frames is a cross-modal matching problem. See
[`examples/README.md`](examples/README.md) for how each modality is loaded.

```bash
python examples/demo.py --checkpoint checkpoints/tartanmatch_v1.safetensors                    # rgb→event, event→depth, depth→thermal, thermal→lidar, lidar→rgb
python examples/demo.py --checkpoint checkpoints/tartanmatch_v1.safetensors --all              # all 20 cross-modality pairs
python examples/demo.py --checkpoint checkpoints/tartanmatch_v1.safetensors --pairs rgb:lidar event:thermal
```

Each pair produces `examples/outputs/<source>_to_<target>.png`: the source view in its native modality, colour-coded flow, and the
target frame warped into the source frame (black where the model predicts the pixel is not covisible).
Reference panels for the default pairs are in [`examples/expected_outputs/`](examples/expected_outputs/).

![examples](examples/assets/preview.jpg)

## Numerical notes

`predict` runs the backbone under float16 autocast on CUDA (`mixed_precision=True`, the default), matching
how the model was evaluated. With `mixed_precision=False` and TF32 disabled, TartanMatch reproduces the
training code's float32 outputs to within 0.01 px on every modality pair (bit-exact for non-image pairs).

## Converting a training checkpoint

Checkpoints from the Multimodal-UFM training code are converted with:

```bash
python scripts/convert_lightning_checkpoint.py path/to/last.ckpt checkpoints/tartanmatch_v1.safetensors
```

The script strips the Lightning wrapper, renames the heads, verifies every dropped or deduplicated tensor
against the model definition, and refuses to write anything that does not load strictly.

## Tests

```bash
pytest tests                                            # architecture / preprocessing tests, no weights needed
TARTANMATCH_CKPT=checkpoints/tartanmatch_v1.safetensors \
TARTANMATCH_REFERENCE_DIR=path/to/reference pytest tests  # also checks equivalence with the training code
```

## License

The code is released under the [BSD-3-Clause license](LICENSE). The model weights inherit the licenses of
the training datasets and may not be used for commercial purposes.

## Acknowledgements

TartanMatch builds on [UFM](https://github.com/UniFlowMatch/UFM), [UniCeption](https://github.com/castacks/UniCeption),
[DINOv2](https://github.com/facebookresearch/dinov2), and [DUSt3R](https://github.com/naver/dust3r).

## Citation

If you use TartanMatch, please cite:

```bibtex
@article{kwon2026tartanmatch,
 title={TartanMatch: Towards Universal Dense Correspondence Across Modalities},
 author={Kwon, Hyeokjoon and Cai, Jiting and Li, Ruogu and Bhandari, Kritan and Hemkumar, Geethika and Maheshwari, Parv and Qiu, Yuheng and Zhang, Yuchen and Scherer, Sebastian and Wang, Wenshan},
 journal={arXiv preprint},
 year={2026}
}
```

TartanMatch builds on UFM; please consider citing it as well:

```bibtex
@article{zhang2025ufm,
 title={UFM: A Simple Path towards Unified Dense Correspondence with Flow},
 author={Zhang, Yuchen and Keetha, Nikhil and Lyu, Chenwei and Jhamb, Bhuvan and Chen, Yutian and Qiu, Yuheng and Karhade, Jay and Jha, Shreyas and Hu, Yaoyu and Ramanan, Deva and Scherer, Sebastian and Wang, Wenshan},
 journal={arXiv preprint arXiv:2506.09278},
 year={2025}
}
```
