"""
DPT dense prediction head (feature fusion + regression), as used by DUSt3R/UFM.

Trimmed from castacks/UniCeption (BSD-3-Clause), which follows CroCo v2 / DPT.
Attribute names are kept so released checkpoints load without key remapping.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvUnit(nn.Module):
    """Two 3x3 convolutions with a residual connection (ReLU before each conv)."""

    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.activation(x))
        out = self.conv2(self.activation(out))
        return out + x


class FeatureFusionBlock(nn.Module):
    """Fuse a coarser path with a skip feature, then upsample 2x."""

    def __init__(self, features: int, has_skip_unit: bool = True):
        """
        Args:
            features: Channel count of all fused features.
            has_skip_unit: Whether this block has the residual unit applied to the skip
                feature. The coarsest block receives no skip feature and omits it.
        """
        super().__init__()
        self.out_conv = nn.Conv2d(features, features, kernel_size=1)
        if has_skip_unit:
            self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, path: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        if skip is not None:
            path = path + self.resConfUnit1(skip)
        path = self.resConfUnit2(path)
        path = F.interpolate(path, scale_factor=2, mode="bilinear", align_corners=True)
        return self.out_conv(path)


class DPTFeature(nn.Module):
    """
    Turn four transformer feature maps (at patch resolution) into one feature map
    upsampled 8x relative to the patch grid.
    """

    def __init__(
        self,
        input_feature_dims: List[int],
        layer_dims: List[int],
        feature_dim: int,
    ):
        """
        Args:
            input_feature_dims: Channel counts of the four input feature maps.
            layer_dims: Channel counts of the four re-projected pyramid levels.
            feature_dim: Channel count of the fused output.
        """
        super().__init__()
        if len(input_feature_dims) != 4 or len(layer_dims) != 4:
            raise ValueError("DPTFeature expects exactly four input levels.")

        # Fusion blocks (coarse to fine). ``scratch`` mirrors the DPT reference layout.
        self.scratch = nn.Module()
        self.scratch.refinenet1 = FeatureFusionBlock(feature_dim)
        self.scratch.refinenet2 = FeatureFusionBlock(feature_dim)
        self.scratch.refinenet3 = FeatureFusionBlock(feature_dim)
        self.scratch.refinenet4 = FeatureFusionBlock(feature_dim, has_skip_unit=False)

        # Per-level resampling to build a 4x / 2x / 1x / 0.5x pyramid from the patch grid
        resample_layers = [
            nn.Sequential(
                nn.Conv2d(input_feature_dims[0], layer_dims[0], kernel_size=1),
                nn.ConvTranspose2d(layer_dims[0], layer_dims[0], kernel_size=4, stride=4),
            ),
            nn.Sequential(
                nn.Conv2d(input_feature_dims[1], layer_dims[1], kernel_size=1),
                nn.ConvTranspose2d(layer_dims[1], layer_dims[1], kernel_size=2, stride=2),
            ),
            nn.Sequential(nn.Conv2d(input_feature_dims[2], layer_dims[2], kernel_size=1)),
            nn.Sequential(
                nn.Conv2d(input_feature_dims[3], layer_dims[3], kernel_size=1),
                nn.Conv2d(layer_dims[3], layer_dims[3], kernel_size=3, stride=2, padding=1),
            ),
        ]
        # Per-level resampling followed by a 3x3 projection into the fused width
        self.input_process = nn.ModuleList(
            [
                nn.Sequential(resample, nn.Conv2d(layer_dim, feature_dim, kernel_size=3, padding=1, bias=False))
                for resample, layer_dim in zip(resample_layers, layer_dims)
            ]
        )

    def forward(self, layered_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            layered_features: Four feature maps (B, C_i, h, w) on the same patch grid.

        Returns:
            Fused feature map of shape (B, feature_dim, 8h, 8w).
        """
        if len(layered_features) != 4:
            raise ValueError(f"DPTFeature expects 4 feature maps, got {len(layered_features)}.")
        levels = [process(feature) for process, feature in zip(self.input_process, layered_features)]

        path_4 = self.scratch.refinenet4(levels[3])[:, :, : levels[2].shape[2], : levels[2].shape[3]]
        path_3 = self.scratch.refinenet3(path_4, levels[2])
        path_2 = self.scratch.refinenet2(path_3, levels[1])
        return self.scratch.refinenet1(path_2, levels[0])


class DPTRegressionProcessor(nn.Module):
    """Refine the 8x DPT feature and regress ``output_dim`` channels at the target resolution."""

    def __init__(self, input_feature_dim: int, hidden_dims: Tuple[int, int], output_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(input_feature_dim, hidden_dims[0], kernel_size=3, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dims[1], output_dim, kernel_size=1),
        )

    def forward(self, features_8x: torch.Tensor, output_shape_hw: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            features_8x: Fused DPT feature map (B, C, 8h, 8w).
            output_shape_hw: Target (H, W) of the dense prediction.

        Returns:
            Dense prediction of shape (B, output_dim, H, W).
        """
        x = self.conv1(features_8x)
        x = F.interpolate(x, size=output_shape_hw, mode="bilinear", align_corners=True)
        return self.conv2(x)


class DPTHead(nn.Module):
    """DPT feature fusion followed by dense regression."""

    def __init__(
        self,
        input_feature_dims: List[int],
        layer_dims: List[int],
        feature_dim: int,
        hidden_dims: Tuple[int, int],
        output_dim: int,
    ):
        super().__init__()
        self.dpt = DPTFeature(input_feature_dims, layer_dims, feature_dim)
        self.regressor = DPTRegressionProcessor(feature_dim, hidden_dims, output_dim)

    def forward(self, layered_features: List[torch.Tensor], output_shape_hw: Tuple[int, int]) -> torch.Tensor:
        return self.regressor(self.dpt(layered_features), output_shape_hw)
