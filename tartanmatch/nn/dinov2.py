"""
DINOv2 ViT encoder returning the last-layer patch tokens as a feature map.

The architecture is fetched from ``facebookresearch/dinov2`` via torch.hub with
``pretrained=False`` (the TartanMatch checkpoint already contains the fine-tuned
encoder weights, so the 1.1 GB DINOv2 weights are never downloaded). The hub
repository code itself is downloaded once and cached by torch.hub.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DINOV2_HUB_MODELS = {"small": "dinov2_vits14", "base": "dinov2_vitb14", "large": "dinov2_vitl14"}
DINOV2_EMBED_DIMS = {"small": 384, "base": 768, "large": 1024}


def _use_pytorch_sdpa_attention(attention_module: nn.Module) -> nn.Module:
    """Swap the DINOv2 attention forward for PyTorch's fused scaled-dot-product kernel."""

    class _SDPAAttention(attention_module.__class__):
        def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
            batch_size, num_tokens, channels = x.shape
            qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, channels // self.num_heads)
            query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            x = F.scaled_dot_product_attention(query, key, value, attn_bias)
            x = x.permute(0, 2, 1, 3).reshape(batch_size, num_tokens, channels)
            return self.proj_drop(self.proj(x))

    attention_module.__class__ = _SDPAAttention
    return attention_module


class DINOv2Encoder(nn.Module):
    """DINOv2 ViT backbone producing (B, embed_dim, H/patch, W/patch) features."""

    def __init__(self, size: str = "large", patch_size: int = 14):
        super().__init__()
        if size not in DINOV2_HUB_MODELS:
            raise ValueError(f"size must be one of {list(DINOV2_HUB_MODELS)}, got {size!r}.")
        self.patch_size = patch_size
        self.embed_dim = DINOV2_EMBED_DIMS[size]

        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            DINOV2_HUB_MODELS[size],
            pretrained=False,
            trust_repo=True,
            verbose=False,
        )
        # The mask token is only used for masked-image pretraining.
        del self.model.mask_token
        for block in self.model.blocks:
            block.attn = _use_pytorch_sdpa_attention(block.attn)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Normalized images of shape (B, 3, H, W); H and W must be multiples of the patch size.

        Returns:
            Last-layer patch features of shape (B, embed_dim, H // patch_size, W // patch_size).
        """
        _, channels, height, width = images.shape
        if channels != 3:
            raise ValueError(f"Encoder expects 3-channel input, got {channels}.")
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(f"Input size {(height, width)} must be divisible by patch size {self.patch_size}.")
        return self.model.get_intermediate_layers(images, n=1, reshape=True, norm=True)[0]
