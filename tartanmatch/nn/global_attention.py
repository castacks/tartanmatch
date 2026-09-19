"""
Multi-view global-attention transformer (the UFM "information sharing" stage).

Tokens of all views are concatenated and processed jointly by self-attention
blocks. Trimmed from castacks/UniCeption (BSD-3-Clause); attribute names are
kept so released checkpoints load without key remapping.
"""

from functools import partial
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from tartanmatch.nn.transformer import SelfAttentionBlock


def sinusoid_encoding_table(num_positions: int, dim: int, base: float = 10000.0) -> torch.Tensor:
    """
    Fixed sinusoidal positional encoding table.

    Args:
        num_positions: Number of positions (rows).
        dim: Encoding dimension (columns).
        base: Frequency base.

    Returns:
        Float tensor of shape (num_positions, dim).
    """
    position = np.arange(num_positions)[:, None]
    channel = np.arange(dim)[None, :]
    angles = position / np.power(base, 2 * (channel // 2) / dim)
    table = np.zeros_like(angles)
    table[:, 0::2] = np.sin(angles[:, 0::2])
    table[:, 1::2] = np.cos(angles[:, 1::2])
    return torch.tensor(table, dtype=torch.float32)


class MultiViewGlobalAttentionTransformer(nn.Module):
    """
    Joint self-attention over the patch tokens of several views.

    Each view receives a fixed per-view positional embedding (so the network can
    tell the reference view from the others) before all tokens are concatenated
    and passed through ``depth`` self-attention blocks.
    """

    def __init__(
        self,
        input_embed_dim: int,
        dim: int,
        depth: int,
        num_heads: int,
        max_num_views: int,
        intermediate_layer_indices: List[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        norm_intermediate: bool = True,
    ):
        """
        Args:
            input_embed_dim: Channel dimension of the incoming encoder features.
            dim: Transformer width.
            depth: Number of self-attention blocks.
            num_heads: Attention heads per block.
            max_num_views: Size of the per-view positional embedding table.
            intermediate_layer_indices: Block indices whose outputs are also returned
                (used as multi-scale inputs by the DPT heads).
            mlp_ratio: MLP hidden width relative to ``dim``.
            qkv_bias: Whether the qkv projection has a bias.
            qk_norm: Whether to layer-normalize queries and keys.
            norm_intermediate: Whether intermediate outputs pass through the final norm.
        """
        super().__init__()
        self.input_embed_dim = input_embed_dim
        self.dim = dim
        self.depth = depth
        self.max_num_views = max_num_views
        self.intermediate_layer_indices = list(intermediate_layer_indices)
        self.norm_intermediate = norm_intermediate

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.proj_embed = nn.Linear(input_embed_dim, dim) if input_embed_dim != dim else nn.Identity()
        self.self_attention_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(dim)
        self.register_buffer("view_pos_table", sinusoid_encoding_table(max_num_views, dim), persistent=False)

    def _tokens_to_views(
        self, tokens: torch.Tensor, num_views: int, height: int, width: int
    ) -> List[torch.Tensor]:
        """Reshape (B, V*H*W, C) tokens back into a list of V feature maps (B, C, H, W)."""
        batch_size = tokens.shape[0]
        feature_maps = tokens.reshape(batch_size, num_views, height, width, self.dim)
        feature_maps = feature_maps.permute(0, 1, 4, 2, 3).contiguous()
        return [view.squeeze(1) for view in feature_maps.split(1, dim=1)]

    def forward(
        self, view_features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            view_features: One feature map per view, each of shape (B, input_embed_dim, H, W).
                All views must share the same spatial shape.

        Returns:
            final_features: Per-view output feature maps (B, dim, H, W) after the last block and norm.
            intermediate_features: One entry per index in ``intermediate_layer_indices``; each entry is
                the per-view list of feature maps produced by that block.
        """
        num_views = len(view_features)
        if num_views > self.max_num_views:
            raise ValueError(f"Got {num_views} views but the model supports at most {self.max_num_views}.")
        batch_size, _, height, width = view_features[0].shape
        tokens_per_view = height * width

        # (B, V, C, H, W) -> (B, V*H*W, C), then project to the transformer width
        tokens = torch.stack(view_features, dim=1).permute(0, 1, 3, 4, 2)
        tokens = tokens.reshape(batch_size, num_views * tokens_per_view, self.input_embed_dim)
        tokens = self.proj_embed(tokens)

        # Add the fixed per-view positional embedding (view i gets table row i). The table is
        # float32, which keeps the residual stream in float32 under mixed-precision autocast.
        view_embedding = self.view_pos_table[:num_views].repeat_interleave(tokens_per_view, dim=0)
        tokens = tokens + view_embedding.unsqueeze(0)

        # Run the blocks, collecting the requested intermediate outputs
        intermediate_features: List[List[torch.Tensor]] = []
        for block_index, block in enumerate(self.self_attention_blocks):
            tokens = block(tokens)
            if block_index in self.intermediate_layer_indices:
                intermediate = self.norm(tokens) if self.norm_intermediate else tokens
                intermediate_features.append(self._tokens_to_views(intermediate, num_views, height, width))

        final_features = self._tokens_to_views(self.norm(tokens), num_views, height, width)
        return final_features, intermediate_features
