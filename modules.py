# Copyright (c) 2021 Kemal Kurniawan

from typing import Optional

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import LongTensor, Tensor
import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Transformer encoder layer that uses distance-aware self-attention layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_size: int = 2048,
        dropout: float = 0.1,
        kv_size: int = 64,
    ) -> None:
        # call with fake n_heads to avoid error creating MultiheadAttention
        super().__init__(d_model, 1, dim_feedforward=ff_size, dropout=dropout)
        # replace with custom attention
        self.self_attn = DistanceAwareSelfAttention(
            d_model, n_heads, dropout=dropout, kv_size=kv_size
        )


class DistanceAwareSelfAttention(nn.Module):
    """Distance-aware self-attention layer from Ahmad et al. (2019)."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float = 0.0,
        clip_dist: int = 10,
        kv_size: int = 64,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(embed_dim, 3 * n_heads * kv_size),
            Rearrange("slen bsz (n nhead dim) -> n bsz nhead slen dim", n=3, dim=kv_size),
        )
        self.k_dist_emb = nn.Embedding(clip_dist + 1, kv_size)
        self.v_dist_emb = nn.Embedding(clip_dist + 1, kv_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(n_heads * kv_size, embed_dim)

    def forward(self, inputs, inputs2, inputs3, attn_mask=None, key_padding_mask=None):
        assert inputs is inputs2 and inputs is inputs3, "must be a self-attention"
        assert attn_mask is None, "attn_mask should not be given"

        # shape: (slen, bsz, embed_dim)
        assert inputs.dim() == 3
        # shape: (bsz, slen)
        assert key_padding_mask is None or key_padding_mask.shape == (
            inputs.size(1),
            inputs.size(0),
        )

        # each shape: (bsz, nhead, slen, qdim/vdim)
        q, k, v = self.in_proj(inputs)
        # shape: (slen, slen)
        distances = self._get_distances(inputs.size(0)).to(inputs.device)

        q *= q.size(-1) ** -0.5
        k = rearrange(k, "bsz nhead slen qdim -> bsz nhead qdim slen")

        # shape: (bsz, nhead, slen, slen)
        attn_weights = q @ k + self._get_dist_attn_weights(q, distances)

        if key_padding_mask is not None:
            # broadcast over heads and queries
            mask = rearrange(key_padding_mask, "bsz slen -> bsz () () slen")
            attn_weights.masked_fill_(mask, float("-inf"))

        # shape: (bsz, nhead, slen, slen)
        attn_weights = attn_weights.softmax(dim=-1)
        # shape: (bsz, nhead, slen, slen)
        attn_weights = self.attn_dropout(attn_weights)
        # shape: (bsz, nhead, slen, vdim)
        attn_outputs = attn_weights @ v + self._get_dist_attn_outputs(attn_weights, distances)

        attn_outputs = rearrange(attn_outputs, "bsz nhead slen vdim -> slen bsz (nhead vdim)")
        # shape: (slen, bsz, embed_dim)
        attn_outputs = self.out_proj(attn_outputs)

        return attn_outputs, None  # attn_weights is not needed

    def _get_distances(self, slen: int) -> LongTensor:
        x = rearrange(torch.arange(slen), "slen -> () slen")
        y = rearrange(torch.arange(slen), "slen -> slen ()")
        # shape: (slen, slen)
        dist = torch.abs(x - y)
        clip_dist = self.k_dist_emb.num_embeddings - 1
        # shape: (slen, slen)
        return dist.clamp(max=clip_dist).long()  # type: ignore

    def _get_dist_attn_weights(self, q: Tensor, dist: LongTensor) -> Tensor:
        # shape: (bsz, nhead, slen, qdim)
        assert q.dim() == 4
        # shape: (slen, slen)
        assert dist.shape == (q.size(2), q.size(2))

        # shape: (slen, slen, qdim)
        k_dist = self.k_dist_emb(dist)

        q_dist = rearrange(q, "bsz nhead slen qdim -> bsz nhead slen () qdim")  # bc over keys
        k_dist = rearrange(k_dist, "slen slen2 qdim -> slen qdim slen2")
        weights = q_dist @ k_dist
        return rearrange(weights, "bsz nhead slen () slen2 -> bsz nhead slen slen2")

    def _get_dist_attn_outputs(self, attn_weights: Tensor, dist: LongTensor) -> Tensor:
        # shape: (bsz, nhead, slen, slen)
        assert attn_weights.dim() == 4
        assert attn_weights.size(-2) == attn_weights.size(-1)
        # shape: (slen, slen)
        assert dist.shape == attn_weights.shape[-2:]

        # shape: (slen, slen, vdim)
        v_dist = self.v_dist_emb(dist)

        attn_dist = rearrange(attn_weights, "bsz nhead slen slen2 -> bsz nhead slen () slen2")
        outputs = attn_dist @ v_dist
        return rearrange(outputs, "bsz nhead slen () vdim -> bsz nhead slen vdim")
