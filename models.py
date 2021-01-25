# Copyright (c) 2021 Kemal Kurniawan

from typing import Optional, Tuple

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import BoolTensor, LongTensor, Tensor
import torch
import torch.nn as nn

from modules import TransformerEncoderLayer


class SelfAttGraph(nn.Module):
    """Self-attention graph-based parser based on Ahmad et al. (2019)."""

    def __init__(
        self,
        n_words: int,
        n_types: int,
        n_tags: int = 0,
        word_size: int = 300,
        tag_size: int = 50,
        n_heads: int = 10,
        n_layers: int = 6,
        ff_size: int = 2048,
        kv_size: int = 64,
        word_dropout: float = 0.5,
        outdim_dropout: float = 0.5,
        arc_size: int = 128,
        type_size: int = 128,
    ) -> None:
        super().__init__()
        self.word_emb = nn.Embedding(n_words, word_size)
        enc_in_size = word_size
        self.tag_emb = None
        if n_tags > 0:
            self.tag_emb = nn.Embedding(n_tags, tag_size)
            enc_in_size += tag_size
        self.encoder = nn.TransformerEncoder(
            TransformerEncoderLayer(enc_in_size, n_heads, ff_size=ff_size, kv_size=kv_size),
            n_layers,
        )

        enc_out_size = self.encoder.layers[-1].norm2.normalized_shape[-1]
        self.word_dropout = nn.Dropout2d(p=word_dropout)
        self.mlp_layer = nn.Sequential(
            nn.Linear(enc_out_size, 2 * arc_size + 2 * type_size),
            nn.ReLU(),
            Rearrange("bsz slen dim -> bsz dim slen"),
            nn.Dropout2d(p=outdim_dropout),  # drop some dims entirely
            Rearrange("bsz dim slen -> bsz slen dim"),
        )

        self.arc_score_mix = nn.Parameter(torch.empty(arc_size, arc_size))  # type: ignore
        self.arc_score_bias = nn.Parameter(torch.empty(arc_size))  # type: ignore
        self.type_score_mix = nn.Bilinear(type_size, type_size, n_types)
        self.type_score_bias = nn.Linear(2 * type_size, n_types, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(self.arc_score_mix, -0.01, 0.01)
        torch.nn.init.constant_(self.arc_score_bias, 0.0)

    def forward(self, words, tags=None, mask=None, heads=None):
        assert words.dim() == 2
        assert tags is None or tags.shape == words.shape
        assert mask is None or mask.shape == words.shape
        assert heads is None or heads.shape == words.shape

        if mask is None:
            mask = torch.full_like(words, 1).bool()

        bsz, slen = words.shape

        # shape: (bsz, slen, dim)
        inputs = self._embed(words, tags)
        # each shape: (bsz, slen, dim)
        arc_h, arc_d, type_h, type_d = self._encode(inputs, mask)  # type: ignore

        # shape: (bsz, slen, slen)
        arc_scores = self._compute_arc_scores(arc_h, arc_d)

        if heads is None:
            # broadcast over dependents
            type_h = type_h.unsqueeze(2).expand(bsz, slen, slen, -1).contiguous()
            # broadcast over heads
            type_d = type_d.unsqueeze(1).expand(bsz, slen, slen, -1).contiguous()
        else:
            # broadcast over dimensions
            heads = rearrange(heads, "bsz slen -> bsz slen ()").expand_as(type_h)
            # reorder type_h according to heads
            type_h = type_h.gather(1, heads)
        # shape: (bsz, slen, n_types) or (bsz, slen, slen, n_types)
        type_scores = self._compute_type_scores(type_h, type_d)

        return arc_scores, type_scores

    def _embed(self, words: LongTensor, tags: Optional[LongTensor] = None) -> Tensor:
        # shape: (bsz, slen)
        assert words.dim() == 2
        assert tags is None or tags.shape == words.shape

        # shape: (bsz, slen, wdim)
        outputs = self.word_emb(words)
        if self.tag_emb is not None:
            assert tags is not None
            # shape: (bsz, slen, wdim+tdim)
            outputs = torch.cat([outputs, self.tag_emb(tags)], dim=-1)

        return outputs

    def _encode(
        self, inputs: Tensor, mask: BoolTensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # shape: (bsz, slen, dim)
        assert inputs.dim() == 3
        # shape: (bsz, slen)
        assert mask.shape == inputs.shape[:-1]

        # drop some words entirely
        outputs = self.word_dropout(inputs)

        outputs = rearrange(outputs, "bsz slen dim -> slen bsz dim")
        outputs = self.encoder(outputs, src_key_padding_mask=~mask)
        outputs = rearrange(outputs, "slen bsz dim -> bsz slen dim")

        # shape: (bsz, slen, dim)
        outputs = self.mlp_layer(outputs)
        arc_size = self.arc_score_bias.numel()
        outputs_arc = outputs[:, :, : 2 * arc_size]
        outputs_type = outputs[:, :, 2 * arc_size :]

        arc_h, arc_d = rearrange(outputs_arc, "bsz slen (n asz) -> n bsz slen asz", n=2)
        type_h, type_d = rearrange(outputs_type, "bsz slen (n tsz) -> n bsz slen tsz", n=2)
        return arc_h, arc_d, type_h, type_d

    def _compute_arc_scores(self, arc_h: Tensor, arc_d: Tensor) -> Tensor:
        """Compute scores of arcs for all heads and dependents.

        This method implements equation 6 in (Dozat and Manning, 2017).
        """
        # shape: (bsz, slen, dim)
        assert arc_h.dim() == 3
        assert arc_d.shape == arc_h.shape

        arc_d = rearrange(arc_d, "bsz slen asz -> bsz asz slen")

        # shape: (bsz, slen, slen)
        mix_scores = arc_h @ self.arc_score_mix @ arc_d
        # shape: (bsz, slen, 1)
        # broadcast bias scores over every possible dependent
        bias_scores = arc_h @ rearrange(self.arc_score_bias, "asz -> asz ()")

        # shape: (bsz, slen, slen)
        return mix_scores + bias_scores

    def _compute_type_scores(self, type_h: Tensor, type_d: Tensor) -> Tensor:
        """Compute scores of types for all dependents of given/all heads.

        This method implements equation 3 in (Dozat and Manning, 2017).
        """
        # shape: (bsz, slen, dim) or (bsz, slen, slen, dim)
        assert type_h.dim() in (3, 4)
        assert type_d.shape == type_h.shape

        # shape: (bsz, slen, n_types) or (bsz, slen, slen, n_types)
        mix_scores = self.type_score_mix(type_h, type_d)
        # shape: (bsz, slen, n_types) or (bsz, slen, slen, n_types)
        bias_scores = self.type_score_bias(
            rearrange([type_h, type_d], "n ... dim -> ... (n dim)")
        )

        # shape: (bsz, slen, n_types) or (bsz, slen, slen, n_types)
        return mix_scores + bias_scores
