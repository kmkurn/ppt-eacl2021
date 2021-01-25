# Copyright (c) 2021 Kemal Kurniawan

from typing import List, Optional, Tuple
import warnings

from torch import BoolTensor, LongTensor, Tensor
from torch_struct import DependencyCRF
from torch_struct.deptree import _convert, _unconvert
import torch

from matrix_tree import compute_log_partitions, compute_marginals
from mst import decode_mst


class DepTreeCRF:
    """Dependency tree CRF.

    This CRF defines a (conditional) probability distribution over labeled dependency
    trees. A labeled dependency tree is represented as a sequence of head positions
    and a sequence of dependency types for the corresponding arc. The first position
    in the sequence (position 0) is assumed to be the tree's root.

    Args:
        scores: Tensor of shape (B, N, N, L) containing scores of all labeled
            head-dependent arcs.
        mask: Boolean tensor of shape (B, N) indicating valid positions.
        projective: Whether to operate in the space of projective trees.
        multiroot: Whether to consider multi-root case, where the (symbolic) root can have
            more than one child.

    Note:
        B = batch size, N = sequence length, L = number of dependency labels/types.
    """

    ROOT = 0

    def __init__(
        self,
        scores: Tensor,
        mask: Optional[BoolTensor] = None,
        projective: bool = False,
        multiroot: bool = True,
    ) -> None:
        assert scores.dim() == 4
        bsz, slen = scores.shape[:2]
        assert scores.size(2) == slen
        assert mask is None or mask.shape == (bsz, slen)

        if mask is None:
            mask = scores.new_full([bsz, slen], 1).bool()  # type: ignore

        self.scores = scores
        self.mask = mask
        self.proj = projective
        self.multiroot = multiroot

    def log_probs(
        self, heads: LongTensor, types: LongTensor, score_only: bool = False
    ) -> Tensor:
        """Compute the log probability of a labeled dependency tree.

        Args:
            heads: Tensor of shape (B, N) containing the index/position of the head of
                each word.
            types: Tensor of shape (B, N) containing the dependency types for the
                corresponding head-dependent relation.
            score_only: Whether to compute only the score of the tree. Useful for training
                with cross-entropy loss.

        Returns:
            1-D tensor of length B containing the log probabilities.
        """
        assert heads.dim() == 2
        assert types.shape == heads.shape
        assert self.mask is not None

        scores = self.scores
        bsz, slen, _, n_types = self.scores.shape

        # broadcast over types
        heads = heads.unsqueeze(2).expand(bsz, slen, n_types)  # type: ignore
        # shape: (bsz, slen, n_types)
        scores = scores.gather(1, heads.unsqueeze(1)).squeeze(1)
        # shape: (bsz, slen)
        scores = scores.gather(2, types.unsqueeze(2)).squeeze(2)
        # mask scores from invalid dependents
        scores = scores.masked_fill(~self.mask, 0)
        # mask scores of root as dependents
        scores = scores.masked_fill(torch.arange(slen).to(scores.device) == self.ROOT, 0)

        return scores.sum(dim=1) - (0 if score_only else self.log_partitions())

    def argmax(self) -> Tuple[LongTensor, LongTensor]:
        """Compute the most probable labeled dependency tree.

        Returns:
            - Tensor of shape (B, N) containing the head positions of the best tree.
            - Tensor of shape (B, N) containing the dependency types for the
              corresponding head-dependent relation.
        """
        assert self.mask is not None

        # each shape: (bsz, slen, slen)
        scores, best_types = self.scores.max(dim=3)
        lengths = self.mask.long().sum(dim=1)

        if self.proj:
            crf = DependencyCRF(_unconvert(scores), lengths - 1, multiroot=self.multiroot)
            # shape: (bsz, slen)
            _, pred_heads = _convert(crf.argmax).max(dim=1)
            pred_heads[:, self.ROOT] = self.ROOT
        else:
            if not self.multiroot:
                warnings.warn(
                    "argmax for non-projective is still multiroot although multiroot=False"
                )
            # shape: (bsz, slen)
            pred_heads = find_mst(scores, lengths.tolist())

        # shape: (bsz, slen)
        pred_types = best_types.gather(1, pred_heads.unsqueeze(1)).squeeze(1)

        return pred_heads, pred_types  # type: ignore

    def log_partitions(self) -> Tensor:
        """Compute the log partition function.

        Returns:
            1-D tensor of length B containing the log partition functions.
        """
        assert self.mask is not None

        if self.proj:
            lengths = self.mask.long().sum(dim=1)
            crf = DependencyCRF(_unconvert(self.scores), lengths - 1, multiroot=self.multiroot)
            return crf.partition

        return compute_log_partitions(self.scores, self.mask, self.multiroot)

    def marginals(self) -> Tensor:
        """Compute the arc marginal probabilities.

        Returns:
            Tensor of shape (B, N, N, L) containing the arc marginal probabilities.
        """
        assert self.mask is not None

        if self.proj:
            lengths = self.mask.long().sum(dim=1)
            crf = DependencyCRF(_unconvert(self.scores), lengths - 1, multiroot=self.multiroot)
            margs = _convert(crf.marginals)

            # marginals of incoming arcs to root are zero
            margs[:, :, self.ROOT] = 0
            # marginals of self-loops are zero
            self_loop_mask = torch.eye(margs.size(1)).to(margs.device).unsqueeze(2).bool()
            margs = margs.masked_fill(self_loop_mask, 0)

            return margs

        return compute_marginals(self.scores, self.mask, self.multiroot)


def find_mst(scores: Tensor, slens: Optional[List[int]] = None, root: int = 0) -> LongTensor:
    """Find maximum spanning tree with Tarjan's implementation of Edmond's algorithm.

    Args:
        scores: Tensor of shape (B, N, N) containing the scores of all possible arcs.
        slens: List of sequence lengths.
        root: Index/position of the root.

    Returns:
        Tensor of shape (B, N) containing the head positions of the maximum tree.
    """
    bsz, maxlen, _ = scores.shape
    heads = scores.new_zeros((bsz, maxlen)).long()

    for b in range(bsz):
        slen = maxlen if slens is None else slens[b]
        hs, _ = decode_mst(scores[b].cpu().numpy(), slen, has_labels=False)
        heads[b] = torch.from_numpy(hs).to(heads)

    heads[:, root] = root

    return heads  # type: ignore
