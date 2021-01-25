# Copyright (c) 2021 Kemal Kurniawan

from typing import Optional

from torch import BoolTensor, Tensor
import torch


def compute_log_partitions(
    log_weights: Tensor,
    mask: Optional[BoolTensor] = None,
    multiroot: bool = True,
    delta: float = 1e-8,
) -> Tensor:
    """Compute log partition function with Matrix-Tree Theorem.

    By convention, node with index 0 is assumed to be the (dummy) root. See
    (McDonald and Satta, 2007; Koo et al., 2007) for reference.

    Args:
        log_weights: Tensor of shape (B, N, N, L) which contains the log weight of each
            pair of labeled arcs in the batch.
        mask: Tensor of shape (B, N) indicating valid positions.
        multiroot: Whether to compute partition over multi-rooted trees, which means the
            (dummy) root is allowed to have more than one child. If False, the (dummy) root
            will have exactly one child.
        delta: Small positive number to ensure weights are not too small.

    Returns:
        1-D tensor of length B containing the log partition function values.

    Note:
        B = batch size, N = sequence length, L = number of dependency labels/types.
    """
    assert log_weights.dim() == 4
    bsz, slen, _, _ = log_weights.shape
    assert slen >= 2, "number of nodes is at least 2"
    assert log_weights.size(2) == slen
    assert mask is None or mask.shape == (bsz, slen)

    if mask is None:
        mask = log_weights.new_ones([bsz, slen]).bool()  # type: ignore
    assert mask is not None

    # clone to allow in-place operations
    log_weights = log_weights.clone()

    # mask incoming arcs to root
    log_weights[:, :, 0] = float("-inf")

    # mask invalid arcs
    arc_mask = mask.reshape(bsz, 1, slen) & mask.reshape(bsz, slen, 1)
    log_weights.masked_fill_(~arc_mask.reshape(bsz, slen, slen, 1), float("-inf"))

    # mask self-loops
    loop_mask = torch.eye(slen).to(log_weights.device).bool().reshape(slen, slen, 1)
    log_weights.masked_fill_(loop_mask, float("-inf"))

    # shift log weights to lie in a safe range for exp
    max_logw, _ = log_weights.reshape(bsz, -1).max(dim=1)
    weights = (log_weights - max_logw.reshape(bsz, 1, 1, 1)).exp() + delta
    weights = weights.sum(dim=3)

    # ensure heads of invalid nodes are always node 1, and the arc weights equal one
    weights[:, 1].masked_fill_(~mask, 1)

    # compute log partition functions via matrix-tree theorem
    Q = to_laplacian(weights[:, 1:, 1:])
    if not multiroot:
        Q[:, 0] = weights[:, 0, 1:]
    else:
        Q = Q + torch.diag_embed(weights[:, 0, 1:])
    log_partitions = Q.logdet()

    # shift back to correct the result
    lengths = mask.long().sum(dim=1)
    log_partitions = log_partitions + (lengths.float() - 1) * max_logw

    return log_partitions


def compute_marginals(
    log_weights: Tensor,
    mask: Optional[BoolTensor] = None,
    multiroot: bool = True,
    delta: float = 1e-8,
) -> Tensor:
    """Compute marginal probabilities of all labeled arcs.

    By convention, node with index 0 is assumed to be the (dummy) root. See
    (McDonald and Satta, 2007; Koo et al., 2007) for reference.

    Args:
        log_weights: Tensor of shape (B, N, N, L) which contains the log weight of each
            pair of labeled arcs in the batch.
        mask: Tensor of shape (B, N) indicating valid positions.
        multiroot: Whether to compute marginals over multi-rooted trees, which means the
            (dummy) root is allowed to have more than one child. If False, the (dummy) root
            will have exactly one child.
        delta: Small positive number to ensure weights are not too small.

    Returns:
        Tensor of shape (B, N, N, L) containing the arc marginal probabilities.

    Note:
        B = batch size, N = sequence length, L = number of dependency labels/types.
    """
    assert log_weights.dim() == 4
    bsz, slen, _, _ = log_weights.shape
    assert slen >= 2, "number of nodes is at least 2"
    assert log_weights.size(2) == slen
    assert mask is None or mask.shape == (bsz, slen)

    if mask is None:
        mask = log_weights.new_ones([bsz, slen]).bool()  # type: ignore
    assert mask is not None

    # clone to allow in-place operations
    log_weights = log_weights.clone()

    # mask incoming arcs to root
    log_weights[:, :, 0] = float("-inf")

    # mask invalid arcs
    arc_mask = mask.reshape(bsz, 1, slen) & mask.reshape(bsz, slen, 1)
    log_weights.masked_fill_(~arc_mask.reshape(bsz, slen, slen, 1), float("-inf"))

    # mask self-loops
    loop_mask = torch.eye(slen).to(log_weights.device).bool().reshape(slen, slen, 1)
    log_weights.masked_fill_(loop_mask, float("-inf"))

    # shift log weights to lie in a safe range for exp
    max_logw, _ = log_weights.reshape(bsz, -1).max(dim=1)
    log_weights = log_weights - max_logw.reshape(bsz, 1, 1, 1)
    weights_ = log_weights.exp() + delta
    weights = weights_.sum(dim=3)

    # ensure heads of invalid nodes are always node 1, and the arc weights equal one
    weights[:, 1].masked_fill_(~mask, 1)

    # compute log marginals via matrix-tree theorem
    Q = to_laplacian(weights[:, 1:, 1:])
    if not multiroot:
        Q[:, 0] = weights[:, 0, 1:]
    else:
        Q = Q + torch.diag_embed(weights[:, 0, 1:])
    Q_inv = Q.inverse()
    marginals = torch.zeros_like(log_weights)

    # arcs outgoing from root
    term = diag(Q_inv) if multiroot else Q_inv[:, :, 0]
    marginals[:, 0, 1:] = weights_[:, 0, 1:] * term.unsqueeze(2)  # bc over types

    # arcs not involving root
    term1 = diag(Q_inv).reshape(bsz, 1, slen - 1).clone()
    term2 = Q_inv.transpose(1, 2).reshape(bsz, slen - 1, slen - 1).clone()
    if not multiroot:
        term1[:, :, 0] = 0
        term2[:, 0] = 0
    marginals[:, 1:, 1:] = weights_[:, 1:, 1:] * (term1 - term2).unsqueeze(3)  # bc over types

    return marginals


def to_laplacian(weights: Tensor) -> Tensor:
    # shape: (b, n, n)
    assert weights.dim() == 3
    assert weights.size(1) == weights.size(2)

    n = weights.size(1)
    # zero out weights of self-loops
    weights = weights.masked_fill(torch.eye(n).bool().to(weights.device), 0)
    # fill diagonals with the correct weights
    weights -= torch.diag_embed(weights.sum(dim=1))

    return -weights


def diag(t: Tensor) -> Tensor:
    """Batch version of torch.diag."""
    return t.diagonal(dim1=-2, dim2=-1)
