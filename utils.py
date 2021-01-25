# Copyright (c) 2021 Kemal Kurniawan

from typing import Mapping, Optional, Sequence
import logging

from sacred.run import Run
from text2array import BucketIterator
from torch import Tensor
from tqdm import tqdm
import numpy as np
import torch

from crf import DepTreeCRF


logger = logging.getLogger(__name__)


def extend_word_embedding(
    weight: Tensor,
    words: Sequence[str],
    kv: Optional[Mapping[str, np.ndarray]] = None,
    unk_id: Optional[int] = None,
) -> Tensor:
    assert weight.dim() == 2
    if kv is None:
        kv = {}

    new_weight = torch.randn(len(words), weight.size(1))
    new_weight[: weight.size(0)] = weight
    cnt_pre, cnt_unk = 0, 0
    for w in words:
        wid = words.index(w)
        if wid < weight.size(0):
            continue
        if w in kv:
            new_weight[wid] = torch.from_numpy(kv[w])
            cnt_pre += 1
        elif w.lower() in kv:
            new_weight[wid] = torch.from_numpy(kv[w.lower()])
            cnt_pre += 1
        else:
            cnt_unk += 1
            if unk_id is not None:
                new_weight[wid] = weight[unk_id]

    logger.info("Initialized %d target words with pre-trained embedding", cnt_pre)
    logger.info("Found %d unknown words", cnt_unk)

    return new_weight


def report_log_ntrees_stats(
    samples: Sequence[dict],
    aa_mask_field: str,
    batch_size: int = 1,
    projective: bool = False,
    multiroot: bool = False,
) -> None:
    log_ntrees: list = []
    pbar = tqdm(total=sum(len(s["words"]) for s in samples), leave=False)
    for batch in BucketIterator(samples, lambda s: len(s["words"]), batch_size):
        arr = batch.to_array()
        aaet_mask = torch.from_numpy(arr[aa_mask_field]).bool()
        cnt_scores = torch.zeros_like(aaet_mask).float().masked_fill(~aaet_mask, -1e9)
        log_ntrees.extend(
            DepTreeCRF(cnt_scores, projective=projective, multiroot=multiroot)
            .log_partitions()
            .tolist()
        )
        pbar.update(arr["words"].size)
    pbar.close()
    logger.info(
        "Log number of trees: min %.2f | q1 %.2f | q2 %.2f | q3 %.2f | max %.2f",
        np.min(log_ntrees),
        np.quantile(log_ntrees, 0.25),
        np.quantile(log_ntrees, 0.5),
        np.quantile(log_ntrees, 0.75),
        np.max(log_ntrees),
    )


def print_accs(
    accs: Mapping[str, float],
    on: str = "dev",
    run: Optional[Run] = None,
    step: Optional[int] = None,
) -> None:
    for key, acc in accs.items():
        logger.info(f"{on}_{key}: {acc:.2%}")
        if run is not None:
            run.log_scalar(f"{on}_{key}", acc, step=step)
