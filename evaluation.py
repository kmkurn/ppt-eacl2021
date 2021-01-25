# Copyright (c) 2021 Kemal Kurniawan

from dataclasses import astuple, dataclass
from typing import Dict, Optional, Union

import torch
from torch import BoolTensor, LongTensor


def count_correct(
    heads: LongTensor,
    types: LongTensor,
    pred_heads: LongTensor,
    pred_types: LongTensor,
    mask: BoolTensor,
    nopunct_mask: BoolTensor,
    proj_mask: BoolTensor,
    root_idx: int = 0,
    type_idx: Optional[int] = None,
) -> Union["Counts", "TypeWiseCounts"]:
    # shape: (bsz, slen)
    assert heads.dim() == 2
    assert types.shape == heads.shape
    assert pred_heads.shape == heads.shape
    assert pred_types.shape == heads.shape
    assert mask.shape == heads.shape
    assert nopunct_mask.shape == heads.shape
    assert proj_mask.shape == heads.shape

    corr_heads = heads == pred_heads
    corr_types = types == pred_types

    if type_idx is None:
        root_mask = heads == root_idx
        nonproj_mask = ~torch.all(proj_mask | (~mask), dim=1, keepdim=True)

        usents = int(torch.all(corr_heads | (~mask), dim=1).long().sum())
        usents_nopunct = int(
            torch.all(corr_heads | (~mask) | (~nopunct_mask), dim=1).long().sum()
        )
        lsents = int(torch.all(corr_heads & corr_types | (~mask), dim=1).long().sum())
        lsents_nopunct = int(
            torch.all(corr_heads & corr_types | (~mask) | (~nopunct_mask), dim=1).long().sum()
        )
        uarcs = int((corr_heads & mask).long().sum())
        uarcs_nopunct = int((corr_heads & mask & nopunct_mask).long().sum())
        uarcs_nonproj = int((corr_heads & mask & nonproj_mask).long().sum())
        larcs = int((corr_heads & corr_types & mask).long().sum())
        larcs_nopunct = int((corr_heads & corr_types & mask & nopunct_mask).long().sum())
        larcs_nonproj = int((corr_heads & corr_types & mask & nonproj_mask).long().sum())
        roots = int((corr_heads & mask & root_mask).long().sum())
        n_sents = heads.size(0)
        n_arcs = int(mask.long().sum())
        n_arcs_nopunct = int((mask & nopunct_mask).long().sum())
        n_arcs_nonproj = int((mask & nonproj_mask).long().sum())
        n_roots = int((mask & root_mask).long().sum())

        return Counts(
            usents,
            usents_nopunct,
            lsents,
            lsents_nopunct,
            uarcs,
            uarcs_nopunct,
            uarcs_nonproj,
            larcs,
            larcs_nopunct,
            larcs_nonproj,
            roots,
            n_sents,
            n_arcs,
            n_arcs_nopunct,
            n_arcs_nonproj,
            n_roots,
        )

    assert type_idx is not None
    type_mask = types == type_idx

    uarcs = int((corr_heads & type_mask & mask).long().sum())
    uarcs_nopunct = int((corr_heads & type_mask & nopunct_mask & mask).long().sum())
    larcs = int((corr_heads & corr_types & type_mask & mask).long().sum())
    larcs_nopunct = int(
        (corr_heads & corr_types & type_mask & nopunct_mask & mask).long().sum()
    )
    n_arcs = int((type_mask & mask).long().sum())
    n_arcs_nopunct = int((type_mask & nopunct_mask & mask).long().sum())

    return TypeWiseCounts(
        type_idx, uarcs, uarcs_nopunct, larcs, larcs_nopunct, n_arcs, n_arcs_nopunct
    )


@dataclass
class Counts:
    usents: int
    usents_nopunct: int
    lsents: int
    lsents_nopunct: int
    uarcs: int
    uarcs_nopunct: int
    uarcs_nonproj: int
    larcs: int
    larcs_nopunct: int
    larcs_nonproj: int
    roots: int
    n_sents: int
    n_arcs: int
    n_arcs_nopunct: int
    n_arcs_nonproj: int
    n_roots: int

    @property
    def accs(self) -> Dict[str, float]:
        accs = {
            "uem": self.usents / self.n_sents,
            "uem_nopunct": self.usents_nopunct / self.n_sents,
            "uas": self.uarcs / self.n_arcs,
            "uas_nopunct": self.uarcs_nopunct / self.n_arcs_nopunct,
            "lem": self.lsents / self.n_sents,
            "lem_nopunct": self.lsents_nopunct / self.n_sents,
            "las": self.larcs / self.n_arcs,
            "las_nopunct": self.larcs_nopunct / self.n_arcs_nopunct,
        }
        if self.n_arcs_nonproj:
            accs["uas_nonproj"] = self.uarcs_nonproj / self.n_arcs_nonproj
            accs["las_nonproj"] = self.larcs_nonproj / self.n_arcs_nonproj
        accs["root"] = self.roots / self.n_roots
        return accs

    def __add__(self, c):
        if not isinstance(c, Counts):
            raise TypeError
        x = torch.tensor(astuple(self))
        y = torch.tensor(astuple(c))
        z = x + y
        return Counts(*z.tolist())


@dataclass
class TypeWiseCounts:
    type_idx: int
    uarcs: int
    uarcs_nopunct: int
    larcs: int
    larcs_nopunct: int
    n_arcs: int
    n_arcs_nopunct: int

    def __add__(self, c):
        if not isinstance(c, self.__class__):
            raise TypeError
        if self.type_idx != c.type_idx:
            raise ValueError("cannot add counts with different type idx")

        x = torch.tensor(astuple(self))
        y = torch.tensor(astuple(c))
        z = x + y
        return self.__class__(self.type_idx, *z[1:].tolist())

    @property
    def accs(self) -> Dict[str, float]:
        accs = {}
        if self.n_arcs:
            accs["uas"] = self.uarcs / self.n_arcs
            accs["las"] = self.larcs / self.n_arcs
        if self.n_arcs_nopunct:
            accs["uas_nopunct"] = self.uarcs_nopunct / self.n_arcs_nopunct
            accs["las_nopunct"] = self.larcs_nopunct / self.n_arcs_nopunct
        return accs
