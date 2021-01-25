#!/usr/bin/env python

# Copyright (c) 2021 Kemal Kurniawan

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Collection, Dict, List, Optional
import json
import sys

from tqdm import tqdm
import networkx as nx
import networkx.algorithms.isomorphism as iso

from readers import UDReader


def main(
    ud_path: Path,
    langs: Optional[Collection[str]] = None,
    src_train_max_length: int = 100,
    tgt_test_max_length: int = 150,
) -> Dict[str, float]:
    if langs is None:
        langs = "fa ar id ko tr hi hr he bg it pt fr es no da sv nl de".split()

    r = UDReader(ud_path)

    print("Reading en train samples", file=sys.stderr)
    src_graphs: Dict[int, List[nx.Graph]] = defaultdict(list)
    src_samples = list(r.read_samples("en"))
    for s in tqdm(src_samples, unit="sample"):
        g = nx.Graph([(v, u) for u, v in enumerate(s["heads"], start=1)])
        src_graphs[len(s["words"])].append(g)

    res, cnt = {}, 0
    for lang in langs:
        cnt += 1
        print(
            f"[{cnt}/{len(langs)}] Computing leakage for {lang} test samples", file=sys.stderr,
        )
        n_samples, n_leaks = 0, 0
        tgt_samples = list(r.read_samples(lang, "test"))
        for s in tqdm(tgt_samples, unit="sample"):
            n_samples += 1
            g = nx.Graph([(v, u) for u, v in enumerate(s["heads"], start=1)])
            for g_ in src_graphs[len(s["words"])]:
                if iso.tree_isomorphism(g, g_):
                    n_leaks += 1
                    break
        print(f"[{cnt}/{len(langs)}] Leakage is {n_leaks/n_samples:.1%}", file=sys.stderr)
        res[lang] = n_leaks / n_samples
    print("Done", file=sys.stderr)

    return res


if __name__ == "__main__":
    p = ArgumentParser(description="Compute cross-lingual treebank leakage with en as source.")
    p.add_argument("ud_path", type=Path, help="path to UD data")
    p.add_argument("--langs", nargs="+", help="target languages")
    p.add_argument(
        "--src-train-max-length",
        type=int,
        default=100,
        help="max sent length of source train set",
    )
    p.add_argument(
        "--tgt-test-max-length",
        type=int,
        default=150,
        help="max sent length of target test set",
    )
    p.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        type=Path,
        default="leakage.json",
        help="path to JSON output file",
    )
    args = p.parse_args()
    res = main(args.ud_path, args.langs, args.src_train_max_length, args.tgt_test_max_length)
    print(f"Writing output to {args.output}", file=sys.stderr)
    with open(args.output, "w", encoding="utf8") as f:
        json.dump(res, f, indent=2)
