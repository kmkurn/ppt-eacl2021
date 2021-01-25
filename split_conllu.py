#!/usr/bin/env python

# Copyright (c) 2021 Kemal Kurniawan

from pathlib import Path
import argparse
import sys
import random

from conllu import parse
from tqdm import tqdm


def split(path, out_path=None, n_parts=5, rng=None):
    """Split a CoNLL-U file into parts."""
    if out_path is None:
        out_path = Path("output")
    if rng is None:
        rng = random.Random()

    out_path.mkdir()

    print(f"Reading {path}", file=sys.stderr)
    with open(path, encoding="utf-8") as f:
        sents = parse(f.read())
    rng.shuffle(sents)

    count = [0] * n_parts
    for i, sent in enumerate(tqdm(sents)):
        count[i % n_parts] += 1
        with open(out_path / f"{i % n_parts:02}.conllu", "a", encoding="utf-8") as f:
            print(sent.serialize(), file=f, end="")

    for i, cnt in enumerate(count):
        print(f"Part {i:02}: {cnt}", file=sys.stderr)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=split.__doc__)
    p.add_argument("path", type=Path, help="CoNLL-U file to split")
    p.add_argument("-o", "--output", type=Path, default=Path.cwd(), help="output directory")
    p.add_argument(
        "-n", "--num-parts", dest="n_parts", type=int, default=5, help="number of parts"
    )
    p.add_argument("-s", "--seed", type=int, default=0, help="random seed")
    args = p.parse_args()

    split(args.path, args.output, args.n_parts, random.Random(args.seed))
