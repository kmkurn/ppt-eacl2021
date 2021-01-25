#!/usr/bin/env python

# Copyright (c) 2021 Kemal Kurniawan

from pathlib import Path
import argparse
import random
import sys

from conllu import parse
from tqdm import tqdm


def sample(path, size, out_path=None, rng=None):
    """Sample sentences from a CoNLL-U file."""
    if out_path is None:
        out_path = Path("output.conllu")
    if rng is None:
        rng = random.Random()

    print(f"Reading {path}", file=sys.stderr)
    with open(path, encoding="utf-8") as f:
        sents = parse(f.read())
    rng.shuffle(sents)

    with open(out_path, "w", encoding="utf-8") as f:
        for sent in tqdm(sents[:size]):
            print(sent.serialize(), file=f, end="")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=sample.__doc__)
    p.add_argument("path", type=Path, help="CoNLL-U file to sample from")
    p.add_argument("size", type=int, help="sample size")
    p.add_argument(
        "-o", "--output", type=Path, default="output.conllu", help="output CoNLL-U file"
    )
    p.add_argument("-s", "--seed", type=int, default=0, help="random seed")
    args = p.parse_args()

    sample(args.path, args.size, args.output, random.Random(args.seed))
