# Copyright (c) 2021 Kemal Kurniawan

from pathlib import Path
import re

from sacred import Ingredient
from tqdm import tqdm

from readers import UDReader, get_proj_edges

ing = Ingredient("corpus")


@ing.config
def cfg():
    # path to UD directory
    ud_path = "ud-treebanks-v2.2"
    # UD version to use
    version = 2
    # language to load
    lang = "en"
    # how many portion of samples to read (0-1)
    portion = 1.0
    # whether to convert digits to zeros
    normalize_digits = False
    # whether to skip MWEs and empty words
    skip_mwe_and_empty = True


@ing.capture
def read_samples(
    ud_path,
    _log,
    lang="en",
    portion=1.0,
    normalize_digits=False,
    which="train",
    prep=True,
    max_length=None,
    skip_mwe_and_empty=True,
    version=2,
):
    if max_length is None:
        max_length = float("inf")

    _log.info("Reading %s %s samples from %s", lang, which, ud_path)
    samples = UDReader(Path(ud_path), skip_mwe_and_empty, version).read_samples(
        langcode=lang, which=which
    )
    if portion < 1:
        samples = list(samples)
        n = int(len(samples) * portion)
        samples = samples[:n]

    for s in samples:
        if len(s["words"]) > max_length:
            continue
        if normalize_digits:
            s["words"] = [re.sub(r"\d", "0", w) for w in s["words"]]
        if prep:
            s = prep_for_parsing(s)
        yield s


@ing.command(unobserved=True)
def print_stats():
    """Print corpus statistics."""
    for which in "train dev test".split():
        n_sents, n_toks, n_toks_nopunct, n_sents_nonproj, n_toks_nonproj = get_stats(
            read_samples(which=which)
        )
        # avoid counting ROOT element
        n_toks -= n_sents
        n_toks_nopunct -= n_sents
        n_toks_nonproj -= n_sents_nonproj
        print(f"** {which}")
        print(f"   n_sents: {n_sents}")
        print(f"   n_toks: {n_toks}")
        print(f"   n_toks_nopunct: {n_toks_nopunct}")
        print(f"   n_sents_nonproj: {n_sents_nonproj} ({n_sents_nonproj/n_sents:.1%})")
        print(f"   n_toks_nonproj: {n_toks_nonproj} ({n_toks_nonproj/n_toks:.1%})")


def prep_for_parsing(sample):
    # Add ROOT element
    for key in "words tags types".split():
        sample[key].insert(0, "<root>")
    sample["heads"].insert(0, 0)

    # Get only the main dependency types
    sample["types"] = [t.split(":")[0] for t in sample["types"]]

    # Mark punctuations
    sample["punct?"] = [False for _ in sample["tags"]]
    for i, t in enumerate(sample["tags"]):
        if t in ("SYM", "PUNCT"):
            sample["punct?"][i] = True

    # Mark projective arcs
    sample["proj?"] = [False for _ in sample["heads"]]
    sample["proj?"][0] = True
    edges = [(h, d) for d, h in enumerate(sample["heads"]) if d != 0]
    for _, d in get_proj_edges(edges):
        sample["proj?"][d] = True

    return sample


def get_stats(samples):
    n_sents = n_toks = n_toks_nopunct = 0
    n_sents_nonproj = n_toks_nonproj = 0
    for s in tqdm(samples, leave=False):
        n_sents += 1
        n_toks += len(s["words"])
        assert len(s["words"]) == len(s["tags"])
        n_toks_nopunct += sum(0 if p else 1 for p in s["punct?"])
        is_proj = all(s["proj?"])
        n_sents_nonproj += 1 if not is_proj else 0
        n_toks_nonproj += len(s["words"]) if not is_proj else 0

    return n_sents, n_toks, n_toks_nopunct, n_sents_nonproj, n_toks_nonproj
