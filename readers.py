# Copyright (c) 2021 Kemal Kurniawan

from collections import defaultdict
from typing import Collection, Iterator, Optional, Set, Tuple
from pathlib import Path

from conllu import parse_incr


class UDReader:
    DEFAULT_TREEBANK = {
        "ar": ("Arabic", ["PADT"]),
        "bg": ("Bulgarian", ["BTB"]),
        "ca": ("Catalan", ["AnCora"]),
        "cs": ("Czech", ["PDT", "CAC", "CLTT", "FicTree"]),
        "da": ("Danish", ["DDT"]),
        "de": ("German", ["GSD"]),
        "en": ("English", ["EWT"]),
        "es": ("Spanish", ["GSD", "AnCora"]),
        "et": ("Estonian", ["EDT"]),
        "eu": ("Basque", ["BDT"]),
        "fa": ("Persian", ["Seraji"]),
        "fi": ("Finnish", ["TDT"]),
        "fr": ("French", ["GSD"]),
        "he": ("Hebrew", ["HTB"]),
        "hi": ("Hindi", ["HDTB"]),
        "hr": ("Croatian", ["SET"]),
        "id": ("Indonesian", ["GSD"]),
        "it": ("Italian", ["ISDT"]),
        "ja": ("Japanese", ["GSD"]),
        "ko": ("Korean", ["GSD", "Kaist"]),
        "la": ("Latin", ["PROIEL"]),
        "lv": ("Latvian", ["LVTB"]),
        "nl": ("Dutch", ["Alpino", "LassySmall"]),
        "no": ("Norwegian", ["Bokmaal", "Nynorsk"]),
        "pl": ("Polish", ["LFG", "SZ"]),
        "pt": ("Portuguese", ["Bosque", "GSD"]),
        "ro": ("Romanian", ["RRT"]),
        "ru": ("Russian", ["SynTagRus"]),
        "sk": ("Slovak", ["SNK"]),
        "sl": ("Slovenian", ["SSJ", "SST"]),
        "sv": ("Swedish", ["Talbanken"]),
        "tr": ("Turkish", ["IMST"]),
        "uk": ("Ukrainian", ["IU"]),
        "zh": ("Chinese", ["GSD"]),
    }

    def __init__(
        self,
        ud_path: Path,
        skip_mwe_and_empty: bool = True,
        version: int = 2,
        treebank_dict: Optional[dict] = None,
    ) -> None:
        if treebank_dict is None:
            treebank_dict = self.DEFAULT_TREEBANK
        if version not in (1, 2):
            raise ValueError("version must be 1 or 2")

        self.ud_path = ud_path
        self.treebank_dict = treebank_dict
        self.skip_mwe_and_empty = skip_mwe_and_empty
        self.version = version

    def read_samples(self, langcode: str, which: str = "train") -> Iterator[dict]:
        if self.version == 1:
            return self._read_samples_v1(langcode, which)
        return self._read_samples_v2(langcode, which)

    def _read_samples_v2(self, langcode: str, which: str = "train") -> Iterator[dict]:
        langname, tbnames = self.treebank_dict[langcode]
        for tbname in tbnames:
            path = (
                self.ud_path
                / f"UD_{langname}-{tbname}"
                / f"{langcode}_{tbname.lower()}-ud-{which}.conllu"
            )
            with open(path, encoding="utf-8") as f:
                for sent in parse_incr(f):
                    sample: dict = defaultdict(list)
                    for tok in sent:
                        # skip MWEs or empty words
                        if self.skip_mwe_and_empty and not isinstance(tok["id"], int):
                            continue
                        sample["words"].append(tok["form"])
                        sample["tags"].append(tok["upostag"])
                        sample["heads"].append(tok["head"])
                        sample["types"].append(tok["deprel"])
                    yield sample

    def _read_samples_v1(self, langcode: str, which: str = "train") -> Iterator[dict]:
        langname, _ = self.treebank_dict[langcode]
        path = self.ud_path / f"UD_{langname}" / f"{langcode}-ud-{which}.conllu"
        with open(path, encoding="utf-8") as f:
            for sent in parse_incr(f):
                sample: dict = defaultdict(list)
                for tok in sent:
                    # skip MWEs or empty words
                    if self.skip_mwe_and_empty and not isinstance(tok["id"], int):
                        continue
                    sample["words"].append(tok["form"])
                    sample["tags"].append(tok["upostag"])
                    sample["heads"].append(tok["head"])
                    sample["types"].append(tok["deprel"])
                yield sample


def get_proj_edges(edges: Collection[Tuple[int, int]]) -> Iterator[Tuple[int, int]]:
    """Obtain projective edges from a collection of edges of a dependency tree."""
    adj_set: dict = defaultdict(set)
    for u, v in edges:
        adj_set[u].add(v)

    def dfs(root: int) -> Set[int]:
        stack, seen = [root], set()
        while stack:
            u = stack.pop()
            seen.add(u)
            for v in adj_set[u]:
                if v not in seen:
                    stack.append(v)
        return seen

    nodes = {u for e in edges for u in e}
    reachable_from = {u: dfs(u) for u in nodes}
    for u, v in edges:
        for w in range(min(u, v) + 1, max(u, v)):
            if w not in reachable_from[u]:
                break
        else:
            yield (u, v)
