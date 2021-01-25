#!/usr/bin/env python

# Copyright (c) 2021 Kemal Kurniawan

from itertools import chain
import os

from gensim.models.keyedvectors import KeyedVectors
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from text2array import Vocab

from ingredients.corpus import ing as corpus_ing, read_samples

ex = Experiment("xduft-minimize-vectors-file-testrun", ingredients=[corpus_ing])
ex.captured_out_filter = apply_backspaces_and_linefeeds

# Setup mongodb observer
mongo_url = os.getenv("SACRED_MONGO_URL")
db_name = os.getenv("SACRED_DB_NAME")
if None not in (mongo_url, db_name):
    ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))


@ex.config
def default():
    # path to vectors file in word2vec format
    vectors_path = "wiki.en.vec"
    # write minimized vectors to this file path
    output_path = "wiki.min.en.vec"


@ex.automain
def minimize(_log, vectors_path="wiki.en.vec", output_path="wiki.min.en.vec"):
    """Minimize the given vectors file to contain only words in the given corpus."""
    samples = {wh: list(read_samples(which=wh)) for wh in ["train", "test"]}
    try:
        samples["dev"] = list(read_samples(which="dev"))
    except FileNotFoundError:
        pass  # skip if not exist

    vocab = Vocab.from_samples(chain(*samples.values()))
    kv = KeyedVectors.load_word2vec_format(vectors_path)

    _log.info("Creating new, minimized word vectors")
    min_kv = KeyedVectors(kv.vector_size)
    for w in kv.vocab:
        if w in vocab["words"]:
            min_kv[w] = kv[w]

    _log.info("Saving the new word vectors to %s", output_path)
    min_kv.save_word2vec_format(output_path)
