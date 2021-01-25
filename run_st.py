#!/usr/bin/env python

# Copyright (c) 2021 Kemal Kurniawan

from itertools import chain
from pathlib import Path
import os

from einops import rearrange
from gensim.models.keyedvectors import KeyedVectors
from rnnr import Event, Runner
from rnnr.attachments import EpochTimer, ProgressBar, SumReducer
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from text2array import BucketIterator, ShuffleIterator
import torch

from callbacks import (
    batch2tensors,
    compute_l2_loss,
    compute_total_arc_type_scores,
    evaluate_batch,
    get_n_items,
    log_grads,
    log_stats,
    predict_batch,
    save_state_dict,
    set_train_mode,
    update_params,
)
from ingredients.corpus import ing as corpus_ing, read_samples
from serialization import dump, load
from utils import extend_word_embedding, print_accs

ex = Experiment("xduft-st-testrun", ingredients=[corpus_ing])
ex.captured_out_filter = apply_backspaces_and_linefeeds

# Setup mongodb observer
mongo_url = os.getenv("SACRED_MONGO_URL")
db_name = os.getenv("SACRED_DB_NAME")
if None not in (mongo_url, db_name):
    ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))


@ex.config
def default():
    # directory to save finetuning artifacts
    artifacts_dir = "ft_artifacts"
    # whether to overwrite existing artifacts directory
    overwrite = False
    # discard train/dev/test samples with length greater than these numbers
    max_length = {}
    # load training artifacts from this artifacts directory
    load_from = "artifacts"
    # load trained model parameters from this file under 'load_from' directory
    load_params = "model.pth"
    # device to run on [cpu, cuda]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # path to word embedding in word2vec format
    word_emb_path = "wiki.en.vec"
    # whether to freeze word and tag embedding
    freeze = False
    # whether to operate in the space of projective trees
    projective = False
    # whether to consider multi-root trees (otherwise only single-root trees)
    multiroot = False
    # batch size
    batch_size = 16
    # learning rate
    lr = 1e-5
    # coefficient of L2 regularization against initial parameters
    l2_coef = 1.0
    # max number of epochs
    max_epoch = 5


@ex.named_config
def ahmadetal():
    max_length = {"train": 100}
    batch_size = 80
    corpus = {"normalize_digits": True}


@ex.named_config
def heetal_eval_setup():
    max_length = {"dev": 150, "test": 150}


@ex.named_config
def nearby():
    max_length = {"train": 60}
    lr = 5.6e-4
    l2_coef = 3e-4


@ex.named_config
def distant():
    max_length = {"train": 60}
    lr = 3.7e-4
    l2_coef = 2.8e-4


@ex.named_config
def testrun():
    seed = 12345
    max_epoch = 2
    corpus = dict(portion=0.05)


@ex.capture
def run_eval(
    model, vocab, samples, device="cpu", projective=False, multiroot=True, batch_size=32,
):
    runner = Runner()
    runner.on(
        Event.BATCH,
        [
            batch2tensors(device, vocab),
            set_train_mode(model, training=False),
            compute_total_arc_type_scores(model, vocab),
            predict_batch(projective, multiroot),
            evaluate_batch(),
            get_n_items(),
        ],
    )

    n_tokens = sum(len(s["words"]) for s in samples)
    ProgressBar(leave=False, total=n_tokens, unit="tok").attach_on(runner)
    SumReducer("counts").attach_on(runner)

    with torch.no_grad():
        runner.run(BucketIterator(samples, lambda s: len(s["words"]), batch_size))

    return runner.state


@ex.automain
def finetune(
    _log,
    _run,
    _rnd,
    max_length=None,
    artifacts_dir="ft_artifacts",
    overwrite=False,
    load_from="artifacts",
    load_params="model.pth",
    device="cpu",
    word_emb_path="wiki.id.vec",
    freeze=False,
    projective=False,
    multiroot=True,
    batch_size=32,
    lr=1e-5,
    l2_coef=1.0,
    max_epoch=5,
):
    """Finetune a trained model with self-training."""
    if max_length is None:
        max_length = {}

    artifacts_dir = Path(artifacts_dir)
    _log.info("Creating artifacts directory %s", artifacts_dir)
    artifacts_dir.mkdir(exist_ok=overwrite)

    samples = {
        wh: list(read_samples(which=wh, max_length=max_length.get(wh)))
        for wh in ["train", "dev", "test"]
    }
    for wh in samples:
        n_toks = sum(len(s["words"]) for s in samples[wh])
        _log.info("Read %d %s samples and %d tokens", len(samples[wh]), wh, n_toks)

    path = Path(load_from) / "vocab.yml"
    _log.info("Loading vocabulary from %s", path)
    vocab = load(path.read_text(encoding="utf8"))
    for name in vocab:
        _log.info("Found %d %s", len(vocab[name]), name)

    _log.info("Extending vocabulary with target words")
    vocab.extend(chain(*samples.values()), ["words"])
    _log.info("Found %d words now", len(vocab["words"]))

    path = artifacts_dir / "vocab.yml"
    _log.info("Saving vocabulary to %s", path)
    path.write_text(dump(vocab), encoding="utf8")

    samples = {wh: list(vocab.stoi(samples[wh])) for wh in samples}

    path = Path(load_from) / "model.yml"
    _log.info("Loading model from metadata %s", path)
    model = load(path.read_text(encoding="utf8"))

    path = Path(load_from) / load_params
    _log.info("Loading model parameters from %s", path)
    model.load_state_dict(torch.load(path, "cpu"))

    _log.info("Creating extended word embedding layer")
    kv = KeyedVectors.load_word2vec_format(word_emb_path)
    assert model.word_emb.embedding_dim == kv.vector_size
    with torch.no_grad():
        model.word_emb = torch.nn.Embedding.from_pretrained(
            extend_word_embedding(model.word_emb.weight, vocab["words"], kv)
        )

    path = artifacts_dir / "model.yml"
    _log.info("Saving model metadata to %s", path)
    path.write_text(dump(model), encoding="utf8")

    model.word_emb.requires_grad_(not freeze)
    model.tag_emb.requires_grad_(not freeze)
    model.to(device)

    for wh in ["train"]:
        for i, s in enumerate(samples[wh]):
            s["_id"] = i

        runner = Runner()
        runner.state.update({"st_heads": [], "st_types": [], "_ids": []})
        runner.on(
            Event.BATCH,
            [
                batch2tensors(device, vocab),
                set_train_mode(model, training=False),
                compute_total_arc_type_scores(model, vocab),
                predict_batch(projective, multiroot),
            ],
        )

        @runner.on(Event.BATCH)
        def save_st_trees(state):
            state["st_heads"].extend(state["pred_heads"].tolist())
            state["st_types"].extend(state["pred_types"].tolist())
            state["_ids"].extend(state["batch"]["_id"].tolist())
            state["n_items"] = state["batch"]["words"].numel()

        n_toks = sum(len(s["words"]) for s in samples[wh])
        ProgressBar(total=n_toks, unit="tok").attach_on(runner)

        _log.info("Computing ST trees for %s set", wh)
        with torch.no_grad():
            runner.run(BucketIterator(samples[wh], lambda s: len(s["words"]), batch_size))

        assert len(runner.state["st_heads"]) == len(samples[wh])
        assert len(runner.state["st_types"]) == len(samples[wh])
        assert len(runner.state["_ids"]) == len(samples[wh])
        for i, st_heads, st_types in zip(
            runner.state["_ids"], runner.state["st_heads"], runner.state["st_types"]
        ):
            assert len(samples[wh][i]["words"]) == len(st_heads)
            assert len(samples[wh][i]["words"]) == len(st_types)
            samples[wh][i]["st_heads"] = st_heads
            samples[wh][i]["st_types"] = st_types

    _log.info("Creating optimizer")
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    finetuner = Runner()
    origin_params = {name: p.clone().detach() for name, p in model.named_parameters()}
    finetuner.on(
        Event.BATCH,
        [
            batch2tensors(device, vocab),
            set_train_mode(model),
            compute_l2_loss(model, origin_params),
        ],
    )

    @finetuner.on(Event.BATCH)
    def compute_loss(state):
        bat = state["batch"]
        words, tags, heads, types = bat["words"], bat["tags"], bat["st_heads"], bat["st_types"]
        mask = bat["mask"]

        arc_scores, type_scores = model(words, tags, mask, heads)
        arc_scores = arc_scores.masked_fill(~mask.unsqueeze(2), -1e9)  # mask padding heads
        type_scores[..., vocab["types"].index(vocab.PAD_TOKEN)] = -1e9

        # remove root
        arc_scores, type_scores = arc_scores[:, :, 1:], type_scores[:, 1:]
        heads, types, mask = heads[:, 1:], types[:, 1:], mask[:, 1:]

        arc_scores = rearrange(arc_scores, "bsz slen1 slen2 -> (bsz slen2) slen1")
        heads = heads.reshape(-1)
        arc_loss = torch.nn.functional.cross_entropy(arc_scores, heads, reduction="none")

        type_scores = rearrange(type_scores, "bsz slen ntypes -> (bsz slen) ntypes")
        types = types.reshape(-1)
        type_loss = torch.nn.functional.cross_entropy(type_scores, types, reduction="none")

        arc_loss = arc_loss.masked_select(mask.reshape(-1)).mean()
        type_loss = type_loss.masked_select(mask.reshape(-1)).mean()
        loss = arc_loss + type_loss + l2_coef * state["l2_loss"]

        state["loss"] = loss
        state["stats"] = {
            "arc_ppl": arc_loss.exp().item(),
            "type_ppl": type_loss.exp().item(),
            "l2_loss": state["l2_loss"].item(),
        }
        state["extra_stats"] = {"arc_loss": arc_loss.item(), "type_loss": type_loss.item()}

    finetuner.on(
        Event.BATCH,
        [get_n_items(), update_params(opt), log_grads(_run, model), log_stats(_run)],
    )

    @finetuner.on(Event.EPOCH_FINISHED)
    def eval_on_dev(state):
        _log.info("Evaluating on dev")
        eval_state = run_eval(model, vocab, samples["dev"])
        accs = eval_state["counts"].accs
        print_accs(accs, run=_run, step=state["n_iters"])
        state["dev_accs"] = accs

    @finetuner.on(Event.EPOCH_FINISHED)
    def maybe_eval_on_test(state):
        if state["epoch"] != max_epoch:
            return

        _log.info("Evaluating on test")
        eval_state = run_eval(model, vocab, samples["test"])
        print_accs(eval_state["counts"].accs, on="test", run=_run, step=state["n_iters"])

    finetuner.on(Event.EPOCH_FINISHED, save_state_dict("model", model, under=artifacts_dir))

    EpochTimer().attach_on(finetuner)
    n_tokens = sum(len(s["words"]) for s in samples["train"])
    ProgressBar(stats="stats", total=n_tokens, unit="tok").attach_on(finetuner)

    bucket_key = lambda s: (len(s["words"]) - 1) // 10
    trn_iter = ShuffleIterator(
        BucketIterator(samples["train"], bucket_key, batch_size, shuffle_bucket=True, rng=_rnd),
        rng=_rnd,
    )
    _log.info("Starting finetuning")
    try:
        finetuner.run(trn_iter, max_epoch)
    except KeyboardInterrupt:
        _log.info("Interrupt detected, training will abort")
    else:
        return finetuner.state["dev_accs"]["las_nopunct"]
