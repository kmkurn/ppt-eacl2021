#!/usr/bin/env python

# Copyright (c) 2021 Kemal Kurniawan

from itertools import chain
from pathlib import Path
import os
import math

from einops import rearrange
from gensim.models.keyedvectors import KeyedVectors
from rnnr import Event, Runner
from rnnr.attachments import EpochTimer, LambdaReducer, ProgressBar, SumReducer
from rnnr.callbacks import maybe_stop_early
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from text2array import BucketIterator, ShuffleIterator, Vocab
import torch

from callbacks import (
    batch2tensors,
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
from models import SelfAttGraph
from serialization import dump, load
from utils import extend_word_embedding, print_accs

ex = Experiment("xduft-parser-testrun", ingredients=[corpus_ing])
ex.captured_out_filter = apply_backspaces_and_linefeeds

# Setup mongodb observer
mongo_url = os.getenv("SACRED_MONGO_URL")
db_name = os.getenv("SACRED_DB_NAME")
if None not in (mongo_url, db_name):
    ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))


@ex.config
def default():
    # directory to save training artifacts
    artifacts_dir = "artifacts"
    # whether to overwrite existing artifacts directory
    overwrite = False
    # discard train/dev/test samples with length greater than these numbers
    max_length = {}
    # path to word embedding in word2vec format
    word_emb_path = "wiki.en.vec"
    # size of POS tag embedding
    tag_size = 50
    # number of heads in transformer encoder
    n_heads = 10
    # number of layers in transformer encoder
    n_layers = 6
    # size of feedforward hidden layer in transformer encoder
    ff_size = 2048
    # size of keys and values in the transformer encoder
    kv_size = 35
    # word dropout rate
    p_word = 0.5
    # output dim dropout rate
    p_out = 0.5
    # size of dep arc representation
    arc_size = 128
    # size of dep type representation
    type_size = 128
    # batch size
    batch_size = 16
    # learning rate
    lr = 1e-4
    # max number of epochs
    max_epoch = 1000
    # device to run on [cpu, cuda]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # how many epochs to wait before early stopping
    patience = 50
    # whether to operate in the space of projective trees
    projective = False
    # whether to consider multi-root trees (otherwise only single-root trees)
    multiroot = True
    # load parameters from this file under artifacts directory (only for evaluate)
    load_params = "model.pth"
    # whether to do type-wise evaluation (only for evaluate)
    type_wise = False
    # load types vocabulary from this file
    load_types_vocab_from = ""


@ex.named_config
def testrun():
    seed = 12345
    tag_size = 10
    n_heads = 2
    n_layers = 2
    ff_size = 7
    kv_size = 6
    arc_size = 3
    type_size = 3
    max_epoch = 3
    corpus = dict(portion=0.05)


@ex.named_config
def ahmadetal():
    max_length = dict(train=100, dev=140, test=140)
    tag_size = 50
    n_heads = 8
    n_layers = 6
    ff_size = 512
    kv_size = 64
    p_word = 0.2
    p_out = 0.2
    arc_size = 512
    type_size = 128
    batch_size = 80
    lr = 1e-4
    corpus = dict(normalize_digits=True)


@ex.named_config
def heetal_eval_setup():
    max_length = dict(dev=150, test=150)


@ex.capture
def make_model(
    vocab,
    _log,
    word_emb_path="wiki.en.vec",
    artifacts_dir="artifacts",
    tag_size=50,
    n_heads=10,
    n_layers=6,
    ff_size=2048,
    kv_size=64,
    p_word=0.5,
    p_out=0.5,
    arc_size=128,
    type_size=128,
):
    kv = KeyedVectors.load_word2vec_format(word_emb_path)

    _log.info("Creating model")
    model = SelfAttGraph(
        len(vocab["words"]),
        len(vocab["types"]),
        len(vocab["tags"]),
        word_size=kv.vector_size,
        tag_size=tag_size,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_size=ff_size,
        kv_size=kv_size,
        word_dropout=p_word,
        outdim_dropout=p_out,
        arc_size=arc_size,
        type_size=type_size,
    )
    _log.info("Model created with %d parameters", sum(p.numel() for p in model.parameters()))

    weight = torch.randn(len(vocab["words"]), kv.vector_size)
    cnt_pre, cnt_unk = 0, 0
    for w in vocab["words"]:
        wid = vocab["words"].index(w)
        if w in kv:
            weight[wid] = torch.from_numpy(kv[w])
            cnt_pre += 1
        elif w.lower() in kv:
            weight[wid] = torch.from_numpy(kv[w.lower()])
            cnt_pre += 1
        else:
            cnt_unk += 1

    with torch.no_grad():
        # freeze embedding to preserve alignment
        model.word_emb = torch.nn.Embedding.from_pretrained(weight, freeze=True)
    _log.info("Initialized %d words with pre-trained embedding", cnt_pre)
    _log.info("Found %d unknown words", cnt_unk)

    path = Path(artifacts_dir) / "model.yml"
    _log.info("Saving model metadata to %s", path)
    path.write_text(dump(model), encoding="utf8")

    return model


@ex.capture
def run_eval(
    model,
    vocab,
    samples,
    device="cpu",
    projective=False,
    multiroot=True,
    batch_size=32,
    type_wise=False,
):
    runner = Runner()
    runner.on(
        Event.BATCH,
        [
            batch2tensors(device, vocab),
            set_train_mode(model, training=False),
            compute_total_arc_type_scores(model, vocab),
            predict_batch(projective, multiroot),
            evaluate_batch(vocab["types"] if type_wise else None),
            get_n_items(),
        ],
    )

    n_tokens = sum(len(s["words"]) for s in samples)
    ProgressBar(leave=False, total=n_tokens, unit="tok").attach_on(runner)
    SumReducer("counts").attach_on(runner)
    if type_wise:
        LambdaReducer(
            "type2counts",
            lambda o1, o2: {y: o1[y] + o2[y] for y in vocab["types"]},
            value="tw_output",
        ).attach_on(runner)

    with torch.no_grad():
        runner.run(BucketIterator(samples, lambda s: len(s["words"]), batch_size))

    return runner.state


@ex.command
def evaluate(
    _log,
    _run,
    max_length=None,
    artifacts_dir="artifacts",
    load_params="model.pth",
    word_emb_path="wiki.id.vec",
    device="cpu",
):
    """Evaluate a trained self-attention graph-based parser."""
    if max_length is None:
        max_length = {}

    artifacts_dir = Path(artifacts_dir)

    samples = {}
    try:
        samples["dev"] = list(read_samples(which="dev", max_length=max_length.get("dev")))
    except FileNotFoundError:
        _log.info("Dev set is not found, skipping")
    samples["test"] = list(read_samples(which="test", max_length=max_length.get("test")))

    for wh in samples:
        n_toks = sum(len(s["words"]) for s in samples[wh])
        _log.info("Read %d %s samples and %d tokens", len(samples[wh]), wh, n_toks)

    path = artifacts_dir / "vocab.yml"
    _log.info("Loading source vocabulary from %s", path)
    vocab = load(path.read_text(encoding="utf8"))
    for name in vocab.keys():
        _log.info("Found %d %s", len(vocab[name]), name)

    _log.info("Extending vocab with target words")
    old_n_words = len(vocab["words"])
    vocab.extend(chain(*samples.values()), ["words"])
    _log.info("Found %d words now", len(vocab["words"]))

    samples = {wh: list(vocab.stoi(samples[wh])) for wh in samples}

    path = artifacts_dir / "model.yml"
    _log.info("Loading model from metadata %s", path)
    model = load(path.read_text(encoding="utf8"))

    path = artifacts_dir / load_params
    _log.info("Loading model parameters from %s", path)
    model.load_state_dict(torch.load(path, "cpu"))

    if len(vocab["words"]) > old_n_words:
        _log.info("Creating extended word embedding layer")
        if word_emb_path:
            kv = KeyedVectors.load_word2vec_format(word_emb_path)
            assert model.word_emb.embedding_dim == kv.vector_size
        else:
            _log.warning(
                "Word embedding file not specified; any extra target words will be treated as unks"
            )
            kv = None
        with torch.no_grad():
            model.word_emb = torch.nn.Embedding.from_pretrained(
                extend_word_embedding(
                    model.word_emb.weight,
                    vocab["words"],
                    kv,
                    vocab["words"].index(vocab.UNK_TOKEN),
                )
            )

    model.to(device)
    dev_accs = {}
    for wh in samples:
        _log.info("Evaluating on %s", wh)
        state = run_eval(model, vocab, samples[wh])
        accs = state["counts"].accs
        if wh == "dev":
            dev_accs = accs
        print_accs(accs, on=wh, run=_run)

        if "type2counts" in state:
            _log.info("Type-wise accuracies:")
            for type_, c in state["type2counts"].items():
                for key, acc in c.accs.items():
                    metric_name = f"{wh}_{type_}_{key}"
                    _log.info(f"{metric_name}: {acc:.2%}")
                    _run.log_scalar(metric_name, acc)

                for suffix in ("", "_nopunct"):
                    metric_name = f"{wh}_{type_}_n_arcs{suffix}"
                    _log.info("%s: %d", metric_name, getattr(c, f"n_arcs{suffix}"))
                    _run.log_scalar(metric_name, getattr(c, f"n_arcs{suffix}"))

    return dev_accs.get("las_nopunct")


@ex.automain
def train(
    _log,
    _run,
    _rnd,
    artifacts_dir="artifacts",
    overwrite=False,
    max_length=None,
    load_types_vocab_from=None,
    batch_size=16,
    device="cpu",
    lr=0.001,
    patience=5,
    max_epoch=1000,
):
    """Train a self-attention graph-based parser."""
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

    _log.info("Creating vocabulary")
    vocab = Vocab.from_samples(chain(*samples.values()))
    if load_types_vocab_from:
        path = Path(load_types_vocab_from)
        _log.info("Loading types vocab from %s", path)
        vocab["types"] = load(path.read_text(encoding="utf8"))["types"]

    _log.info("Vocabulary created")
    for name in vocab:
        _log.info("Found %d %s", len(vocab[name]), name)

    path = artifacts_dir / "vocab.yml"
    _log.info("Saving vocabulary to %s", path)
    path.write_text(dump(vocab), encoding="utf8")

    samples = {wh: list(vocab.stoi(samples[wh])) for wh in samples}

    model = make_model(vocab)
    model.to(device)

    _log.info("Creating optimizer")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5)

    trainer = Runner()
    trainer.state.update({"dev_larcs_nopunct": -1, "dev_uarcs_nopunct": -1})
    trainer.on(Event.BATCH, [batch2tensors(device, vocab), set_train_mode(model)])

    @trainer.on(Event.BATCH)
    def compute_loss(state):
        bat = state["batch"]
        words, tags, heads, types = bat["words"], bat["tags"], bat["heads"], bat["types"]
        mask = bat["mask"]

        arc_scores, type_scores = model(words, tags, mask, heads)
        arc_scores = arc_scores.masked_fill(~mask.unsqueeze(2), -1e9)  # mask padding heads
        type_scores[..., vocab["types"].index(Vocab.PAD_TOKEN)] = -1e9

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
        loss = arc_loss + type_loss

        state["loss"] = loss
        arc_loss, type_loss = arc_loss.item(), type_loss.item()
        state["stats"] = {
            "arc_ppl": math.exp(arc_loss),
            "type_ppl": math.exp(type_loss),
        }
        state["extra_stats"] = {"arc_loss": arc_loss, "type_loss": type_loss}
        state["n_items"] = bat["mask"].long().sum().item()

    trainer.on(Event.BATCH, [update_params(opt), log_grads(_run, model), log_stats(_run)])

    @trainer.on(Event.EPOCH_FINISHED)
    def eval_on_dev(state):
        _log.info("Evaluating on dev")
        eval_state = run_eval(model, vocab, samples["dev"])
        accs = eval_state["counts"].accs
        print_accs(accs, run=_run, step=state["n_iters"])

        scheduler.step(accs["las_nopunct"])

        if eval_state["counts"].larcs_nopunct > state["dev_larcs_nopunct"]:
            state["better"] = True
        elif eval_state["counts"].larcs_nopunct < state["dev_larcs_nopunct"]:
            state["better"] = False
        elif eval_state["counts"].uarcs_nopunct > state["dev_uarcs_nopunct"]:
            state["better"] = True
        else:
            state["better"] = False

        if state["better"]:
            _log.info("Found new best result on dev!")
            state["dev_larcs_nopunct"] = eval_state["counts"].larcs_nopunct
            state["dev_uarcs_nopunct"] = eval_state["counts"].uarcs_nopunct
            state["dev_accs"] = accs
            state["dev_epoch"] = state["epoch"]
        else:
            _log.info("Not better, the best so far is epoch %d:", state["dev_epoch"])
            print_accs(state["dev_accs"])
            print_accs(state["test_accs"], on="test")

    @trainer.on(Event.EPOCH_FINISHED)
    def maybe_eval_on_test(state):
        if not state["better"]:
            return

        _log.info("Evaluating on test")
        eval_state = run_eval(model, vocab, samples["test"])
        state["test_accs"] = eval_state["counts"].accs
        print_accs(state["test_accs"], on="test", run=_run, step=state["n_iters"])

    trainer.on(
        Event.EPOCH_FINISHED,
        [
            maybe_stop_early(patience=patience),
            save_state_dict("model", model, under=artifacts_dir, when="better"),
        ],
    )

    EpochTimer().attach_on(trainer)
    n_tokens = sum(len(s["words"]) for s in samples["train"])
    ProgressBar(stats="stats", total=n_tokens, unit="tok").attach_on(trainer)

    bucket_key = lambda s: (len(s["words"]) - 1) // 10
    trn_iter = ShuffleIterator(
        BucketIterator(samples["train"], bucket_key, batch_size, shuffle_bucket=True, rng=_rnd),
        rng=_rnd,
    )
    _log.info("Starting training")
    try:
        trainer.run(trn_iter, max_epoch)
    except KeyboardInterrupt:
        _log.info("Interrupt detected, training will abort")
    else:
        return trainer.state["dev_accs"]["las_nopunct"]
