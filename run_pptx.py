#!/usr/bin/env python

# Copyright (c) 2021 Kemal Kurniawan

from itertools import chain
from pathlib import Path
import os
import pickle

from gensim.models.keyedvectors import KeyedVectors
from rnnr import Event, Runner
from rnnr.attachments import EpochTimer, MeanReducer, ProgressBar, SumReducer
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from text2array import ShuffleIterator
import text2array
import torch

from aatrn import compute_aatrn_loss, compute_ambiguous_arcs_mask
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
from utils import extend_word_embedding, print_accs, report_log_ntrees_stats

ex = Experiment("xduft-pptx-testrun", ingredients=[corpus_ing])
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
    # load source models from these directories and parameters {key: (load_from, load_params)}
    load_src = {}
    # whether to treat keys in load_src as lang codes
    src_key_as_lang = False
    # the main source to start finetuning from
    main_src = ""
    # device to run on [cpu, cuda]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # path to word embedding in word2vec format
    word_emb_path = "wiki.en.vec"
    # whether to freeze word and tag embedding
    freeze = False
    # cumulative prob threshold
    thresh = 0.95
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
    # whether to save the final samples as an artifact
    save_samples = False
    # load samples from this file (*.pkl)
    load_samples_from = ""


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
    max_length = {"train": 30}
    lr = 2.1e-5
    l2_coef = 0.079


@ex.named_config
def distant():
    max_length = {"train": 30}
    lr = 5.9e-5
    l2_coef = 1.2e-4


@ex.named_config
def repr_nearby():
    max_length = {"train": 30}
    lr = 1.7e-5
    l2_coef = 4e-4


@ex.named_config
def repr_distant():
    max_length = {"train": 30}
    lr = 9.7e-5
    l2_coef = 0.084


@ex.named_config
def prag_nearby():
    max_length = {"train": 30}
    lr = 4.4e-5
    l2_coef = 2.7e-4


@ex.named_config
def prag_distant():
    max_length = {"train": 30}
    lr = 8.5e-5
    l2_coef = 2.8e-5


@ex.named_config
def prag_proj_nearby():
    projective = True
    max_length = {"train": 20}
    lr = 9.4e-5
    l2_coef = 2.4e-4


@ex.named_config
def prag_proj_distant():
    projective = True
    max_length = {"train": 20}
    lr = 9.4e-5
    l2_coef = 2.4e-4


@ex.named_config
def testrun():
    seed = 12345
    max_epoch = 2
    corpus = dict(portion=0.05)


class BucketIterator(text2array.BucketIterator):
    def __iter__(self):
        for ss in self._buckets:
            if self._shuf and len(ss) > 1:
                ss = ShuffleIterator(ss, key=lambda s: len(s["words"]), rng=self._rng)
            yield from text2array.BatchIterator(ss, self._bsz)


@ex.capture
def run_eval(
    model,
    vocab,
    samples,
    compute_loss=True,
    device="cpu",
    projective=False,
    multiroot=True,
    batch_size=32,
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

    @runner.on(Event.BATCH)
    def maybe_compute_loss(state):
        if not compute_loss:
            return

        pptx_loss = compute_aatrn_loss(
            state["total_arc_type_scores"],
            state["batch"]["pptx_mask"].bool(),
            projective=projective,
            multiroot=multiroot,
        )
        state["pptx_loss"] = pptx_loss.item()
        state["size"] = state["batch"]["words"].size(0)

    n_tokens = sum(len(s["words"]) for s in samples)
    ProgressBar(leave=False, total=n_tokens, unit="tok").attach_on(runner)
    SumReducer("counts").attach_on(runner)
    if compute_loss:
        MeanReducer("mean_pptx_loss", value="pptx_loss").attach_on(runner)

    with torch.no_grad():
        runner.run(BucketIterator(samples, lambda s: len(s["words"]), batch_size))

    return runner.state


@ex.automain
def finetune(
    corpus,
    _log,
    _run,
    _rnd,
    max_length=None,
    artifacts_dir="ft_artifacts",
    load_samples_from=None,
    overwrite=False,
    load_src=None,
    src_key_as_lang=False,
    main_src=None,
    device="cpu",
    word_emb_path="wiki.id.vec",
    freeze=False,
    thresh=0.95,
    projective=False,
    multiroot=True,
    batch_size=32,
    save_samples=False,
    lr=1e-5,
    l2_coef=1.0,
    max_epoch=5,
):
    """Finetune a trained model with PPTX."""
    if max_length is None:
        max_length = {}
    if load_src is None:
        load_src = {"src": ("artifacts", "model.pth")}
        main_src = "src"
    elif main_src not in load_src:
        raise ValueError(f"{main_src} not found in load_src")

    artifacts_dir = Path(artifacts_dir)
    _log.info("Creating artifacts directory %s", artifacts_dir)
    artifacts_dir.mkdir(exist_ok=overwrite)

    if load_samples_from:
        _log.info("Loading samples from %s", load_samples_from)
        with open(load_samples_from, "rb") as f:
            samples = pickle.load(f)
    else:
        samples = {
            wh: list(read_samples(which=wh, max_length=max_length.get(wh)))
            for wh in ["train", "dev", "test"]
        }
    for wh in samples:
        n_toks = sum(len(s["words"]) for s in samples[wh])
        _log.info("Read %d %s samples and %d tokens", len(samples[wh]), wh, n_toks)

    kv = KeyedVectors.load_word2vec_format(word_emb_path)

    if load_samples_from:
        _log.info("Skipping non-main src because samples are processed and loaded")
        srcs = []
    else:
        srcs = [src for src in load_src if src != main_src]
        if src_key_as_lang and corpus["lang"] in srcs:
            _log.info("Removing %s from src parsers because it's the tgt", corpus["lang"])
            srcs.remove(corpus["lang"])
    srcs.append(main_src)

    for src_i, src in enumerate(srcs):
        _log.info("Processing src %s [%d/%d]", src, src_i + 1, len(srcs))
        load_from, load_params = load_src[src]
        path = Path(load_from) / "vocab.yml"
        _log.info("Loading %s vocabulary from %s", src, path)
        vocab = load(path.read_text(encoding="utf8"))
        for name in vocab:
            _log.info("Found %d %s", len(vocab[name]), name)

        _log.info("Extending %s vocabulary with target words", src)
        vocab.extend(chain(*samples.values()), ["words"])
        _log.info("Found %d words now", len(vocab["words"]))

        samples_ = {wh: list(vocab.stoi(samples[wh])) for wh in samples}

        path = Path(load_from) / "model.yml"
        _log.info("Loading %s model from metadata %s", src, path)
        model = load(path.read_text(encoding="utf8"))

        path = Path(load_from) / load_params
        _log.info("Loading %s model parameters from %s", src, path)
        model.load_state_dict(torch.load(path, "cpu"))

        _log.info("Creating %s extended word embedding layer", src)
        assert model.word_emb.embedding_dim == kv.vector_size
        with torch.no_grad():
            model.word_emb = torch.nn.Embedding.from_pretrained(
                extend_word_embedding(model.word_emb.weight, vocab["words"], kv)
            )
        model.to(device)

        for wh in ["train", "dev"]:
            if load_samples_from:
                assert all("pptx_mask" in s for s in samples[wh])
                continue

            for i, s in enumerate(samples_[wh]):
                s["_id"] = i

            runner = Runner()
            runner.state.update({"pptx_masks": [], "_ids": []})
            runner.on(
                Event.BATCH,
                [
                    batch2tensors(device, vocab),
                    set_train_mode(model, training=False),
                    compute_total_arc_type_scores(model, vocab),
                ],
            )

            @runner.on(Event.BATCH)
            def compute_pptx_ambiguous_arcs_mask(state):
                assert state["batch"]["mask"].all()
                scores = state["total_arc_type_scores"]
                pptx_mask = compute_ambiguous_arcs_mask(scores, thresh, projective, multiroot)
                state["pptx_masks"].extend(pptx_mask)
                state["_ids"].extend(state["batch"]["_id"].tolist())
                state["n_items"] = state["batch"]["words"].numel()

            n_toks = sum(len(s["words"]) for s in samples_[wh])
            ProgressBar(total=n_toks, unit="tok").attach_on(runner)

            _log.info("Computing PPTX ambiguous arcs mask for %s set with source %s", wh, src)
            with torch.no_grad():
                runner.run(BucketIterator(samples_[wh], lambda s: len(s["words"]), batch_size))

            assert len(runner.state["pptx_masks"]) == len(samples_[wh])
            assert len(runner.state["_ids"]) == len(samples_[wh])
            for i, pptx_mask in zip(runner.state["_ids"], runner.state["pptx_masks"]):
                samples_[wh][i]["pptx_mask"] = pptx_mask.tolist()

            _log.info("Computing (log) number of trees stats on %s set", wh)
            report_log_ntrees_stats(
                samples_[wh], "pptx_mask", batch_size, projective, multiroot
            )

            _log.info("Combining the ambiguous arcs mask")
            assert len(samples_[wh]) == len(samples[wh])
            for i in range(len(samples_[wh])):
                pptx_mask = torch.tensor(samples_[wh][i]["pptx_mask"])
                assert pptx_mask.dim() == 3
                if "pptx_mask" in samples[wh][i]:
                    old_mask = torch.tensor(samples[wh][i]["pptx_mask"])
                else:
                    old_mask = torch.zeros(1, 1, 1).bool()
                samples[wh][i]["pptx_mask"] = (old_mask | pptx_mask).tolist()

    assert src == main_src
    _log.info("Main source is %s", src)

    path = artifacts_dir / "vocab.yml"
    _log.info("Saving vocabulary to %s", path)
    path.write_text(dump(vocab), encoding="utf8")

    path = artifacts_dir / "model.yml"
    _log.info("Saving model metadata to %s", path)
    path.write_text(dump(model), encoding="utf8")

    if save_samples:
        path = artifacts_dir / "samples.pkl"
        _log.info("Saving samples to %s", path)
        with open(path, "wb") as f:
            pickle.dump(samples, f)

    samples = {wh: list(vocab.stoi(samples[wh])) for wh in samples}

    for wh in ["train", "dev"]:
        _log.info("Computing (log) number of trees stats on %s set", wh)
        report_log_ntrees_stats(samples[wh], "pptx_mask", batch_size, projective, multiroot)

    model.word_emb.requires_grad_(not freeze)
    model.tag_emb.requires_grad_(not freeze)

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
            compute_total_arc_type_scores(model, vocab),
        ],
    )

    @finetuner.on(Event.BATCH)
    def compute_loss(state):
        mask = state["batch"]["mask"]
        pptx_mask = state["batch"]["pptx_mask"].bool()
        scores = state["total_arc_type_scores"]

        pptx_loss = compute_aatrn_loss(scores, pptx_mask, mask, projective, multiroot)
        pptx_loss /= mask.size(0)
        loss = pptx_loss + l2_coef * state["l2_loss"]

        state["loss"] = loss
        state["stats"] = {
            "pptx_loss": pptx_loss.item(),
            "l2_loss": state["l2_loss"].item(),
        }
        state["extra_stats"] = {"loss": loss.item()}
        state["n_items"] = mask.long().sum().item()

    finetuner.on(Event.BATCH, [update_params(opt), log_grads(_run, model), log_stats(_run)])

    @finetuner.on(Event.EPOCH_FINISHED)
    def eval_on_dev(state):
        _log.info("Evaluating on dev")
        eval_state = run_eval(model, vocab, samples["dev"])
        accs = eval_state["counts"].accs
        print_accs(accs, run=_run, step=state["n_iters"])

        pptx_loss = eval_state["mean_pptx_loss"]
        _log.info("dev_pptx_loss: %.4f", pptx_loss)
        _run.log_scalar("dev_pptx_loss", pptx_loss, step=state["n_iters"])

        state["dev_accs"] = accs

    @finetuner.on(Event.EPOCH_FINISHED)
    def maybe_eval_on_test(state):
        if state["epoch"] != max_epoch:
            return

        _log.info("Evaluating on test")
        eval_state = run_eval(model, vocab, samples["test"], compute_loss=False)
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
