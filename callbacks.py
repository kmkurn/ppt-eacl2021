# Copyright (c) 2021 Kemal Kurniawan

from typing import Callable, Dict, Optional, Sequence

from rnnr.callbacks import save
from sacred.run import Run
from text2array import Vocab
import torch

from crf import DepTreeCRF
from evaluation import count_correct


def set_train_mode(model: torch.nn.Module, training: bool = True) -> Callable[[dict], None]:
    def callback(state):
        model.train(training)

    return callback


def batch2tensors(device: str, vocab: Vocab) -> Callable[[dict], None]:
    def callback(state):
        batch = state["batch"].to_array()
        for k in batch:
            batch[k] = torch.from_numpy(batch[k]).long().to(device)
        batch["proj?"] = batch["proj?"].bool()
        batch["punct?"] = batch["punct?"].bool()
        batch["mask"] = batch["words"] != vocab["words"].index(Vocab.PAD_TOKEN)
        state["batch"] = batch

    return callback


def update_params(opt: torch.optim.Optimizer) -> Callable[[dict], None]:
    def callback(state):
        opt.zero_grad()
        state["loss"].backward()
        opt.step()

    return callback


def log_grads(run: Run, model: torch.nn.Module, every: int = 10) -> Callable[[dict], None]:
    def callback(state):
        if state["n_iters"] % every != 0:
            return

        for name, p in model.named_parameters():
            if p.requires_grad:
                run.log_scalar(f"grad_{name}", p.grad.norm().item(), state["n_iters"])

    return callback


def log_stats(run: Run, every: int = 10) -> Callable[[dict], None]:
    def callback(state):
        if state["n_iters"] % every != 0:
            return

        for name, value in state["stats"].items():
            run.log_scalar(f"batch_{name}", value, state["n_iters"])
        for name, value in state.get("extra_stats", {}).items():
            run.log_scalar(f"batch_{name}", value, state["n_iters"])

    return callback


def save_state_dict(*args, **kwargs) -> Callable[[dict], None]:
    kwargs.update({"using": lambda m, p: torch.save(m.state_dict(), p), "ext": "pth"})
    return save(*args, **kwargs)


def compute_total_arc_type_scores(
    model: torch.nn.Module, vocab: Vocab
) -> Callable[[dict], None]:
    def callback(state):
        bat = state["batch"]
        words, tags, mask = bat["words"], bat["tags"], bat["mask"]

        arc_scores, type_scores = model(words, tags, mask)
        type_scores[..., vocab["types"].index(vocab.PAD_TOKEN)] = -1e9  # mask padding type
        _, HEAD, DEPD, TYPE = range(4)
        arc_scores = arc_scores.masked_fill(~mask.unsqueeze(DEPD), -1e9)  # mask padding heads
        arc_scores = arc_scores.log_softmax(dim=HEAD)
        type_scores = type_scores.log_softmax(dim=TYPE)
        state["total_arc_type_scores"] = arc_scores.unsqueeze(TYPE) + type_scores

    return callback


def predict_batch(
    projective=False, multiroot=False, scores="total_arc_type_scores"
) -> Callable[[dict], None]:
    def callback(state):
        assert state["batch"]["mask"].all()
        crf = DepTreeCRF(state[scores], projective=projective, multiroot=multiroot)
        pred_heads, pred_types = crf.argmax()
        state["pred_heads"] = pred_heads
        state["pred_types"] = pred_types

    return callback


def get_n_items() -> Callable[[dict], None]:
    def callback(state):
        state["n_items"] = state["batch"]["mask"].long().sum().item()

    return callback


def evaluate_batch(type_vocab: Optional[Sequence[str]] = None) -> Callable[[dict], None]:
    def callback(state):
        bat = state["batch"]
        words, tags, heads, types = bat["words"], bat["tags"], bat["heads"], bat["types"]
        pred_heads, pred_types = state["pred_heads"], state["pred_types"]
        mask, proj_mask, punct_mask = bat["mask"], bat["proj?"], bat["punct?"]

        # remove root
        words, tags, heads, types = words[:, 1:], tags[:, 1:], heads[:, 1:], types[:, 1:]
        pred_heads, pred_types = pred_heads[:, 1:], pred_types[:, 1:]
        mask, proj_mask, punct_mask = mask[:, 1:], proj_mask[:, 1:], punct_mask[:, 1:]

        counts = count_correct(
            heads, types, pred_heads, pred_types, mask, ~punct_mask, proj_mask
        )
        state["output"] = counts

        if type_vocab is not None:
            state["tw_output"] = {
                y: count_correct(
                    heads,
                    types,
                    pred_heads,
                    pred_types,
                    mask,
                    ~punct_mask,
                    proj_mask,
                    type_idx=type_vocab.index(y),
                )
                for y in type_vocab
            }

    return callback


def compute_l2_loss(
    model: torch.nn.Module, origin: Optional[Dict[str, torch.Tensor]] = None
) -> Callable[[dict], None]:
    if origin is None:
        origin = {}

    def callback(state):
        loss = 0
        for name, p in model.named_parameters():
            loss += (p - origin.get(name, 0)).pow(2).sum()
        state["l2_loss"] = loss

    return callback
