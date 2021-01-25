# Copyright (c) 2021 Kemal Kurniawan

from camel import Camel, CamelRegistry
from text2array import Vocab
from text2array.vocab import StringStore

from models import SelfAttGraph

reg = CamelRegistry()


def dump(obj):
    return Camel([reg]).dump(obj)


def load(data):
    return Camel([reg]).load(data)


@reg.dumper(Vocab, "vocab", version=1)
def _dump_vocab(vocab):
    return dict(vocab)


@reg.loader("vocab", version=1)
def _load_vocab(data, version):
    return Vocab(data)


@reg.dumper(StringStore, "strstore", version=1)
def _dump_strstore(store):
    return {"initial": list(store), "default": store.default}


@reg.loader("strstore", version=1)
def _load_strstore(data, version):
    return StringStore(**data)


@reg.dumper(SelfAttGraph, "self_att_graph", version=1)
def _dump_self_att_graph(model):
    kv_size = model.encoder.layers[0].self_attn.k_dist_emb.embedding_dim
    return {
        "n_words": model.word_emb.num_embeddings,
        "n_types": model.type_score_mix.out_features,
        "n_tags": 0 if model.tag_emb is None else model.tag_emb.num_embeddings,
        "word_size": model.word_emb.embedding_dim,
        "tag_size": 0 if model.tag_emb is None else model.tag_emb.embedding_dim,
        "n_heads": model.encoder.layers[0].self_attn.in_proj[0].out_features // (3 * kv_size),
        "n_layers": model.encoder.num_layers,
        "ff_size": model.encoder.layers[0].linear1.out_features,
        "kv_size": kv_size,
        "word_dropout": model.word_dropout.p,
        "outdim_dropout": model.mlp_layer[-2].p,
        "arc_size": model.arc_score_bias.numel(),
        "type_size": model.type_score_mix.in1_features,
    }


@reg.loader("self_att_graph", version=1)
def _load_self_att_graph(data, version):
    return SelfAttGraph(data.pop("n_words"), data.pop("n_types"), **data)
