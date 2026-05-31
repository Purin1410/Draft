import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import posformer
from utils.position_labels import PositionTokenIds, build_position_targets_from_padded
from utils.utils import to_bi_tgt_out_from_padded
from utils.vocab_info import VocabInfo


class _Words:
    def __init__(self):
        tokens = [
            "<pad>",
            "<sos>",
            "<eos>",
            "x",
            "2",
            "i",
            "{",
            "}",
            "^",
            "_",
            "[",
            "]",
            "\\frac",
            "\\sqrt",
        ]
        self.word2idx = {token: idx for idx, token in enumerate(tokens)}


def _vocab_info():
    words = _Words()
    return VocabInfo(
        vocab_size=len(words.word2idx),
        pad_id=words.word2idx["<pad>"],
        sos_id=words.word2idx["<sos>"],
        eos_id=words.word2idx["<eos>"],
        words=words,
    )


def _config():
    return {
        "model": {
            "name": "PosFormer",
            "d_model": 16,
            "growth_rate": 4,
            "num_layers": 1,
            "reduction": 0.5,
            "bottleneck": False,
            "use_dropout": False,
            "encoder_dropout": 0.0,
            "nhead": 4,
            "num_decoder_layers": 1,
            "dim_feedforward": 32,
            "decoder_dropout": 0.0,
            "dc": 8,
            "cross_coverage": True,
            "self_coverage": True,
            "position": {
                "enabled": True,
                "label_depth": 5,
                "num_layer_classes": 5,
                "num_pos_classes": 6,
                "mlp_hidden_dim": 16,
                "dropout": 0.0,
            },
            "implicit_attention_correction": {
                "enabled": True,
                "mask_tokens": ["{", "^", "_"],
            },
        }
    }


def _batch_tensors(vocab_info):
    words = vocab_info.words.word2idx
    seqs = [
        [words["x"], words["^"], words["{"], words["2"], words["}"]],
        [words["x"], words["_"], words["{"], words["i"], words["}"]],
    ]
    lengths = torch.tensor([len(seq) for seq in seqs], dtype=torch.long)
    labels = torch.full((2, int(lengths.max())), vocab_info.pad_id, dtype=torch.long)
    for i, seq in enumerate(seqs):
        labels[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    tgt, out = to_bi_tgt_out_from_padded(
        labels,
        lengths,
        sos_id=vocab_info.sos_id,
        eos_id=vocab_info.eos_id,
        pad_id=vocab_info.pad_id,
    )
    pos_targets = build_position_targets_from_padded(
        labels,
        lengths,
        PositionTokenIds.from_vocab_info(vocab_info),
        sos_id=vocab_info.sos_id,
        eos_id=vocab_info.eos_id,
        pad_id=vocab_info.pad_id,
        label_depth=5,
    )
    return tgt, out, pos_targets


def test_posformer_forward_shapes():
    vocab_info = _vocab_info()
    model = posformer.PosFormer(_config(), vocab_info=vocab_info)
    model.eval()
    tgt, _, pos_targets = _batch_tensors(vocab_info)
    imgs = torch.randn(2, 1, 32, 32)
    mask = torch.zeros(2, 32, 32, dtype=torch.bool)

    with torch.inference_mode():
        word_logits, layer_logits, pos_logits = model(imgs, mask, tgt, pos_targets.pos_tgt)

    assert word_logits.shape == (4, tgt.shape[1], vocab_info.vocab_size)
    assert layer_logits.shape == (4, tgt.shape[1], 5)
    assert pos_logits.shape == (4, tgt.shape[1], 6)


def test_beam_search_skips_position_decoder():
    vocab_info = _vocab_info()
    model = posformer.PosFormer(_config(), vocab_info=vocab_info)

    def fail_position_forward(*args, **kwargs):
        raise AssertionError("position decoder must not run during beam search")

    def fake_expression_beam_search(src, src_mask, beam_size, max_len, alpha, early_stopping, temperature, **kwargs):
        return ["expression-only"]

    model.pos_decoder.forward = fail_position_forward
    model.decoder.beam_search = fake_expression_beam_search
    model.eval()

    imgs = torch.randn(1, 1, 32, 32)
    mask = torch.zeros(1, 32, 32, dtype=torch.bool)
    with torch.inference_mode():
        result = model.beam_search(
            imgs,
            mask,
            beam_size=1,
            max_len=4,
            alpha=1.0,
            early_stopping=True,
            temperature=1.0,
        )

    assert result == ["expression-only"]


if __name__ == "__main__":
    test_posformer_forward_shapes()
    test_beam_search_skips_position_decoder()
    print("[PASS] PosFormer smoke tests")
