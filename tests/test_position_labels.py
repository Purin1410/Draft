import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.position_labels import (
    PositionTokenIds,
    build_position_targets_from_padded,
    indices_to_multi_label,
)
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
            "a",
            "b",
            "n",
            "y",
            "[",
            "]",
            "{",
            "}",
            "^",
            "_",
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


def _encode(words, expr):
    return [words.word2idx[token] for token in expr.split()]


def test_position_targets_for_required_expressions():
    vocab_info = _vocab_info()
    token_ids = PositionTokenIds.from_vocab_info(vocab_info)
    words = vocab_info.words
    expressions = [
        r"x ^ { 2 }",
        r"x _ { i }",
        r"\frac { a } { b }",
        r"\sqrt { x }",
        r"\sqrt [ n ] { x }",
        r"\frac { x ^ { 2 } } { y _ { i } }",
    ]
    seqs = [_encode(words, expr) for expr in expressions]
    lengths = torch.tensor([len(seq) for seq in seqs], dtype=torch.long)
    max_len = int(lengths.max())
    labels = torch.full((len(seqs), max_len), vocab_info.pad_id, dtype=torch.long)
    for i, seq in enumerate(seqs):
        labels[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    targets = build_position_targets_from_padded(
        labels=labels,
        lengths=lengths,
        token_ids=token_ids,
        sos_id=vocab_info.sos_id,
        eos_id=vocab_info.eos_id,
        pad_id=vocab_info.pad_id,
        label_depth=5,
    )

    assert targets.pos_tgt.shape == (2 * len(seqs), max_len + 1, 5)
    assert targets.pos_layer.shape == (2 * len(seqs), max_len + 1)
    assert targets.pos_pos.shape == targets.pos_layer.shape
    assert torch.all((0 <= targets.pos_layer) & (targets.pos_layer < 5))
    assert torch.all((0 <= targets.pos_pos) & (targets.pos_pos < 6))


def test_indices_to_multi_label_does_not_mutate_input():
    vocab_info = _vocab_info()
    token_ids = PositionTokenIds.from_vocab_info(vocab_info)
    seq = [vocab_info.eos_id, vocab_info.words.word2idx["}"], vocab_info.words.word2idx["x"]]
    original = list(seq)

    labels = indices_to_multi_label(seq, token_ids, label_depth=5)

    assert seq == original
    assert len(labels) == len(seq)
    assert all(len(label) == 5 for label in labels)


def test_collate_fn_builds_position_targets():
    from datamodule.datamodule import CROHMEDatamodule
    from sconf import Config

    with tempfile.TemporaryDirectory() as tmpdir:
        dict_path = Path(tmpdir) / "dictionary.txt"
        dict_path.write_text(
            "\n".join(["x", "2", "a", "b", "^", "_", "{", "}", "[", "]", "\\frac", "\\sqrt"]),
            encoding="utf-8",
        )
        config = Config(
            {
                "seed_everything": 7,
                "model": {
                    "max_len": 200,
                    "position": {
                        "enabled": True,
                        "label_depth": 5,
                    },
                },
                "data": {
                    "zipfile_path": tmpdir,
                    "test_year": "2014",
                    "dictionary_txt": str(dict_path),
                    "train_batch_size": 2,
                    "eval_batch_size": 2,
                    "num_workers": 0,
                    "scale_aug": False,
                    "max_pixels_per_batch": 1280000,
                    "lazy_load": False,
                    "k_min": 0.7,
                    "k_max": 1.4,
                    "w_lo": 16,
                    "w_hi": 1024,
                    "h_lo": 16,
                    "h_hi": 256,
                    "pin_memory": False,
                    "persistent_workers": False,
                },
            }
        )
        CROHMEDatamodule.shared_vocab = None
        dm = CROHMEDatamodule(config)
        batch = dm.collate_fn(
            [
                ("img1", torch.randn(1, 10, 20), ["x", "^", "{", "2", "}"]),
                ("img2", torch.randn(1, 15, 12), ["\\frac", "{", "a", "}", "{", "b", "}"]),
            ]
        )
        CROHMEDatamodule.shared_vocab = None

    assert batch.imgs.shape == (2, 1, 15, 20)
    assert batch.mask.shape == (2, 15, 20)
    assert batch.tgt.shape[0] == 4
    assert batch.out.shape == batch.tgt.shape
    assert batch.pos_tgt.shape[:2] == batch.tgt.shape
    assert batch.pos_tgt.shape[-1] == 5
    assert batch.pos_layer.shape == batch.tgt.shape
    assert batch.pos_pos.shape == batch.tgt.shape


if __name__ == "__main__":
    test_position_targets_for_required_expressions()
    test_indices_to_multi_label_does_not_mutate_input()
    test_collate_fn_builds_position_targets()
    print("[PASS] position label tests")
