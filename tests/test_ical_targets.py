import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.utils import make_ical_targets_from_padded, make_implicit_labels_from_padded
from utils.vocab_info import VocabInfo


def test_ical_target_construction():
    info = VocabInfo(
        vocab_size=10,
        sos_id=1,
        eos_id=2,
        pad_id=0,
        space_id=3,
        structural_token_ids=(5, 7, 8, 9),
    )
    labels = torch.tensor([[4, 5, 6, 7, 8, 9]], dtype=torch.long)
    original = labels.clone()
    lengths = torch.tensor([6], dtype=torch.long)

    implicit = make_implicit_labels_from_padded(labels, lengths, info)
    assert torch.equal(labels, original)
    assert implicit.tolist() == [[3, 5, 3, 7, 8, 9]]

    exp_tgt, exp_out, imp_tgt, imp_out, fusion_tgt, fusion_out = (
        make_ical_targets_from_padded(labels, lengths, info)
    )

    assert exp_tgt.tolist() == [[1, 4, 5, 6, 7, 8, 9], [2, 9, 8, 7, 6, 5, 4]]
    assert exp_out.tolist() == [[4, 5, 6, 7, 8, 9, 2], [9, 8, 7, 6, 5, 4, 1]]
    assert fusion_tgt.tolist() == exp_tgt.tolist()
    assert fusion_out.tolist() == exp_out.tolist()
    assert imp_tgt.tolist() == [[1, 3, 5, 3, 7, 8, 9], [2, 9, 8, 7, 3, 5, 3]]
    assert imp_out.tolist() == [[3, 5, 3, 7, 8, 9, 2], [9, 8, 7, 3, 5, 3, 1]]


def test_ical_target_padding():
    info = VocabInfo(
        vocab_size=8,
        sos_id=1,
        eos_id=2,
        pad_id=0,
        space_id=3,
        structural_token_ids=(5,),
    )
    labels = torch.tensor([[4, 5, 0], [6, 0, 0]], dtype=torch.long)
    lengths = torch.tensor([2, 1], dtype=torch.long)
    implicit = make_implicit_labels_from_padded(labels, lengths, info)

    assert implicit.tolist() == [[3, 5, 0], [3, 0, 0]]
    exp_tgt, exp_out, imp_tgt, imp_out, _, _ = make_ical_targets_from_padded(
        labels, lengths, info
    )
    assert exp_tgt.tolist() == [
        [1, 4, 5, 0],
        [1, 6, 0, 0],
        [2, 5, 4, 0],
        [2, 6, 0, 0],
    ]
    assert exp_out.tolist() == [
        [4, 5, 2, 0],
        [6, 2, 0, 0],
        [5, 4, 1, 0],
        [6, 1, 0, 0],
    ]
    assert imp_tgt.tolist() == [
        [1, 3, 5, 0],
        [1, 3, 0, 0],
        [2, 5, 3, 0],
        [2, 3, 0, 0],
    ]
    assert imp_out.tolist() == [
        [3, 5, 2, 0],
        [3, 2, 0, 0],
        [5, 3, 1, 0],
        [3, 1, 0, 0],
    ]


if __name__ == "__main__":
    test_ical_target_construction()
    test_ical_target_padding()
    print("[PASS] ICAL target construction")
