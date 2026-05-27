import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datamodule.utils import Batch
from lit_ical import LitICAL
from utils.utils import make_ical_targets_from_padded
from utils.vocab_info import VocabInfo


def test_lit_ical_training_step_smoke():
    vocab_info = VocabInfo(
        vocab_size=12,
        sos_id=1,
        eos_id=2,
        pad_id=0,
        space_id=3,
        structural_token_ids=(5, 7, 8, 9),
    )
    config = {
        "model": {
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
            "dc": 4,
            "cross_coverage": True,
            "self_coverage": True,
            "sccm": {
                "nhead": 4,
                "dim_feedforward": 32,
                "dropout": 0.0,
                "num_layers": 1,
            },
            "fusion": {"type": "gated"},
            "loss": {
                "explicit_weight": 1.0,
                "implicit_weight": 1.0,
                "fusion_weight": 1.0,
                "dynamic_implicit_weight": True,
            },
            "optimizer": {"use": "SGD", "SGD": {"lr": 0.01}},
            "scheduler": {
                "use": "ReduceLROnPlateau",
                "interval": "epoch",
                "monitor": "val_ExpRate",
                "ReduceLROnPlateau": {"mode": "max", "factor": 0.25, "patience": 1},
                "warmup": {"enabled": False, "interval": "step", "epochs": 0, "steps": 0},
            },
        }
    }
    labels = torch.tensor([[4, 5, 6], [8, 9, 0]], dtype=torch.long)
    lengths = torch.tensor([3, 2], dtype=torch.long)
    exp_tgt, exp_out, imp_tgt, imp_out, fusion_tgt, fusion_out = (
        make_ical_targets_from_padded(labels, lengths, vocab_info)
    )
    batch = Batch(
        img_bases=["a", "b"],
        imgs=torch.randn(2, 1, 32, 32),
        mask=torch.zeros(2, 32, 32, dtype=torch.bool),
        indices=[[4, 5, 6], [8, 9]],
        labels=labels,
        lengths=lengths,
        exp_tgt=exp_tgt,
        exp_out=exp_out,
        imp_tgt=imp_tgt,
        imp_out=imp_out,
        fusion_tgt=fusion_tgt,
        fusion_out=fusion_out,
    )

    model = LitICAL(config=config, vocab_info=vocab_info)
    loss = model.training_step(batch, 0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


if __name__ == "__main__":
    test_lit_ical_training_step_smoke()
    print("[PASS] LitICAL training_step smoke")
