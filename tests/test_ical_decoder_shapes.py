import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.decoder import Decoder
from utils.vocab_info import VocabInfo


def test_ical_decoder_shapes():
    batch = 2
    bi_batch = 2 * batch
    length = 5
    vocab_size = 20
    d_model = 32
    height = 3
    width = 4

    info = VocabInfo(
        vocab_size=vocab_size,
        sos_id=1,
        eos_id=2,
        pad_id=0,
        space_id=3,
        structural_token_ids=(4, 5, 6, 7),
    )
    decoder = Decoder(
        d_model=d_model,
        nhead=4,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        dc=8,
        cross_coverage=True,
        self_coverage=True,
        vocab_info=info,
        sccm={
            "nhead": 4,
            "dim_feedforward": 64,
            "dropout": 0.0,
            "num_layers": 1,
        },
        fusion={"type": "gated"},
    )
    decoder.eval()

    src = torch.randn(bi_batch, height, width, d_model)
    src_mask = torch.zeros(bi_batch, height, width, dtype=torch.bool)
    tgt = torch.randint(1, vocab_size, (bi_batch, length), dtype=torch.long)

    with torch.inference_mode():
        exp_logits, imp_logits, fusion_logits = decoder(src, src_mask, tgt)
        transform_logits = decoder.transform([src], [src_mask], tgt)

    assert exp_logits.shape == (bi_batch, length, vocab_size)
    assert imp_logits.shape == (bi_batch, length, vocab_size)
    assert fusion_logits.shape == (bi_batch, length, vocab_size)
    assert transform_logits.shape == (bi_batch, length, vocab_size)
    assert not isinstance(transform_logits, tuple)
    assert torch.isfinite(fusion_logits).all()
    assert torch.equal(transform_logits, fusion_logits)


if __name__ == "__main__":
    test_ical_decoder_shapes()
    print("[PASS] ICAL decoder shapes")
