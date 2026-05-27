import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datamodule.vocab import Vocab


def test_vocab_ical_special_order():
    with tempfile.TemporaryDirectory() as tmp:
        dict_path = Path(tmp) / "dictionary.txt"
        dict_path.write_text(
            "\n".join(["<pad>", "a", "{", "}", "^", "_", "b", "<space>"]),
            encoding="utf-8",
        )

        vocab = Vocab(
            dict_path=str(dict_path),
            force_ical_special_token_order=True,
            special_tokens={
                "pad": "<pad>",
                "sos": "<sos>",
                "eos": "<eos>",
                "space": "<space>",
            },
            implicit_structural_tokens=["{", "}", "^", "_"],
        )
        info = vocab.get_info()

        assert info.pad_id == 0
        assert info.sos_id == 1
        assert info.eos_id == 2
        assert info.space_id == 3
        assert vocab.word2idx["<space>"] == 3
        assert info.structural_token_ids == (
            vocab.word2idx["{"],
            vocab.word2idx["}"],
            vocab.word2idx["^"],
            vocab.word2idx["_"],
        )
        assert all(token_id != info.space_id for token_id in info.structural_token_ids)
        assert info.vocab_size == len(vocab)


if __name__ == "__main__":
    test_vocab_ical_special_order()
    print("[PASS] ICAL vocab special order")
