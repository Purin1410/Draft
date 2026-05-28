# datamodule/vocab.py

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from utils.vocab_info import VocabInfo


class Vocab:
    # ICAL contract: these IDs must stay fixed.
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    SPACE_IDX = 3

    # Backward-compatible aliases used by older code/tests.
    ICAL_SPACE_ID = SPACE_IDX

    def __init__(
        self,
        dict_path: str,
        special_tokens: Optional[Dict[str, str]] = None,
        implicit_structural_tokens: Optional[Sequence[str]] = None,
        strict: bool = True,
    ) -> None:
        special_tokens = special_tokens or {}

        self.pad_token = special_tokens.get("pad", "<pad>")
        self.sos_token = special_tokens.get("sos", "<sos>")
        self.eos_token = special_tokens.get("eos", "<eos>")
        self.space_token = special_tokens.get("space", "<space>")
        self.implicit_structural_tokens = tuple(
            implicit_structural_tokens or ("{", "}", "^", "_")
        )
        self.strict = strict

        self._validate_special_token_names()

        self.word2idx: Dict[str, int] = {
            self.pad_token: self.PAD_IDX,
            self.sos_token: self.SOS_IDX,
            self.eos_token: self.EOS_IDX,
            self.space_token: self.SPACE_IDX,
        }

        self._load_dictionary(dict_path)
        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}

        # Instance aliases, so existing code can use either Vocab.PAD_IDX or self.PAD_IDX.
        self.PAD_IDX = self.word2idx[self.pad_token]
        self.SOS_IDX = self.word2idx[self.sos_token]
        self.EOS_IDX = self.word2idx[self.eos_token]
        self.SPACE_IDX = self.word2idx[self.space_token]
        self.space_id = self.SPACE_IDX

        self._validate_fixed_ids()
        self._validate_structural_tokens()

    def _validate_special_token_names(self) -> None:
        tokens = [
            self.pad_token,
            self.sos_token,
            self.eos_token,
            self.space_token,
        ]
        if len(tokens) != len(set(tokens)):
            raise ValueError(
                f"Special tokens must be distinct, got: {tokens}"
            )

    def _load_dictionary(self, dict_path: str) -> None:
        path = Path(dict_path)
        if not path.exists():
            raise FileNotFoundError(f"Dictionary file not found: {path}")

        skipped_duplicates = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                token = line.strip()
                if not token:
                    continue

                if token in self.word2idx:
                    skipped_duplicates.append((line_no, token))
                    continue

                self.word2idx[token] = len(self.word2idx)

        if skipped_duplicates:
            preview = ", ".join(
                f"line {line_no}: {token!r}"
                for line_no, token in skipped_duplicates[:8]
            )
            warnings.warn(
                "Dictionary contains duplicate tokens already reserved by Vocab; "
                f"skipped them to preserve ICAL token IDs. Examples: {preview}",
                RuntimeWarning,
                stacklevel=2,
            )

    def _validate_fixed_ids(self) -> None:
        expected = {
            self.pad_token: self.PAD_IDX,
            self.sos_token: self.SOS_IDX,
            self.eos_token: self.EOS_IDX,
            self.space_token: self.SPACE_IDX,
        }
        actual = {
            self.pad_token: self.word2idx[self.pad_token],
            self.sos_token: self.word2idx[self.sos_token],
            self.eos_token: self.word2idx[self.eos_token],
            self.space_token: self.word2idx[self.space_token],
        }

        if actual != expected:
            raise ValueError(
                "ICAL requires fixed special token IDs: "
                f"expected={expected}, actual={actual}"
            )

    def _validate_structural_tokens(self) -> None:
        missing = [
            token
            for token in self.implicit_structural_tokens
            if token not in self.word2idx
        ]
        if missing:
            raise ValueError(
                "Missing ICAL structural tokens in dictionary: "
                f"{missing}. These are needed to build implicit targets."
            )

    def words2indices(self, words: List[str]) -> List[int]:
        missing = [w for w in words if w not in self.word2idx]
        if missing:
            uniq_missing = sorted(set(missing))
            raise KeyError(
                "Caption contains tokens missing from vocabulary: "
                f"{uniq_missing[:20]}"
            )
        return [self.word2idx[w] for w in words]

    def indices2words(self, id_list: List[int]) -> List[str]:
        missing = [i for i in id_list if int(i) not in self.idx2word]
        if missing:
            uniq_missing = sorted(set(int(i) for i in missing))
            raise KeyError(
                "Predicted/label IDs missing from vocabulary: "
                f"{uniq_missing[:20]}"
            )
        return [self.idx2word[int(i)] for i in id_list]

    def indices2label(self, id_list: List[int]) -> str:
        return " ".join(self.indices2words(id_list))

    def get_info(self) -> VocabInfo:
        structural_token_ids = tuple(
            self.word2idx[token] for token in self.implicit_structural_tokens
        )
        return VocabInfo(
            vocab_size=len(self),
            sos_id=self.SOS_IDX,
            eos_id=self.EOS_IDX,
            pad_id=self.PAD_IDX,
            space_id=self.SPACE_IDX,
            structural_token_ids=structural_token_ids,
            words=self,
        )

    def debug_summary(self) -> str:
        structural = {
            token: self.word2idx[token]
            for token in self.implicit_structural_tokens
        }
        return (
            f"Vocab(size={len(self)}, "
            f"pad={self.PAD_IDX}, sos={self.SOS_IDX}, "
            f"eos={self.EOS_IDX}, space={self.SPACE_IDX}, "
            f"structural={structural})"
        )

    def __len__(self) -> int:
        return len(self.word2idx)