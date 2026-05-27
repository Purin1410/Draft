import warnings
from typing import Dict, List, Optional, Sequence

from utils.vocab_info import VocabInfo


class Vocab:

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    ICAL_SPACE_ID = EOS_IDX + 1

    def __init__(
        self,
        dict_path: str = "crohme_dictionary.txt",
        force_ical_special_token_order: bool = True,
        special_tokens: Optional[Dict[str, str]] = None,
        implicit_structural_tokens: Optional[Sequence[str]] = None,
    ) -> None:
        special_tokens = special_tokens or {}
        self.pad_token = special_tokens.get("pad", "<pad>")
        self.sos_token = special_tokens.get("sos", "<sos>")
        self.eos_token = special_tokens.get("eos", "<eos>")
        self.space_token = special_tokens.get("space", "<space>")
        self.implicit_structural_tokens = tuple(
            implicit_structural_tokens or ("{", "}", "^", "_")
        )
        self.force_ical_special_token_order = force_ical_special_token_order

        self.word2idx = dict()
        self._duplicate_specials_warned = False

        if force_ical_special_token_order:
            self.word2idx[self.pad_token] = self.PAD_IDX
            self.word2idx[self.sos_token] = self.SOS_IDX
            self.word2idx[self.eos_token] = self.EOS_IDX
            self.word2idx[self.space_token] = self.ICAL_SPACE_ID
        else:
            self.word2idx[self.pad_token] = self.PAD_IDX
            self.word2idx[self.sos_token] = self.SOS_IDX
            self.word2idx[self.eos_token] = self.EOS_IDX

        has_space = False
        with open(dict_path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.rstrip("\n\r")
                if w == "":
                    continue
                if w in self.word2idx:
                    if w in {
                        self.pad_token,
                        self.sos_token,
                        self.eos_token,
                        self.space_token,
                    } and not self._duplicate_specials_warned:
                        warnings.warn(
                            "Dictionary contains duplicate special tokens; "
                            "keeping configured special-token ids.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        self._duplicate_specials_warned = True
                    continue
                if w == self.space_token or w == " ":
                    has_space = True
                self.word2idx[w] = len(self.word2idx)

        if not force_ical_special_token_order and not has_space:
            self.word2idx[" "] = len(self.word2idx)

        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}
        self.PAD_IDX = self.word2idx[self.pad_token]
        self.SOS_IDX = self.word2idx[self.sos_token]
        self.EOS_IDX = self.word2idx[self.eos_token]
        self.space_id = self.word2idx.get(self.space_token, self.word2idx.get(" "))


    def words2indices(self, words: List[str]) -> List[int]:
        return [self.word2idx[w] for w in words]

    def indices2words(self, id_list: List[int]) -> List[str]:
        return [self.idx2word[i] for i in id_list]

    def indices2label(self, id_list: List[int]) -> str:
        words = self.indices2words(id_list)
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)

    def get_info(self) -> VocabInfo:
        structural_token_ids = tuple(
            self.word2idx[token]
            for token in self.implicit_structural_tokens
            if token in self.word2idx
        )
        return VocabInfo(
            vocab_size=len(self),
            sos_id=self.SOS_IDX,
            eos_id=self.EOS_IDX,
            pad_id=self.PAD_IDX,
            space_id=self.space_id,
            structural_token_ids=structural_token_ids,
            words=self,
        )
