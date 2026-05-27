from typing import Dict, List, Any
from dataclasses import dataclass

from utils.vocab_info import VocabInfo


class Vocab:

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2

    def __init__(self, dict_path: str = "crohme_dictionary.txt") -> None:
        self.word2idx = dict()
        self.word2idx["<pad>"] = self.PAD_IDX
        self.word2idx["<sos>"] = self.SOS_IDX
        self.word2idx["<eos>"] = self.EOS_IDX

        has_space = False
        with open(dict_path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.rstrip("\n")
                if w == "":
                    continue
                if w == " ":
                    has_space = True
                self.word2idx[w] = len(self.word2idx)

        if not has_space:
            self.word2idx[" "] = len(self.word2idx)

        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}


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
        return VocabInfo(
            vocab_size=len(self),
            sos_id=self.SOS_IDX,
            eos_id=self.EOS_IDX,
            pad_id=self.PAD_IDX,
            words=self,
        )
