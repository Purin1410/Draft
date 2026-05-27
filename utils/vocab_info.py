from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class VocabInfo:
    vocab_size: int
    sos_id: int
    eos_id: int
    pad_id: int
    words: Any = None
