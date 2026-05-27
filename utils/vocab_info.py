from dataclasses import dataclass
from typing import Any, Tuple

@dataclass(frozen=True)
class VocabInfo:
    vocab_size: int
    sos_id: int
    eos_id: int
    pad_id: int
    space_id: int
    structural_token_ids: Tuple[int, ...]
    words: Any = None
