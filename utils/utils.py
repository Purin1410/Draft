from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import LongTensor
from torchmetrics import Metric
from utils.latex2gtd import safe_to_struct
from utils.vocab_info import VocabInfo


class Hypothesis:
    seq: List[int]
    score: float

    def __init__(
        self,
        seq_tensor: LongTensor,
        score: float,
        direction: str,
    ) -> None:
        assert direction in {"l2r", "r2l"}
        raw_seq = seq_tensor.tolist()

        if direction == "r2l":
            result = raw_seq[::-1]
        else:
            result = raw_seq

        self.seq = result
        self.score = score

    def __len__(self):
        if len(self.seq) != 0:
            return len(self.seq)
        else:
            return 1

    def __str__(self):
        return f"seq: {self.seq}, score: {self.score}"


class ExpRateRecorder(Metric):
    def __init__(self, vocab_info, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.vocab_info = vocab_info

        self.add_state("total_line", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rec", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, indices_hat: List[List[int]], indices: List[List[int]]):
        for pred, truth in zip(indices_hat, indices):
            is_same = pred == truth

            if is_same:
                self.rec += 1

            self.total_line += 1

    def compute(self) -> float:
        exp_rate = self.rec / self.total_line
        return exp_rate


def ce_loss(
    output_hat: torch.Tensor,
    output: torch.Tensor,
    ignore_idx: int,
    reduction: str = "mean",
) -> torch.Tensor:
    """comput cross-entropy loss

    Args:
        output_hat (torch.Tensor): [batch, len, e]
        output (torch.Tensor): [batch, len]
        ignore_idx (int):

    Returns:
        torch.Tensor: loss value
    """
    flat_hat = rearrange(output_hat, "b l e -> (b l) e")
    flat = rearrange(output, "b l -> (b l)")
    loss = F.cross_entropy(flat_hat, flat, ignore_index=ignore_idx, reduction=reduction)
    return loss


def to_tgt_output(
    tokens: Union[List[List[int]], List[LongTensor]],
    direction: str,
    device: torch.device,
    sos_id: int,
    eos_id: int,
    pad_id: int,
    pad_to_len: Optional[int] = None,
) -> Tuple[LongTensor, LongTensor]:
    """Generate tgt and out for indices

    Parameters
    ----------
    tokens : Union[List[List[int]], List[LongTensor]]
        indices: [b, l]
    direction : str
        one of "l2f" and "r2l"
    device : torch.device

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        tgt, out: [b, l], [b, l]
    """
    assert direction in {"l2r", "r2l"}

    if isinstance(tokens[0], list):
        tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]

    if direction == "l2r":
        tokens = tokens
        start_w = sos_id
        stop_w = eos_id
    else:
        tokens = [torch.flip(t, dims=[0]) for t in tokens]
        start_w = eos_id
        stop_w = sos_id

    batch_size = len(tokens)
    lens = [len(t) for t in tokens]

    length = max(lens) + 1
    if pad_to_len is not None:
        length = max(length, pad_to_len)

    tgt = torch.full(
        (batch_size, length),
        fill_value=pad_id,
        dtype=torch.long,
        device=device,
    )
    out = torch.full(
        (batch_size, length),
        fill_value=pad_id,
        dtype=torch.long,
        device=device,
    )

    for i, token in enumerate(tokens):
        tgt[i, 0] = start_w
        tgt[i, 1 : (1 + lens[i])] = token

        out[i, : lens[i]] = token
        out[i, lens[i]] = stop_w

    return tgt, out


def to_bi_tgt_out(
    tokens: List[List[int]], 
    device: torch.device,
    sos_id: int,
    eos_id: int,
    pad_id: int,
) -> Tuple[LongTensor, LongTensor]:
    """Generate bidirection tgt and out

    Parameters
    ----------
    tokens : List[List[int]]
        indices: [b, l]
    device : torch.device

    Returns
    -------
    Tuple[LongTensor, LongTensor]
        tgt, out: [2b, l], [2b, l]
    """
    l2r_tgt, l2r_out = to_tgt_output(tokens, "l2r", device, sos_id, eos_id, pad_id)
    r2l_tgt, r2l_out = to_tgt_output(tokens, "r2l", device, sos_id, eos_id, pad_id)

    tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
    out = torch.cat((l2r_out, r2l_out), dim=0)

    return tgt, out


def to_bi_tgt_out_from_padded(
    labels: LongTensor,
    lengths: LongTensor,
    sos_id: int,
    eos_id: int,
    pad_id: int,
) -> Tuple[LongTensor, LongTensor]:
    """Vectorized bidirectional target/output construction from padded labels.

    Parameters
    ----------
    labels : LongTensor [B, L]
        Padded label tensor (pad positions filled with pad_id).
    lengths : LongTensor [B]
        Actual label lengths (non-pad count per sample).
    sos_id, eos_id, pad_id : int
        Token ids.

    Returns
    -------
    Tuple[LongTensor, LongTensor]
        tgt: [2B, L+1], out: [2B, L+1]
        First B rows are l2r, last B rows are r2l.
    """
    B, L = labels.shape
    device = labels.device
    out_len = L + 1  # space for start/stop token

    # ---- l2r ----
    l2r_tgt = torch.full((B, out_len), pad_id, dtype=torch.long, device=device)
    l2r_out = torch.full((B, out_len), pad_id, dtype=torch.long, device=device)

    l2r_tgt[:, 0] = sos_id
    l2r_tgt[:, 1:L+1] = labels  # labels already padded with pad_id

    l2r_out[:, :L] = labels
    # Place EOS at position lengths[i] for each sample
    l2r_out[torch.arange(B, device=device), lengths] = eos_id

    # ---- r2l (reversed labels) ----
    r2l_tgt = torch.full((B, out_len), pad_id, dtype=torch.long, device=device)
    r2l_out = torch.full((B, out_len), pad_id, dtype=torch.long, device=device)

    # Build reversed labels using gather with reversed valid indices
    # For sample i with length L_i, we want labels[i, L_i-1], labels[i, L_i-2], ..., labels[i, 0]
    positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)  # [B, L]
    # rev_positions[i, j] = lengths[i] - 1 - j  (clamped to 0 for pad positions)
    rev_positions = (lengths.unsqueeze(1) - 1 - positions).clamp(min=0)  # [B, L]
    reversed_labels = labels.gather(1, rev_positions)
    # Mask out pad positions (where j >= lengths[i])
    valid_mask = positions < lengths.unsqueeze(1)  # [B, L]
    reversed_labels = reversed_labels.masked_fill(~valid_mask, pad_id)

    r2l_tgt[:, 0] = eos_id
    r2l_tgt[:, 1:L+1] = reversed_labels

    r2l_out[:, :L] = reversed_labels
    r2l_out[torch.arange(B, device=device), lengths] = sos_id

    # ---- combine ----
    tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
    out = torch.cat((l2r_out, r2l_out), dim=0)
    return tgt, out


def _tokens_to_lists(tokens: Union[List[List[int]], List[LongTensor]]) -> List[List[int]]:
    if len(tokens) == 0:
        return []
    if isinstance(tokens[0], torch.Tensor):
        return [t.detach().cpu().tolist() for t in tokens]
    return [list(t) for t in tokens]


def to_struct_output(
    tokens: Union[List[List[int]], List[LongTensor]],
    device: torch.device,
    vocab_info: VocabInfo,
    pad_to_len: Optional[int] = None,
    ignore_index: int = -1,
) -> Tuple[LongTensor, torch.BoolTensor]:
    """Build bidirectional structural parent targets from token ids.

    ``struct_out`` is shaped ``[2B, L + 1]`` and aligns with decoder sequence
    outputs. The extra position supervises the terminal token to point at the
    final real token, matching TAMER's reference training target.
    """
    indices = _tokens_to_lists(tokens)
    if len(indices) == 0:
        empty = torch.empty((0, 0), dtype=torch.long, device=device)
        return empty, torch.empty((0,), dtype=torch.bool, device=device)

    structs: List[List[int]] = []
    illegal: List[bool] = []
    for seq in indices:
        try:
            words = vocab_info.words.indices2words(seq)
            struct, is_illegal = safe_to_struct(words)
        except Exception:
            struct, is_illegal = [ignore_index for _ in seq], True
        structs.append(struct)
        illegal.append(is_illegal)

    batch_size = len(structs)
    lens = [len(struct) for struct in structs]
    length = (max(lens) if lens else 0) + 1
    if pad_to_len is not None:
        length = max(length, int(pad_to_len))

    l2r_out = torch.full(
        (batch_size, length), fill_value=ignore_index, dtype=torch.long, device=device
    )
    r2l_out = torch.full(
        (batch_size, length), fill_value=ignore_index, dtype=torch.long, device=device
    )

    for i, struct in enumerate(structs):
        seq_len = lens[i]
        if seq_len == 0 or illegal[i]:
            continue
        l2r_out[i, :seq_len] = torch.tensor(struct, dtype=torch.long, device=device)
        l2r_out[i, seq_len] = seq_len - 1

        reversed_struct = [
            seq_len - parent - 1 if parent != ignore_index else ignore_index
            for parent in reversed(struct)
        ]
        r2l_out[i, :seq_len] = torch.tensor(
            reversed_struct, dtype=torch.long, device=device
        )
        r2l_out[i, seq_len] = seq_len - 1

    out = torch.cat((l2r_out, r2l_out), dim=0)
    illegal_tensor = torch.tensor(illegal, dtype=torch.bool, device=device)
    return out, illegal_tensor


def to_struct_output_from_labels(
    labels: LongTensor,
    lengths: LongTensor,
    device: torch.device,
    vocab_info: VocabInfo,
    ignore_index: int = -1,
) -> Tuple[LongTensor, torch.BoolTensor]:
    """Build structural targets from padded label tensors."""
    label_cpu = labels.detach().cpu()
    length_cpu = lengths.detach().cpu()
    tokens = [
        label_cpu[i, : int(length_cpu[i].item())].tolist()
        for i in range(label_cpu.shape[0])
    ]
    return to_struct_output(
        tokens=tokens,
        device=device,
        vocab_info=vocab_info,
        pad_to_len=labels.shape[1] + 1,
        ignore_index=ignore_index,
    )
