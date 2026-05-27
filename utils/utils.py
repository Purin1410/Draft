from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import LongTensor
from torchmetrics import Metric


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
    need_weight: bool = False,
    class_of_interest: Optional[int] = None,
) -> torch.Tensor:
    """comput cross-entropy loss

    Args:
        output_hat (torch.Tensor): [batch, len, e]
        output (torch.Tensor): [batch, len]
        ignore_idx (int):

    Returns:
        torch.Tensor: loss value
    """
    vocab_size = output_hat.shape[-1]
    flat_hat = output_hat.reshape(-1, vocab_size)
    flat = output.reshape(-1)
    weight = None
    if need_weight:
        weight = torch.ones(vocab_size, device=output_hat.device, dtype=output_hat.dtype)
        if class_of_interest is not None:
            valid = flat.ne(ignore_idx)
            total = valid.sum().clamp_min(1).to(dtype=output_hat.dtype)
            class_count = flat[valid].eq(class_of_interest).sum().clamp_min(1)
            frequency = class_count.to(dtype=output_hat.dtype) / total
            smooth_weight = 1.0 + torch.log1p(1.0 / ((1.0 - frequency) + 1e-6))
            smooth_weight = torch.clamp(smooth_weight, max=10.0)
            weight[:] = smooth_weight
            weight[class_of_interest] = 1.0
    loss = F.cross_entropy(
        flat_hat,
        flat,
        weight=weight,
        ignore_index=ignore_idx,
        reduction=reduction,
    )
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


def make_implicit_labels_from_padded(
    labels: LongTensor,
    lengths: LongTensor,
    vocab_info,
) -> LongTensor:
    """Replace non-structural valid label tokens with the configured space id."""
    implicit = labels.clone()
    if labels.numel() == 0:
        return implicit

    _, max_len = labels.shape
    positions = torch.arange(max_len, device=labels.device).unsqueeze(0)
    valid = positions < lengths.unsqueeze(1)

    structural = torch.zeros_like(labels, dtype=torch.bool)
    for token_id in vocab_info.structural_token_ids:
        structural |= labels.eq(int(token_id))

    implicit[valid & ~structural] = int(vocab_info.space_id)
    implicit[~valid] = int(vocab_info.pad_id)
    return implicit


def make_ical_targets_from_padded(
    labels: LongTensor,
    lengths: LongTensor,
    vocab_info,
) -> Tuple[LongTensor, LongTensor, LongTensor, LongTensor, LongTensor, LongTensor]:
    """Build explicit, implicit, and fusion bidirectional targets once per batch."""
    exp_tgt, exp_out = to_bi_tgt_out_from_padded(
        labels=labels,
        lengths=lengths,
        sos_id=vocab_info.sos_id,
        eos_id=vocab_info.eos_id,
        pad_id=vocab_info.pad_id,
    )
    implicit_labels = make_implicit_labels_from_padded(labels, lengths, vocab_info)
    imp_tgt, imp_out = to_bi_tgt_out_from_padded(
        labels=implicit_labels,
        lengths=lengths,
        sos_id=vocab_info.sos_id,
        eos_id=vocab_info.eos_id,
        pad_id=vocab_info.pad_id,
    )
    fusion_tgt = exp_tgt.clone()
    fusion_out = exp_out.clone()
    return exp_tgt, exp_out, imp_tgt, imp_out, fusion_tgt, fusion_out
