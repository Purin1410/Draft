from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import FloatTensor, LongTensor


@dataclass(frozen=True)
class PositionTokenIds:
    pad: int
    sos: int
    eos: int
    left_square: int
    right_square: int
    left_brace: int
    right_brace: int
    caret: int
    underscore: int
    frac: int
    sqrt: int

    @classmethod
    def from_vocab_info(cls, vocab_info) -> "PositionTokenIds":
        words = getattr(vocab_info, "words", None)
        word2idx = getattr(words, "word2idx", None)
        if word2idx is None:
            raise ValueError("vocab_info.words.word2idx is required for PosFormer position labels.")

        required = ["[", "]", "{", "}", "^", "_", "\\frac", "\\sqrt"]
        missing = [token for token in required if token not in word2idx]
        if missing:
            raise KeyError(f"Missing required PosFormer structural tokens: {missing}")

        return cls(
            pad=int(vocab_info.pad_id),
            sos=int(vocab_info.sos_id),
            eos=int(vocab_info.eos_id),
            left_square=int(word2idx["["]),
            right_square=int(word2idx["]"]),
            left_brace=int(word2idx["{"]),
            right_brace=int(word2idx["}"]),
            caret=int(word2idx["^"]),
            underscore=int(word2idx["_"]),
            frac=int(word2idx["\\frac"]),
            sqrt=int(word2idx["\\sqrt"]),
        )


@dataclass(frozen=True)
class PositionTargets:
    pos_tgt: FloatTensor
    pos_layer: LongTensor
    pos_pos: LongTensor


def make_position_token_ids(vocab_info) -> PositionTokenIds:
    return PositionTokenIds.from_vocab_info(vocab_info)


def _find_end(indices: Sequence[int], start_i: int, end: int, left: int, right: int) -> int:
    count = 1
    i = start_i + 1
    while count > 0 and i < end:
        if indices[i] == left:
            count += 1
        elif indices[i] == right:
            count -= 1
        i += 1
    return i - 1 if count == 0 else 0


def _find_end_square(indices: Sequence[int], start_i: int, end: int, token_ids: PositionTokenIds) -> int:
    return _find_end(indices, start_i, end, token_ids.left_square, token_ids.right_square)


def _find_end_brace(indices: Sequence[int], start_i: int, end: int, token_ids: PositionTokenIds) -> int:
    return _find_end(indices, start_i, end, token_ids.left_brace, token_ids.right_brace)


def _append_range(result: List[List[int]], start: int, end: int, value: int) -> None:
    for j in range(start, end):
        if 0 <= j < len(result):
            result[j].append(value)


def _helper(
    indices: Sequence[int],
    start: int,
    end: int,
    result: List[List[int]],
    token_ids: PositionTokenIds,
) -> List[List[int]]:
    flag = [0, 1, 2, 3, 4, 5]
    special = True
    i = start + 1
    while i < end:
        token = indices[i]
        if token == token_ids.caret:
            end1 = _find_end_brace(indices, i + 1, end, token_ids)
            if special:
                _append_range(result, i, i + 2, flag[3])
                if 0 <= end1 < len(result):
                    result[end1].append(flag[3])
            _append_range(result, i + 2, end1, flag[4])
            result = _helper(indices, i + 1, end1, result, token_ids)
            i = end1 + 1
        elif token == token_ids.underscore:
            end1 = _find_end_brace(indices, i + 1, end, token_ids)
            if special:
                _append_range(result, i, i + 2, flag[3])
                if 0 <= end1 < len(result):
                    result[end1].append(flag[3])
            _append_range(result, i + 2, end1, flag[5])
            result = _helper(indices, i + 1, end1, result, token_ids)
            i = end1 + 1
        elif token == token_ids.frac:
            result[i].append(flag[3])
            end1 = _find_end_brace(indices, i + 1, end, token_ids)
            _append_range(result, i + 2, end1, flag[4])
            end2 = _find_end_brace(indices, end1 + 1, end, token_ids)
            _append_range(result, end1 + 2, end2, flag[5])
            if special:
                for pos in (i + 1, end1, end1 + 1, end2):
                    if 0 <= pos < len(result):
                        result[pos].append(flag[3])
            result = _helper(indices, i + 1, end1, result, token_ids)
            result = _helper(indices, end1 + 1, end2, result, token_ids)
            i = end2 + 1
        elif token == token_ids.sqrt:
            result[i].append(flag[3])
            if i + 1 < end and indices[i + 1] == token_ids.left_square:
                end1 = _find_end_square(indices, i + 1, end, token_ids)
                _append_range(result, i + 2, end1, flag[4])
                end2 = _find_end_brace(indices, end1 + 1, end, token_ids)
                _append_range(result, end1 + 2, end2, flag[5])
                if special:
                    for pos in (i + 1, end1, end1 + 1, end2):
                        if 0 <= pos < len(result):
                            result[pos].append(flag[3])
                result = _helper(indices, i + 1, end1, result, token_ids)
                result = _helper(indices, end1 + 1, end2, result, token_ids)
                i = end2 + 1
            else:
                end1 = _find_end_brace(indices, i + 1, end, token_ids)
                _append_range(result, i + 2, end1, flag[5])
                if special:
                    for pos in (i + 1, end1):
                        if 0 <= pos < len(result):
                            result[pos].append(flag[3])
                result = _helper(indices, i + 1, end1, result, token_ids)
                i = end1 + 1
        elif token == token_ids.pad:
            result[i].append(flag[0])
            i += 1
        elif token == token_ids.sos:
            result[i].append(flag[1])
            i += 1
        elif token == token_ids.eos:
            result[i].append(flag[2])
            i += 1
        else:
            result[i].append(flag[3])
            i += 1
    return result


def indices_to_multi_label(
    indices: Sequence[int],
    token_ids: PositionTokenIds,
    label_depth: int = 5,
) -> List[List[int]]:
    working = [int(x) for x in indices]
    result: List[List[int]] = [[] for _ in range(len(working))]
    is_reverse = bool(working and working[0] == token_ids.eos)
    if is_reverse:
        working = list(reversed(working))

    result = _helper(working, -1, len(working), result, token_ids)
    if is_reverse:
        result.reverse()
    return [_pad_or_trim(label, label_depth) for label in result]


def _raw_multi_label(indices: Sequence[int], token_ids: PositionTokenIds) -> List[List[int]]:
    working = [int(x) for x in indices]
    result: List[List[int]] = [[] for _ in range(len(working))]
    is_reverse = bool(working and working[0] == token_ids.eos)
    if is_reverse:
        working = list(reversed(working))
    result = _helper(working, -1, len(working), result, token_ids)
    if is_reverse:
        result.reverse()
    return result


def _pad_or_trim(label: Sequence[int], label_depth: int) -> List[int]:
    values = [int(x) for x in label[:label_depth]]
    values.extend([0] * (label_depth - len(values)))
    return values


def tgt_to_multi_label(
    tgt: Sequence[Sequence[int]],
    token_ids: PositionTokenIds,
    label_depth: int = 5,
) -> List[List[List[int]]]:
    return [indices_to_multi_label(row, token_ids, label_depth) for row in tgt]


def _layer_and_pos_from_labels(labels: Sequence[Sequence[int]]) -> Tuple[List[int], List[int]]:
    layer_num = []
    final_pos = []
    for label in labels:
        if len(label) == 1:
            layer_num.append(0)
            final_pos.append(int(label[0]))
        elif len(label) <= 5:
            layer_num.append(len(label) - 1)
            final_pos.append(int(label[-2]))
        else:
            layer_num.append(4)
            final_pos.append(int(label[3]))
    return layer_num, final_pos


def out_to_layernum_and_pos(
    tgt: Sequence[Sequence[int]],
    token_ids: PositionTokenIds,
) -> Tuple[List[List[int]], List[List[int]]]:
    layer_num = []
    final_pos = []
    for row in tgt:
        labels = _raw_multi_label(row, token_ids)
        if labels:
            labels = labels[1:] + labels[:1]
        row_layer, row_pos = _layer_and_pos_from_labels(labels)
        layer_num.append(row_layer)
        final_pos.append(row_pos)
    return layer_num, final_pos


def _build_bi_tgt_from_padded(
    labels: LongTensor,
    lengths: LongTensor,
    sos_id: int,
    eos_id: int,
    pad_id: int,
) -> LongTensor:
    batch_size, max_len = labels.shape
    out_len = max_len + 1

    l2r_tgt = torch.full((batch_size, out_len), pad_id, dtype=torch.long)
    l2r_tgt[:, 0] = sos_id
    l2r_tgt[:, 1:] = labels.cpu()

    r2l_tgt = torch.full((batch_size, out_len), pad_id, dtype=torch.long)
    positions = torch.arange(max_len).unsqueeze(0).expand(batch_size, -1)
    rev_positions = (lengths.cpu().unsqueeze(1) - 1 - positions).clamp(min=0)
    reversed_labels = labels.cpu().gather(1, rev_positions)
    valid_mask = positions < lengths.cpu().unsqueeze(1)
    reversed_labels = reversed_labels.masked_fill(~valid_mask, pad_id)
    r2l_tgt[:, 0] = eos_id
    r2l_tgt[:, 1:] = reversed_labels

    return torch.cat((l2r_tgt, r2l_tgt), dim=0)


def build_position_targets_from_padded(
    labels: LongTensor,
    lengths: LongTensor,
    token_ids: PositionTokenIds,
    sos_id: int,
    eos_id: int,
    pad_id: int,
    label_depth: int = 5,
) -> PositionTargets:
    tgt = _build_bi_tgt_from_padded(labels, lengths, sos_id, eos_id, pad_id)
    rows = tgt.tolist()
    pos_tgt = torch.tensor(tgt_to_multi_label(rows, token_ids, label_depth), dtype=torch.float32)
    pos_layer, pos_pos = out_to_layernum_and_pos(rows, token_ids)
    return PositionTargets(
        pos_tgt=pos_tgt,
        pos_layer=torch.tensor(pos_layer, dtype=torch.long),
        pos_pos=torch.tensor(pos_pos, dtype=torch.long),
    )
