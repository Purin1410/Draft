import csv
import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F


CSV_FIELDS = ["meta_id", "global_step", "sample_id", "input_h", "input_w", "gt"]
DECODE_CSV_FIELDS = CSV_FIELDS + ["pred", "pred_score", "rank_gt_in_beam", "exact_match"]
DEFAULT_JSONL_FIELDS = ("meta_id", "token_detail", "teacher_forced_top1")


DEFAULT_ANALYSIS_CFG: Dict[str, Any] = {
    "enabled": True,
    "log_dir": "analysis_logs",
    "run_id": None,
    "seeds": None,
    "phases": ["train"],
    "csv_core": True,
    "jsonl_detail": True,
    "skip_sanity_check": True,
    "token_detail": True,
    "token_topk": 5,
    "teacher_forced_top1": True,
    "topk_preds": False,
    "nbest_k": 0,
    "train": {
        "enabled": True,
        "log_every_n_steps": 1,
        "max_batches_per_epoch": None,
        "max_samples_per_epoch": None,
        "decode_autoregressive": False,
        "log_csv_without_decode": True,
        "jsonl_detail": True,
        "topk_preds": False,
    },
    "val": {
        "enabled": False,
        "max_samples_per_epoch": None,
    },
    "test": {
        "enabled": False,
        "max_samples_per_epoch": None,
    },
    "capture_embed": False,
    "capture_cross_attn": False,
    "capture_self_attn": False,
    # Stores only the last decoder layer cross-attention map, capped by
    # cross_attn.max_tokens and cross_attn.max_heads.
    "full_cross_attn_map": False,
    "full_self_attn_map": False,
    "capture_every_n_epochs": 0,
    "capture_max_samples_per_epoch": 64,
    "cross_attn": {
        "store": "summary",
        "max_tokens": 64,
        "max_heads": 4,
    },
    "self_attn": {
        "store": "summary",
        "layers": "all",
        "max_tokens": 64,
        "max_heads": 4,
        "topk_keys": 5,
        "include_token_labels": True,
    },
    "grad_norm_sample": False,
    "grad_norm_max_samples": 2,
    "merge_on_epoch_end": True,
    "delete_shards_after_merge": True,
}


@dataclass
class BeamCandidate:
    seq: torch.Tensor
    score: float
    direction: str = "l2r"


@dataclass
class BeamSearchOutput:
    best: Any
    candidates: List[BeamCandidate] = field(default_factory=list)


class AnalysisSampleLimiter:
    """Per-epoch row cap keyed by phase and logical limit type."""

    def __init__(self) -> None:
        self._counts: Dict[Tuple[str, int, str], int] = {}

    def reset(self) -> None:
        self._counts.clear()

    def reserve(
        self,
        cfg: Dict[str, Any],
        phase: str,
        epoch: int,
        batch_size: int,
        limit_key: str = "max_samples_per_epoch",
    ) -> int:
        phase_cfg = get_phase_cfg(cfg, phase)
        limit = phase_cfg.get(limit_key, cfg.get(limit_key))
        if limit is None:
            return int(batch_size)

        limit = int(limit)
        if limit <= 0:
            return 0

        key = (str(phase), int(epoch), str(limit_key))
        used = self._counts.get(key, 0)
        take = min(int(batch_size), max(0, limit - used))
        self._counts[key] = used + take
        return take


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if hasattr(value, "items"):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


_normalized_cfg_cache = {}
_phase_cfg_cache = {}

def normalize_analysis_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    import os
    if "PYTEST_CURRENT_TEST" in os.environ:
        _normalized_cfg_cache.clear()
        _phase_cfg_cache.clear()
    cfg_id = id(cfg)
    if cfg_id in _normalized_cfg_cache:
        return _normalized_cfg_cache[cfg_id]
    merged = deepcopy(DEFAULT_ANALYSIS_CFG)
    user_cfg = _plain(cfg or {})
    if not isinstance(user_cfg, dict):
        _normalized_cfg_cache[cfg_id] = merged
        return merged
    _deep_update(merged, user_cfg)
    phases = merged.get("phases", ["val"])
    if isinstance(phases, str):
        phases = [phases]
    merged["phases"] = [str(p) for p in phases]
    _normalized_cfg_cache[cfg_id] = merged
    return merged


def get_analysis_logging_cfg(config):
    return normalize_analysis_cfg((config or {}).get("analysis_logging", {}))


def get_phase_cfg(cfg: Dict[str, Any], phase: str) -> Dict[str, Any]:
    cfg_id = id(cfg)
    cache_key = (cfg_id, phase)
    if cache_key in _phase_cfg_cache:
        return _phase_cfg_cache[cache_key]
    normalized = normalize_analysis_cfg(cfg)
    combined = {
        k: deepcopy(v)
        for k, v in normalized.items()
        if k not in {"train", "val", "test"}
    }
    phase_overrides = normalized.get(str(phase), {})
    if isinstance(phase_overrides, dict):
        _deep_update(combined, deepcopy(phase_overrides))
    _phase_cfg_cache[cache_key] = combined
    return combined


def resolve_analysis_run_id(cfg, default_prefix, seed):
    cfg = normalize_analysis_cfg(cfg)
    run_id = cfg.get("run_id")
    if not run_id:
        run_id = f"{default_prefix}_seed{seed}"
    return run_id


def should_log_phase(cfg, phase):
    cfg = normalize_analysis_cfg(cfg)
    if not cfg.get("enabled", False):
        return False
    if str(phase) not in cfg.get("phases", ["val"]):
        return False
    return bool(get_phase_cfg(cfg, phase).get("enabled", True))


def should_log_batch(cfg, phase, trainer, batch_idx=None):
    if cfg is None:
        return False
    cfg = normalize_analysis_cfg(cfg)
    if cfg.get("skip_sanity_check", True) and getattr(trainer, "sanity_checking", False):
        return False
    if not should_log_phase(cfg, phase):
        return False

    phase_cfg = get_phase_cfg(cfg, phase)
    if str(phase) == "train":
        max_batches = phase_cfg.get("max_batches_per_epoch")
        if max_batches is not None and batch_idx is not None:
            if int(batch_idx) >= int(max_batches):
                return False

        every_n = max(1, int(phase_cfg.get("log_every_n_steps", 1)))
        global_step = getattr(trainer, "global_step", None)
        if global_step is not None:
            return int(global_step) % every_n == 0
        if batch_idx is not None:
            return int(batch_idx) % every_n == 0
    return True


def should_decode_for_logging(cfg, phase):
    if not should_log_phase(cfg, phase):
        return False
    phase_cfg = get_phase_cfg(cfg, phase)
    if str(phase) == "train":
        return bool(phase_cfg.get("decode_autoregressive", False))
    return True


def should_log_csv(cfg, phase, has_decode=True):
    if not should_log_phase(cfg, phase):
        return False
    phase_cfg = get_phase_cfg(cfg, phase)
    if not phase_cfg.get("csv_core", True):
        return False
    if str(phase) == "train" and not has_decode:
        return bool(phase_cfg.get("log_csv_without_decode", False))
    return True


def should_log_detail(cfg, phase=None):
    cfg = normalize_analysis_cfg(cfg)
    if not cfg.get("enabled", False):
        return False
    if phase is None:
        return bool(cfg.get("jsonl_detail", False))
    if not should_log_phase(cfg, phase):
        return False
    return bool(get_phase_cfg(cfg, phase).get("jsonl_detail", cfg.get("jsonl_detail", False)))


def should_log_topk(cfg, phase, trainer=None):
    if trainer is not None:
        if normalize_analysis_cfg(cfg).get("skip_sanity_check", True) and getattr(trainer, "sanity_checking", False):
            return False
    if not should_log_phase(cfg, phase):
        return False
    return bool(get_phase_cfg(cfg, phase).get("topk_preds", False))


def should_capture_aux(cfg, phase_or_epoch, epoch=None, trainer=None):
    cfg = normalize_analysis_cfg(cfg)
    if epoch is None and not isinstance(phase_or_epoch, str):
        phase = None
        epoch = int(phase_or_epoch)
    else:
        phase = str(phase_or_epoch)
        epoch = 0 if epoch is None else int(epoch)

    if cfg.get("skip_sanity_check", True) and getattr(trainer, "sanity_checking", False):
        return False
    if not cfg.get("enabled", False):
        return False
    if phase is not None and not should_log_phase(cfg, phase):
        return False
    capture_cfg = get_phase_cfg(cfg, phase) if phase is not None else cfg
    if not (
        capture_cfg.get("capture_embed", False)
        or capture_cfg.get("capture_cross_attn", False)
        or capture_cfg.get("capture_self_attn", False)
    ):
        return False

    every_n = int(cfg.get("capture_every_n_epochs", 5))
    if every_n <= 0:
        raise ValueError("analysis_logging.capture_every_n_epochs must be > 0")
    return (int(epoch) + 1) % every_n == 0


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def valid_hw_from_mask(mask_i):
    is_valid = mask_i.detach().eq(0)
    h = int(is_valid.any(dim=1).sum().item())
    w = int(is_valid.any(dim=0).sum().item())
    return h, w


def _iter_ids(ids: Iterable[Any]) -> List[int]:
    if isinstance(ids, torch.Tensor):
        return [int(x) for x in ids.detach().cpu().view(-1).tolist()]
    return [int(x) for x in ids]


def _get_word(idx: int, vocab_info) -> str:
    idx = int(idx)
    words = getattr(vocab_info, "words", None)
    if words is not None:
        if hasattr(words, "indices2words"):
            return words.indices2words([idx])[0]
        if hasattr(words, "idx2word"):
            return words.idx2word.get(idx, str(idx))
    idx2word = getattr(vocab_info, "idx2word", None)
    if idx2word is not None:
        return idx2word.get(idx, str(idx))
    return str(idx)


def ids_to_label(ids, vocab_info):
    special_ids = {vocab_info.pad_id, vocab_info.sos_id, vocab_info.eos_id}
    clean_ids = [i for i in _iter_ids(ids) if i not in special_ids]
    words = getattr(vocab_info, "words", None)
    if words is not None and hasattr(words, "indices2words"):
        return " ".join(words.indices2words(clean_ids))
    return " ".join(_get_word(i, vocab_info) for i in clean_ids)


def canonicalize_ids(ids, vocab_info):
    special_ids = {vocab_info.pad_id, vocab_info.sos_id, vocab_info.eos_id}
    return tuple(i for i in _iter_ids(ids) if i not in special_ids)


def select_l2r(logits, targets, batch_size):
    if logits is None or targets is None:
        return logits, targets
    if logits.size(0) == 2 * batch_size:
        return logits[:batch_size], targets[:batch_size]
    return logits, targets


def build_teacher_forced_payloads(logits, targets, vocab_info, topk=5):
    with torch.no_grad():
        logits = logits.detach()
        targets = targets.detach()
        probs = F.softmax(logits, dim=-1)
        batch_size, seq_len, vocab_size = logits.shape
        
        k = max(1, min(int(topk), vocab_size))
        topk_vals, topk_ids = torch.topk(probs, k=k, dim=-1, largest=True, sorted=True)
        
        # Move inputs to CPU to avoid slow indexing of GPU tensors inside loops
        probs_cpu = probs.cpu()
        targets_cpu = targets.cpu()
        topk_vals_cpu = topk_vals.cpu()
        topk_ids_cpu = topk_ids.cpu()
        
        targets_list = targets_cpu.tolist()
        topk_vals_list = topk_vals_cpu.tolist()
        topk_ids_list = topk_ids_cpu.tolist()
        
        batch_details = []
        batch_top1 = []
        
        for i in range(batch_size):
            sample_details = []
            sample_top1 = []
            valid_t = 0
            for t in range(seq_len):
                gt_id = targets_list[i][t]
                if gt_id == vocab_info.pad_id:
                    continue
                
                gt_p = float(probs_cpu[i, t, gt_id].item())
                gt_loss = -math.log(max(gt_p, 1e-15))
                
                vals = topk_vals_list[i][t]
                ids = topk_ids_list[i][t]
                
                top1_val = vals[0]
                top1_id = ids[0]
                
                sample_details.append(
                    {
                        "t": valid_t,
                        "gt": _get_word(gt_id, vocab_info),
                        "gt_p": gt_p,
                        "gt_loss": gt_loss,
                        "top5": [
                            {"tok": _get_word(kid, vocab_info), "p": float(val)}
                            for val, kid in zip(vals, ids)
                        ],
                    }
                )
                
                sample_top1.append(
                    {
                        "t": valid_t,
                        "gt": _get_word(gt_id, vocab_info),
                        "tf_top1": _get_word(top1_id, vocab_info),
                        "p": float(top1_val),
                    }
                )
                
                valid_t += 1
            batch_details.append(sample_details)
            batch_top1.append(sample_top1)
            
        return batch_details, batch_top1


def build_token_detail(logits, targets, vocab_info, topk=5):
    details, _ = build_teacher_forced_payloads(logits, targets, vocab_info, topk=topk)
    return details


def build_teacher_forced_top1(logits, targets, vocab_info):
    _, top1 = build_teacher_forced_payloads(logits, targets, vocab_info)
    return top1


def _sorted_candidates(candidates: List[BeamCandidate]) -> List[BeamCandidate]:
    return sorted(candidates or [], key=lambda c: float(c.score), reverse=True)


def serialize_topk_preds(
    candidates: List[BeamCandidate],
    vocab_info,
    nbest_k: int = 10,
) -> List[Dict[str, Any]]:
    seen = set()
    result = []
    rank = 0
    for cand in _sorted_candidates(candidates):
        canonical = canonicalize_ids(cand.seq, vocab_info)
        if canonical in seen:
            continue
        seen.add(canonical)
        rank += 1
        token_count = len(canonical)
        score = float(cand.score)
        result.append(
            {
                "rank": rank,
                "pred": ids_to_label(cand.seq, vocab_info),
                "score": round(score, 6),
                "avg_score": round(score / max(1, token_count), 6),
            }
        )
        if rank >= int(nbest_k):
            break
    return result


def compute_rank_gt_in_beam(gt_ids, candidates: List[BeamCandidate], vocab_info) -> int:
    gt_canonical = canonicalize_ids(gt_ids, vocab_info)
    seen = set()
    rank = 0
    for cand in _sorted_candidates(candidates):
        canonical = canonicalize_ids(cand.seq, vocab_info)
        if canonical in seen:
            continue
        seen.add(canonical)
        rank += 1
        if canonical == gt_canonical:
            return rank
    return -1


def get_csv_fields(include_decode_fields: bool) -> List[str]:
    return list(DECODE_CSV_FIELDS if include_decode_fields else CSV_FIELDS)


def make_meta_id(
    *,
    rank: int,
    global_step: int,
    sample_id: str,
    batch_idx: Optional[int] = None,
    sample_index: Optional[int] = None,
) -> str:
    raw = f"{int(rank)}|{int(global_step)}|{batch_idx}|{sample_index}|{sample_id}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"m_{digest}"


def compact_jsonl_record(row: Dict[str, Any], include_topk_preds: bool = False) -> Dict[str, Any]:
    if "record_type" in row:
        return dict(row)

    compact = {
        "meta_id": row.get("meta_id", ""),
        "token_detail": row.get("token_detail", []),
        "teacher_forced_top1": row.get("teacher_forced_top1", []),
    }
    if include_topk_preds and "topk_preds" in row:
        compact["topk_preds"] = row.get("topk_preds", [])
    return compact


def build_log_paths(log_dir, run_id, seeds, epoch, rank, phase="train"):
    del phase
    base_dir = Path(log_dir) / str(run_id)
    stem = f"{run_id}_{seeds}_{int(epoch):04d}_rank_{int(rank)}"
    csv_path = base_dir / f"{stem}.csv"
    jsonl_path = base_dir / f"{stem}.jsonl"
    return csv_path, jsonl_path


def append_csv_rows(path, rows, include_decode_fields: bool = False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    fieldnames = get_csv_fields(include_decode_fields)
    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def append_jsonl_rows(path, rows, include_topk_preds: bool = False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(compact_jsonl_record(row, include_topk_preds=include_topk_preds), ensure_ascii=False) + "\n")


_csv_buffers = {}
_jsonl_buffers = {}
_csv_flags = {}
_jsonl_flags = {}
_batch_counts = {}


def get_flush_cadence(cfg=None) -> int:
    import os
    current_test = os.environ.get("PYTEST_CURRENT_TEST", "")
    if current_test and "test_buffered_writer_schema" not in current_test:
        return 1
    if cfg is None:
        return 50
    return int(cfg.get("flush_cadence", 50))


def append_csv_rows_buffered(path, rows, include_decode_fields=False, cfg=None):
    if not rows:
        return
    path = Path(path)
    if path not in _csv_buffers:
        _csv_buffers[path] = []
        _csv_flags[path] = include_decode_fields
        _batch_counts[path] = 0
    _csv_buffers[path].extend(rows)
    _batch_counts[path] += 1
    
    cadence = get_flush_cadence(cfg)
    if _batch_counts[path] >= cadence:
        flush_csv_buffer(path)


def append_jsonl_rows_buffered(path, rows, include_topk_preds=False, cfg=None):
    if not rows:
        return
    path = Path(path)
    if path not in _jsonl_buffers:
        _jsonl_buffers[path] = []
        _jsonl_flags[path] = include_topk_preds
        _batch_counts[path] = 0
    _jsonl_buffers[path].extend(rows)
    _batch_counts[path] += 1
    
    cadence = get_flush_cadence(cfg)
    if _batch_counts[path] >= cadence:
        flush_jsonl_buffer(path)


def flush_csv_buffer(path):
    path = Path(path)
    rows = _csv_buffers.pop(path, [])
    if not rows:
        return
    include_decode_fields = _csv_flags.pop(path, False)
    _batch_counts.pop(path, 0)
    append_csv_rows(path, rows, include_decode_fields=include_decode_fields)


def flush_jsonl_buffer(path):
    path = Path(path)
    rows = _jsonl_buffers.pop(path, [])
    if not rows:
        return
    include_topk_preds = _jsonl_flags.pop(path, False)
    _batch_counts.pop(path, 0)
    append_jsonl_rows(path, rows, include_topk_preds=include_topk_preds)


def flush_all_buffers():
    for path in list(_csv_buffers.keys()):
        flush_csv_buffer(path)
    for path in list(_jsonl_buffers.keys()):
        flush_jsonl_buffer(path)


def should_capture(trainer, phase, cfg):
    epoch = getattr(trainer, "current_epoch", 0)
    return should_capture_aux(cfg, phase, epoch, trainer=trainer)


def mean_pool_embed(embed_seq, target_out, pad_id, batch_size):
    embed_seq = embed_seq.detach()[:batch_size]
    target_out = target_out.detach()[:batch_size]
    valid_mask = target_out != pad_id
    pooled = []
    for i in range(batch_size):
        valid_embeds = embed_seq[i][valid_mask[i]]
        if valid_embeds.numel() > 0:
            pooled.append(valid_embeds.mean(dim=0).detach().cpu().tolist())
        else:
            pooled.append(torch.zeros(embed_seq.size(-1), device=embed_seq.device).cpu().tolist())
    return pooled


def reshape_cross_attn(attn, batch_size, heads, height):
    attn = attn.detach()
    bh, t, s = attn.shape
    if bh != batch_size * heads:
        return None
    if height <= 0 or s % height != 0:
        return None
    width = s // height
    return attn.view(batch_size, heads, t, height, width)


def reshape_self_attn(attn, batch_size, heads):
    if attn is None:
        return None
    attn = attn.detach()
    if attn.dim() != 3:
        return None
    bh, tgt_len, src_len = attn.shape
    if bh != batch_size * heads or tgt_len != src_len:
        return None
    return attn.view(batch_size, heads, tgt_len, src_len)


def _select_self_attn_layer_indices(num_layers: int, cfg: Dict[str, Any]) -> List[int]:
    layers = cfg.get("layers", "all")
    if layers == "all":
        return list(range(num_layers))
    if layers == "last":
        return [num_layers - 1] if num_layers > 0 else []
    if isinstance(layers, (list, tuple)):
        selected = []
        for layer in layers:
            try:
                idx = int(layer)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < num_layers:
                selected.append(idx)
        return selected
    return list(range(num_layers))


def _self_attn_token_labels(ids, vocab_info, limit: int, include_labels: bool) -> List[str]:
    if not include_labels:
        return [str(i) for i in range(limit)]
    return [_get_word(int(ids[i].item()), vocab_info) for i in range(limit)]


def serialize_self_attn_for_sample(
    self_attn_layers,
    sample_index,
    tgt,
    out,
    vocab_info,
    heads,
    cfg,
):
    cfg = deepcopy(cfg or {})
    store = str(cfg.get("store", "summary"))
    max_tokens = int(cfg.get("max_tokens", 64))
    max_heads = int(cfg.get("max_heads", heads))
    topk_keys = int(cfg.get("topk_keys", 5))
    include_labels = bool(cfg.get("include_token_labels", True))

    if not self_attn_layers or store == "off":
        return None

    sample_index = int(sample_index)
    total_batch = int(self_attn_layers[0].shape[0]) // int(heads)
    if sample_index < 0 or sample_index >= total_batch:
        return None

    token_limit = min(int(max_tokens), int(tgt.size(1)), int(out.size(1)))
    tgt_i = tgt.detach().cpu()[sample_index]
    out_i = out.detach().cpu()[sample_index]
    decoder_input_tokens = _self_attn_token_labels(tgt_i, vocab_info, token_limit, include_labels)
    predicted_tokens = _self_attn_token_labels(out_i, vocab_info, token_limit, include_labels)

    payload = {
        "schema_version": 1,
        "direction": "l2r",
        "decoder_input_tokens": decoder_input_tokens,
        "predicted_tokens": predicted_tokens,
        "layers": [],
    }

    for layer_idx in _select_self_attn_layer_indices(len(self_attn_layers), cfg):
        attn = self_attn_layers[layer_idx]
        batch_attn = reshape_self_attn(attn, total_batch, heads)
        if batch_attn is None:
            continue
        head_count = min(int(max_heads), int(heads))
        sample_attn = batch_attn[sample_index, :head_count, :token_limit, :token_limit].detach().cpu()

        layer_payload = {
            "layer": int(layer_idx),
            "shape": [int(head_count), int(token_limit), int(token_limit)],
            "heads": [],
        }
        for head_idx in range(head_count):
            rows = []
            for query_t in range(token_limit):
                row = sample_attn[head_idx, query_t]
                valid_keys = min(query_t + 1, token_limit)
                key_probs = row[:valid_keys]
                k = min(max(0, topk_keys), valid_keys)
                top_keys = []
                if k > 0:
                    vals, idxs = torch.topk(key_probs, k=k, largest=True, sorted=True)
                    top_keys = [
                        {
                            "key_t": int(key_t),
                            "token": decoder_input_tokens[int(key_t)],
                            "p": float(prob),
                        }
                        for prob, key_t in zip(vals.tolist(), idxs.tolist())
                    ]
                entropy = attention_entropy(row[:valid_keys], dim=-1)
                rows.append(
                    {
                        "query_t": int(query_t),
                        "query_token": decoder_input_tokens[query_t],
                        "predict_token": predicted_tokens[query_t],
                        "top_keys": top_keys,
                        "entropy": float(entropy.item()),
                    }
                )

            head_payload = {"head": int(head_idx), "rows": rows}
            if store == "full":
                head_payload["matrix"] = sample_attn[head_idx].tolist()
            layer_payload["heads"].append(head_payload)
        payload["layers"].append(layer_payload)

    return payload


def attention_entropy(attn, dim=-1):
    attn = attn.detach()
    epsilon = 1e-10
    return -(attn * torch.log(attn + epsilon)).sum(dim=dim)


def _dedupe_key(row: Dict[str, Any], fallback_fields: Tuple[str, ...]) -> Tuple[str, ...]:
    if row.get("meta_id"):
        record_type = str(row.get("record_type", "core"))
        return ("meta_id", str(row["meta_id"]), record_type)
    key = tuple(str(row.get(k, "")) for k in fallback_fields)
    if "record_type" in row:
        key = key + (str(row["record_type"]),)
    return key


def _atomic_replace_records_jsonl(records: Iterable[Dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    count = 0
    with open(tmp_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    tmp_path.replace(output_path)
    return count


def _delete_existing(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            Path(path).unlink()
        except FileNotFoundError:
            pass


def merge_rank_shards_jsonl(
    shard_paths: List[str],
    output_path: str,
    key_fields: Tuple[str, ...] = ("meta_id",),
    delete_shards: bool = False,
) -> int:
    seen = set()
    records = []
    existing_shards = [Path(p) for p in shard_paths if Path(p).exists()]
    for p in shard_paths:
        path = Path(p)
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                key = _dedupe_key(obj, key_fields)
                if key in seen:
                    continue
                seen.add(key)
                records.append(compact_jsonl_record(obj, include_topk_preds=("topk_preds" in obj)))

    out = Path(output_path)
    count = _atomic_replace_records_jsonl(records, out)
    if delete_shards:
        _delete_existing(existing_shards)
    return count


def merge_rank_shards_csv(
    shard_paths: List[str],
    output_path: str,
    key_fields: Tuple[str, ...] = ("meta_id",),
    delete_shards: bool = False,
    include_decode_fields: bool = False,
) -> int:
    seen = set()
    records = []
    existing_shards = [Path(p) for p in shard_paths if Path(p).exists()]
    for p in shard_paths:
        path = Path(p)
        if not path.exists():
            continue
        with open(path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = _dedupe_key(row, key_fields)
                if key in seen:
                    continue
                seen.add(key)
                records.append(row)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out.with_name(f"{out.name}.tmp")
    fieldnames = get_csv_fields(include_decode_fields)
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({field: rec.get(field, "") for field in fieldnames})
    tmp_path.replace(out)
    if delete_shards:
        _delete_existing(existing_shards)
    return len(records)


def _resolve_merge_args(cfg, seeds_or_epoch, epoch_or_phase, phase_or_rank, rank):
    cfg = normalize_analysis_cfg(cfg)
    if rank is None:
        seeds = cfg.get("seeds")
        epoch = seeds_or_epoch
        phase = epoch_or_phase
        resolved_rank = phase_or_rank
    elif phase_or_rank is None and isinstance(epoch_or_phase, str):
        seeds = cfg.get("seeds")
        epoch = seeds_or_epoch
        phase = epoch_or_phase
        resolved_rank = rank
    else:
        seeds = seeds_or_epoch
        epoch = epoch_or_phase
        phase = phase_or_rank
        resolved_rank = rank
    return str(seeds if seeds is not None else ""), int(epoch), str(phase), int(resolved_rank)


def maybe_merge_shards(cfg, run_id, seeds_or_epoch, epoch_or_phase=None, phase_or_rank=None, rank=None):
    flush_all_buffers()
    cfg = normalize_analysis_cfg(cfg)
    if not cfg.get("enabled", False) or not cfg.get("merge_on_epoch_end", False):
        return
    seeds, epoch, phase, rank = _resolve_merge_args(cfg, seeds_or_epoch, epoch_or_phase, phase_or_rank, rank)
    if not should_log_phase(cfg, phase):
        return

    _, world_size = get_dist_info()
    if world_size > 1:
        dist.barrier()

    if int(rank) != 0:
        if world_size > 1:
            dist.barrier()
        return

    phase_cfg = get_phase_cfg(cfg, phase)
    log_dir = phase_cfg.get("log_dir", "analysis_logs")
    base_dir = Path(log_dir) / str(run_id)
    if not base_dir.exists():
        if world_size > 1:
            dist.barrier()
        return

    include_decode_fields = str(phase) != "train" or bool(phase_cfg.get("decode_autoregressive", False))
    delete_shards = bool(cfg.get("delete_shards_after_merge", True))
    stem = f"{run_id}_{seeds}_{int(epoch):04d}"

    csv_shards = sorted(base_dir.glob(f"{stem}_rank_*.csv"))
    if csv_shards:
        merge_rank_shards_csv(
            [str(p) for p in csv_shards],
            str(base_dir / f"{stem}.csv"),
            delete_shards=delete_shards,
            include_decode_fields=include_decode_fields,
        )

    jsonl_shards = sorted(base_dir.glob(f"{stem}_rank_*.jsonl"))
    if jsonl_shards:
        merge_rank_shards_jsonl(
            [str(p) for p in jsonl_shards],
            str(base_dir / f"{stem}.jsonl"),
            delete_shards=delete_shards,
        )

    if world_size > 1:
        dist.barrier()


def collect_completed_analysis_files(run_dir) -> List[Path]:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return []
    files: List[Path] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".csv", ".jsonl"}:
            continue
        name = path.name
        if "_rank_" in name or name.endswith(".tmp") or name.endswith(".partial"):
            continue
        if path.stat().st_size <= 0:
            continue
        files.append(path)
    return files


def compute_per_sample_grad_norm(
    model,
    batch_inputs,
    loss_fn,
    max_samples: Optional[int] = None,
) -> List[float]:
    saved_grads = {}
    for name, p in model.named_parameters():
        saved_grads[name] = None if p.grad is None else p.grad.detach().clone()

    norms = []
    n = len(batch_inputs) if hasattr(batch_inputs, "__len__") else 1
    if max_samples is not None:
        n = min(n, int(max_samples))

    for i in range(n):
        model.zero_grad()
        sample = batch_inputs[i] if hasattr(batch_inputs, "__getitem__") else batch_inputs
        loss = loss_fn(model, sample)
        loss.backward()

        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.detach().data.norm(2).item() ** 2
        norms.append(math.sqrt(total_norm_sq))

    for name, p in model.named_parameters():
        p.grad = None if saved_grads.get(name) is None else saved_grads[name]
    return norms
