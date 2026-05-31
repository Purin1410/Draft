import re
from typing import Any, Optional, Union


_SAFE_PART_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def _ensure_mapping(obj: Any, key: str):
    value = _get(obj, key)
    if value is None:
        value = {}
        _set(obj, key, value)
    return value


def sanitize_run_part(value: object) -> str:
    if value is None:
        raise ValueError("run name parts cannot be None")
    text = str(value).strip()
    if not text:
        raise ValueError("run name parts cannot be empty")
    text = _SAFE_PART_RE.sub("_", text).strip("._-")
    if not text:
        raise ValueError("run name parts cannot sanitize to empty")
    return text


def build_run_name(
    model_name: str,
    datasets: str,
    seeds: Union[int, str],
    run_type: str,
    ablation_type: Optional[str] = None,
) -> str:
    run_type = sanitize_run_part(run_type)
    if run_type not in {"baseline", "ablation"}:
        raise ValueError("run_type must be 'baseline' or 'ablation'")
    if run_type == "baseline" and ablation_type is not None and str(ablation_type).strip():
        raise ValueError("ablation_type is only valid for ablation runs")
    if run_type == "ablation" and (ablation_type is None or not str(ablation_type).strip()):
        raise ValueError("ablation_type is required for ablation runs")

    seed_part = sanitize_run_part(seeds)
    int(seed_part)
    parts = [
        sanitize_run_part(model_name),
        sanitize_run_part(datasets),
        seed_part,
        run_type,
    ]
    if ablation_type is not None and str(ablation_type).strip():
        parts.append(sanitize_run_part(ablation_type))
    return "_".join(parts)


def build_checkpoint_filename(run_name: str) -> str:
    return f"{sanitize_run_part(run_name)}_{{epoch}}-{{val_ExpRate:.4f}}"


def _find_checkpoint_callback(config):
    trainer = _ensure_mapping(config, "trainer")
    callbacks = _get(trainer, "callbacks", [])
    for callback in callbacks:
        class_path = str(_get(callback, "class_path", ""))
        init_args = _get(callback, "init_args", {})
        if class_path.endswith("ModelCheckpoint") or _get(init_args, "filename") is not None:
            return callback
    if len(callbacks) > 1:
        return callbacks[1]
    raise ValueError("Could not find ModelCheckpoint callback config")


def apply_runtime_overrides(
    config,
    *,
    model_name,
    datasets,
    seeds,
    run_type,
    ablation_type=None,
):
    run_name = build_run_name(model_name, datasets, seeds, run_type, ablation_type)
    seed_int = int(str(seeds).strip())
    _set(config, "seed_everything", seed_int)

    checkpoint_callback = _find_checkpoint_callback(config)
    init_args = _ensure_mapping(checkpoint_callback, "init_args")
    _set(init_args, "filename", build_checkpoint_filename(run_name))

    wandb_cfg = _ensure_mapping(config, "wandb")
    _set(wandb_cfg, "name", run_name)

    analysis_cfg = _ensure_mapping(config, "analysis_logging")
    _set(analysis_cfg, "run_id", run_name)
    _set(analysis_cfg, "seeds", str(seeds))
    return run_name
