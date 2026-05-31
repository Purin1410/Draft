import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def join_rclone_path(remote_dir: str, rel_path: str) -> str:
    remote_dir = str(remote_dir).rstrip("/")
    rel_path = str(rel_path).lstrip("/")
    if remote_dir.endswith(":"):
        return f"{remote_dir}{rel_path}"
    return f"{remote_dir}/{rel_path}"


def build_remote_run_dir(rclone_cfg: Dict[str, Any], run_name: str) -> str:
    remote_root = _get(rclone_cfg, "remote_root", None)
    if not remote_root:
        raise ValueError("Missing required config: remote_root in rclone_cfg")
    run_dir = _get(rclone_cfg, "run_dir", None) or run_name
    return join_rclone_path(remote_root, str(run_dir))


def checkpoint_epoch_from_name(name: str, run_name: str) -> Optional[int]:
    path_name = str(name).replace("\\", "/")
    pattern = re.compile(rf"(^|/){re.escape(run_name)}_(?:epoch=)?(?P<epoch>\d+).*\.ckpt$")
    match = pattern.search(path_name)
    if match is None:
        return None
    return int(match.group("epoch"))


def select_latest_checkpoint(items: List[Dict[str, Any]], run_name: str) -> Optional[Dict[str, Any]]:
    candidates = []
    for item in items:
        rel_path = item.get("Path") or item.get("Name")
        if not rel_path:
            continue
        epoch = checkpoint_epoch_from_name(rel_path, run_name)
        if epoch is None:
            continue
        candidates.append({"epoch": epoch, "path": rel_path, "item": item})
    if not candidates:
        return None
    return max(candidates, key=lambda x: x["epoch"])


def is_completed_upload_file(path: Path, run_name: Optional[str] = None) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    if path.suffix not in {".ckpt", ".csv", ".jsonl"}:
        return False
    name = path.name
    if "_rank_" in name or name.endswith(".tmp") or name.endswith(".partial"):
        return False
    if run_name is not None and not name.startswith(run_name):
        return False
    return path.stat().st_size > 0


def collect_uploadable_files(paths: Iterable[Path], run_name: Optional[str] = None) -> List[Path]:
    files: List[Path] = []
    for root in paths:
        root_path = Path(root)
        if root_path.is_file():
            candidates = [root_path]
        elif root_path.exists():
            candidates = sorted(p for p in root_path.rglob("*") if p.is_file())
        else:
            candidates = []
        for path in candidates:
            if is_completed_upload_file(path, run_name=run_name):
                files.append(path)
    return sorted(dict.fromkeys(files))


def _run_rclone(cmd: List[str], fail_on_error: bool = False, **kwargs):
    try:
        result = subprocess.run(cmd, check=False, **kwargs)
    except (OSError, subprocess.SubprocessError):
        if fail_on_error:
            raise
        return None
    if getattr(result, "returncode", 0) != 0 and fail_on_error:
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            getattr(result, "stdout", None),
            getattr(result, "stderr", None),
        )
    return result


def find_and_download_latest_checkpoint(
    *,
    remote_run_dir: str,
    run_name: str,
    local_dir,
    rclone_command: str = "rclone",
    recursive: bool = False,
    download_flags: Optional[List[str]] = None,
    fail_on_error: bool = False,
) -> Optional[str]:
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    cmd = [rclone_command, "lsjson", remote_run_dir, "--files-only"]
    if recursive:
        cmd.append("-R")
    result = _run_rclone(cmd, fail_on_error=fail_on_error, capture_output=True, text=True)
    if result is None or getattr(result, "returncode", 0) != 0:
        return None

    try:
        items = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        if fail_on_error:
            raise
        return None

    selected = select_latest_checkpoint(items, run_name)
    if selected is None:
        return None

    remote_ckpt = join_rclone_path(remote_run_dir, selected["path"])
    local_ckpt = local_dir / Path(selected["path"]).name
    copy_cmd = [rclone_command, "copy", remote_ckpt, str(local_dir)]
    copy_cmd.extend(download_flags or ["--progress"])
    result = _run_rclone(copy_cmd, fail_on_error=fail_on_error)
    if result is None or getattr(result, "returncode", 0) != 0:
        return None
    return str(local_ckpt)


def ensure_remote_dir(
    remote_run_dir: str,
    *,
    rclone_command: str = "rclone",
    fail_on_error: bool = False,
) -> bool:
    result = _run_rclone([rclone_command, "mkdir", remote_run_dir], fail_on_error=fail_on_error)
    return result is not None and getattr(result, "returncode", 0) == 0


def upload_files(
    files: Iterable[Path],
    *,
    remote_run_dir: str,
    local_root=None,
    rclone_command: str = "rclone",
    copy_flags: Optional[List[str]] = None,
    fail_on_error: bool = False,
    run_name: Optional[str] = None,
) -> bool:
    ok = True
    flags = copy_flags or ["--update", "--verbose", "--no-traverse"]
    local_root = Path(local_root).resolve() if local_root is not None else None
    for file_path in collect_uploadable_files(files, run_name=run_name):
        file_path = Path(file_path)
        rel_parent = "."
        if local_root is not None:
            try:
                rel_parent = str(file_path.resolve().parent.relative_to(local_root))
            except ValueError:
                rel_parent = "."
        remote_dir = remote_run_dir if rel_parent in {"", "."} else join_rclone_path(remote_run_dir, rel_parent)
        cmd = [rclone_command, "move", str(file_path), remote_dir]
        cmd.extend(flags)
        result = _run_rclone(cmd, fail_on_error=fail_on_error)
        if result is None or getattr(result, "returncode", 0) != 0:
            ok = False
    return ok


def log_wandb_files(wandb_run, files: Iterable[Path], *, fail_on_error: bool = False, run_name: Optional[str] = None) -> bool:
    if wandb_run is None:
        return True
    try:
        for file_path in collect_uploadable_files(files, run_name=run_name):
            if hasattr(wandb_run, "save"):
                wandb_run.save(str(file_path), base_path=str(file_path.parent), policy="now")
            elif hasattr(wandb_run, "log"):
                wandb_run.log({"uploaded_file": str(file_path)})
        return True
    except Exception:
        if fail_on_error:
            raise
        return False


def cleanup_uploaded_files(files: Iterable[Path], *, keep_last_local_checkpoints: int = 1, run_name: Optional[str] = None) -> List[Path]:
    uploadable = collect_uploadable_files(files, run_name=run_name)
    ckpts = sorted(
        (p for p in uploadable if p.suffix == ".ckpt"),
        key=lambda p: (p.stat().st_mtime, p.name),
        reverse=True,
    )
    keep = set(ckpts[: max(0, int(keep_last_local_checkpoints))])
    removed: List[Path] = []
    for path in uploadable:
        if path in keep:
            continue
        try:
            path.unlink()
            removed.append(path)
        except FileNotFoundError:
            pass
    return removed


def maybe_cleanup_uploaded_files(
    files: Iterable[Path],
    *,
    rclone_ok: bool,
    wandb_ok: bool,
    cleanup: bool,
    keep_last_local_checkpoints: int = 1,
    run_name: Optional[str] = None,
) -> List[Path]:
    if not cleanup or not rclone_ok or not wandb_ok:
        return []
    return cleanup_uploaded_files(files, keep_last_local_checkpoints=keep_last_local_checkpoints, run_name=run_name)
