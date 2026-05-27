import pytorch_lightning as pl
from datamodule import CROHMEDatamodule
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    Callback
)
import subprocess
from pytorch_lightning.loggers import WandbLogger as Logger
import argparse
from sconf import Config
import os
import re
import json
import subprocess
from pathlib import Path

class MoreValidationCallback(pl.Callback):
    def __init__(self, monitor="val_ExpRate"):
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer, pl_module):
        metric = trainer.callback_metrics.get(self.monitor)
        if trainer.current_epoch >= 220:
            if metric is not None:
                if metric > 0.55:
                    trainer.check_val_every_n_epoch = 1
            else:
                trainer.check_val_every_n_epoch = 2
        elif trainer.current_epoch >= 148:
            trainer.check_val_every_n_epoch = 25

class RcloneUploadCallback(Callback):
    """
    Copy all checkpoint files from local_dir to remote_dir.

    Important:
      - use rclone copy, not move, so local checkpoints stay available
      - do not use --ignore-existing, so rewritten files can be updated remotely
    """
    def __init__(self, local_dir, remote_dir, every_n_epochs=1, upload_on_train_end=True):
        super().__init__()
        self.local_dir = str(local_dir)
        self.remote_dir = remote_dir
        self.every_n_epochs = every_n_epochs
        self.upload_on_train_end = upload_on_train_end

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        if self.every_n_epochs is None:
            return

        epoch_num = trainer.current_epoch + 1
        if epoch_num % self.every_n_epochs == 0:
            self._rclone_upload()

    def on_train_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        if self.upload_on_train_end:
            self._rclone_upload()

    def _rclone_upload(self):
        print("Uploading checkpoints to remote...")
        command = [
            "rclone",
            "copy",
            "--update",
            "--verbose",
            self.local_dir,
            self.remote_dir,
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during upload: {e}")
        print("Upload completed.")


class ConditionalLastCheckpointCallback(Callback):
    """
    Save one extra final checkpoint only when the last epoch metric is worse
    than the best metric already saved by ModelCheckpoint.

    For mode="max":
      save last iff last_score < best_score

    For mode="min":
      save last iff last_score > best_score
    """
    def __init__(
        self,
        checkpoint_callback,
        dirpath,
        filename_template,
        monitor="val_ExpRate",
        mode="max",
    ):
        super().__init__()
        self.checkpoint_callback = checkpoint_callback
        self.dirpath = Path(dirpath)
        self.filename_template = filename_template
        self.monitor = monitor
        self.mode = mode

        self.last_score = None
        self.last_metric_epoch = None
        self.last_train_epoch = None

    @staticmethod
    def _to_float(value):
        if value is None:
            return None

        try:
            if hasattr(value, "detach"):
                return float(value.detach().cpu().item())
            return float(value)
        except (TypeError, ValueError):
            return None

    def on_validation_epoch_end(self, trainer, pl_module):
        metric = trainer.callback_metrics.get(self.monitor)
        metric = self._to_float(metric)

        if metric is None:
            return

        self.last_score = metric
        self.last_metric_epoch = trainer.current_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        self.last_train_epoch = trainer.current_epoch

    def _last_is_worse_than_best(self, last_score, best_score):
        if self.mode == "min":
            return last_score > best_score

        return last_score < best_score

    def _make_filename(self, epoch, score):
        """
        Try to reuse the same filename template you use for the best checkpoint.
        Example template:
          LiSRB_CROHME_seed7_{epoch}-{val_ExpRate:.4f}
        """
        try:
            filename = self.filename_template.format(
                epoch=epoch,
                **{self.monitor: score},
            )
        except Exception:
            prefix = self.filename_template.split("{", 1)[0]
            safe_monitor = self.monitor.replace("/", "_")
            filename = f"{prefix}last_epoch={epoch}-{safe_monitor}={score:.4f}"

        if not filename.endswith(".ckpt"):
            filename = f"{filename}.ckpt"

        return filename

    def on_train_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        best_score = self._to_float(self.checkpoint_callback.best_model_score)

        if best_score is None:
            print("[last-ckpt] Skip: best checkpoint score is not available.")
            return

        if self.last_score is None:
            print(f"[last-ckpt] Skip: monitor metric '{self.monitor}' is not available.")
            return

        # This keeps the condition strict: only compare against the metric
        # from the actual final train epoch. If the final epoch was not validated,
        # we do not guess.
        if self.last_train_epoch is not None and self.last_metric_epoch != self.last_train_epoch:
            print(
                "[last-ckpt] Skip: final epoch has no validation metric. "
                f"last_metric_epoch={self.last_metric_epoch}, "
                f"last_train_epoch={self.last_train_epoch}"
            )
            return

        if not self._last_is_worse_than_best(self.last_score, best_score):
            print(
                "[last-ckpt] Skip: last checkpoint is not worse than best. "
                f"last_score={self.last_score:.6f}, best_score={best_score:.6f}"
            )
            return

        self.dirpath.mkdir(parents=True, exist_ok=True)

        epoch = self.last_metric_epoch
        filename = self._make_filename(epoch=epoch, score=self.last_score)
        ckpt_path = self.dirpath / filename

        trainer.save_checkpoint(str(ckpt_path))
        print(
            "[last-ckpt] Saved extra final checkpoint because last is worse than best:\n"
            f"  last_score = {self.last_score:.6f}\n"
            f"  best_score = {best_score:.6f}\n"
            f"  path       = {ckpt_path}"
        )


def _join_rclone_path(remote_dir: str, rel_path: str) -> str:
    remote_dir = remote_dir.rstrip("/")
    rel_path = rel_path.lstrip("/")
    if remote_dir.endswith(":"):
        return f"{remote_dir}{rel_path}"
    return f"{remote_dir}/{rel_path}"


def find_latest_remote_checkpoint(
    remote_dir: str,
    filename_prefix: str,
    local_dir: str = "checkpoints",
    recursive: bool = False,
):
    """
    Find latest checkpoint on rclone remote by epoch number and download it.

    Supports names like:
      LiSRB_CROHME_seed7_12-0.5678.ckpt
      LiSRB_CROHME_seed7_epoch=12-val_ExpRate=0.5678.ckpt
    """
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    cmd = ["rclone", "lsjson", remote_dir, "--files-only"]
    if recursive:
        cmd.append("-R")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[auto-resume] Could not list remote checkpoints: {e}")
        print(e.stderr)
        return None

    try:
        items = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as e:
        print(f"[auto-resume] Could not parse rclone lsjson output: {e}")
        return None

    # Match both:
    #   LiSRB_CROHME_seed7_12-0.1234.ckpt
    #   LiSRB_CROHME_seed7_epoch=12-val_ExpRate=0.1234.ckpt
    pattern = re.compile(
        rf"(^|/){re.escape(filename_prefix)}(?:epoch=)?(?P<epoch>\d+).*\.ckpt$"
    )

    candidates = []
    for item in items:
        rel_path = item.get("Path") or item.get("Name")
        if not rel_path:
            continue

        m = pattern.search(rel_path)
        if m is None:
            continue

        epoch = int(m.group("epoch"))
        candidates.append((epoch, rel_path))

    if not candidates:
        print(f"[auto-resume] No checkpoint matching prefix: {filename_prefix}")
        return None

    latest_epoch, latest_rel_path = max(candidates, key=lambda x: x[0])

    remote_ckpt = _join_rclone_path(remote_dir, latest_rel_path)
    local_ckpt = str(Path(local_dir) / Path(latest_rel_path).name)

    print(f"[auto-resume] Found latest remote checkpoint:")
    print(f"  epoch      = {latest_epoch}")
    print(f"  remote     = {remote_ckpt}")
    print(f"  local path = {local_ckpt}")

    try:
        subprocess.run(
            ["rclone", "copyto", remote_ckpt, local_ckpt, "--progress"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[auto-resume] Failed to download checkpoint: {e}")
        return None

    return local_ckpt

def get_ckpt_prefix_from_config(config):
    filename = config.trainer.callbacks[1].init_args.filename
    return filename.split("{", 1)[0]

def train(config):
    pl.seed_everything(config.seed_everything, workers=True)

    # Auto resume from GDrive/rclone if local resume path is not set
    if config.trainer.resume_from_checkpoint is None:
        latest_ckpt = find_latest_remote_checkpoint(
            remote_dir=config.trainer.get("resume_remote_dir", "purin_gdrive:"),
            filename_prefix=config.trainer.get(
                "resume_ckpt_prefix",
                get_ckpt_prefix_from_config(config),
                # f"LiSRB_CROHME_seed{config.seed_everything}_",
            ),
            local_dir=config.trainer.default_root_dir,
            recursive=config.trainer.get("resume_remote_recursive", False),
        )

        if latest_ckpt is not None:
            config.trainer.resume_from_checkpoint = latest_ckpt
            print(f"[auto-resume] Will resume from: {latest_ckpt}")
        else:
            print("[auto-resume] No remote checkpoint found. Training from scratch.")

    # Data
    data_module = CROHMEDatamodule(config=config)

    # Model
    from lit_comer import LitCoMER
    from utils.callbacks import (GradNormCallback)
    if config.trainer.resume_from_checkpoint is not None:
        print(f"Resuming full training state from: {config.trainer.resume_from_checkpoint}")
    else:
        print("Training from new weights")

    model_module = LitCoMER(
        config=config,
        beam_size=config.model.beam_size,
        max_len=config.model.max_len,
        alpha=config.model.alpha,
        early_stopping=config.model.early_stopping,
        temperature=config.model.temperature,
        vocab_info=data_module.vocab.get_info(),
    )

   # Logger
    logger = Logger(config.wandb.name, project=config.wandb.project, config=dict(config), log_model=False)
    if config.wandb.get("wandb_watch", False):
        logger.watch(model_module.comer_model, log=config.wandb.get("wandb_watch_log", "gradients"), log_freq=config.wandb.get("wandb_watch_log_freq", 1000))

   # Callback
    lr_callback = LearningRateMonitor(logging_interval=config.trainer.callbacks[0].init_args.logging_interval)

    ckpt_dir = config.trainer.default_root_dir
    remote_dir = config.trainer.get("resume_remote_dir", "purin_gdrive:")

    checkpoint_callback = ModelCheckpoint(
        save_top_k = config.trainer.callbacks[1].init_args.save_top_k,
        monitor    = config.trainer.callbacks[1].init_args.monitor,
        mode       = config.trainer.callbacks[1].init_args.mode,
        filename   = config.trainer.callbacks[1].init_args.filename,
        dirpath    = ckpt_dir,
    )

    conditional_last_callback = ConditionalLastCheckpointCallback(
        checkpoint_callback = checkpoint_callback,
        dirpath             = ckpt_dir,
        filename_template   = config.trainer.callbacks[1].init_args.filename,
        monitor             = config.trainer.callbacks[1].init_args.monitor,
        mode                = config.trainer.callbacks[1].init_args.mode,
    )

    rclone_callback = RcloneUploadCallback(
        local_dir           = ckpt_dir,
        remote_dir          = remote_dir,
        every_n_epochs      = config.trainer.get("rclone_every_n_epochs", 1),
        upload_on_train_end = True,
    )

    callback = [lr_callback, checkpoint_callback, conditional_last_callback, rclone_callback]

    callback.append(MoreValidationCallback())
    
    if config.trainer.get("log_grad_norm", False):
        grad_norm_callback = GradNormCallback()
        callback.append(grad_norm_callback)
    
    trainer = pl.Trainer(
        gpus                    = config.trainer.gpus,
        accelerator             = config.trainer.accelerator,
        val_check_interval      = config.trainer.val_check_interval,
        check_val_every_n_epoch = config.trainer.check_val_every_n_epoch,
        max_epochs              = config.trainer.max_epochs,
        logger                  = logger,
        deterministic           = config.trainer.deterministic,
        callbacks               = callback,
        # default_root_dir        = config.trainer.default_root_dir,
        resume_from_checkpoint  = config.trainer.resume_from_checkpoint,
        replace_sampler_ddp=False,
    )
    
    trainer.fit(model_module,data_module)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = Config(args.config)
    print(config.dumps())
    train(config)