import pytorch_lightning as pl
from datamodule import CROHMEDatamodule
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    Callback
)
from pytorch_lightning.loggers import WandbLogger as Logger
import argparse
from sconf import Config
from pathlib import Path
from utils.remote_sync import (
    build_remote_run_dir,
    collect_uploadable_files,
    ensure_remote_dir,
    find_and_download_latest_checkpoint,
    log_wandb_files,
    maybe_cleanup_uploaded_files,
    upload_files,
)
from utils.run_identity import apply_runtime_overrides

def _lightning_barrier(trainer, name=None):
    # Lightning >= 1.5 / 2.x
    strategy = getattr(trainer, "strategy", None)
    if strategy is not None and hasattr(strategy, "barrier"):
        return strategy.barrier(name)

    # Lightning 1.x old APIs
    plugin = getattr(trainer, "training_type_plugin", None)
    if plugin is None:
        accelerator = getattr(trainer, "accelerator", None)
        plugin = getattr(accelerator, "training_type_plugin", None)

    if plugin is not None and hasattr(plugin, "barrier"):
        return plugin.barrier(name)

    # Final fallback
    import torch
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available() and dist.get_backend() == "nccl":
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()

class MoreValidationCallback(pl.Callback):
    def __init__(self, monitor="val_ExpRate"):
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer, pl_module):
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is not None:
            if metric > 0.57:
                trainer.check_val_every_n_epoch = 1

class RcloneUploadCallback(Callback):
    """Upload completed checkpoints and merged analysis logs only."""

    def __init__(
        self,
        checkpoint_dir,
        analysis_dir,
        remote_run_dir,
        rclone_cfg=None,
        wandb_cfg=None,
    ):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.analysis_dir = Path(analysis_dir)
        self.remote_run_dir = remote_run_dir
        self.rclone_cfg = rclone_cfg or {}
        self.wandb_cfg = wandb_cfg or {}
        self.every_n_epochs = _cfg_get(self.rclone_cfg, "every_n_epochs", 1)
        self.upload_on_train_end = _cfg_get(self.rclone_cfg, "upload_on_train_end", True)
        self._last_upload_epoch = None

    def on_train_epoch_end(self, trainer, pl_module):
        if self.every_n_epochs is None:
            return
        epoch_num = int(trainer.current_epoch) + 1
        if epoch_num % int(self.every_n_epochs) == 0:
            self._merge_and_upload(trainer, pl_module)
            self._last_upload_epoch = int(trainer.current_epoch)

    def on_train_end(self, trainer, pl_module):
        if not self.upload_on_train_end:
            return
        current_epoch = int(getattr(trainer, "current_epoch", 0))
        if self._last_upload_epoch == current_epoch:
            return
        self._merge_and_upload(trainer, pl_module)
        self._last_upload_epoch = current_epoch

    def _merge_and_upload(self, trainer, pl_module):
        from utils.analysis_logging import (
            flush_all_buffers,
            get_dist_info,
            maybe_merge_shards,
            resolve_analysis_run_id,
        )

        # local process buffers are per-rank; every rank must flush its own buffers
        flush_all_buffers()

        cfg = getattr(pl_module, "analysis_logging_cfg", None) or {}
        rank, world_size = get_dist_info()

        if cfg.get("enabled", False) and cfg.get("merge_on_epoch_end", False):
            run_id = resolve_analysis_run_id(cfg, "CoMER", pl_module.config.get("seed_everything", ""))
            seeds = str(cfg.get("seeds") or pl_module.config.get("seed_everything", ""))
            epoch = int(trainer.current_epoch)
            # IMPORTANT: all ranks call this; rank 0 merges, others wait/return inside function.
            maybe_merge_shards(cfg, run_id, seeds, epoch, "train", rank)

        # Keep non-zero ranks parked until rank 0 upload finishes, so all ranks leave callback together.
        if rank == 0:
            self._upload_completed(trainer)

        if world_size > 1:
            # trainer.strategy.barrier("analysis_upload_done")
            _lightning_barrier(trainer, "analysis_upload_done")

    
    def _upload_completed(self, trainer):
        if not trainer.is_global_zero:
            return
        pl_module = trainer.lightning_module
        run_name = None
        if pl_module is not None:
            cfg = getattr(pl_module, "analysis_logging_cfg", None) or {}
            from utils.analysis_logging import resolve_analysis_run_id
            run_name = resolve_analysis_run_id(cfg, "CoMER", pl_module.config.get("seed_everything", ""))
        
        if not run_name:
            run_name = _cfg_get(self.wandb_cfg, "name", None)

        paths = []
        if _cfg_get(self.rclone_cfg, "upload_checkpoints", True):
            paths.append(self.checkpoint_dir)
        if _cfg_get(self.rclone_cfg, "upload_analysis_logs", True):
            paths.append(self.analysis_dir)
        if not paths:
            return
        files = collect_uploadable_files(paths, run_name=run_name)
        if not files:
            return

        rclone_ok = True
        if _cfg_get(self.rclone_cfg, "enabled", True):
            rclone_ok = upload_files(
                files,
                remote_run_dir=self.remote_run_dir,
                rclone_command=_cfg_get(self.rclone_cfg, "command", "rclone"),
                copy_flags=_cfg_get(self.rclone_cfg, "copy_flags", ["--update", "--verbose", "--no-traverse"]),
                fail_on_error=_cfg_get(self.rclone_cfg, "fail_on_error", False),
                run_name=run_name,
            )

        wandb_ok = True
        if _cfg_get(self.wandb_cfg, "upload_artifacts", True):
            wandb_run = getattr(getattr(trainer, "logger", None), "experiment", None)
            wandb_ok = log_wandb_files(
                wandb_run,
                files,
                fail_on_error=_cfg_get(self.wandb_cfg, "fail_on_error", False),
                run_name=run_name,
            )

        maybe_cleanup_uploaded_files(
            files,
            rclone_ok=rclone_ok,
            wandb_ok=wandb_ok,
            cleanup=_cfg_get(self.wandb_cfg, "artifact_cleanup_local", False),
            keep_last_local_checkpoints=_cfg_get(self.wandb_cfg, "keep_last_local_checkpoints", 1),
            run_name=run_name,
        )


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


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)

def train(config):
    pl.seed_everything(config.seed_everything, workers=True)
    run_name = config.analysis_logging.run_id
    rclone_cfg = _cfg_get(config, "rclone", {})
    remote_run_dir = build_remote_run_dir(rclone_cfg, run_name)

    # Auto resume from GDrive/rclone if local resume path is not set
    if _cfg_get(rclone_cfg, "enabled", True) and _cfg_get(rclone_cfg, "resume", True) and config.trainer.resume_from_checkpoint is None:
        latest_ckpt = find_and_download_latest_checkpoint(
            remote_run_dir=remote_run_dir,
            run_name=run_name,
            local_dir=config.trainer.default_root_dir,
            rclone_command=_cfg_get(rclone_cfg, "command", "rclone"),
            recursive=_cfg_get(rclone_cfg, "recursive_list", True),
            download_flags=_cfg_get(rclone_cfg, "download_flags", ["--progress"]),
            fail_on_error=_cfg_get(rclone_cfg, "fail_on_error", False),
        )

        if latest_ckpt is not None:
            config.trainer.resume_from_checkpoint = latest_ckpt
            print(f"[auto-resume] Will resume from: {latest_ckpt}")
        else:
            ensure_remote_dir(
                remote_run_dir,
                rclone_command=_cfg_get(rclone_cfg, "command", "rclone"),
                fail_on_error=_cfg_get(rclone_cfg, "fail_on_error", False),
            )
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
    analysis_dir = Path(config.analysis_logging.log_dir) / str(run_name)

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
        checkpoint_dir      = ckpt_dir,
        analysis_dir        = analysis_dir,
        remote_run_dir      = remote_run_dir,
        rclone_cfg          = rclone_cfg,
        wandb_cfg           = _cfg_get(config, "wandb", {}),
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


def _non_empty_arg(value: str) -> str:
    value = str(value).strip()
    if not value:
        raise argparse.ArgumentTypeError("value cannot be empty")
    return value


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=_non_empty_arg, required=True)
    parser.add_argument("--model_name", type=_non_empty_arg, required=True)
    parser.add_argument("--datasets", type=_non_empty_arg, required=True)
    parser.add_argument("--seeds", type=_non_empty_arg, required=True)
    parser.add_argument("--run_type", type=_non_empty_arg, choices=["baseline", "ablation"], required=True)
    parser.add_argument("--ablation_type", type=_non_empty_arg, required=False, default=None)
    args = parser.parse_args()
    config = Config(args.config)
    apply_runtime_overrides(
        config,
        model_name=args.model_name,
        datasets=args.datasets,
        seeds=args.seeds,
        run_type=args.run_type,
        ablation_type=args.ablation_type,
    )
    print(config.dumps())
    train(config)
