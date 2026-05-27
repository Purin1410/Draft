import zipfile
from typing import List, Dict, Any, Optional

import pytorch_lightning as pl
import torch.optim as optim
from torch import FloatTensor, LongTensor
import torch

from datamodule.datamodule import CROHMEDatamodule
from datamodule.utils import Batch
from datamodule.vocab import VocabInfo

from models.tamer import TAMER
from utils.utils import (
    ExpRateRecorder,
    Hypothesis,
    ce_loss,
    to_struct_output_from_labels,
)


class LitTAMER(pl.LightningModule):
    def __init__(
        self,
        config: Dict[str, Any],
        beam_size: int = 10,
        max_len: int = 200,
        alpha: float = 1.0,
        early_stopping: bool = True,
        temperature: float = 1.0,
        vocab_info: VocabInfo = None,
    ):
        super().__init__()
        mcfg = config["model"]
        self.vocab_info = vocab_info
        # Ignore vocab_info in save_hyperparameters to avoid deep serialization issues
        self.save_hyperparameters(ignore=["vocab_info"])
        

        self.tamer_cfg = mcfg.get("tamer", {})
        self.struct_head_cfg = self.tamer_cfg.get("struct_head", {})
        self.struct_loss_cfg = self.tamer_cfg.get("struct_loss", {})
        self.tree_rescore_cfg = self.tamer_cfg.get("tree_rescore", {})
        self.tamer_debug_cfg = self.tamer_cfg.get("debug", {})
        self.struct_loss_enabled = bool(self.struct_loss_cfg.get("enabled", False))
        self.struct_loss_weight = float(self.struct_loss_cfg.get("weight", 1.0))
        self.struct_ignore_index = int(self.struct_loss_cfg.get("ignore_index", -1))
        self.log_illegal_struct_rate = bool(
            self.tamer_debug_cfg.get("log_illegal_struct_rate", False)
        )

        # model
        self.tamer_model = TAMER(config, vocab_info=vocab_info)
        #- -------------------------Optimizer config---------------------------------
        self.optimizer_cfg = mcfg.get("optimizer", {})
        self.optimizer_use = self.optimizer_cfg.get("use", "SGD")
        self.exprate_recorder = ExpRateRecorder(vocab_info)

        # -------------------------Scheduler config---------------------------------
        self.scheduler_cfg = mcfg.get("scheduler", {})
        self.scheduler_use = self.scheduler_cfg.get("use", "ReduceLROnPlateau")
        self.scheduler_interval = self.scheduler_cfg.get("interval", "epoch")
        self.scheduler_monitor = self.scheduler_cfg.get("monitor", "val_ExpRate")

        warmup_cfg = self.scheduler_cfg.get("warmup", {})
        self.warmup_enabled = warmup_cfg.get("enabled", True)
        self.warmup_interval = warmup_cfg.get("interval", "step")
        self.warmup_epochs = int(warmup_cfg.get("epochs", 1))
        self.warmup_steps = int(warmup_cfg.get("steps", 555))

        self._target_lrs = None
        self._warmup_total_steps = None
        self._warmup_finished = not self.warmup_enabled
        self._plateau_scheduler = None

    def forward(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        tgt: LongTensor,
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        return self.tamer_model(img, img_mask, tgt)

    def training_step(self, batch: Batch, _):
        output = self(batch.imgs, batch.mask, batch.tgt)

        seq_loss = ce_loss(output.logits, batch.out, ignore_idx=self.vocab_info.pad_id)
        struct_loss = self._struct_loss(output.struct_logits, batch)
        loss = seq_loss + self.struct_loss_weight * struct_loss
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/seq_loss", seq_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "train/struct_loss",
            struct_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self._log_illegal_struct_rate(batch, "train")

        return loss

    def validation_step(self, batch: Batch, _):
        output = self(batch.imgs, batch.mask, batch.tgt)

        seq_loss = ce_loss(output.logits, batch.out, ignore_idx=self.vocab_info.pad_id)
        struct_loss = self._struct_loss(output.struct_logits, batch)
        loss = seq_loss + self.struct_loss_weight * struct_loss
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val/seq_loss",
            seq_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "val/struct_loss",
            struct_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self._log_illegal_struct_rate(batch, "val")

        hyps = self.approximate_joint_search(batch.imgs, batch.mask)

        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
    
    def on_fit_start(self):
        self._init_warmup_if_needed()

    def on_train_batch_start(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0):
        self._apply_linear_warmup_if_needed()

    def on_validation_epoch_end(self):
        self._step_plateau_after_warmup()
    
    def on_train_epoch_start(self):
        sampler = None
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None:
            sampler = getattr(datamodule, "train_batch_sampler", None)

        if sampler is None:
            loaders = getattr(self.trainer, "train_dataloaders", None)
            if loaders is None:
                loaders = getattr(self.trainer, "train_dataloader", None)
            if loaders is not None and not isinstance(loaders, (list, tuple)):
                loaders = [loaders]
            if loaders:
                for loader in loaders:
                    candidate = getattr(loader, "batch_sampler", None)
                    if hasattr(candidate, "set_epoch"):
                        sampler = candidate
                        break

        if not hasattr(sampler, "set_epoch"):
            raise RuntimeError(
                "Could not find BucketedBatchSampler in on_train_epoch_start; "
                "epoch-dependent shuffling would be frozen."
            )

        sampler.set_epoch(int(self.current_epoch))
    def validation_epoch_end(self, *args, **kwargs):
        pass
    def training_step_end(self, *args, **kwargs):
        pass
    def validation_step_end(self, *args, **kwargs):
        pass

    def test_step(self, batch: Batch, _):
        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        return batch.img_bases, [self.vocab_info.words.indices2label(h.seq) for h in hyps]

    def test_epoch_end(self, test_outputs) -> None:
        exprate = self.exprate_recorder.compute()
        print(f"Validation ExpRate: {exprate}")

        with zipfile.ZipFile("result.zip", "w") as zip_f:
            for img_bases, preds in test_outputs:
                for img_base, pred in zip(img_bases, preds):
                    content = f"%{img_base}\n${pred}$".encode()
                    with zip_f.open(f"{img_base}.txt", "w") as f:
                        f.write(content)

    def _get_train_batches_per_epoch(self) -> int:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            raise RuntimeError("Trainer is not available, cannot estimate warmup steps.")
        num_batches = getattr(trainer, "num_training_batches", None)
        if isinstance(num_batches, int) and num_batches > 0:
            return num_batches
        if isinstance(num_batches, (list, tuple)):
            valid_batches = [int(x) for x in num_batches if isinstance(x, int) and x > 0]
            if valid_batches:
                return sum(valid_batches)
        loaders = getattr(trainer, "train_dataloaders", None)
        if loaders is None:
            loaders = getattr(trainer, "train_dataloader", None)
        if loaders is not None and not isinstance(loaders, (list, tuple)):
            loaders = [loaders]
        if loaders:
            lengths = []
            for loader in loaders:
                try:
                    lengths.append(len(loader))
                except TypeError:
                    pass
            if lengths:
                return sum(lengths)
        raise RuntimeError(
            "Cannot estimate number of train batches per epoch. "
            "If you use IterableDataset or dynamic dataloader length, set warmup.interval='step' "
            "and provide warmup.steps explicitly."
        )
    def _resolve_warmup_total_steps(self) -> int:
        if not self.warmup_enabled:
            return 0
        interval = str(self.warmup_interval).lower()
        if interval == "step":
            return max(int(self.warmup_steps), 0)
        if interval == "epoch":
            train_batches_per_epoch = self._get_train_batches_per_epoch()
            return max(int(self.warmup_epochs) * int(train_batches_per_epoch), 0)
        raise ValueError(f"Unknown warmup interval: {self.warmup_interval}. Use 'step' or 'epoch'.")
        
    def _set_optimizer_lrs(self, optimizer, lrs: List[float]) -> None:
        for param_group, lr in zip(optimizer.param_groups, lrs):
            param_group["lr"] = float(lr)
            
    def _get_optimizer(self):
        if self.trainer is None or len(self.trainer.optimizers) == 0:
            return None
        return self.trainer.optimizers[0]

    def _init_warmup_if_needed(self) -> None:
        if self._warmup_total_steps is not None:
            return
        optimizer = self._get_optimizer()
        if optimizer is None:
            return
        if self._target_lrs is None:
            self._target_lrs = [float(pg["lr"]) for pg in optimizer.param_groups]
        self._warmup_total_steps = self._resolve_warmup_total_steps()
        self._warmup_finished = (not self.warmup_enabled) or self._warmup_total_steps <= 0
        if self._warmup_finished:
            self._set_optimizer_lrs(optimizer, self._target_lrs)
        else:
            self._set_optimizer_lrs(optimizer, [0.0 for _ in self._target_lrs])
            
    def _apply_linear_warmup_if_needed(self) -> None:
        if not self.warmup_enabled:
            return
        self._init_warmup_if_needed()
        if self._warmup_finished:
            return
        optimizer = self._get_optimizer()
        if optimizer is None:
            return
        if self._warmup_total_steps is None or self._warmup_total_steps <= 0:
            self._warmup_finished = True
            self._set_optimizer_lrs(optimizer, self._target_lrs)
            return
        current_step = int(self.global_step)
        warmup_factor = min(float(current_step) / float(self._warmup_total_steps), 1.0)
        new_lrs = [target_lr * warmup_factor for target_lr in self._target_lrs]
        self._set_optimizer_lrs(optimizer, new_lrs)
        if current_step >= self._warmup_total_steps:
            self._warmup_finished = True
            self._set_optimizer_lrs(optimizer, self._target_lrs)

    def _get_monitor_value(self):
        metrics = getattr(self.trainer, "callback_metrics", {})
        monitor_value = metrics.get(self.scheduler_monitor)
        if monitor_value is None and self.scheduler_monitor == "val_ExpRate":
            try:
                monitor_value = self.exprate_recorder.compute()
            except Exception:
                monitor_value = None
        return monitor_value
    def _step_plateau_after_warmup(self) -> None:
        if self._plateau_scheduler is None:
            return
        self._init_warmup_if_needed()
        if not self._warmup_finished:
            return
        monitor_value = self._get_monitor_value()
        if monitor_value is None:
            raise RuntimeError(
                f"ReduceLROnPlateau expected monitor '{self.scheduler_monitor}', "
                "but it was not found in trainer.callback_metrics. "
                "Check self.log(...) in validation_step or change scheduler.monitor."
            )
        if isinstance(monitor_value, torch.Tensor):
            monitor_value = monitor_value.detach()
            if monitor_value.numel() == 1:
                monitor_value = monitor_value.item()
        old_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self._plateau_scheduler.step(monitor_value)
        new_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("plateau_monitor", float(monitor_value), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("lr_after_plateau", float(new_lr), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["lit_tamer_warmup_state"] = {
            "target_lrs": self._target_lrs,
            "warmup_total_steps": self._warmup_total_steps,
            "warmup_finished": self._warmup_finished,
            "plateau_scheduler": (
                self._plateau_scheduler.state_dict()
                if self._plateau_scheduler is not None
                else None
            ),
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        state = checkpoint.get("lit_tamer_warmup_state", {})
        self._target_lrs = state.get("target_lrs", self._target_lrs)
        self._warmup_total_steps = state.get("warmup_total_steps", self._warmup_total_steps)
        self._warmup_finished = state.get("warmup_finished", self._warmup_finished)
        self._loaded_plateau_scheduler_state = state.get("plateau_scheduler", None)

    def approximate_joint_search(
        self, 
        img: FloatTensor, 
        mask: LongTensor,
    ) -> List[Hypothesis]:
        return self.tamer_model.beam_search(
            img,
            mask,
            **self.hparams,
            current_epoch=int(self.current_epoch),
        )

    def _get_struct_targets(self, batch: Batch):
        if batch.struct_out is not None:
            return batch.struct_out, batch.struct_illegal
        if batch.labels is None or batch.lengths is None:
            return None, None
        return to_struct_output_from_labels(
            labels=batch.labels,
            lengths=batch.lengths,
            device=batch.labels.device,
            vocab_info=self.vocab_info,
            ignore_index=self.struct_ignore_index,
        )

    def _struct_loss(self, struct_logits, batch: Batch) -> torch.Tensor:
        if not self.struct_loss_enabled:
            return torch.zeros((), dtype=torch.float, device=self.device)
        if struct_logits is None:
            raise RuntimeError(
                "model.tamer.struct_loss.enabled=true requires "
                "model.tamer.struct_head.enabled=true"
            )
        struct_out, _ = self._get_struct_targets(batch)
        if struct_out is None:
            raise RuntimeError("Structural loss is enabled but no structural targets exist.")
        return ce_loss(
            struct_logits,
            struct_out,
            ignore_idx=self.struct_ignore_index,
        )

    def _log_illegal_struct_rate(self, batch: Batch, stage: str) -> None:
        if not self.log_illegal_struct_rate:
            return
        _, struct_illegal = self._get_struct_targets(batch)
        if struct_illegal is None:
            return
        rate = struct_illegal.float().mean()
        self.log(
            f"{stage}/illegal_struct_rate",
            rate,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
