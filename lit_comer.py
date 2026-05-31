import zipfile
from typing import List, Dict, Any, Optional

import pytorch_lightning as pl
import torch.optim as optim
from torch import FloatTensor, LongTensor
import torch

from datamodule.datamodule import CROHMEDatamodule
from datamodule.utils import Batch
from datamodule.vocab import VocabInfo

from models.comer import CoMER
from utils.utils import (ExpRateRecorder, Hypothesis, ce_loss, to_tgt_output)


class LitCoMER(pl.LightningModule):
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
        from utils.analysis_logging import get_analysis_logging_cfg
        self.analysis_logging_cfg = get_analysis_logging_cfg(config)
        self.config = config
        # Ignore vocab_info in save_hyperparameters to avoid deep serialization issues
        self.save_hyperparameters(ignore=["vocab_info"])
        

        # model
        self.comer_model = CoMER(config, vocab_info=vocab_info)
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
        return_aux: bool = False,
        capture_embed: bool = False,
        capture_cross_attn: bool = False,
        capture_self_attn: bool = False,
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
        return self.comer_model(
            img,
            img_mask,
            tgt,
            return_aux=return_aux,
            capture_embed=capture_embed,
            capture_cross_attn=capture_cross_attn,
            capture_self_attn=capture_self_attn,
        )

    def training_step(self, batch: Batch, batch_idx):
        out_hat = self(batch.imgs, batch.mask, batch.tgt)

        loss = ce_loss(out_hat, batch.out, ignore_idx=self.vocab_info.pad_id)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        cfg = getattr(self, "analysis_logging_cfg", None) or {}
        if cfg.get("enabled", False) and cfg.get("grad_norm_sample", False):
            raise NotImplementedError(
                "analysis_logging.grad_norm_sample is off by default and is not safely implemented for LitCoMER."
            )

        self._analysis_log_train_batch(batch, out_hat, batch_idx)

        return loss

    def validation_step(self, batch: Batch, _):
        out_hat = self(batch.imgs, batch.mask, batch.tgt)

        loss = ce_loss(out_hat, batch.out, ignore_idx=self.vocab_info.pad_id)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        cfg = getattr(self, "analysis_logging_cfg", None) or {}
        from utils.analysis_logging import should_log_topk
        topk_enabled = should_log_topk(cfg, "val", trainer=self._analysis_trainer_or_none())

        beam_results = self.approximate_joint_search(
            batch.imgs, batch.mask, return_nbest=topk_enabled
        )

        if topk_enabled and beam_results and hasattr(beam_results[0], 'candidates'):
            hyps = [r.best for r in beam_results]
            nbest_outputs = beam_results
        else:
            hyps = beam_results
            nbest_outputs = None

        self._analysis_log_batch("val", batch, out_hat, hyps, nbest_outputs=nbest_outputs)
        self._maybe_log_analysis_aux(batch, "val")

        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def _analysis_trainer_or_none(self):
        try:
            return self.trainer
        except RuntimeError:
            return None

    def _analysis_epoch(self):
        trainer = self._analysis_trainer_or_none()
        return int(getattr(trainer, "current_epoch", getattr(self, "current_epoch", 0)))

    def _reserve_analysis_sample_indices(self, cfg, phase: str, batch: Batch, batch_idx=None):
        from utils.analysis_logging import AnalysisSampleLimiter, should_log_batch

        if not should_log_batch(cfg, phase, self._analysis_trainer_or_none(), batch_idx=batch_idx):
            return []
        if not hasattr(self, "_analysis_sample_limiter"):
            self._analysis_sample_limiter = AnalysisSampleLimiter()
        count = self._analysis_sample_limiter.reserve(
            cfg, phase, self._analysis_epoch(), len(batch.img_bases)
        )
        return list(range(count))

    def _reserve_analysis_aux_sample_indices(self, cfg, phase: str, batch: Batch):
        from utils.analysis_logging import AnalysisSampleLimiter

        if not hasattr(self, "_analysis_aux_limiter"):
            self._analysis_aux_limiter = AnalysisSampleLimiter()
        count = self._analysis_aux_limiter.reserve(
            cfg,
            phase,
            self._analysis_epoch(),
            len(batch.img_bases),
            limit_key="capture_max_samples_per_epoch",
        )
        return list(range(count))

    def _split_analysis_decode_results(self, results, topk_enabled: bool):
        if topk_enabled and results and hasattr(results[0], "candidates"):
            return [r.best for r in results], results
        return results, None

    def _analysis_decode_no_side_effect(self, img, mask, return_nbest: bool = False):
        was_training = self.training
        try:
            self.eval()
            with torch.no_grad():
                return self.approximate_joint_search(img, mask, return_nbest=return_nbest)
        finally:
            self.train(was_training)

    def _analysis_log_train_batch(self, batch: Batch, logits_for_logging, batch_idx):
        trainer = self._analysis_trainer_or_none()
        if trainer is not None and getattr(trainer, "profiler", None) is not None:
            with trainer.profiler.profile("analysis_log_train_batch"):
                self._analysis_log_train_batch_core(batch, logits_for_logging, batch_idx)
        else:
            self._analysis_log_train_batch_core(batch, logits_for_logging, batch_idx)

    def _analysis_log_train_batch_core(self, batch: Batch, logits_for_logging, batch_idx):
        cfg = getattr(self, "analysis_logging_cfg", None) or {}
        sample_indices = self._reserve_analysis_sample_indices(cfg, "train", batch, batch_idx=batch_idx)
        if not sample_indices:
            return

        from utils.analysis_logging import should_decode_for_logging, should_log_topk

        hyps = None
        nbest_outputs = None
        if should_decode_for_logging(cfg, "train"):
            topk_enabled = should_log_topk(cfg, "train", trainer=self._analysis_trainer_or_none())
            take = len(sample_indices)
            decoded = self._analysis_decode_no_side_effect(
                batch.imgs[:take], batch.mask[:take], return_nbest=topk_enabled
            )
            hyps, nbest_outputs = self._split_analysis_decode_results(decoded, topk_enabled)

        self._analysis_log_batch(
            "train",
            batch,
            logits_for_logging,
            hyps,
            nbest_outputs=nbest_outputs,
            sample_indices=sample_indices,
            batch_idx=batch_idx,
        )

    def _analysis_log_batch(self, phase: str, batch: Batch, logits_for_logging, hyps: List[Hypothesis],
                            nbest_outputs=None, sample_indices=None, batch_idx=None):
        cfg = getattr(self, "analysis_logging_cfg", None)
        if cfg is None or not cfg.get("enabled", False):
            return
        trainer = self._analysis_trainer_or_none()
        if cfg.get("skip_sanity_check", True) and getattr(trainer, "sanity_checking", False):
            return

        from utils.analysis_logging import (
            should_log_phase,
            resolve_analysis_run_id,
            get_dist_info,
            get_phase_cfg,
            valid_hw_from_mask,
            ids_to_label,
            canonicalize_ids,
            select_l2r,
            build_teacher_forced_payloads,
            make_meta_id,
            build_log_paths,
            append_csv_rows_buffered,
            append_jsonl_rows_buffered,
            serialize_topk_preds,
            compute_rank_gt_in_beam,
            should_log_csv,
            should_log_detail,
            should_log_topk,
        )

        if not should_log_phase(cfg, phase):
            return
        if sample_indices is None:
            sample_indices = self._reserve_analysis_sample_indices(cfg, phase, batch, batch_idx=batch_idx)
        else:
            sample_indices = list(sample_indices)
        if not sample_indices:
            return

        # 1. Dist info
        rank, world_size = get_dist_info()

        # 2. run_id resolution
        run_id = resolve_analysis_run_id(cfg, "CoMER", self.config.get("seed_everything", ""))

        epoch = int(self.current_epoch)
        global_step = int(self.global_step)
        seed = self.config.get("seed_everything", "")
        seeds = str(cfg.get("seeds") or seed)

        # 3. Handle bidirectional tensors
        batch_size = len(batch.img_bases)
        l_logits, l_targets = select_l2r(logits_for_logging, getattr(batch, "fusion_out", getattr(batch, "out", None)), batch_size)

        phase_cfg = get_phase_cfg(cfg, phase)
        nbest_k = phase_cfg.get("nbest_k", 10)
        topk_enabled = should_log_topk(cfg, phase, trainer=trainer)
        has_decode = hyps is not None

        # 4. CSV scalar rows
        csv_rows = []
        if should_log_csv(cfg, phase, has_decode=has_decode):
            for j, i in enumerate(sample_indices):
                sample_id = batch.img_bases[i]

                # Ground truth
                gt_ids = batch.indices[i]
                gt_label = ids_to_label(gt_ids, self.vocab_info)

                pred_ids = []
                pred_label = ""
                pred_score = ""
                if has_decode:
                    hyp_idx = j if len(hyps) == len(sample_indices) else i
                    pred_ids = hyps[hyp_idx].seq
                    pred_label = ids_to_label(pred_ids, self.vocab_info)
                    pred_score = getattr(hyps[hyp_idx], "score", "")

                # Exact match
                exact_match = (
                    canonicalize_ids(pred_ids, self.vocab_info)
                    == canonicalize_ids(gt_ids, self.vocab_info)
                    if has_decode else ""
                )

                # rank_gt_in_beam from actual beam candidates
                rank_gt = ""
                if topk_enabled and nbest_outputs is not None:
                    out_idx = j if len(nbest_outputs) == len(sample_indices) else i
                    if out_idx < len(nbest_outputs):
                        rank_gt = compute_rank_gt_in_beam(
                            gt_ids, nbest_outputs[out_idx].candidates, self.vocab_info
                        )

                # input_h, input_w from mask[i]
                input_h, input_w = valid_hw_from_mask(batch.mask[i])
                meta_id = make_meta_id(
                    rank=rank,
                    global_step=global_step,
                    batch_idx=batch_idx,
                    sample_index=i,
                    sample_id=sample_id,
                )

                csv_rows.append({
                    "meta_id": meta_id,
                    "global_step": global_step,
                    "sample_id": sample_id,
                    "input_h": input_h,
                    "input_w": input_w,
                    "gt": gt_label,
                    "pred": pred_label,
                    "pred_score": pred_score,
                    "rank_gt_in_beam": rank_gt,
                    "exact_match": exact_match
                })

        # 5. JSONL token details
        jsonl_rows = []
        if should_log_detail(cfg, phase):
            log_token_detail = bool(phase_cfg.get("token_detail", False))
            log_teacher_forced_top1 = bool(phase_cfg.get("teacher_forced_top1", False))
            if log_token_detail or log_teacher_forced_top1 or (topk_enabled and nbest_outputs is not None):
                token_topk = phase_cfg.get("token_topk", 5)
                token_details = [[] for _ in range(batch_size)]
                teacher_forced_top1s = [[] for _ in range(batch_size)]
                
                if (log_token_detail or log_teacher_forced_top1) and l_logits is not None and l_targets is not None:
                    token_details, teacher_forced_top1s = build_teacher_forced_payloads(
                        l_logits, l_targets, self.vocab_info, topk=token_topk
                    )

                for j, i in enumerate(sample_indices):
                    sample_id = batch.img_bases[i]
                    meta_id = make_meta_id(
                        rank=rank,
                        global_step=global_step,
                        batch_idx=batch_idx,
                        sample_index=i,
                        sample_id=sample_id,
                    )
                    row = {
                        "meta_id": meta_id,
                    }
                    if log_token_detail:
                        row["token_detail"] = token_details[i]
                    if log_teacher_forced_top1:
                        row["teacher_forced_top1"] = teacher_forced_top1s[i]
                    # topk_preds from actual beam candidates
                    if topk_enabled and nbest_outputs is not None:
                        out_idx = j if len(nbest_outputs) == len(sample_indices) else i
                        if out_idx < len(nbest_outputs):
                            row["topk_preds"] = serialize_topk_preds(
                                nbest_outputs[out_idx].candidates, self.vocab_info, nbest_k=nbest_k
                            )
                            row["rank_gt_in_beam"] = compute_rank_gt_in_beam(
                                batch.indices[i], nbest_outputs[out_idx].candidates, self.vocab_info
                            )
                    jsonl_rows.append(row)

        # 6. Build paths and write
        log_dir = phase_cfg.get("log_dir", "analysis_logs")
        csv_path, jsonl_path = build_log_paths(log_dir, run_id, seeds, epoch, rank, phase=phase)

        if len(csv_rows) > 0:
            append_csv_rows_buffered(csv_path, csv_rows, include_decode_fields=has_decode, cfg=cfg)
        if len(jsonl_rows) > 0:
            append_jsonl_rows_buffered(jsonl_path, jsonl_rows, include_topk_preds=topk_enabled and nbest_outputs is not None, cfg=cfg)

    def _maybe_log_analysis_aux(self, batch, phase: str):
        cfg = getattr(self, "analysis_logging_cfg", None) or {}
        trainer = self._analysis_trainer_or_none()
        if cfg.get("skip_sanity_check", True) and getattr(trainer, "sanity_checking", False):
            return
        from utils.analysis_logging import (
            should_log_phase, should_capture_aux, resolve_analysis_run_id,
            get_phase_cfg,
            mean_pool_embed, reshape_cross_attn,
            attention_entropy, get_dist_info, build_log_paths, append_jsonl_rows_buffered,
            serialize_self_attn_for_sample, make_meta_id,
        )
        if not should_log_phase(cfg, phase):
            return
        epoch = getattr(trainer, "current_epoch", 0)
        if not should_capture_aux(cfg, phase, epoch, trainer=trainer):
            return
        sample_indices = self._reserve_analysis_aux_sample_indices(cfg, phase, batch)
        if not sample_indices:
            return
        phase_cfg = get_phase_cfg(cfg, phase)
        capture_embed = bool(phase_cfg.get("capture_embed", False))
        capture_cross_attn = bool(phase_cfg.get("capture_cross_attn", False))
        capture_self_attn = bool(phase_cfg.get("capture_self_attn", False))
            
        with torch.no_grad():
            if hasattr(batch, "pos_tgt"):
                out = self.forward(
                    batch.imgs, batch.mask, batch.tgt, batch.pos_tgt,
                    return_aux=True,
                    capture_embed=capture_embed,
                    capture_cross_attn=capture_cross_attn,
                    capture_self_attn=capture_self_attn,
                )
            elif hasattr(batch, "exp_tgt"):
                out = self.forward(
                    batch.imgs, batch.mask, batch.exp_tgt,
                    return_aux=True,
                    capture_embed=capture_embed,
                    capture_cross_attn=capture_cross_attn,
                    capture_self_attn=capture_self_attn,
                )
            else:
                out = self.forward(
                    batch.imgs, batch.mask, batch.tgt,
                    return_aux=True,
                    capture_embed=capture_embed,
                    capture_cross_attn=capture_cross_attn,
                    capture_self_attn=capture_self_attn,
                )
            
        logits, aux = out
        rank, _ = get_dist_info()
        run_id = resolve_analysis_run_id(cfg, "CoMER", self.config.get("seed_everything", ""))
        seeds = str(cfg.get("seeds") or self.config.get("seed_everything", ""))
        log_dir = phase_cfg.get("log_dir", "analysis_logs")
        epoch = getattr(trainer, "current_epoch", 0)
        global_step = getattr(trainer, "global_step", 0)
        batch_size = len(batch.img_bases)
        
        embeds = None
        if capture_embed and aux.get("embed_seq") is not None:
            embeds = mean_pool_embed(aux["embed_seq"], getattr(batch, "fusion_out", getattr(batch, "out", None)), self.vocab_info.pad_id, batch_size)
            
        cross_attns = None
        store_full_attn = bool(phase_cfg.get("full_cross_attn_map", False) or phase_cfg.get("cross_attn", {}).get("store") == "full")
        if capture_cross_attn and aux.get("cross_attn") is not None and len(aux["cross_attn"]) > 0:
            attn_maps = aux["cross_attn"]
            h_out = aux["height"]
            
            model = getattr(self, "comer_model", getattr(self, "tamer_model", getattr(self, "posformer_model", getattr(self, "ical_model", None))))
            heads = model.decoder.model.layers[0].multihead_attn.num_heads
            
            last_attn = attn_maps[-1]
            reshaped = reshape_cross_attn(last_attn, batch.imgs.size(0) * 2, heads, h_out)
            
            if reshaped is not None:
                reshaped = reshaped[:batch_size]
                attn_probs = last_attn.view(batch.imgs.size(0) * 2, heads, last_attn.size(1), -1)[:batch_size]
                entropy = attention_entropy(attn_probs, dim=-1)
                mean_entropy = entropy.mean(dim=1)
                
                max_tokens = phase_cfg.get("cross_attn", {}).get("max_tokens", 64)
                max_heads = phase_cfg.get("cross_attn", {}).get("max_heads", heads)
                mean_entropy = mean_entropy[:, :max_tokens]
                
                cross_attns = []
                for i in range(batch_size):
                    t_len = int((getattr(batch, "fusion_out", getattr(batch, "out", None))[i] != self.vocab_info.pad_id).sum().item())
                    t_len = min(t_len, max_tokens)
                    if store_full_attn:
                        head_count = min(int(max_heads), heads)
                        cross_attns.append({
                            "layers": {"last": reshaped[i, :head_count, :t_len].detach().cpu().tolist()},
                            "shape": {"tokens": t_len, "height": h_out, "width": reshaped.size(4), "heads": head_count},
                            "stored": "full"
                        })
                    else:
                        valid_entropy = mean_entropy[i, :t_len].detach().cpu().tolist()
                        cross_attns.append({
                            "layers": {"last": {"entropy": valid_entropy}},
                            "shape": {"tokens": t_len, "height": h_out, "width": reshaped.size(4), "heads": heads},
                            "stored": "summary"
                        })

        self_attns = None
        store_full_self_attn = bool(
            phase_cfg.get("full_self_attn_map", False)
            or phase_cfg.get("self_attn", {}).get("store") == "full"
        )
        if capture_self_attn and aux.get("self_attn") is not None and len(aux["self_attn"]) > 0:
            model = getattr(self, "comer_model", getattr(self, "tamer_model", getattr(self, "posformer_model", getattr(self, "ical_model", None))))
            layer0 = model.decoder.model.layers[0]
            self_attn_mod = getattr(layer0, "self_attn", None)
            heads = getattr(self_attn_mod, "num_heads", layer0.multihead_attn.num_heads)
            self_cfg = dict(phase_cfg.get("self_attn", {}))
            if store_full_self_attn:
                self_cfg["store"] = "full"
            tgt_for_self = getattr(batch, "exp_tgt", getattr(batch, "tgt", None))
            out_for_self = getattr(batch, "fusion_out", getattr(batch, "out", None))
            self_attns = []
            for i in range(batch_size):
                self_attns.append(
                    serialize_self_attn_for_sample(
                        aux["self_attn"],
                        sample_index=i,
                        tgt=tgt_for_self,
                        out=out_for_self,
                        vocab_info=self.vocab_info,
                        heads=heads,
                        cfg=self_cfg,
                    )
                )

        jsonl_rows = []
        if embeds is not None or cross_attns is not None or self_attns is not None:
            for i in sample_indices:
                if embeds is None and cross_attns is None and (self_attns is None or self_attns[i] is None):
                    continue
                row = {
                    "meta_id": make_meta_id(
                        rank=rank,
                        global_step=global_step,
                        batch_idx=None,
                        sample_index=i,
                        sample_id=batch.img_bases[i],
                    ),
                    "run_id": run_id,
                    "record_type": "aux",
                    "schema_version": "hmer-analysis-v1",
                    "repo": "Baseline_CoMER",
                    "phase": phase,
                    "epoch": epoch,
                    "global_step": global_step,
                    "rank": rank,
                    "sample_id": batch.img_bases[i],
                }
                if embeds is not None:
                    row["embed"] = embeds[i]
                if cross_attns is not None:
                    if store_full_attn:
                        row["cross_attn_map"] = cross_attns[i]
                    else:
                        row["cross_attn_summary"] = cross_attns[i]
                if self_attns is not None and self_attns[i] is not None:
                    if store_full_self_attn:
                        row["self_attn_map"] = self_attns[i]
                    else:
                        row["self_attn_summary"] = self_attns[i]
                jsonl_rows.append(row)
            
        if jsonl_rows:
            _, jsonl_path = build_log_paths(log_dir, run_id, seeds, epoch, rank, phase=phase)
            append_jsonl_rows_buffered(jsonl_path, jsonl_rows, cfg=cfg)

    def on_fit_start(self):
        self._init_warmup_if_needed()

    def on_train_batch_start(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0):
        self._apply_linear_warmup_if_needed()

    def on_validation_epoch_end(self):
        self._step_plateau_after_warmup()
        cfg = getattr(self, "analysis_logging_cfg", None) or {}
        if cfg.get("enabled", False) and cfg.get("merge_on_epoch_end", False):
            from utils.analysis_logging import maybe_merge_shards, resolve_analysis_run_id, get_dist_info
            run_id = resolve_analysis_run_id(cfg, "CoMER", self.config.get("seed_everything", ""))
            seeds = str(cfg.get("seeds") or self.config.get("seed_everything", ""))
            epoch = int(self.current_epoch)
            rank, _ = get_dist_info()
            maybe_merge_shards(cfg, run_id, seeds, epoch, "val", rank)
    
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
    
    def on_train_epoch_end(self):
        from utils.analysis_logging import flush_all_buffers
        flush_all_buffers()

    def on_train_end(self):
        from utils.analysis_logging import flush_all_buffers
        flush_all_buffers()
        

    def validation_epoch_end(self, *args, **kwargs):
        pass
    def training_step_end(self, *args, **kwargs):
        pass
    def validation_step_end(self, *args, **kwargs):
        pass

    def test_step(self, batch: Batch, _):
        cfg = getattr(self, "analysis_logging_cfg", None) or {}
        from utils.analysis_logging import should_log_topk
        topk_enabled = should_log_topk(cfg, "test", trainer=self._analysis_trainer_or_none())
        
        logits_for_logging = None
        from utils.analysis_logging import should_log_phase
        if should_log_phase(cfg, "test"):
            with torch.inference_mode():
                out = self(batch.imgs, batch.mask, getattr(batch, "tgt", None))
                logits_for_logging = out.logits if hasattr(out, "logits") else out
                if isinstance(logits_for_logging, tuple):
                    logits_for_logging = logits_for_logging[0]

        hyps_or_nbest = self.approximate_joint_search(
            batch.imgs, batch.mask, return_nbest=topk_enabled
        )

        if topk_enabled and hyps_or_nbest and hasattr(hyps_or_nbest[0], 'candidates'):
            hyps = [r.best for r in hyps_or_nbest]
            nbest_outputs = hyps_or_nbest
        else:
            hyps = hyps_or_nbest
            nbest_outputs = None

        self._analysis_log_batch("test", batch, logits_for_logging, hyps, nbest_outputs=nbest_outputs)
        self._maybe_log_analysis_aux(batch, "test")
        
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        return batch.img_bases, [self.vocab_info.words.indices2label(h.seq) for h in hyps]

    def test_epoch_end(self, test_outputs) -> None:
        exprate = self.exprate_recorder.compute()
        print(f"Validation ExpRate: {exprate}")
        cfg = getattr(self, "analysis_logging_cfg", None) or {}
        if cfg.get("enabled", False) and cfg.get("merge_on_epoch_end", False):
            from utils.analysis_logging import maybe_merge_shards, resolve_analysis_run_id, get_dist_info
            run_id = resolve_analysis_run_id(cfg, "CoMER", self.config.get("seed_everything", ""))
            seeds = str(cfg.get("seeds") or self.config.get("seed_everything", ""))
            epoch = int(self.current_epoch)
            rank, _ = get_dist_info()
            maybe_merge_shards(cfg, run_id, seeds, epoch, "test", rank)

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
        checkpoint["lit_comer_warmup_state"] = {
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
        state = checkpoint.get("lit_comer_warmup_state", {})
        self._target_lrs = state.get("target_lrs", self._target_lrs)
        self._warmup_total_steps = state.get("warmup_total_steps", self._warmup_total_steps)
        self._warmup_finished = state.get("warmup_finished", self._warmup_finished)
        self._loaded_plateau_scheduler_state = state.get("plateau_scheduler", None)

    def approximate_joint_search(
        self, 
        img: FloatTensor, 
        mask: LongTensor,
        return_nbest: bool = False,
    ):
        return self.comer_model.beam_search(
            img, mask, **self.hparams, return_nbest=return_nbest
        )

    def configure_optimizers(self):
        name = self.optimizer_use
        if name == "SGD":
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.optimizer_cfg.get("SGD", {}).get("lr", 0.08),
                momentum=self.optimizer_cfg.get("SGD", {}).get("momentum", 0.9),
                weight_decay=self.optimizer_cfg.get("SGD", {}).get("weight_decay", 1e-4),
            )
        elif name == "Adam":
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.optimizer_cfg.get("Adam", {}).get("lr", 0.08),
                betas=self.optimizer_cfg.get("Adam", {}).get("betas", (0.9, 0.999)),
            )
        elif name == "AdamW":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.optimizer_cfg.get("AdamW", {}).get("lr", 0.08),
                betas=self.optimizer_cfg.get("AdamW", {}).get("betas", (0.9, 0.999)),
                weight_decay=self.optimizer_cfg.get("AdamW", {}).get("weight_decay", 1e-4),
            )
        elif name == "Adadelta":
            optimizer = optim.Adadelta(
                self.parameters(),
                lr=self.optimizer_cfg.get("Adadelta", {}).get("lr", 1),
                weight_decay=self.optimizer_cfg.get("Adadelta", {}).get("weight_decay", 1e-4),
                eps=self.optimizer_cfg.get("Adadelta", {}).get("eps", 1e-6),
            )
        else:
            raise ValueError(f"Unknown optimizer: {name}")

        self._target_lrs = [float(pg["lr"]) for pg in optimizer.param_groups]

        sched_name = self.scheduler_use
        if sched_name == "ReduceLROnPlateau":
            self._plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode=self.scheduler_cfg.get("ReduceLROnPlateau", {}).get("mode", "max"),
                factor=self.scheduler_cfg.get("ReduceLROnPlateau", {}).get("factor", 0.25),
                patience=self.scheduler_cfg.get("ReduceLROnPlateau", {}).get("patience", 6),
            )
            loaded_state = getattr(self, "_loaded_plateau_scheduler_state", None)
            if loaded_state is not None:
                self._plateau_scheduler.load_state_dict(loaded_state)

        else:
            raise ValueError(f"Unknown scheduler: {sched_name}")

        return optimizer
