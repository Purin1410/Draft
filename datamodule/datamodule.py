from typing import Optional
import pytorch_lightning as pl
from .dataset import CROHMEDataset
from torch.utils.data.dataloader import DataLoader
from .utils import (build_train_dataset, 
                    build_validation_dataset,
                    BucketedBatchSampler)
from .vocab import Vocab
from .utils import Batch
from models.transformer.tree_bias import TreeRelationBuilder
import torch

class CROHMEDatamodule(pl.LightningDataModule):
    # shared_vocab is a class-level cache kept for backward compatibility.
    # It lives ONLY inside the datamodule; no utility or model module should
    # import or read it at import time.  Token IDs used by generation must be
    # passed explicitly via VocabInfo (see datamodule.vocab.VocabInfo).
    shared_vocab: Optional[Vocab] = None  # class-level cache

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.config = config
        data_config = self.config.data
        self.zipfile_path               = data_config.zipfile_path
        self.test_year                  = data_config.test_year
        self.train_batch_size           = data_config.train_batch_size
        self.eval_batch_size            = data_config.eval_batch_size
        self.num_workers                = data_config.num_workers
        self.scale_aug                  = data_config.scale_aug
        self.max_pixels_per_batch       = int(data_config.get("max_pixels_per_batch", data_config.get("gpu_max_memory", 1280000)))
        self.maxlen                     = self.config.model.max_len
        self.lazy_load                  = data_config.lazy_load
        self.k_min                      = data_config.k_min
        self.k_max                      = data_config.k_max
        self.w_lo                       = data_config.w_lo
        self.w_hi                       = data_config.w_hi
        self.h_lo                       = data_config.h_lo
        self.h_hi                       = data_config.h_hi
        self.pin_memory                 = data_config.pin_memory
        self.persistent_workers         = data_config.persistent_workers
        if CROHMEDatamodule.shared_vocab is None:
            CROHMEDatamodule.shared_vocab = Vocab(dict_path=data_config.dictionary_txt)
        self.vocab = CROHMEDatamodule.shared_vocab

        # Tree bias builder for precomputing rel_ids in DataLoader workers
        mcfg = self.config.model
        if mcfg.get("use_tree_bias", True):
            self.tree_builder = TreeRelationBuilder(
                id2tok=self.vocab.idx2word,
                pad_id=self.vocab.PAD_IDX,
                num_buckets=mcfg.get("tree_bias_num_buckets", 16),
                mode=mcfg.get("tree_bias_mode", "full"),
                rel_set=mcfg.get("tree_bias_rel_set", "full"),
            )
        else:
            self.tree_builder = None
        
        self.train_batch_sampler = None
        self.val_batch_sampler = None
        self.test_batch_sampler = None
        
        print(f"Load data from: {self.zipfile_path}")
    
    def collate_fn(self, batch):
        fnames = [b[0] for b in batch]
        images_x = [b[1] for b in batch]
        seqs_y = [self.vocab.words2indices(b[2]) for b in batch]
        
        heights_x = [s.size(1) for s in images_x]
        widths_x = [s.size(2) for s in images_x]

        n_samples = len(heights_x)
        max_height_x = max(heights_x) if n_samples > 0 else 0
        max_width_x = max(widths_x) if n_samples > 0 else 0
        
        # pad_strategy = self.config.data.get("pad_strategy", "batch_max")
        # pad_to_multiple = self.config.data.get("pad_to_multiple", 32)
        # static_pad_height = self.config.data.get("static_pad_height", None)
        # static_pad_width = self.config.data.get("static_pad_width", None)

        # if pad_strategy == "bucket":
        #     max_height_x = ((max_height_x + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
        #     max_width_x = ((max_width_x + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
        # elif pad_strategy == "static":
        #     if static_pad_height is not None:
        #         max_height_x = max(max_height_x, static_pad_height)
        #     if static_pad_width is not None:
        #         max_width_x = max(max_width_x, static_pad_width)

        x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
        x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)
        for idx, s_x in enumerate(images_x):
            x[idx, :, : heights_x[idx], : widths_x[idx]] = s_x
            x_mask[idx, : heights_x[idx], : widths_x[idx]] = 0

        from utils.utils import to_bi_tgt_out_from_padded
        
        lengths_x = [len(s) for s in seqs_y]
        max_len = max(lengths_x) if len(lengths_x) > 0 else 0
        labels = torch.full((n_samples, max_len), fill_value=self.vocab.PAD_IDX, dtype=torch.long)
        for i, s in enumerate(seqs_y):
            labels[i, :lengths_x[i]] = torch.tensor(s, dtype=torch.long)
        lengths = torch.tensor(lengths_x, dtype=torch.long)
        
        # Vectorized bidirectional tgt/out from padded labels (D1)
        tgt, out = to_bi_tgt_out_from_padded(
            labels, lengths,
            sos_id=self.vocab.SOS_IDX,
            eos_id=self.vocab.EOS_IDX,
            pad_id=self.vocab.PAD_IDX,
        )

        # Precompute tree relation IDs on CPU (offloaded to DataLoader workers)
        rel_ids = None
        if self.tree_builder is not None:
            rel_ids = self.tree_builder.build(tgt)

        return Batch(
            img_bases=fnames,
            imgs=x,
            mask=x_mask,
            indices=seqs_y,
            tgt=tgt,
            out=out,
            labels=labels,
            lengths=lengths,
            rel_ids=rel_ids
        )

        

    def setup(self, stage: Optional[str] = None) -> None:
        # with ZipFile(self.zipfile_path) as archive:
        if stage == "fit" or stage is None:
            # Train_dataset
            self.train_dataset = CROHMEDataset(
                dataset = build_train_dataset(
                            archive=self.zipfile_path,
                            folder="train",
                            lazy_load=self.lazy_load,
                        ),
                is_train    = True,
                scale_aug   = self.scale_aug,
                k_min       = self.config.data.k_min,
                k_max       = self.config.data.k_max,
                w_lo        = self.config.data.w_lo,
                w_hi        = self.config.data.w_hi,
                h_lo        = self.config.data.h_lo,
                h_hi        = self.config.data.h_hi,
                lazy_load   = self.lazy_load,
                cache_transforms = self.config.data.get("cache_transforms", True),
            )
            # Val_dataset
            self.val_dataset = CROHMEDataset(
                dataset = build_validation_dataset(
                    archive=self.zipfile_path,
                    folder=self.test_year,
                    lazy_load=self.lazy_load,
                ),
                is_train    = False,
                scale_aug   = self.scale_aug,
                k_min       = self.config.data.k_min,
                k_max       = self.config.data.k_max,
                w_lo        = self.config.data.w_lo,
                w_hi        = self.config.data.w_hi,
                h_lo        = self.config.data.h_lo,
                h_hi        = self.config.data.h_hi,
                lazy_load   = self.lazy_load,
                cache_transforms = self.config.data.get("cache_transforms", True),
            )
        if stage == "test" or stage is None:
            self.test_dataset = CROHMEDataset(
                dataset = build_validation_dataset(
                    archive=self.zipfile_path, 
                    folder=self.test_year, 
                    lazy_load=self.lazy_load,
                                            ),
                is_train    = False,
                scale_aug   = self.scale_aug,
                k_min       = self.config.data.k_min,
                k_max       = self.config.data.k_max,
                w_lo        = self.config.data.w_lo,
                w_hi        = self.config.data.w_hi,
                h_lo        = self.config.data.h_lo,
                h_hi        = self.config.data.h_hi,
                lazy_load   = self.lazy_load,
                cache_transforms = self.config.data.get("cache_transforms", True),
            )

    def _get_worker_init_fn(self):
        def worker_init_fn(worker_id):
            try:
                import cv2
                cv2.setNumThreads(self.config.data.get("opencv_num_threads_per_worker", 0))
            except Exception:
                pass
        return worker_init_fn

    def train_dataloader(self):
        self.train_batch_sampler = BucketedBatchSampler(
            data=self.train_dataset.dataset,
            max_pixels_per_batch=self.max_pixels_per_batch,
            max_batch_size=self.train_batch_size,
            shuffle=True,
            maxlen=self.maxlen,
            max_image_size=self.max_pixels_per_batch,
            seed=self.config.seed_everything
        )
        return DataLoader(
            dataset             = self.train_dataset,
            batch_sampler       = self.train_batch_sampler,
            num_workers         = self.num_workers,
            collate_fn          = self.collate_fn,
            pin_memory          = self.pin_memory,
            persistent_workers  = self.persistent_workers,
            worker_init_fn      = self._get_worker_init_fn(),
        )

    # NOTE: In DDP, BucketedBatchSampler may pad by repeating batches so each rank
    # has the same number of steps. This is fine for training-time validation used
    # as a rough signal, but official ExpRate should be computed with a single
    # process to avoid counting duplicated samples.
    def val_dataloader(self):
        self.val_batch_sampler = BucketedBatchSampler(
            data=self.val_dataset.dataset,
            max_pixels_per_batch=self.max_pixels_per_batch,
            max_batch_size=self.eval_batch_size,
            shuffle=False,
            maxlen=self.maxlen,
            max_image_size=self.max_pixels_per_batch,
            seed=self.config.seed_everything
        )
        return DataLoader(
            dataset             = self.val_dataset,
            batch_sampler       = self.val_batch_sampler,
            num_workers         = self.num_workers,
            collate_fn          = self.collate_fn,
            pin_memory          = self.pin_memory,
            persistent_workers  = self.persistent_workers,
            worker_init_fn      = self._get_worker_init_fn(),
        )

    # NOTE: In DDP, BucketedBatchSampler may pad by repeating batches so each rank
    # has the same number of steps. This is fine for training-time validation used
    # as a rough signal, but official ExpRate should be computed with a single
    # process to avoid counting duplicated samples.
    def test_dataloader(self):
        self.test_batch_sampler = BucketedBatchSampler(
            data=self.test_dataset.dataset,
            max_pixels_per_batch=self.max_pixels_per_batch,
            max_batch_size=self.eval_batch_size,
            shuffle=False,
            maxlen=self.maxlen,
            max_image_size=self.max_pixels_per_batch,
            seed=self.config.seed_everything
        )
        return DataLoader(
            dataset             = self.test_dataset,
            batch_sampler       = self.test_batch_sampler,
            num_workers         = self.num_workers,
            collate_fn          = self.collate_fn,
            pin_memory          = self.pin_memory,
            persistent_workers  = self.persistent_workers,
            worker_init_fn      = self._get_worker_init_fn(),
        )