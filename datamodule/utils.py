from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional
import numpy as np
from PIL import Image
from torch import FloatTensor, LongTensor
from pathlib import Path

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
# - lazy_load=False: List[(str, Image.Image, List[str])]
# - lazy_load=True:  List[(str, Tuple[int,int], List[str])]
Data = List[Tuple[str, object, List[str]]]


from torch.utils.data import Sampler
import random
import math
import torch.distributed as dist

class BucketedBatchSampler(Sampler):
    def __init__(
        self,
        data: Data,
        max_pixels_per_batch: int,
        max_batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        maxlen: int = 200,
        max_image_size: int = 32e4,
        seed: int = 42,
    ):
        self.data = data
        self.max_pixels_per_batch = max_pixels_per_batch
        self.max_batch_size = max_batch_size

        # BatchSampler interface compat
        self.batch_size = max_batch_size
        self.drop_last = drop_last

        self.shuffle = shuffle
        self.maxlen = maxlen
        self.max_image_size = max_image_size
        self.seed = seed
        self.epoch = 0
        self.batches = self._build_batches()

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling across DDP ranks."""
        self.epoch = epoch

    def _ddp_info(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size(), dist.get_rank()
        return 1, 0

    def __iter__(self):
        batches = list(self.batches)

        if self.shuffle:
            # Use local RNG seeded by (seed + epoch) for deterministic cross-rank ordering
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(batches)

        world_size, rank = self._ddp_info()

        if world_size > 1:
            if self.drop_last:
                total = (len(batches) // world_size) * world_size
                batches = batches[:total]
            else:
                remainder = len(batches) % world_size
                if remainder != 0:
                    padding = world_size - remainder
                    batches += batches[:padding]

            batches = batches[rank::world_size]

        return iter(batches)


    def __len__(self):
        world_size, _ = self._ddp_info()
        if world_size == 1:
            return len(self.batches)

        if self.drop_last:
            return len(self.batches) // world_size
        return math.ceil(len(self.batches) / world_size)

    def _build_batches(self):
        # Create list of indices
        indices = list(range(len(self.data)))
        
        # Function to get area
        def get_area(idx):
            fea = self.data[idx][1]
            if hasattr(fea, "size"):
                return fea.size[0] * fea.size[1]
            return fea[0] * fea[1]
            
        # Filter indices by maxlen and max_image_size
        valid_indices = []
        for idx in indices:
            _, fea, lab = self.data[idx]
            size = get_area(idx)
            if len(lab) > self.maxlen:
                continue
            if size > self.max_image_size:
                continue
            valid_indices.append(idx)
            
        valid_indices.sort(key=get_area)
        
        batches = []
        current_batch = []
        biggest_image_size = 0
        
        for idx in valid_indices:
            size = get_area(idx)
            if size > biggest_image_size:
                biggest_image_size = size
            
            batch_image_size = biggest_image_size * (len(current_batch) + 1)
            
            if batch_image_size > self.max_pixels_per_batch or len(current_batch) == self.max_batch_size:
                if len(current_batch) > 0:
                    batches.append(current_batch)
                current_batch = []
                biggest_image_size = size
                
            current_batch.append(idx)
            
        if len(current_batch) > 0 and not self.drop_last:
            batches.append(current_batch)
            
        print(f"total {len(batches)} batch data loaded")
        return batches

def _resolve_image_path(
    img_dir: Path,
    stem: str,
    exts: Sequence[str] = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
) -> Optional[Path]:
    p = img_dir / stem
    if p.exists():
        return p

    for ext in exts:
        cand = img_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None

def extract_data(
    root_dir: str,
    split: str,              
    caption_name: str = "caption.txt",
    img_subdir: str = "img",
    convert_mode: str = "L",
    lazy_load: bool = False
) -> Data:
    """
    Read from:
      root_dir/
        split/
          img/
          caption.txt
    caption.txt format: "<image_stem> token1 token2 ..."
    Return: 
      If lazy_load=False: [(img_name, Image, tokens)]
      If lazy_load=True:  [(img_name, (w,h), tokens)]
    """
    split_dir = Path(root_dir) / split
    cap_path = split_dir / caption_name
    img_dir = split_dir / img_subdir

    assert cap_path.exists(), f"Not found: {cap_path}"
    assert img_dir.exists(), f"Not found: {img_dir}"

    with cap_path.open("r", encoding="utf-8") as f:
        captions = f.readlines()

    data: Data = []
    missing = 0

    for line in captions:
        parts = line.strip().split()
        if len(parts) == 0:
            continue
        img_stem = parts[0]
        tokens = parts[1:]

        stem_no_ext = Path(img_stem).stem
        img_path = _resolve_image_path(img_dir, stem_no_ext) or _resolve_image_path(img_dir, img_stem)
        if img_path is None:
            missing += 1
            print(f"[WARN] Missing image for '{img_stem}' in {img_dir}")
            continue

        if lazy_load:
            with Image.open(img_path) as im:
                size = im.size
            data.append((str(img_path), size, tokens))
        else:
            with Image.open(img_path) as im:
                if convert_mode is not None:
                    im = im.convert(convert_mode)
                im = im.copy()
            data.append((img_path.stem, im, tokens))

    print(f"Extract data from dir: {split_dir}, size: {len(data)} (missing: {missing})")
    return data


def build_validation_dataset(
    archive: str,
    folder: str,
    lazy_load: bool = False,
):
    if folder != "all":
        data = extract_data(root_dir=archive, split=folder, lazy_load=lazy_load)
    else:
        data = []
        for folder_name in ["2014", "2016", "2019"]:
            data += extract_data(root_dir=archive, split=folder_name, lazy_load=lazy_load)

    return data


def build_train_dataset(
    archive: str,
    folder: str,
    lazy_load: bool = False,
):
    if folder == "train":
        data = extract_data(root_dir=archive, split=folder, lazy_load=lazy_load)
    return data


# ---------------------------------------------------------------------
# Batch class
# ---------------------------------------------------------------------
@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 1, H, W]
    mask: LongTensor  # [b, H, W]
    indices: List[List[int]]  # [b, l]
    tgt: Optional[LongTensor] = None
    out: Optional[LongTensor] = None
    labels: Optional[LongTensor] = None
    lengths: Optional[LongTensor] = None

    def __len__(self) -> int:
        return len(self.img_bases)

    def pin_memory(self):
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.pin_memory(),
            mask=self.mask.pin_memory(),
            indices=self.indices,
            tgt=None if self.tgt is None else self.tgt.pin_memory(),
            out=None if self.out is None else self.out.pin_memory(),
            labels=None if self.labels is None else self.labels.pin_memory(),
            lengths=None if self.lengths is None else self.lengths.pin_memory(),
        )

    def to(self, device, non_blocking=True) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device, non_blocking=non_blocking),
            mask=self.mask.to(device, non_blocking=non_blocking),
            indices=self.indices,
            tgt=None if self.tgt is None else self.tgt.to(device, non_blocking=non_blocking),
            out=None if self.out is None else self.out.to(device, non_blocking=non_blocking),
            labels=None if self.labels is None else self.labels.to(device, non_blocking=non_blocking),
            lengths=None if self.lengths is None else self.lengths.to(device, non_blocking=non_blocking),
        )
