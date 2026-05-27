import torchvision.transforms as tr
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
from .transforms import ScaleAugmentation, ScaleToLimitRange

class CROHMEDataset(Dataset):
    def __init__(self, 
                dataset,
                is_train: bool,
                scale_aug: bool,
                k_min: float,
                k_max: float,
                w_lo: float,
                w_hi: float,
                h_lo: float,
                h_hi: float,
                lazy_load: bool = False,
                cache_transforms: bool = False) -> None:
        super().__init__()
        self.dataset = dataset
        self.lazy_load = lazy_load
        self.scale_aug = scale_aug
        
        # We only cache if scale_aug is disabled to avoid caching an augmented image forever
        self.use_cache = cache_transforms and not scale_aug
        if self.use_cache:
            self.cache = {}
        else:
            self.cache = None

        trans_list = []
        if is_train and scale_aug:
            trans_list.append(ScaleAugmentation(lo = k_min,
                                                hi = k_max)
                              )

        trans_list += [
            ScaleToLimitRange(w_lo=w_lo, 
                              w_hi=w_hi, 
                              h_lo=h_lo, 
                              h_hi=h_hi),
            tr.ToTensor(),
        ]
        self.transform = tr.Compose(trans_list)

    def __getitem__(self, idx):
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]
            
        fname, im, caption = self.dataset[idx]

        if self.lazy_load:
            try:
                im = Image.open(fname).convert("L")
                im = np.array(im)
            except Exception as e:
                raise RuntimeError(
                    f"Could not open image {fname}: {e}. "
                    "Fix or remove the broken record from the dataset."
                ) from e

        if not isinstance(im, np.ndarray):
            im = np.array(im)
            
        processed_img = self.transform(im)
        res = (fname, processed_img, caption)
        
        if self.cache is not None:
            self.cache[idx] = res

        return res

    def __len__(self):
        return len(self.dataset)