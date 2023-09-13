from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset

from .utils import generate_splits

NUM_SAMPLES = 195000
SPLIT_SIZES = {"train": 0.95, "val": 0.05}
SPLIT_SEED = 42


class ActivityTransform(torch.nn.Module):
    def __init__(self, vmin: float = -2.5, vmax: float = 2.5):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax

    def forward(self, act: torch.Tensor):
        act = act.to(torch.float32) / 255.0
        act = (self.vmax - self.vmin) * act + self.vmin
        return act

    def inverse(self, act: torch.Tensor):
        act = (act - self.vmin) / (self.vmax - self.vmin)
        act = (act * 255.0).to(torch.uint8)
        return act


@lru_cache
def load_nsd_flat_splits() -> Dict[str, torch.Tensor]:
    """
    Load NSD-Flat split indices.
    """
    split_indices = generate_splits(
        NUM_SAMPLES, list(SPLIT_SIZES.values()), seed=SPLIT_SEED
    )
    split_indices_map = {split: ind for split, ind in zip(SPLIT_SIZES, split_indices)}
    return split_indices_map


def load_nsd_flat(
    keep_in_memory: bool = False,
    columns: Optional[List[str]] = ["subject_id", "nsd_id", "activity"],
) -> Tuple[Dict[str, Dataset], torch.Tensor]:
    """
    Load NSD-Flat train/val dataset splits. Optionally keep in memory. Returns a
    dictionary mapping split names to datasets and a mask of pixels with fMRI data.
    """
    ds = load_dataset("clane9/NSD-Flat", split="train", keep_in_memory=keep_in_memory)
    if columns:
        ds = ds.select_columns(columns)
    ds.set_format("torch")

    split_indices_map = load_nsd_flat_splits()
    dsets = {
        split: ds.select(ind, keep_in_memory=keep_in_memory)
        for split, ind in split_indices_map.items()
    }

    # mask of pixels with fMRI data
    # missing data are coded as all 0
    example_activity = ds[:100]["activity"]
    mask = ~torch.all(example_activity == 127, dim=0)
    return dsets, mask
