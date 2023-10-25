from collections import defaultdict
from functools import lru_cache
from importlib import resources
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import boldgpt.resources


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


class Collate(torch.nn.Module):
    def __init__(
        self,
        image_transform: Optional[torch.nn.Module] = None,
        activity_transform: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.image_transform = image_transform
        self.activity_transform = activity_transform

    def forward(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        collated = defaultdict(list)
        for sample in batch:
            for k, v in sample.items():
                collated[k].append(v)

        if "image" in collated and self.image_transform is not None:
            collated["image"] = [self.image_transform(img) for img in collated["image"]]

        if "activity" in collated and self.activity_transform is not None:
            collated["activity"] = [
                self.activity_transform(img) for img in collated["activity"]
            ]

        collated = {k: torch.stack(v) for k, v in collated.items()}
        return collated


@lru_cache
def load_nsd_flat_splits() -> Dict[str, torch.Tensor]:
    """
    Load NSD-Flat split indices. Validation samples are those from shared1000.
    """
    ds = load_dataset("clane9/NSD-Flat", split="train")
    ds.set_format("torch")
    shared1000_mask = ds["shared1000"]
    split_indices_map = {
        "train": torch.where(~shared1000_mask)[0],
        "val": torch.where(shared1000_mask)[0],
    }
    return split_indices_map


def load_nsd_flat_mask() -> torch.Tensor:
    """
    Load the NSD-Flat activity data mask. Returns a boolean tensor of shape (215, 200).
    """
    with (resources.files(boldgpt.resources) / "nsd_flat_mask.npy").open("rb") as f:
        mask = torch.from_numpy(np.load(f))
    return mask


def load_nsd_flat(
    keep_in_memory: bool = False,
    columns: Optional[List[str]] = ["subject_id", "nsd_id", "activity"],
) -> Dict[str, Dataset]:
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
    return dsets


def get_mask(activity: torch.Tensor):
    """
    Get the mask of pixels with fMRI data (not all zero across the first dimension).
    """
    return ~torch.all(activity == 127, dim=0)


def get_default_collate() -> Collate:
    """
    Get the default collate function for NSD flat.
    """
    return Collate(
        image_transform=transforms.Compose(
            [
                transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
                ),
            ]
        ),
        activity_transform=ActivityTransform(),
    )
