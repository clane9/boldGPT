from collections import defaultdict
from dataclasses import dataclass, field
from importlib import resources
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, RandomSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms as T
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


class ToFloatTensor(torch.nn.Module):
    def forward(self, img: torch.Tensor):
        img = img.to(torch.float32) / 255.0
        # HWC -> CHW
        img = torch.permute(img, (2, 0, 1)).contiguous()
        return img


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


def get_collate(
    img_size: int = 224,
    img_mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    img_std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD,
    interp_mode: InterpolationMode = InterpolationMode.BICUBIC,
    crop_scale: float = 1.0,
    train: bool = True,
) -> Collate:
    transforms = [ToFloatTensor()]
    if train and crop_scale < 1.0:
        transforms.append(
            T.RandomResizedCrop(
                size=img_size,
                scale=(crop_scale, 1.0),
                ratio=(1.0, 1.0),
                interpolation=interp_mode,
                antialias=True,
            )
        )
    else:
        transforms.extend(
            [
                T.Resize(img_size, interpolation=interp_mode, antialias=True),
                # images should be square but just to be safe
                T.CenterCrop(img_size),
            ]
        )
    transforms.append(T.Normalize(mean=img_mean, std=img_std))

    return Collate(
        image_transform=T.Compose(transforms), activity_transform=ActivityTransform()
    )


def load_nsd_flat_mask() -> torch.Tensor:
    """
    Load the NSD-Flat activity data mask. Returns a boolean tensor of shape (215, 200).
    """
    with (resources.files(boldgpt.resources) / "nsd_flat_mask.npy").open("rb") as f:
        mask = torch.from_numpy(np.load(f))
    return mask


def load_nsd_flat(
    splits: List[str] = ["train", "val"],
    columns: Optional[List[str]] = ["subject_id", "nsd_id", "activity"],
    keep_in_memory: bool = False,
) -> Dict[str, Dataset]:
    """
    Load NSD-Flat train/val dataset splits. Optionally keep in memory. Returns a
    dictionary mapping split names to datasets.
    """
    dsets = {}

    if "train" in splits or "val" in splits:
        ds = load_dataset(
            "clane9/NSD-Flat", split="train", keep_in_memory=keep_in_memory
        )
        if columns:
            ds = ds.select_columns(columns)
        ds.set_format("torch")

        if "train" in splits:
            indices = torch.where(~ds["shared1000"])[0]
            dsets["train"] = ds.select(indices, keep_in_memory=keep_in_memory)
        if "val" in splits:
            indices = torch.where(ds["shared1000"])[0]
            dsets["val"] = ds.select(indices, keep_in_memory=keep_in_memory)

    if "test" in splits:
        ds = load_dataset(
            "clane9/NSD-Flat", split="test", keep_in_memory=keep_in_memory
        )
        if columns:
            ds = ds.select_columns(columns)
        ds.set_format("torch")
        dsets["test"] = ds
    return dsets


def load_coco(
    splits: List[str] = ["train", "val"],
    columns: Optional[List[str]] = ["image_id", "activity"],
    keep_in_memory: bool = False,
) -> Dict[str, Dataset]:
    """
    Load COCO train/val dataset splits. Optionally keep in memory. Returns a dictionary
    mapping split names to datasets.
    """
    dsets = {}
    for split in splits:
        ds = load_dataset(
            "detection-datasets/coco", split=split, keep_in_memory=keep_in_memory
        )
        if columns:
            ds = ds.select_columns(columns)
        ds.set_format("torch")
        dsets[split] = ds
    return dsets


@dataclass
class DataConfig:
    dataset: str = "NSD-Flat"
    splits: List[str] = field(default_factory=lambda: ["train", "val"])
    columns: List[str] = field(
        default_factory=lambda: ["subject_id", "nsd_id", "activity"]
    )
    img_size: int = 224
    img_mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    img_std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD
    interp_mode: InterpolationMode = InterpolationMode.BICUBIC


def create_data_loaders(
    cfg: DataConfig,
    keep_in_memory: bool = False,
    crop_scale: float = 1.0,
    distributed: bool = False,
    batch_size: int = 512,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = True,
) -> Dict[str, DataLoader]:
    dataset = cfg.dataset.lower()
    if dataset == "nsd-flat":
        dsets = load_nsd_flat(
            splits=cfg.splits, columns=cfg.columns, keep_in_memory=keep_in_memory
        )
    elif dataset == "coco":
        dsets = load_coco(
            splits=cfg.splits, columns=cfg.columns, keep_in_memory=keep_in_memory
        )
    else:
        raise ValueError(f"Unsupported dataset {cfg.dataset}")

    samplers = {}
    for split, ds in dsets.items():
        samplers[split] = _get_sampler(
            ds, train=split == "train", distributed=distributed
        )

    collate_fns = {}
    for split in dsets:
        collate_fns[split] = get_collate(
            img_size=cfg.img_size,
            img_mean=cfg.img_mean,
            img_std=cfg.img_std,
            interp_mode=cfg.interp_mode,
            crop_scale=crop_scale,
            train=split == "train",
        )

    loaders = {}
    for split in dsets:
        loaders[split] = DataLoader(
            dsets[split],
            batch_size=batch_size,
            sampler=samplers[split],
            num_workers=num_workers,
            collate_fn=collate_fns[split],
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    return loaders


def _get_sampler(
    dataset: Dataset, train: bool = True, distributed: bool = False
) -> Optional[Sampler]:
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=train)
    else:
        sampler = RandomSampler(dataset) if train else None
    return sampler
