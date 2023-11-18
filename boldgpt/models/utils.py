import re
from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from matplotlib import colormaps
from matplotlib import pyplot as plt

from boldgpt.utils import resize_and_pad

CMAP = colormaps.get_cmap("turbo")
CMAP.set_bad("gray")


def get_no_decay_keys(model: torch.nn.Module) -> List[str]:
    """
    Get all no decay keys for a model recursively.
    """
    func = getattr(model, "no_decay_keys", None)
    if func is not None:
        return func()

    keys = []
    for name, module in model.named_children():
        keys.extend([f"{name}.{key}" for key in get_no_decay_keys(module)])

    return keys


def infer_embed_dim(arch: str) -> int:
    dims = {
        "tiny": 192,
        "small": 384,
        "base": 768,
        "large": 1024,
    }

    pattern = f"_({'|'.join(dims)})_"
    match = re.search(pattern, arch)
    if match is None:
        raise ValueError(f"Arch {arch} doesn't match any expected dims: {list(dims)}")
    dim = dims[match.group(1)]
    return dim


def r2_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    dim: Union[int, Tuple[int, ...]] = 1,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    var = torch.var(target, dim=dim, correction=0)
    res = torch.mean((target - pred) ** 2, dim=dim)
    score = 1.0 - res / var

    if reduction == "mean":
        score = score.mean()
    elif reduction == "sum":
        score = score.sum()
    elif reduction == "none":
        pass
    else:
        raise ValueError(f"Invalid reduction {reduction}")

    return score


def to_numpy(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x


def imshow(img: np.ndarray, img_shape: Optional[Tuple[int, int]] = None, **kwargs):
    kwargs = {"interpolation": "nearest", "cmap": CMAP, **kwargs}
    if img.ndim == 2:
        # (H, W)
        img = np.where(img == 0, np.nan, img)
    else:
        # (C, H, W)
        img = np.transpose(img, (1, 2, 0))
    if img_shape and img.shape[-2:] != img_shape:
        img = resize_and_pad(img, img_shape)
    plt.imshow(img, **kwargs)
    plt.axis("off")
