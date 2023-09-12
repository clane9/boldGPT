from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from boldgpt.tokenizer import BoldTokenizer

plt.rcParams["figure.dpi"] = 150
plt.style.use("ggplot")


def plot_examples(
    tokenizer: BoldTokenizer,
    subid: torch.Tensor,
    nsdid: torch.Tensor,
    activity: torch.Tensor,
    tokens: torch.Tensor,
    pred: Optional[torch.Tensor] = None,
    order: Optional[torch.Tensor] = None,
    fname: Optional[str] = None,
) -> Figure:
    """
    Plot a grid of examples.

    Args:
        subid: subject IDs, (B,)
        nsdid: NSD stimulus IDs, (B,)
        activity: activity maps, (B, H, W)
        tokens: discretized activity tokens, (B, N)
        pred: predicted tokens, (B, N)
        order: per-sample token order, (B, N)
        fname: output figure filename

    Returns:
        A matplotlib figure
    """
    assert activity.ndim == 3, "Invalid activity shape"
    assert tokens.ndim == 2, "Invalid tokens shape"
    assert pred is None or pred.ndim == 2, "Invalid pred shape"
    assert order is None or order.ndim == 2, "Invalid order shape"

    B = activity.shape[0]

    activity = activity.cpu()
    tokens = tokens.cpu()
    if pred is not None:
        pred = pred.detach().cpu()
    if order is not None:
        order = order.cpu()
    else:
        order = torch.arange(tokenizer.num_patches).expand(B, -1)

    mask = tokenizer.patchify.mask.numpy()
    subid = subid.numpy()
    nsdid = nsdid.numpy()

    tokens = tokenizer.inverse(tokens)
    if pred is not None:
        pred = tokenizer.inverse(pred)
    order = inverse_order(tokenizer, order)

    plotw = 3.0
    ploth = 3.5
    nr = B
    nc = 3 if pred is None else 4
    f, axs = plt.subplots(nr, nc, figsize=(nc * plotw, nr * ploth), squeeze=False)

    textdict = {"fontsize": 10, "color": "w"}

    for ii in range(B):
        label = f"sub{subid[ii]+1:02d} nsd{nsdid[ii]:05d}"

        plt.sca(axs[ii, 0])
        tform = axs[ii, 0].transAxes
        show_img(mask, order[ii])
        plt.text(0.5, 0.98, "Order", ha="center", va="top", transform=tform, **textdict)
        plt.text(0.02, 0.0, label, ha="left", va="bottom", transform=tform, **textdict)

        plt.sca(axs[ii, 1])
        tform = axs[ii, 1].transAxes
        show_img(mask, activity[ii])
        plt.text(
            0.5, 0.98, "Activity", ha="center", va="top", transform=tform, **textdict
        )

        plt.sca(axs[ii, 2])
        tform = axs[ii, 2].transAxes
        show_img(mask, tokens[ii])
        plt.text(
            0.5, 0.98, "Tokens", ha="center", va="top", transform=tform, **textdict
        )

        if pred is not None:
            plt.sca(axs[ii, 3])
            tform = axs[ii, 3].transAxes
            show_img(mask, pred[ii])
            plt.text(
                0.5, 0.98, "Pred", ha="center", va="top", transform=tform, **textdict
            )

    plt.tight_layout(pad=0.2, h_pad=0.05)

    if fname is not None:
        plt.savefig(fname, bbox_inches="tight")
    return f


def inverse_vector(tokenizer: BoldTokenizer, values: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct a patch map from a vector of patch values
    """
    assert values.ndim == 2, "Invalid values shape"
    values = values[..., None].expand(-1, -1, tokenizer.dim)
    values_map = tokenizer.inverse_patches(values)
    return values_map


def inverse_order(tokenizer: BoldTokenizer, order: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct a patch ranking map from patch order indices.
    """
    ranking = torch.argsort(order, dim=-1)
    return inverse_vector(tokenizer, ranking)


def show_img(mask: np.ndarray, img: Optional[np.ndarray] = None):
    """
    Show a masked image.
    """
    plt.imshow(mask, vmin=-1.0, cmap="gray", interpolation="nearest")
    if img is not None:
        img = np.asarray(img)
        img = np.where(mask, img, np.nan)
        plt.imshow(img, cmap="turbo", interpolation="nearest")
    plt.axis("off")
