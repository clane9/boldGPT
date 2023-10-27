import logging
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import nn

from boldgpt.data import load_nsd_flat_mask
from boldgpt.patching import MaskedPatchify
from boldgpt.shuffle import random_order, shuffle
from boldgpt.tokenizer import KMeansTokenizer

from . import constants as C
from .registry import register_model
from .transformer import Transformer
from .utils import r2_score

CMAP = colormaps.get_cmap("turbo")
CMAP.set_bad("gray")


class ImageGPT(nn.Module):
    def __init__(
        self,
        patchify: MaskedPatchify,
        tokenizer: Optional[KMeansTokenizer],
        decoder: Transformer,
        shuffle: bool = True,
        modality: Literal["image", "bold"] = "bold",
    ):
        super().__init__()
        self.shuffle = shuffle
        self.modality = modality
        self.is_categorical = tokenizer is not None
        self.with_sub_embed = decoder.with_sub_embed

        self.patchify = patchify
        if tokenizer is None:
            self.register_module("tokenizer", None)
        else:
            self.tokenizer = tokenizer
        self.decoder = decoder

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        # Unpack data
        images = batch["activity"] if self.modality == "bold" else batch["image"]
        sub_indices = batch["subject_id"] if self.with_sub_embed else None

        # Get patches and optionally tokens
        patches = self.patchify(images)
        B, N = patches.shape[:2]
        device = patches.device
        if self.shuffle and self.training:
            order, ranking = random_order(B, N, device=device)
            patches = shuffle(patches, order)
        else:
            order = ranking = None
        tokens = self.tokenizer(patches) if self.is_categorical else None

        # Forward pass
        output = self.decoder(patches, sub_indices=sub_indices, order=order)
        # Drop the trailing EOS token (and any registers)
        output = output[:, :N]

        state = dict(
            patches=patches,
            order=order,
            ranking=ranking,
            tokens=tokens,
            output=output.detach(),
        )
        return output, state

    def loss_fn(
        self,
        batch: Dict[str, torch.Tensor],
        output: torch.Tensor,
        state: Dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        if self.is_categorical:
            tokens = state["tokens"]
            loss = F.cross_entropy(output.flatten(0, 1), tokens.flatten())
        else:
            # Masked MSE loss
            patches = state["patches"]
            order = state["order"]
            mask = self.patchify.patch_mask.expand_as(patches)
            # If patches are shuffled, the mask must be shuffled as well.
            if order is not None:
                mask = shuffle(mask, order)
            loss = torch.sum(mask * (output - patches) ** 2) / mask.sum()
        return loss

    @torch.no_grad()
    def prepare_examples(
        self,
        batch: Dict[str, torch.Tensor],
        state: Dict[str, torch.Tensor],
    ) -> Dict[str, np.ndarray]:
        """
        Prepare a batch of examples for figure generation.
        """
        subind = batch["subject_id"]
        nsdind = batch["nsd_id"]
        images = batch["activity"] if self.modality == "bold" else batch["image"]

        patches = state["patches"]
        order = state["order"]
        ranking = state["ranking"]
        tokens = state["tokens"]
        output = state["output"]

        B, N = patches.shape[:2]
        device = patches.device

        # Ensure images are masked
        mask = self.patchify.mask
        images = mask * images

        # Unshuffle if necessary
        if order is not None:
            patches = shuffle(patches, ranking)
            if tokens is not None:
                tokens = shuffle(tokens, ranking)
            output = shuffle(output, ranking)
        else:
            ranking = torch.arange(N, device=device).expand(B, -1)

        # Invert order, target, reconstruction and compute MSE
        order_map = self.patchify.inverse_vector(ranking.float() + 1.0)
        target = self.inverse_target(patches, tokens)
        recon = self.inverse_recon(output)
        target_r2 = r2_score(target[:, mask], images[:, mask], reduction="none")
        recon_r2 = r2_score(recon[:, mask], images[:, mask], reduction="none")

        examples = {
            "subject_id": subind,
            "nsd_id": nsdind,
            "images": images,
            "order_map": order_map,
            "target": target,
            "recon": recon,
            "target_r2": target_r2,
            "recon_r2": recon_r2,
        }
        examples = {k: v.cpu().numpy() for k, v in examples.items()}
        return examples

    def inverse_target(
        self, patches: torch.Tensor, tokens: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.is_categorical:
            patches = self.tokenizer.inverse(tokens)
        target = self.patchify.inverse(patches)
        return target

    def inverse_recon(self, output: torch.Tensor) -> torch.Tensor:
        if self.is_categorical:
            pred = torch.argmax(output, dim=-1)
            output = self.tokenizer.inverse(pred)
        recon = self.patchify.inverse(output)
        return recon

    def plot_examples(
        self,
        examples: Dict[str, np.ndarray],
        num_examples: int = 10,
        fname: Optional[str] = None,
    ) -> Figure:
        """
        Plot a grid of samples and predictions.
        """
        examples = {k: v[:num_examples] for k, v in examples.items()}

        subind = examples["subject_id"]
        nsdind = examples["nsd_id"]
        images = examples["images"]
        order_map = examples["order_map"]
        target = examples["target"]
        recon = examples["recon"]
        target_r2 = examples["target_r2"]
        recon_r2 = examples["recon_r2"]

        B = images.shape[0]
        plotw = 3.0
        ploth = 3.5
        nr = B
        nc = 4 if self.is_categorical else 3
        f, axs = plt.subplots(nr, nc, figsize=(nc * plotw, nr * ploth), squeeze=False)

        textdict = {"fontsize": 10, "color": "w"}

        for ii in range(B):
            label = f"sub{subind[ii]+1:02d} nsd{nsdind[ii]:05d}"

            plt.sca(axs[ii, 0])
            tform = axs[ii, 0].transAxes
            _imshow(order_map[ii])
            plt.text(
                0.5, 0.98, "Order", ha="center", va="top", transform=tform, **textdict
            )
            plt.text(
                0.02, 0.0, label, ha="left", va="bottom", transform=tform, **textdict
            )

            plt.sca(axs[ii, 1])
            tform = axs[ii, 1].transAxes
            _imshow(images[ii])
            plt.text(
                0.5,
                0.98,
                "Activity" if self.modality == "bold" else "Image",
                ha="center",
                va="top",
                transform=tform,
                **textdict,
            )

            if self.is_categorical:
                plt.sca(axs[ii, 2])
                tform = axs[ii, 2].transAxes
                _imshow(target[ii])
                plt.text(
                    0.5,
                    0.98,
                    "Target",
                    ha="center",
                    va="top",
                    transform=tform,
                    **textdict,
                )
                plt.text(
                    0.98,
                    0.0,
                    f"R2={target_r2[ii]:.2e}",
                    ha="right",
                    va="bottom",
                    transform=tform,
                    **textdict,
                )

            col = 3 if self.is_categorical else 2
            plt.sca(axs[ii, col])
            tform = axs[ii, col].transAxes
            _imshow(recon[ii])
            plt.text(
                0.5, 0.98, "Pred", ha="center", va="top", transform=tform, **textdict
            )
            plt.text(
                0.98,
                0.0,
                f"R2={recon_r2[ii]:.2e}",
                ha="right",
                va="bottom",
                transform=tform,
                **textdict,
            )

        plt.tight_layout(pad=0.2, h_pad=0.05)

        if fname is not None:
            plt.savefig(fname, bbox_inches="tight")
        return f

    def extra_repr(self) -> str:
        return (
            f"is_categorical={self.is_categorical}, shuffle={self.shuffle}, "
            f"modality={self.modality}"
        )


def _imshow(img: np.ndarray, **kwargs):
    kwargs = {"interpolation": "nearest", "cmap": CMAP, **kwargs}
    img = np.where(img == 0, np.nan, img)
    plt.imshow(img, **kwargs)
    plt.axis("off")


def _create_image_gpt(
    *,
    modality: str = "bold",
    mask: Optional[np.ndarray] = None,
    img_size: int = 224,
    patch_size: int = 10,
    ordering: Optional[str] = None,
    categorical: bool = True,
    with_sub_embed: Optional[bool] = None,
    vocab_size: int = 1024,
    shuffle: bool = True,
    num_subs: int = 1024,
    num_registers: int = 0,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    mlp_ratio: float = 4.0,
    drop_rate: float = 0.0,
    sub_drop_rate: float = 0.0,
    proj_drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    **kwargs,
):
    if kwargs:
        logging.warning("Extra unused kwargs: %s", kwargs)

    is_bold = modality == "bold"
    if mask is None:
        if is_bold:
            mask = load_nsd_flat_mask()
        else:
            mask = torch.ones(img_size, img_size, dtype=torch.bool)
    if ordering is None:
        ordering = "radial" if is_bold else "reverse_radial"
    if with_sub_embed is None:
        with_sub_embed = is_bold

    patchify = MaskedPatchify(mask, patch_size=patch_size, ordering=ordering)

    if categorical:
        tokenizer = KMeansTokenizer(vocab_size=vocab_size, dim=patchify.dim)
    else:
        tokenizer = None

    decoder = Transformer(
        num_patches=patchify.num_patches,
        in_features=patchify.dim,
        num_subs=num_subs,
        num_registers=num_registers,
        num_classes=(vocab_size if categorical else patchify.dim),
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        with_sub_embed=with_sub_embed,
        with_next_pos=shuffle,
        with_cross=False,
        is_causal=True,
        is_masked=False,
        drop_rate=drop_rate,
        sub_drop_rate=sub_drop_rate,
        proj_drop_rate=proj_drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
    )

    model = ImageGPT(patchify, tokenizer, decoder, shuffle=shuffle, modality=modality)
    return model


@register_model
def boldgpt_tiny_patch10(**kwargs):
    return _create_image_gpt(
        modality="bold", patch_size=10, **C.TINY_ARCH_KWARGS, **kwargs
    )


@register_model
def boldgpt_small_patch10(**kwargs):
    return _create_image_gpt(
        modality="bold", patch_size=10, **C.SMALL_ARCH_KWARGS, **kwargs
    )


@register_model
def boldgpt_base_patch10(**kwargs):
    return _create_image_gpt(
        modality="bold", patch_size=10, **C.BASE_ARCH_KWARGS, **kwargs
    )


@register_model
def imagegpt_tiny_patch16(**kwargs):
    return _create_image_gpt(
        modality="image", patch_size=16, **C.TINY_ARCH_KWARGS, **kwargs
    )


@register_model
def imagegpt_small_patch16(**kwargs):
    return _create_image_gpt(
        modality="image", patch_size=16, **C.SMALL_ARCH_KWARGS, **kwargs
    )


@register_model
def imagegpt_base_patch16(**kwargs):
    return _create_image_gpt(
        modality="image", patch_size=16, **C.BASE_ARCH_KWARGS, **kwargs
    )
