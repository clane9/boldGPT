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
from boldgpt.shuffle import permute, random_order
from boldgpt.tokenizer import KMeansTokenizer

from . import constants as C
from .generate import generate
from .registry import register_model
from .transformer import Transformer
from .utils import r2_score

CMAP = colormaps.get_cmap("turbo")
CMAP.set_bad("gray")


class BEiT(nn.Module):
    def __init__(
        self,
        patchify: MaskedPatchify,
        tokenizer: Optional[KMeansTokenizer],
        encoder: Transformer,
        mask_ratio: Optional[float] = 0.75,
        modality: Literal["image", "bold"] = "bold",
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.modality = modality
        self.is_categorical = tokenizer is not None
        self.with_sub_embed = encoder.with_sub_embed

        self.patchify = patchify
        if tokenizer is None:
            self.register_module("tokenizer", None)
        else:
            self.tokenizer = tokenizer
        self.encoder = encoder

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
        tokens = self.tokenizer(patches) if self.is_categorical else None

        # Sample mask positions
        bool_masked_pos = _sample_masked_pos(
            B, N, mask_ratio=self.mask_ratio, device=device
        )

        # Forward pass
        output = self.encoder(
            patches, sub_indices=sub_indices, bool_masked_pos=bool_masked_pos
        )
        # Drop the leading subject token (and any registers)
        output = output[:, 1 : N + 1]

        state = dict(
            patches=patches,
            tokens=tokens,
            bool_masked_pos=bool_masked_pos,
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
            loss = F.cross_entropy(
                output.flatten(0, 1), tokens.flatten(), reduction="none"
            )
            mask = state["bool_masked_pos"].flatten()
            loss = torch.sum(mask * loss) / mask.sum()
        else:
            patches = state["patches"]
            mask = self.patchify.patch_mask.expand_as(patches)
            mask = mask & state["bool_masked_pos"].unsqueeze(-1)
            loss = torch.sum(mask * (output - patches) ** 2) / mask.sum()
        return loss

    @torch.no_grad()
    def generate(
        self,
        batch: Dict[str, torch.Tensor],
        prompt_fraction: float = 0.25,
        shuffle: bool = False,
        order: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        images = batch["activity"] if self.modality == "bold" else batch["image"]
        sub_indices = batch["subject_id"] if self.with_sub_embed else None

        patches = self.patchify(images)
        B, N = patches.shape[:2]
        device = patches.device

        if shuffle:
            order, ranking = random_order(B, N, device=device)
        elif order is not None:
            ranking = torch.argsort(order, dim=1)
        else:
            ranking = None

        if order is not None:
            patches = permute(patches, order)
        prompt_length = int(prompt_fraction * N)
        prompt = patches[:, :prompt_length]

        pred = generate(
            model=self.encoder,
            prompt=prompt,
            sub_indices=sub_indices,
            order=order,
            tokenizer=self.tokenizer,
            patch_mask=self.patchify.patch_mask,
            offset=1,
            temperature=temperature,
            top_k=top_k,
        )

        if order is not None:
            recon = permute(pred, ranking)
        recon = self.patchify.inverse(recon)

        state = {
            "patches": patches,
            "order": order,
            "ranking": ranking,
            "prompt_length": prompt_length,
            "pred": pred,
            "recon": recon,
        }
        return recon, state

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
        tokens = state["tokens"]
        bool_masked_pos = state["bool_masked_pos"]
        output = state["output"]

        # Ensure images are masked
        mask = self.patchify.mask
        images = mask * images

        # Invert mask and get observed mask ratio
        visible_mask = self.patchify.inverse_vector(~bool_masked_pos)
        visible = visible_mask * images
        mask_ratio = bool_masked_pos.float().mean(dim=1)

        # Get reconstruction and compute R^2
        if self.is_categorical:
            target = self.tokenizer.inverse(tokens)
            pred = torch.argmax(output, dim=-1)
            recon = self.tokenizer.inverse(pred)
        else:
            target = patches
            recon = output
        target = self.patchify.inverse(target)
        recon = self.patchify.inverse(recon)
        pred_mask = mask & ~visible_mask
        target_r2 = r2_score(
            target * pred_mask, images * pred_mask, dim=(1, 2), reduction="none"
        )
        recon_r2 = r2_score(
            recon * pred_mask, images * pred_mask, dim=(1, 2), reduction="none"
        )

        examples = {
            "subject_id": subind,
            "nsd_id": nsdind,
            "images": images,
            "visible": visible,
            "mask_ratio": mask_ratio,
            "target": target,
            "recon": recon,
            "target_r2": target_r2,
            "recon_r2": recon_r2,
        }
        examples = {k: v.cpu().numpy() for k, v in examples.items()}
        return examples

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
        visible = examples["visible"]
        mask_ratio = examples["mask_ratio"]
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
            plt.text(
                0.02, 0.0, label, ha="left", va="bottom", transform=tform, **textdict
            )

            plt.sca(axs[ii, 1])
            tform = axs[ii, 1].transAxes
            _imshow(visible[ii])
            plt.text(
                0.5, 0.98, "Visible", ha="center", va="top", transform=tform, **textdict
            )
            plt.text(
                0.98,
                0.0,
                f"MR={mask_ratio[ii]:.2f}",
                ha="right",
                va="bottom",
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
                    f"R2={target_r2[ii]:.3f}",
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
                f"R2={recon_r2[ii]:.3f}",
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
            f"is_categorical={self.is_categorical}, mask_ratio={self.mask_ratio}, "
            f"modality={self.modality}"
        )


def _sample_masked_pos(
    batch_size: int,
    num_patches: int,
    mask_ratio: Optional[float] = 0.5,
    device: torch.device = None,
) -> torch.Tensor:
    # Per-sample mask ratio between 0 and 1
    if mask_ratio is None:
        mask_ratio = torch.rand(batch_size, 1, device=device)
    bool_masked_pos = torch.rand(batch_size, num_patches, device=device) < mask_ratio
    return bool_masked_pos


def _imshow(img: np.ndarray, **kwargs):
    kwargs = {"interpolation": "nearest", "cmap": CMAP, **kwargs}
    img = np.where(img == 0, np.nan, img)
    plt.imshow(img, **kwargs)
    plt.axis("off")


def _create_beit(
    *,
    modality: str = "bold",
    mask: Optional[np.ndarray] = None,
    img_size: int = 224,
    patch_size: int = 10,
    ordering: Optional[str] = None,
    categorical: bool = False,
    with_sub_embed: Optional[bool] = None,
    vocab_size: int = 1024,
    mask_ratio: Optional[float] = 0.75,
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
    encoder = Transformer(
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
        with_next_pos=False,
        with_cross=False,
        is_causal=False,
        is_masked=True,
        drop_rate=drop_rate,
        sub_drop_rate=sub_drop_rate,
        proj_drop_rate=proj_drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
    )

    model = BEiT(patchify, tokenizer, encoder, mask_ratio=mask_ratio, modality=modality)
    return model


@register_model
def boldbeit_tiny_patch10(**kwargs):
    return _create_beit(modality="bold", patch_size=10, **C.TINY_ARCH_KWARGS, **kwargs)


@register_model
def boldbeit_small_patch10(**kwargs):
    return _create_beit(modality="bold", patch_size=10, **C.SMALL_ARCH_KWARGS, **kwargs)


@register_model
def boldbeit_base_patch10(**kwargs):
    return _create_beit(modality="bold", patch_size=10, **C.BASE_ARCH_KWARGS, **kwargs)


@register_model
def imagebeit_tiny_patch16(**kwargs):
    return _create_beit(modality="image", patch_size=16, **C.TINY_ARCH_KWARGS, **kwargs)


@register_model
def imagebeit_small_patch16(**kwargs):
    return _create_beit(
        modality="image", patch_size=16, **C.SMALL_ARCH_KWARGS, **kwargs
    )


@register_model
def imagebeit_base_patch16(**kwargs):
    return _create_beit(modality="image", patch_size=16, **C.BASE_ARCH_KWARGS, **kwargs)
