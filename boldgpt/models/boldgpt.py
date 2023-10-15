import logging
from typing import Dict, Optional, Tuple

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

from .registry import register_model
from .transformer import Transformer

CMAP = colormaps.get_cmap("turbo")
CMAP.set_bad("gray")


class BoldGPT(nn.Module):
    def __init__(
        self,
        patchify: MaskedPatchify,
        tokenizer: Optional[KMeansTokenizer],
        decoder: Transformer,
        shuffle: bool = True,
    ):
        super().__init__()
        self.shuffle = shuffle
        self.is_categorical = tokenizer is not None

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
        activity = batch["activity"]
        if self.decoder.with_sub_embed:
            sub_indices = batch["subject_id"]
        else:
            sub_indices = None

        patches = self.patchify(activity)
        B, N = patches.shape[:2]
        device = patches.device

        if self.shuffle and self.training:
            order, ranking = random_order(B, N, device=device)
            patches = shuffle(patches, order)
            patch_mask = self.patchify.patch_mask.expand(B, -1, -1)
            patch_mask = shuffle(patch_mask, order)
        else:
            order = ranking = None
            patch_mask = None

        if self.is_categorical:
            tokens = self.tokenizer(patches)
        else:
            tokens = None

        output = self.decoder(patches, sub_indices=sub_indices, order=order)
        loss = self.loss_fn(output, patches, tokens, patch_mask=patch_mask)

        state = dict(
            patches=patches,
            order=order,
            ranking=ranking,
            tokens=tokens,
            output=output.detach(),
        )
        return loss, state

    def loss_fn(
        self,
        output: torch.Tensor,
        patches: torch.Tensor,
        tokens: Optional[torch.Tensor],
        patch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_categorical:
            loss = F.cross_entropy(output.flatten(0, 1), tokens.flatten())
        else:
            # Mask output by patch_mask
            # We assume the patches are already masked
            # Note that if patches are shuffled, the mask must be shuffled as well.
            if patch_mask is None:
                patch_mask = self.patchify.patch_mask
            output = output * patch_mask
            loss = F.mse_loss(output, patches)
        return loss

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
        activity = batch["activity"]

        patches = state["patches"]
        order = state["order"]
        ranking = state["ranking"]
        tokens = state["tokens"]
        output = state["output"]

        B, N = patches.shape[:2]
        device = patches.device

        # Ensure activity is masked
        mask = self.patchify.mask
        activity = mask * activity

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
        target_error = F.mse_loss(
            activity[:, mask], target[:, mask], reduction="none"
        ).mean(dim=1)
        recon_error = F.mse_loss(
            activity[:, mask], recon[:, mask], reduction="none"
        ).mean(dim=1)

        examples = {
            "subject_id": subind,
            "nsd_id": nsdind,
            "activity": activity,
            "order_map": order_map,
            "target": target,
            "recon": recon,
            "target_error": target_error,
            "recon_error": recon_error,
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
        activity = examples["activity"]
        order_map = examples["order_map"]
        target = examples["target"]
        recon = examples["recon"]
        target_error = examples["target_error"]
        recon_error = examples["recon_error"]

        B = activity.shape[0]
        plotw = 3.0
        ploth = 3.5
        nr = B
        nc = 4
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
            _imshow(activity[ii])
            plt.text(
                0.5,
                0.98,
                "Activity",
                ha="center",
                va="top",
                transform=tform,
                **textdict,
            )

            plt.sca(axs[ii, 2])
            tform = axs[ii, 2].transAxes
            _imshow(target[ii])
            plt.text(
                0.5, 0.98, "Target", ha="center", va="top", transform=tform, **textdict
            )
            plt.text(
                0.98,
                0.0,
                f"mse={target_error[ii]:.2e}",
                ha="right",
                va="bottom",
                transform=tform,
                **textdict,
            )

            plt.sca(axs[ii, 3])
            tform = axs[ii, 3].transAxes
            _imshow(recon[ii])
            plt.text(
                0.5, 0.98, "Pred", ha="center", va="top", transform=tform, **textdict
            )
            plt.text(
                0.98,
                0.0,
                f"mse={recon_error[ii]:.2e}",
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
        return f"is_categorical={self.is_categorical}, shuffle={self.shuffle}"


def _imshow(img: np.ndarray, **kwargs):
    kwargs = {"interpolation": "nearest", "cmap": CMAP, **kwargs}
    img = np.where(img == 0, np.nan, img)
    plt.imshow(img, **kwargs)
    plt.axis("off")


def _create_bold_gpt(
    *,
    mask: Optional[np.ndarray] = None,
    patch_size: int = 10,
    ordering: str = "radial",
    categorical: bool = True,
    with_sub_embed: bool = True,
    vocab_size: int = 1024,
    shuffle: bool = True,
    num_subs: int = 8,
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

    if mask is None:
        mask = load_nsd_flat_mask()
    patchify = MaskedPatchify(mask, patch_size=patch_size, ordering=ordering)

    if categorical:
        tokenizer = KMeansTokenizer(vocab_size=vocab_size, dim=patchify.dim)
    else:
        tokenizer = None

    decoder = Transformer(
        num_patches=patchify.num_patches,
        in_features=patchify.dim,
        num_subs=num_subs,
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

    model = BoldGPT(patchify, tokenizer, decoder, shuffle=shuffle)
    return model


@register_model
def boldgpt_tiny_patch10(**kwargs):
    model_kwargs = dict(
        patch_size=10, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0
    )
    return _create_bold_gpt(**kwargs, **model_kwargs)


@register_model
def boldgpt_small_patch10(**kwargs):
    model_kwargs = dict(
        patch_size=10, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0
    )
    return _create_bold_gpt(**kwargs, **model_kwargs)


@register_model
def boldgpt_base_patch10(**kwargs):
    model_kwargs = dict(
        patch_size=10, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0
    )
    return _create_bold_gpt(**kwargs, **model_kwargs)
