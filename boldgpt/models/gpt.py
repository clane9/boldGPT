import logging
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import timm
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from timm.data import resolve_model_data_config, str_to_interp_mode
from torch import nn

from boldgpt.data import DataConfig, load_nsd_flat_mask
from boldgpt.patching import MaskedPatchify
from boldgpt.shuffle import permute, random_order
from boldgpt.tokenizer import KMeansTokenizer

from . import constants as C
from .generate import generate
from .registry import register_configs, register_model
from .transformer import Transformer
from .utils import imshow, infer_embed_dim, r2_score, to_numpy


class IGPT(nn.Module):
    def __init__(
        self,
        patchify: MaskedPatchify,
        tokenizer: Optional[KMeansTokenizer],
        decoder: Transformer,
        encoder: Optional[nn.Module] = None,
        shuffle: bool = True,
        modality: Literal["image", "bold"] = "bold",
    ):
        super().__init__()
        self.shuffle = shuffle
        self.modality = modality
        self.is_categorical = tokenizer is not None
        self.is_seq2seq = encoder is not None
        self.with_sub_embed = decoder.with_sub_embed

        self.patchify = patchify
        if tokenizer is None:
            self.register_module("tokenizer", None)
        else:
            self.tokenizer = tokenizer
        self.decoder = decoder

        if encoder is None:
            self.register_module("encoder", None)
        else:
            # TODO: May want to use a `FeatureExtractor` to extract intermediate rather
            # than final features. But currently the forward hooks interfere with model
            # compilation.
            self.encoder = encoder

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        # Unpack data
        images = batch["activity"] if self.modality == "bold" else batch["image"]
        if self.is_seq2seq:
            inputs = batch["image"] if self.modality == "bold" else batch["activity"]
        sub_indices = batch["subject_id"] if self.with_sub_embed else None

        # Get patches and optionally tokens
        patches = self.patchify(images)
        B, N = patches.shape[:2]
        device = patches.device
        if self.shuffle and self.training:
            order, ranking = random_order(B, N, device=device)
            patches = permute(patches, order)
        else:
            order = ranking = None
        tokens = self.tokenizer(patches) if self.is_categorical else None

        # Get encoder context
        if self.is_seq2seq:
            context = self.encoder.forward_features(inputs)
        else:
            context = None

        # Forward pass
        output = self.decoder(
            patches, sub_indices=sub_indices, context=context, order=order
        )
        # Drop the trailing EOS token (and any registers)
        output = output[:, :N]

        state = dict(
            patches=patches,
            order=order,
            ranking=ranking,
            tokens=tokens,
            context=context,
            output=output,
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
            patches = state["patches"]
            order = state["order"]
            mask = self.patchify.patch_mask.expand_as(patches)
            # If patches are shuffled, the mask must be shuffled as well.
            if order is not None:
                mask = permute(mask, order)
            loss = torch.sum(mask * (output - patches) ** 2) / mask.sum()
        return loss

    @torch.no_grad()
    def generate(
        self,
        batch: Dict[str, torch.Tensor],
        prompt_fraction: float = 0.25,
        shuffle: bool = False,
        order: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        if not self.shuffle and (shuffle or order is not None):
            raise ValueError(
                "Must train with shuffled patches to generate with "
                "shuffled/custom order"
            )

        images = batch["activity"] if self.modality == "bold" else batch["image"]
        if self.is_seq2seq:
            inputs = batch["image"] if self.modality == "bold" else batch["activity"]
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

        sample_mask = torch.ones(B, N, device=prompt.device, dtype=torch.bool)
        sample_mask[:, :prompt_length] = False

        # Get encoder context
        if self.is_seq2seq and context is None:
            context = self.encoder.forward_features(inputs)

        sample = generate(
            model=self.decoder,
            prompt=prompt,
            sub_indices=sub_indices,
            context=context,
            order=order,
            tokenizer=self.tokenizer,
            patch_mask=self.patchify.patch_mask,
            temperature=temperature,
            top_k=top_k,
            use_cache=use_cache,
        )

        if order is not None:
            sample_mask = permute(sample_mask, ranking)
            sample = permute(sample, ranking)
        sample_mask = self.patchify.inverse_vector(sample_mask)
        sample = self.patchify.inverse(sample)

        state = {
            "patches": patches,
            "order": order,
            "ranking": ranking,
            "prompt_length": prompt_length,
            "context": context,
            "sample": sample,
            "sample_mask": sample_mask,
        }
        return sample, state

    @torch.no_grad()
    def prepare_examples(
        self,
        batch: Dict[str, torch.Tensor],
        state: Dict[str, torch.Tensor],
    ) -> Dict[str, np.ndarray]:
        """
        Prepare a batch of examples for figure generation.
        """
        subid = batch["subject_id"] if "subject_id" in batch else None
        if "nsd_id" in batch:
            imgid = batch["nsd_id"]
        elif "image_id" in batch:
            imgid = batch["image_id"]
        else:
            imgid = None
        images = batch["activity"] if self.modality == "bold" else batch["image"]
        if self.is_seq2seq:
            inputs = batch["image"] if self.modality == "bold" else batch["activity"]
        else:
            inputs = None

        patches = state["patches"]
        order = state["order"]
        ranking = state["ranking"]
        tokens = state["tokens"]
        context = state["context"]
        output = state["output"]

        if context is not None:
            context = context.detach()
        output = output.detach()

        B, N = patches.shape[:2]
        device = patches.device

        # Ensure images are masked
        mask = self.patchify.expanded_mask
        images = mask * images

        # Unshuffle if necessary
        if order is not None:
            patches = permute(patches, ranking)
            if tokens is not None:
                tokens = permute(tokens, ranking)
            output = permute(output, ranking)
        else:
            ranking = torch.arange(N, device=device).expand(B, -1)

        # Invert order, target, reconstruction and compute R^2
        order_map = self.patchify.inverse_vector(ranking.float() + 1.0)
        # Remove channels dimension
        if self.modality == "image":
            order_map = order_map[:, 0]
        if self.is_categorical:
            target = self.tokenizer.inverse(tokens)
            pred = torch.argmax(output, dim=-1)
            recon = self.tokenizer.inverse(pred)
        else:
            target = patches
            recon = output
        target = self.patchify.inverse(target)
        recon = self.patchify.inverse(recon)
        target_r2 = r2_score(target[:, mask], images[:, mask], reduction="none")
        recon_r2 = r2_score(recon[:, mask], images[:, mask], reduction="none")

        sample, sample_state = self.generate(
            batch=batch,
            prompt_fraction=0.0 if self.is_seq2seq else 0.25,
            order=order,
            context=context,
        )
        sample_mask = sample_state["sample_mask"]
        sample_r2 = r2_score(
            (sample_mask * sample)[:, mask],
            (sample_mask * images)[:, mask],
            reduction="none",
        )

        examples = {
            "subject_id": subid,
            "image_id": imgid,
            "images": images,
            "inputs": inputs,
            "order_map": order_map,
            "target": target,
            "recon": recon,
            "sample": sample,
            "sample_mask": sample_mask,
            "target_r2": target_r2,
            "recon_r2": recon_r2,
            "sample_r2": sample_r2,
        }
        examples = {k: to_numpy(v) for k, v in examples.items()}
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
        subid = examples["subject_id"]
        imgid = examples["image_id"]
        images = examples["images"]
        inputs = examples["inputs"]
        order_map = examples["order_map"]
        target = examples["target"]
        recon = examples["recon"]
        sample = examples["sample"]
        sample_mask = examples["sample_mask"]
        target_r2 = examples["target_r2"]
        recon_r2 = examples["recon_r2"]
        sample_r2 = examples["sample_r2"]

        # Remove channels dimension
        if self.modality == "image":
            sample_mask = sample_mask[:, 0]
        sample_mask_rgba = np.where(
            sample_mask[:, None],
            np.zeros((1, 4, 1, 1)),
            np.array([1.0, 1.0, 1.0, 0.5]).reshape(1, 4, 1, 1),
        )
        sample_mask_rgba = self.patchify.mask.cpu().numpy() * sample_mask_rgba

        img_shape = images.shape[-2:]
        plotw = 3.0
        ploth = 3.5
        nr = num_examples
        # Extra plots depending on model
        nc = 4 + self.is_seq2seq + self.is_categorical
        f, axs = plt.subplots(nr, nc, figsize=(nc * plotw, nr * ploth), squeeze=False)

        textdict = {
            "fontsize": 10,
            "color": "w",
            "bbox": {
                "boxstyle": "square",
                "fc": (0.5, 0.5, 0.5),
                "ec": "none",
                "pad": 0,
            },
        }

        for ii in range(num_examples):
            label = []
            if subid is not None:
                label.append(f"s{subid[ii]+1:02d}")
            if imgid is not None:
                label.append(f"{imgid[ii]:05d}")
            label = " ".join(label)
            col = 0

            if self.is_seq2seq:
                plt.sca(axs[ii, col])
                tform = axs[ii, col].transAxes
                imshow(inputs[ii], img_shape=img_shape)
                plt.text(
                    0.5,
                    0.98,
                    "Image" if self.modality == "bold" else "Activity",
                    ha="center",
                    va="top",
                    transform=tform,
                    **textdict,
                )
                col += 1

            plt.sca(axs[ii, col])
            tform = axs[ii, col].transAxes
            imshow(images[ii])
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
                0.04, 0.0, label, ha="left", va="bottom", transform=tform, **textdict
            )
            col += 1

            if self.is_categorical:
                plt.sca(axs[ii, col])
                tform = axs[ii, col].transAxes
                imshow(target[ii])
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
                col += 1

            plt.sca(axs[ii, col])
            tform = axs[ii, col].transAxes
            imshow(recon[ii])
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
            col += 1

            plt.sca(axs[ii, col])
            tform = axs[ii, col].transAxes
            imshow(sample[ii])
            imshow(sample_mask_rgba[ii])
            plt.text(
                0.5, 0.98, "Sample", ha="center", va="top", transform=tform, **textdict
            )
            plt.text(
                0.98,
                0.0,
                f"R2={sample_r2[ii]:.3f}",
                ha="right",
                va="bottom",
                transform=tform,
                **textdict,
            )
            col += 1

            plt.sca(axs[ii, col])
            tform = axs[ii, col].transAxes
            imshow(order_map[ii])
            plt.text(
                0.5, 0.98, "Order", ha="center", va="top", transform=tform, **textdict
            )
            col += 1

        plt.tight_layout(pad=0.2, h_pad=0.05)

        if fname is not None:
            plt.savefig(fname, bbox_inches="tight")
        return f

    def get_data_config(self) -> DataConfig:
        cfg = {}
        if self.modality == "bold" and not self.is_seq2seq:
            # boldgpt
            cfg["dataset"] = "NSD-Flat"
            cfg["columns"] = ["subject_id", "nsd_id", "activity"]
        elif self.modality == "bold" and self.is_seq2seq:
            # image2bold
            cfg["dataset"] = "NSD-Flat"
            cfg["columns"] = ["subject_id", "nsd_id", "image", "activity"]
            timm_data_config = resolve_model_data_config(self.encoder)
            cfg["img_size"] = timm_data_config["input_size"][1]
            cfg["img_mean"] = timm_data_config["mean"]
            cfg["img_std"] = timm_data_config["std"]
            cfg["interp_mode"] = str_to_interp_mode(timm_data_config["interpolation"])
        elif self.modality == "image" and not self.is_seq2seq:
            # imagegpt
            cfg["dataset"] = "COCO"
            cfg["columns"] = ["image_id", "image"]
            cfg["img_size"] = self.patchify.mask.shape[1]
        elif self.modality == "image" and self.is_seq2seq:
            # bold2image
            cfg["dataset"] = "NSD-Flat"
            cfg["columns"] = ["subject_id", "nsd_id", "image", "activity"]
            cfg["img_size"] = self.patchify.mask.shape[1]
        return DataConfig(**cfg)

    def extra_repr(self) -> str:
        return (
            f"is_categorical={self.is_categorical}, "
            f"is_seq2seq={self.is_seq2seq}, "
            f"shuffle={self.shuffle}, "
            f"modality={self.modality}"
        )


def _create_boldgpt(
    *,
    mask: Optional[np.ndarray] = None,
    patch_size: int = 10,
    categorical: bool = False,
    ordering: str = "radial",
    with_sub_embed: bool = True,
    vocab_size: int = 1024,
    shuffle: bool = True,
    num_subs: int = 1024,
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

    model = IGPT(
        patchify=patchify,
        tokenizer=tokenizer,
        decoder=decoder,
        shuffle=shuffle,
        modality="bold",
    )
    return model


def _create_imagegpt(
    *,
    img_size: int = 224,
    patch_size: int = 10,
    ordering: str = "reverse_radial",
    categorical: bool = False,
    vocab_size: int = 1024,
    shuffle: bool = True,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    mlp_ratio: float = 4.0,
    drop_rate: float = 0.0,
    proj_drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    **kwargs,
):
    if kwargs:
        logging.warning("Extra unused kwargs: %s", kwargs)

    mask = torch.ones(img_size, img_size, dtype=torch.bool)
    patchify = MaskedPatchify(
        mask, num_channels=3, patch_size=patch_size, ordering=ordering
    )
    if categorical:
        tokenizer = KMeansTokenizer(vocab_size=vocab_size, dim=patchify.dim)
    else:
        tokenizer = None

    decoder = Transformer(
        num_patches=patchify.num_patches,
        in_features=patchify.dim,
        num_classes=(vocab_size if categorical else patchify.dim),
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        with_sub_embed=False,
        with_next_pos=shuffle,
        with_cross=False,
        is_causal=True,
        is_masked=False,
        drop_rate=drop_rate,
        proj_drop_rate=proj_drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
    )

    model = IGPT(
        patchify=patchify,
        tokenizer=tokenizer,
        decoder=decoder,
        shuffle=shuffle,
        modality="image",
    )
    return model


def _create_image2bold(
    *,
    mask: Optional[np.ndarray] = None,
    patch_size: int = 10,
    ordering: str = "radial",
    categorical: bool = False,
    with_sub_embed: bool = True,
    vocab_size: int = 1024,
    shuffle: bool = True,
    encoder_name: str = "eva02_base_patch14_224.mim_in22k",
    encoder_kwargs: Optional[Dict[str, Any]] = None,
    num_subs: int = 1024,
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

    encoder_kwargs = encoder_kwargs or {}
    encoder_kwargs = {"pretrained": True, **encoder_kwargs}
    encoder = timm.create_model(encoder_name, **encoder_kwargs)
    encoder_dim = infer_embed_dim(encoder_name)

    decoder = Transformer(
        num_patches=patchify.num_patches,
        in_features=patchify.dim,
        num_subs=num_subs,
        num_classes=(vocab_size if categorical else patchify.dim),
        embed_dim=embed_dim,
        context_dim=encoder_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        with_sub_embed=with_sub_embed,
        with_next_pos=shuffle,
        with_cross=True,
        is_causal=True,
        is_masked=False,
        drop_rate=drop_rate,
        sub_drop_rate=sub_drop_rate,
        proj_drop_rate=proj_drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
    )

    model = IGPT(
        patchify=patchify,
        tokenizer=tokenizer,
        decoder=decoder,
        encoder=encoder,
        shuffle=shuffle,
        modality="bold",
    )
    return model


@register_model
def boldgpt_tiny_patch10(**kwargs):
    return _create_boldgpt(patch_size=10, **C.TINY_ARCH_KWARGS, **kwargs)


@register_model
def boldgpt_small_patch10(**kwargs):
    return _create_boldgpt(patch_size=10, **C.SMALL_ARCH_KWARGS, **kwargs)


@register_model
def boldgpt_base_patch10(**kwargs):
    return _create_boldgpt(patch_size=10, **C.BASE_ARCH_KWARGS, **kwargs)


@register_model
def imagegpt_tiny_patch16(**kwargs):
    return _create_imagegpt(patch_size=16, **C.TINY_ARCH_KWARGS, **kwargs)


@register_model
def imagegpt_small_patch16(**kwargs):
    return _create_imagegpt(patch_size=16, **C.SMALL_ARCH_KWARGS, **kwargs)


@register_model
def imagegpt_base_patch16(**kwargs):
    return _create_imagegpt(patch_size=16, **C.BASE_ARCH_KWARGS, **kwargs)


@register_model
def image2bold_tiny_patch10(**kwargs):
    return _create_image2bold(patch_size=10, **C.TINY_ARCH_KWARGS, **kwargs)


@register_model
def image2bold_small_patch10(**kwargs):
    return _create_image2bold(patch_size=10, **C.SMALL_ARCH_KWARGS, **kwargs)


@register_model
def image2bold_base_patch10(**kwargs):
    return _create_image2bold(patch_size=10, **C.BASE_ARCH_KWARGS, **kwargs)


CONFIGS = {
    "boldgpt_small_patch10.kmq": {
        "has_weights": True,
        "kwargs": {"categorical": True},
    },
    "boldgpt_small_patch10.cont": {
        "has_weights": True,
        "kwargs": {"categorical": False},
    },
}

register_configs(CONFIGS)
