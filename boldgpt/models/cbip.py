import logging
import math
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
from boldgpt.distributed_utils import gather_from_all
from boldgpt.patching import MaskedPatchify
from boldgpt.shuffle import permute, random_order

from . import constants as C
from .registry import register_model
from .transformer import Transformer
from .utils import imshow, to_numpy


class CBIP(nn.Module):
    def __init__(
        self,
        patchify: MaskedPatchify,
        bold_encoder: Transformer,
        image_encoder: nn.Module,
        mask_ratio: Optional[float] = None,
        siglip: bool = False,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.siglip = siglip
        self.with_sub_embed = bold_encoder.with_sub_embed

        self.patchify = patchify
        self.bold_encoder = bold_encoder
        self.image_encoder = image_encoder

        self.loss_module = SigLIPLoss() if siglip else CLIPLoss()

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        images = batch["image"]
        activity = batch["activity"]
        subid = batch["subject_id"] if self.with_sub_embed else None

        patches = self.patchify(activity)
        B, N = patches.shape[:2]
        device = patches.device

        if self.mask_ratio and self.training:
            num_keep = int(N * (1 - self.mask_ratio))
            order, ranking = random_order(B, N, device=device)
            patches = permute(patches, order)
            input_patches = patches[:, :num_keep]
            input_order = order[:, :num_keep]
        else:
            num_keep = order = ranking = input_order = None
            input_patches = patches

        output = self.bold_encoder(input_patches, sub_indices=subid, order=input_order)
        target = self.image_encoder(images)

        output = F.normalize(output, dim=-1)
        target = F.normalize(target, dim=-1)

        state = dict(
            patches=patches,
            order=order,
            ranking=ranking,
            num_keep=num_keep,
            target=target,
            output=output,
        )
        return output, state

    def loss_fn(
        self,
        batch: Dict[str, torch.Tensor],
        output: torch.Tensor,
        state: Dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        return self.loss_module(output, state["target"], sample_ids=batch["nsd_id"])

    @torch.no_grad()
    def prepare_examples(
        self,
        batch: Dict[str, torch.Tensor],
        state: Dict[str, torch.Tensor],
    ) -> Dict[str, np.ndarray]:
        """
        Prepare a batch of examples for figure generation.
        """
        images = batch["image"]
        activity = batch["activity"]
        subid = batch["subject_id"] if "subject_id" in batch else None
        imgid = batch["nsd_id"]

        ranking = state["ranking"]
        num_keep = state["num_keep"]
        target = state["target"]
        output = state["output"]

        target = target.detach()
        output = output.detach()

        # Ensure activity are masked
        activity = self.patchify.mask * activity

        # Get retrieval images in batch
        similarity = output @ target.t()
        ret_sim, ret_indices = torch.max(similarity, dim=1)
        ret_images = images[ret_indices]

        # Mask of hidden patches
        if num_keep is not None:
            hidden_mask = ranking >= num_keep
            hidden_mask = self.patchify.inverse_vector(hidden_mask)
        else:
            hidden_mask = None

        examples = {
            "subject_id": subid,
            "image_id": imgid,
            "images": images,
            "activity": activity,
            "ret_sim": ret_sim,
            "ret_images": ret_images,
            "hidden_mask": hidden_mask,
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
        activity = examples["activity"]
        ret_sim = examples["ret_sim"]
        ret_images = examples["ret_images"]
        hidden_mask = examples["hidden_mask"]

        is_masked = hidden_mask is not None
        if is_masked:
            hidden_mask_rgba = np.where(
                hidden_mask[:, None],
                np.array([1.0, 1.0, 1.0, 0.6]).reshape(1, 4, 1, 1),
                np.zeros((1, 4, 1, 1)),
            )
            hidden_mask_rgba = self.patchify.mask.cpu().numpy() * hidden_mask_rgba
        else:
            hidden_mask_rgba = None

        img_shape = activity.shape[-2:]
        plotw = 3.0
        ploth = 3.5
        nr = num_examples
        nc = 3
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

            plt.sca(axs[ii, col])
            tform = axs[ii, col].transAxes
            imshow(activity[ii])
            if is_masked:
                imshow(hidden_mask_rgba[ii])
            plt.text(
                0.5,
                0.98,
                "Activity",
                ha="center",
                va="top",
                transform=tform,
                **textdict,
            )
            col += 1

            plt.sca(axs[ii, col])
            tform = axs[ii, col].transAxes
            imshow(images[ii], img_shape=img_shape)
            plt.text(
                0.5,
                0.98,
                "Image",
                ha="center",
                va="top",
                transform=tform,
                **textdict,
            )
            col += 1

            plt.sca(axs[ii, col])
            tform = axs[ii, col].transAxes
            imshow(ret_images[ii], img_shape=img_shape)
            plt.text(
                0.5,
                0.98,
                "Ret Image",
                ha="center",
                va="top",
                transform=tform,
                **textdict,
            )
            plt.text(
                0.98,
                0.0,
                f"Sim={ret_sim[ii]:.3f}",
                ha="right",
                va="bottom",
                transform=tform,
                **textdict,
            )
            col += 1

        plt.tight_layout(pad=0.2, h_pad=0.05)

        if fname is not None:
            plt.savefig(fname, bbox_inches="tight")
        return f

    def get_data_config(self) -> DataConfig:
        cfg = {}
        cfg["dataset"] = "NSD-Flat"
        cfg["columns"] = ["subject_id", "nsd_id", "image", "activity"]
        timm_data_config = resolve_model_data_config(self.image_encoder)
        cfg["img_size"] = timm_data_config["input_size"][1]
        cfg["img_mean"] = timm_data_config["mean"]
        cfg["img_std"] = timm_data_config["std"]
        cfg["interp_mode"] = str_to_interp_mode(timm_data_config["interpolation"])
        return DataConfig(**cfg)

    def extra_repr(self) -> str:
        return f"mask_ratio={self.mask_ratio}, siglip={self.siglip}"


class CLIPLoss(nn.Module):
    """
    OpenAI CLIP loss.
    """

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([math.log(1 / 0.07)]))

    def forward(
        self,
        source_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        sample_ids: Optional[torch.Tensor] = None,
    ):
        source_embeddings = gather_from_all(source_embeddings)
        target_embeddings = gather_from_all(target_embeddings)
        B = source_embeddings.size(0)

        logits = self.scale.exp() * source_embeddings @ target_embeddings.t()

        if sample_ids is not None:
            sample_ids = gather_from_all(sample_ids)
        else:
            sample_ids = torch.arange(B, device=logits.device)
        id_mask = sample_ids[:, None] == sample_ids
        probs = F.softmax(torch.where(id_mask, 1.0, -float("inf")), dim=-1)

        loss = 0.5 * (
            F.cross_entropy(logits, probs) + F.cross_entropy(logits.t(), probs)
        )
        return loss


class SigLIPLoss(nn.Module):
    """
    SigLIP loss from https://arxiv.org/abs/2303.15343.
    """

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([math.log(10.0)]))
        self.bias = nn.Parameter(torch.tensor([-10.0]))

    def forward(
        self,
        source_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        sample_ids: Optional[torch.Tensor] = None,
    ):
        source_embeddings = gather_from_all(source_embeddings)
        target_embeddings = gather_from_all(target_embeddings)
        B = source_embeddings.size(0)

        logits = self.scale.exp() * source_embeddings @ target_embeddings.t()
        logits = logits + self.bias

        if sample_ids is not None:
            sample_ids = gather_from_all(sample_ids)
        else:
            sample_ids = torch.arange(B, device=logits.device)
        id_mask = (sample_ids[:, None] == sample_ids).to(logits.dtype)
        labels = 2.0 * id_mask - 1.0

        loss = -F.logsigmoid(labels * logits).sum() / B
        return loss


def _create_cbip(
    *,
    mask: Optional[np.ndarray] = None,
    patch_size: int = 10,
    ordering: str = "radial",
    with_sub_embed: bool = True,
    mask_ratio: Optional[float] = 0.5,
    siglip: bool = False,
    encoder_name: str = "vit_large_patch14_clip_224.openai",
    encoder_kwargs: Optional[Dict[str, Any]] = None,
    num_subs: int = 1024,
    num_registers: int = 8,
    global_pool: Optional[Literal["avg", "token", "reg"]] = "reg",
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

    encoder_kwargs = encoder_kwargs or {}
    encoder_kwargs = {"pretrained": True, **encoder_kwargs}
    image_encoder = timm.create_model(encoder_name, **encoder_kwargs)

    bold_encoder = Transformer(
        num_patches=patchify.num_patches,
        in_features=patchify.dim,
        num_subs=num_subs,
        num_registers=num_registers,
        num_classes=image_encoder.num_classes,
        global_pool=global_pool,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        with_sub_embed=with_sub_embed,
        with_next_pos=False,
        with_cross=False,
        is_causal=False,
        is_masked=False,
        drop_rate=drop_rate,
        sub_drop_rate=sub_drop_rate,
        proj_drop_rate=proj_drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
    )

    model = CBIP(
        patchify=patchify,
        bold_encoder=bold_encoder,
        image_encoder=image_encoder,
        mask_ratio=mask_ratio,
        siglip=siglip,
    )
    return model


@register_model
def cbip_tiny_patch10(**kwargs):
    return _create_cbip(patch_size=10, **C.TINY_ARCH_KWARGS, **kwargs)


@register_model
def cbip_small_patch10(**kwargs):
    return _create_cbip(patch_size=10, **C.SMALL_ARCH_KWARGS, **kwargs)


@register_model
def cbip_base_patch10(**kwargs):
    return _create_cbip(patch_size=10, **C.BASE_ARCH_KWARGS, **kwargs)
