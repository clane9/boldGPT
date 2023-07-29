import math

import torch
from einops.layers.torch import Rearrange
from torch import nn


class MaskedPatchify(nn.Module):
    """
    Patchify images restricted to a mask, shape (H, W).
    """

    mask: torch.Tensor
    patch_mask: torch.Tensor
    patch_indices: torch.Tensor

    def __init__(self, mask: torch.Tensor, patch_size: int = 8):
        assert mask.ndim == 2, f"Invalid mask shape {mask.shape}; expected (H, W)"

        super().__init__()
        mask = torch.as_tensor(mask > 0, dtype=torch.bool)
        H, W = mask.shape

        self.patch_size = patch_size
        self.height = H
        self.width = W
        self.dim = patch_size**2

        padding = _infer_padding(H, W, multiple=patch_size)
        self.padding = padding

        self.pad = nn.ConstantPad2d(padding, 0.0)
        self.to_patches = Rearrange(
            "b (h p1) (w p2) -> b (h w) (p1 p2)", p1=patch_size, p2=patch_size
        )
        self.from_patches = Rearrange(
            "b (h w) (p1 p2) -> b (h p1) (w p2)",
            h=math.ceil(H / patch_size),
            w=math.ceil(W / patch_size),
            p1=patch_size,
            p2=patch_size,
        )

        # Patchified mask
        patch_mask = mask.expand(1, -1, -1)
        patch_mask = self.pad(patch_mask)
        patch_mask = self.to_patches(patch_mask).squeeze(0)
        self.num_total_patches = len(patch_mask)

        # Indices of patches intersecting the mask
        patch_indices = torch.where(torch.any(patch_mask, dim=1))[0]
        patch_mask = patch_mask[patch_indices]
        self.num_patches = len(patch_indices)

        self.register_buffer("mask", mask)
        self.register_buffer("patch_mask", patch_mask)
        self.register_buffer("patch_indices", patch_indices)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert images, shape (N, H, W), to patches, shape (N, L, D).
        """
        squeeze = False
        if images.ndim == 2:
            images = images[None, ...]
            squeeze = True

        images = self.pad(images)
        patches = self.to_patches(images)
        patches = patches[..., self.patch_indices, :]

        if squeeze:
            patches = patches.squeeze(0)
        return patches

    def inverse(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Convert patches, shape (N, L, D), back to images, shape (N, H, W).
        """
        squeeze = False
        if patches.ndim == 2:
            patches = patches[None, ...]
            squeeze = True

        N, _, D = patches.shape
        dtype = patches.dtype
        device = patches.device

        # Inverse masking
        expanded = torch.zeros(
            (N, self.num_total_patches, D), dtype=dtype, device=device
        )
        expanded[..., self.patch_indices, :] = patches

        # Crop to original size
        images = self.from_patches(expanded)
        left, _, top, _ = self.padding
        images = images[..., top : top + self.height, left : left + self.width]

        if squeeze:
            images = images.squeeze(0)
        return images

    def extra_repr(self) -> str:
        return (
            f"patch_size={self.patch_size}, height={self.height}, "
            f"width={self.width}, num_patches={self.num_patches}"
        )


def _infer_padding(height: int, width: int, multiple: int = 8):
    """
    Get the pytorch format padding `[left, right, top, bottom]` so that the image shape
    is a multiple of `multiple`.
    """
    new_h = multiple * math.ceil(height / multiple)
    new_w = multiple * math.ceil(width / multiple)
    padding = [new_h - height, new_w - width]
    # [left, right, top, bottom]
    padding = [p for pad in reversed(padding) for p in [pad // 2, pad - pad // 2]]
    return padding
