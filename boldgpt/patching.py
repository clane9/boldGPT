import math

import torch
from einops.layers.torch import Rearrange
from torch import nn


class MaskedPatchify(nn.Module):
    """
    Patchify images restricted to a mask, shape (H, W).
    """

    def __init__(
        self,
        mask: torch.Tensor,
        patch_size: int = 8,
        in_chans: int = 3,
    ):
        super().__init__()
        mask = torch.as_tensor(mask > 0)
        assert mask.ndim == 2, f"Invalid mask shape {mask.shape}; expected (H, W)"
        H, W = mask.shape

        self.patch_size = patch_size
        self.in_chans = in_chans
        self.height = H
        self.width = W
        self.dim = patch_size**2 * in_chans

        padding = _infer_padding(H, W, multiple=patch_size)
        self.padding = padding

        self.pad = nn.ConstantPad2d(padding, 0.0)
        self.to_patches = Rearrange(
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
        )
        self.from_patches = Rearrange(
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=math.ceil(H / patch_size),
            w=math.ceil(W / patch_size),
            p1=patch_size,
            p2=patch_size,
            c=in_chans,
        )

        # Patchified mask
        patch_mask = mask.expand(1, in_chans, -1, -1)
        patch_mask = self.pad(patch_mask)
        patch_mask = self.to_patches(patch_mask).squeeze(0)

        # Indices of patches intersecting the mask
        patch_indices = torch.where(torch.any(patch_mask, dim=1))[0]
        self.num_patches = len(patch_indices)

        self.register_buffer("mask", mask)
        self.register_buffer("patch_mask", patch_mask)
        self.register_buffer("patch_indices", patch_indices)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Convert images, shape (N,[ C,] H, W), to patches, shape (N, L, D).
        """
        if self.in_chans == 1 and img.ndim == 3:
            img = img.unsqueeze(1)
        assert img.shape[-3:] == (self.in_chans, self.height, self.width)

        img = self.pad(img)
        patches = self.to_patches(img)
        patches = patches[..., self.patch_indices, :]
        return patches

    def inverse(self, patches: torch.Tensor, squeeze: bool = True) -> torch.Tensor:
        """
        Convert patches, shape (N, L, D), back to images, shape (N, C, H, W).
        """
        assert patches.shape[-2:] == (self.num_patches, self.dim)
        N, _, D = patches.shape

        # Inverse masking
        num_total_patches = len(self.patch_mask)
        expanded = torch.zeros(
            (N, num_total_patches, D), dtype=patches.dtype, device=patches.device
        )
        expanded[:, self.patch_indices, :] = patches

        # Crop to original size
        img = self.from_patches(expanded)
        left, _, top, _ = self.padding
        img = img[..., top : top + self.height, left : left + self.width]

        if squeeze and self.in_chans == 1:
            img = img.squeeze(1)
        return img

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
