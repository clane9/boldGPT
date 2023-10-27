import math
from typing import Union

import torch
from einops.layers.torch import Rearrange
from torch import nn

ORDERINGS = ("raster", "radial", "reverse_radial", "random")


class MaskedPatchify(nn.Module):
    """
    Patchify images restricted to a mask, shape (H, W).

    Supports different orderings for the patches:

    - raster: original image raster order
    - radial: patches sorted by distance from mask centroid
    - reverse_radial: reverse of radial
    - random: fixed random patch ordering

    Or you can pass a custom priority tensor, shape (H, W), to determine the order.
    """

    # TODO: could add support for a custom ordering provided via a "priority" map

    mask: torch.Tensor
    patch_mask: torch.Tensor
    patch_indices: torch.Tensor
    order: torch.Tensor
    ranking: torch.Tensor

    def __init__(
        self,
        mask: torch.Tensor,
        patch_size: int = 8,
        num_channels: int = 1,
        ordering: Union[str, torch.Tensor] = "raster",
    ):
        assert mask.ndim == 2, f"Invalid mask shape {mask.shape}; expected (H, W)"
        assert (
            ordering in ORDERINGS
        ), f"Unknown ordering {ordering}; expected one of {ORDERINGS}"

        super().__init__()
        mask = torch.as_tensor(mask > 0, dtype=torch.bool)
        H, W = mask.shape
        device = mask.device

        self.patch_size = patch_size
        self.num_channels = num_channels
        self.ordering = ordering
        self.height = H
        self.width = W
        self.dim = num_channels * patch_size**2

        # Padding to multiple of patch size
        self.padding = padding = _infer_padding(H, W, multiple=patch_size)
        self.pad = nn.ConstantPad2d(padding, 0.0)

        # Patchify/unpatchify using einops
        self.to_patches = Rearrange(
            "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=patch_size, p2=patch_size
        )
        self.from_patches = Rearrange(
            "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
            h=math.ceil(H / patch_size),
            w=math.ceil(W / patch_size),
            p1=patch_size,
            p2=patch_size,
        )

        # Patchified mask
        patch_mask = self.forward_raw(mask.expand(1, num_channels, -1, -1))
        patch_mask = patch_mask.squeeze(0)
        self.num_total_patches = len(patch_mask)

        # Indices of patches intersecting the mask
        patch_indices = torch.where(torch.any(patch_mask, dim=1))[0]
        self.num_patches = len(patch_indices)

        # Patch order
        if isinstance(ordering, torch.Tensor):
            order = self._priority_order(ordering, patch_indices)
        elif ordering == "raster":
            order = torch.arange(self.num_patches, device=device)
        elif ordering == "radial":
            priority = distance_map(mask)
            order = self._priority_order(priority, patch_indices)
        elif ordering == "reverse_radial":
            priority = -distance_map(mask)
            order = self._priority_order(priority, patch_indices)
        elif ordering == "random":
            order = torch.randperm(self.num_patches, device=device)
        else:
            raise ValueError(
                f"Unknown ordering {ordering}; expected one of {ORDERINGS}"
            )

        # Apply ordering
        patch_indices = patch_indices[order]
        patch_mask = patch_mask[patch_indices]

        self.register_buffer("mask", mask)
        self.register_buffer("patch_mask", patch_mask)
        self.register_buffer("patch_indices", patch_indices)
        self.register_buffer("order", order)
        self.register_buffer("ranking", torch.argsort(order))

    def _priority_order(
        self, priority: torch.Tensor, patch_indices: torch.Tensor
    ) -> torch.Tensor:
        assert priority.shape == (self.height, self.width), "Invalid priority shape"
        priority = priority.expand(1, self.num_channels, -1, -1)
        patch_priority = self.forward_raw(priority).squeeze(0)
        patch_priority = patch_priority[patch_indices]
        patch_priority = torch.mean(patch_priority, dim=1)
        order = torch.argsort(patch_priority)
        return order

    @property
    def interior_mask(self) -> torch.Tensor:
        """
        Mask of patches contained in the mask interior.
        """
        return torch.all(self.patch_mask, dim=-1)

    def forward_raw(self, images: torch.Tensor) -> torch.Tensor:
        if self.num_channels == 1 and images.ndim == 3:
            images = images.unsqueeze(1)
        assert images.ndim == 4, "Invalid image shape; expected (B, C, H, W)"
        images = self.pad(images)
        patches = self.to_patches(images)
        return patches

    def forward(self, images: torch.Tensor, apply_mask: bool = True) -> torch.Tensor:
        """
        Convert images, shape (B, C, H, W), to patches, shape (B, N, D).
        """
        patches = self.forward_raw(images)
        patches = patches[..., self.patch_indices, :]
        if apply_mask:
            patches = patches * self.patch_mask
        return patches

    def inverse(
        self, patches: torch.Tensor, apply_mask: bool = True, squeeze: bool = True
    ) -> torch.Tensor:
        """
        Convert patches, shape (B, N, D), back to images, shape (B, C, H, W). If squeeze
        is True and C is 1, the channels dimension is squeezed.
        """
        assert patches.ndim == 3, "Invalid patches shape; expected (B, N, D)"
        assert patches.shape[1] == self.num_patches, "Invalid patches shape"
        B = patches.shape[0]
        dtype = patches.dtype
        device = patches.device

        # Inverse masking
        expanded = torch.zeros(
            (B, self.num_total_patches, self.dim), dtype=dtype, device=device
        )
        expanded[..., self.patch_indices, :] = patches

        # Crop to original size
        images = self.from_patches(expanded)
        left, _, top, _ = self.padding
        images = images[..., top : top + self.height, left : left + self.width]

        if apply_mask:
            images = images * self.mask
        if squeeze and self.num_channels == 1:
            images = images.squeeze(1)
        return images

    def inverse_vector(
        self, values: torch.Tensor, apply_mask: bool = True
    ) -> torch.Tensor:
        """
        Reconstruct a patch map from a vector of patch values.
        """
        assert values.ndim == 2, "Invalid values shape"
        patches = values[..., None].expand(-1, -1, self.dim)
        images = self.inverse(patches, apply_mask=apply_mask)
        return images

    def extra_repr(self) -> str:
        return (
            f"patch_size={self.patch_size}, height={self.height}, "
            f"width={self.width}, num_patches={self.num_patches}, "
            f"ordering={self.ordering}"
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


def distance_map(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the distance map from the mask centroid.
    """
    mask = mask
    device = mask.device

    y = torch.arange(mask.shape[0], dtype=torch.float32, device=device)
    x = torch.arange(mask.shape[1], dtype=torch.float32, device=device)
    coord = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1)

    centroid = coord[mask].mean(dim=0)
    dist = torch.linalg.norm(coord - centroid, axis=-1)
    return dist
