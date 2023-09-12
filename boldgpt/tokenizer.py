from typing import Optional, Tuple

import torch
from sklearn.cluster import MiniBatchKMeans
from torch import nn

from .patching import MaskedPatchify

ORDERINGS = ("raster", "radial", "random")


class BoldTokenizer(nn.Module):
    """
    Tokenize BOLD activity maps using a learned k-means vocabulary. Supports different
    orderings for the tokens:

    - raster: original image raster order
    - radial: tokens sorted by distance from mask centroid
    - random: fixed random token ordering
    """

    vocab: torch.Tensor
    default_order: torch.Tensor
    default_ranking: torch.Tensor

    def __init__(
        self,
        mask: torch.Tensor,
        patch_size: int,
        vocab_size: int,
        ordering: str = "raster",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.ordering = ordering

        self.patchify = MaskedPatchify(mask, patch_size=patch_size)

        if ordering == "raster":
            default_order = torch.arange(self.num_patches)
        elif ordering == "radial":
            default_order = self._radial_order()
        elif ordering == "random":
            default_order = torch.randperm(self.num_patches)
        else:
            raise ValueError(
                f"Unknown ordering {ordering}; expected one of {ORDERINGS}"
            )

        self.register_buffer("vocab", torch.randn(vocab_size, self.dim))
        self.register_buffer("default_order", default_order)
        self.register_buffer("default_ranking", torch.argsort(default_order))

    @property
    def mask(self) -> torch.Tensor:
        return self.patchify.mask

    @property
    def num_patches(self) -> int:
        return self.patchify.num_patches

    @property
    def dim(self) -> int:
        return self.patchify.dim

    @property
    def patch_mask(self) -> torch.Tensor:
        """
        Patch mask with patches in the default order.
        """
        return self.patchify.patch_mask[self.default_order]

    @property
    def interior_mask(self) -> torch.Tensor:
        """
        Mask of patches contained in the mask interior, in the default order.
        """
        return self.patchify.interior_mask[self.default_order]

    def fit(self, images: torch.Tensor):
        """
        Fit vocabulary to images by k-means.
        """
        patches = self.patchify(images)
        # Only fit on patches in the mask interior
        patches = patches[:, self.patchify.interior_mask]
        patches = patches.flatten(0, 1)

        kmeans = MiniBatchKMeans(n_clusters=self.vocab_size, n_init="auto")
        kmeans.fit(patches.numpy())
        self.vocab.copy_(torch.from_numpy(kmeans.cluster_centers_))

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Patchify and tokenize `images`. Returns a tuple of `(patches, tokens)`
        """
        patches = self.forward_patches(images)
        tokens = self.forward_tokens(patches)
        return patches, tokens

    def forward_patches(self, images: torch.Tensor) -> torch.Tensor:
        """
        Patchify `images`.
        """
        patches = self.patchify(images)
        patches = patches[..., self.default_order, :]
        return patches

    def forward_tokens(self, patches: torch.Tensor):
        """
        Tokenize `patches` by nearest neighbor. Returns a tensor `tokens`
        """
        dist = torch.cdist(patches, self.vocab)
        tokens = torch.argmin(dist, dim=-1)
        return tokens

    def inverse(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct images from `tokens`. Returns a tensor `images`.
        """
        patches = self.lookup(tokens)
        images = self.inverse_patches(patches)
        return images

    def lookup(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Lookup the patch exemplars in the vocabular for `tokens`.
        """
        patches = self.vocab[tokens]
        return patches

    def inverse_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct images from `patches`. Returns a tensor `images`.
        """
        patches = patches[..., self.default_ranking, :]
        images = self.patchify.inverse(patches)
        return images

    def _radial_order(self) -> torch.Tensor:
        """
        Compute the radial patch ordering.
        """
        mask = self.patchify.mask
        device = mask.device

        y = torch.arange(mask.shape[0], dtype=torch.float32, device=device)
        x = torch.arange(mask.shape[1], dtype=torch.float32, device=device)
        coord = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1)

        centroid = coord[mask].mean(dim=0)
        dist = torch.linalg.norm(coord - centroid, axis=-1)

        dist_patches = self.patchify(dist)
        dist_patches = torch.median(dist_patches, dim=1)[0]
        order = torch.argsort(dist_patches)
        return order

    def load_vocab(self, vocab: torch.Tensor):
        """
        Load learned kmeans vocabulary.
        """
        self.vocab.copy_(vocab)

    def extra_repr(self) -> str:
        return f"vocab_size={self.vocab_size}, ordering={self.ordering}"


class BoldShuffle(nn.Module):
    """
    Shuffle token order.
    """

    _order: Optional[torch.Tensor]

    def __init__(self):
        super().__init__()
        self._order = None

    def forward(
        self, patches: torch.Tensor, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = patches.shape
        device = patches.device

        order = torch.argsort(torch.rand(B, N, device=device), dim=1)
        patches = torch.take_along_dim(patches, order[..., None], dim=1)
        tokens = torch.take_along_dim(tokens, order, dim=1)

        self._order = order
        return patches, tokens, order

    def inverse(
        self, tokens: torch.Tensor, order: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if order is None:
            order = self._order

        assert order is not None, "order required"
        assert tokens.ndim in {2, 3}, "invalid tokens"
        assert tokens.shape[:2] == order.shape, "token and order shapes don't match"

        ranking = torch.argsort(order, dim=1)
        if tokens.ndim == 3:
            ranking = ranking[..., None]
        tokens = torch.take_along_dim(tokens, ranking, dim=1)
        return tokens
