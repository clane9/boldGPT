import math
from typing import Optional

import torch
from sklearn.decomposition import PCA, IncrementalPCA
from torch import nn
from torch.nn import functional as F


class MaskedPCA(nn.Module):
    mask: torch.Tensor

    def __init__(
        self,
        mask: torch.Tensor,
        dim: int = 1024,
        batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.batch_size = batch_size

        self.fc = nn.Linear(dim, mask.sum())
        self.register_buffer("mask", mask)

    def fit(self, activity: torch.Tensor):
        activity = activity[:, self.mask]
        if self.batch_size:
            pca = IncrementalPCA(n_components=self.dim, batch_size=self.batch_size)
        else:
            pca = PCA(n_components=self.dim, svd_solver="randomized")
        pca.fit(activity.cpu().numpy())

        state_dict = {
            "weight": torch.as_tensor(pca.components_.T, dtype=torch.float32),
            "bias": torch.as_tensor(pca.mean_, dtype=torch.float32),
        }
        self.fc.load_state_dict(state_dict)

    def forward(self, embeddings: torch.Tensor):
        values = self.fc(embeddings)
        recon = self.masked_fill(values)
        return recon

    def masked_fill(self, values: torch.Tensor):
        shape = values.shape[:-1]
        filled = torch.zeros(
            shape + self.mask.shape, dtype=values.dtype, device=values.device
        )
        filled[..., self.mask] = values
        return filled

    @torch.no_grad()
    def encode(
        self,
        activity: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_components: Optional[int] = None,
        lambd: float = 0.0,
    ):
        if n_components is None:
            n_components = self.dim
        activity = activity[:, self.mask]
        activity = activity - self.fc.bias

        if mask is None:
            embeddings = activity @ self.fc.weight[:, :n_components]
        else:
            mask = mask[:, self.mask]
            # (N, D, d)
            A = self.fc.weight[:, :n_components]
            A = A.expand(activity.size(0), -1, -1)
            A = A * mask.unsqueeze(2)
            # (N, D, 1)
            B = (activity * mask).unsqueeze(2)
            # (N, d, 1)
            embeddings = _ridge_regression(A, B, lambd=lambd)
            embeddings = embeddings.squeeze(2)

        if n_components < self.dim:
            embeddings = F.pad(embeddings, (0, self.dim - n_components))
        return embeddings

    def components(self):
        return self.masked_fill(self.fc.weight.detach().t())


def _ridge_regression(A: torch.Tensor, B: torch.Tensor, lambd: float = 1.0):
    """
    Solve the ridge regression problem:

        min X || A X - B ||_F^2 + lambd || X ||_F^2

    which can be equivalently formulated as:

        min X || [ A; sqrt(lambd) I ] X - [B; 0] ||_F^2

    Args:
        A: shape (N, D, d)
        B: shape (N, D, K)
        lambd: regularization penalty

    Returns:
        Regression solution X, shape (N, d, K)
    """
    N, _, d = A.shape
    _, _, K = B.shape

    if lambd > 0:
        I = math.sqrt(lambd) * torch.eye(d, d, device=A.device, dtype=A.dtype)
        I = I.expand(N, -1, -1)
        A = torch.cat([A, I], dim=1)

        Z = torch.zeros(N, d, K, device=A.device, dtype=A.dtype)
        B = torch.cat([B, Z], dim=1)

    beta = torch.linalg.lstsq(A, B).solution
    return beta
