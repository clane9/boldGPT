from typing import Optional

import torch
from sklearn.decomposition import PCA, IncrementalPCA
from torch import nn


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
        return self.fc(embeddings)

    @torch.no_grad()
    def encode(self, activity: torch.Tensor, mask: Optional[torch.Tensor] = None):
        activity = activity[:, self.mask]
        activity = activity - self.fc.bias

        if mask is None:
            embeddings = activity @ self.fc.weight
        else:
            mask = mask[:, self.mask]
            # (N, D, d)
            A = self.fc.weight.expand(activity.size(0), -1, -1)
            A = A * mask.unsqueeze(2)
            # (N, D, 1)
            B = (activity * mask).unsqueeze(2)
            # (N, d, 1)
            embeddings = torch.linalg.lstsq(A, B).solution
            embeddings = embeddings.squeeze(2)

        return embeddings
