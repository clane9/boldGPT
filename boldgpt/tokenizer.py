import torch
from sklearn.cluster import MiniBatchKMeans
from torch import nn


class KMeansTokenizer(nn.Module):
    """
    Tokenize patch sequences using a learned k-means vocabulary.
    """

    vocab: torch.Tensor

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.register_buffer("vocab", torch.randn(vocab_size, self.dim))

    def fit(self, patches: torch.Tensor):
        """
        Fit a vocabulary to a dataset of patches with kmeans.
        """
        assert patches.shape[-1] == self.dim, "invalid patch dimension"
        patches = patches.reshape(-1, patches.shape[-1])
        kmeans = MiniBatchKMeans(n_clusters=self.vocab_size, n_init="auto")
        kmeans.fit(patches.cpu().numpy())
        self.vocab.copy_(torch.from_numpy(kmeans.cluster_centers_))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Tokenize patches.
        """
        dist = torch.cdist(patches, self.vocab)
        tokens = torch.argmin(dist, dim=-1)
        return tokens

    def inverse(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Lookup the patch exemplars in the vocabular for `tokens`.
        """
        patches = self.vocab[tokens]
        return patches

    def extra_repr(self) -> str:
        return f"vocab_size={self.vocab_size}, dim={self.dim}"
