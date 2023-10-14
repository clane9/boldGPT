import pytest
import torch

from boldgpt.tokenizer import KMeansTokenizer


def test_kmeans_tokenizer():
    tokenizer = KMeansTokenizer(vocab_size=128, dim=64)
    patches = torch.randn(1024, 8, 64)
    tokenizer.fit(patches)

    vocab = tokenizer.vocab
    assert vocab.shape == (128, 64)
    tokens = tokenizer(vocab)
    assert torch.all(tokens == torch.arange(len(tokens)))
    recon = tokenizer.inverse(tokens)
    assert torch.allclose(vocab, recon)


if __name__ == "__main__":
    pytest.main([__file__])
