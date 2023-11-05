from typing import Optional

import torch

from boldgpt.shuffle import permute
from boldgpt.tokenizer import KMeansTokenizer

from .transformer import Transformer


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    sub_indices: Optional[torch.Tensor] = None,
    context: Optional[torch.Tensor] = None,
    order: Optional[torch.Tensor] = None,
    tokenizer: Optional[KMeansTokenizer] = None,
    patch_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """
    Auto-regressively generate samples from a model.
    """
    if not model.is_causal:
        raise ValueError("Can only generate with a causal model")
    is_categorical = tokenizer is not None
    if is_categorical and tokenizer.vocab_size != model.num_classes:
        raise ValueError("Incompatible model.num_classes for discrete sampling")
    if not is_categorical and model.in_features != model.num_classes:
        raise ValueError("Incompatible model.num_classes for continuous sampling")

    B, start, D = prompt.shape
    N = model.num_patches
    samples = torch.zeros(B, N, D, device=prompt.device, dtype=prompt.dtype)
    if start > 0:
        samples[:, :start] = prompt

    if patch_mask is not None:
        assert patch_mask.shape == (N, D), "Expected patch mask shape (N, D)"
        patch_mask = patch_mask.expand_as(samples)
        if order is not None:
            patch_mask = permute(patch_mask, order)

    model.decoding(use_cache=use_cache)

    for idx in range(start, model.num_patches):
        if use_cache:
            # On first iteration, we pass the full prompt to fill the cache
            # Then we reuse the cache, passing only a single token for each later step
            input = samples[:, :idx] if idx == start else samples[:, idx - 1 : idx]
            offset = -1 if idx == start else idx - 1
        else:
            input = samples
            offset = None

        output = model(
            input,
            sub_indices=sub_indices,
            context=context,
            order=order,
            offset=offset,
        )
        output = output[:, -1:] if use_cache else output[:, idx : idx + 1]

        if is_categorical:
            tokens = sample_tokens(output, temperature=temperature, top_k=top_k)
            output = tokenizer.inverse(tokens)

        if patch_mask is not None:
            output = output * patch_mask[:, idx : idx + 1]

        samples[:, idx : idx + 1] = output

    model.decoding(False)
    return samples


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample a discrete token given a logits.

    Copied from: https://github.com/karpathy/nanoGPT
    """
    K = logits.size(-1)
    shape = logits.shape[:-1]
    logits = logits.reshape(-1, K)

    logits = logits / temperature
    if top_k is not None:
        vals, _ = torch.topk(logits, min(top_k, K))
        logits[logits < vals[..., -1:]] = -float("Inf")
    probs = torch.softmax(logits, dim=-1)
    tokens = torch.multinomial(probs, num_samples=1)
    tokens = tokens.reshape(shape)
    return tokens
