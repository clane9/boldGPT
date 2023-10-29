from typing import Optional

import torch

from ..tokenizer import KMeansTokenizer
from .transformer import Transformer


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    sub_indices: Optional[torch.Tensor] = None,
    context: Optional[torch.Tensor] = None,
    order: Optional[torch.Tensor] = None,
    tokenizer: Optional[KMeansTokenizer] = None,
    offset: int = 0,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Auto-regressively generate samples from a model.

    Args:
        offset: how many leading non-prediction tokens to strip.
    """
    if not (model.is_causal or model.is_masked):
        raise ValueError("Can only generate with a causal or masked model")
    is_categorical = tokenizer is not None
    if is_categorical and tokenizer.vocab_size != model.num_classes:
        raise ValueError("Incompatible model.num_classes for discrete sampling")
    if not is_categorical and model.in_features != model.num_classes:
        raise ValueError("Incompatible model.num_classes for continuous sampling")

    B = prompt.size(0)
    device = prompt.device

    # To sample we always pass a full-length sequence. But we mask the trailing tokens.
    # This way, we can potentially sample from an MAE as well as a GPT. The extra
    # compute cost should not be a big deal (?).
    x = torch.zeros(B, model.num_patches, model.in_features, device=device)
    if model.is_masked:
        bool_mask_pos = torch.ones(
            B, model.num_patches, dtype=torch.bool, device=device
        )
    else:
        bool_mask_pos = None

    start = prompt.size(1)
    if start > 0:
        x[:, :start] = prompt
        if model.is_masked:
            bool_mask_pos[:, :start] = False

    for idx in range(start, model.num_patches):
        output = model(
            x,
            sub_indices=sub_indices,
            context=context,
            order=order,
            bool_masked_pos=bool_mask_pos,
        )
        # Strip leading subject token if MAE.
        output = output[:, offset + idx : offset + idx + 1]

        if is_categorical:
            tokens = sample_tokens(output, temperature=temperature, top_k=top_k)
            patches = tokenizer.inverse(tokens)
        else:
            patches = output
        x[:, idx : idx + 1] = patches
    return x


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
