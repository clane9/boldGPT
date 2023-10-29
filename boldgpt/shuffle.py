from typing import Optional, Tuple

import torch


def random_order(
    batch_size: int,
    seq_length: int,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a random sequence order for each sample in a batch. Returns a tuple of
    (order, ranking), each shape (B, N).
    """
    priority = torch.rand(batch_size, seq_length, device=device)
    order = torch.argsort(priority, dim=1)
    ranking = torch.argsort(order, dim=1)
    return order, ranking


def permute(input: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """
    Shuffle the items in an input sequence, shape (B, N, [C]), by order, shape (B, N).
    """
    assert input.ndim in {2, 3}, "invalid input shape"
    assert input.shape[:2] == order.shape, "input and order shapes don't match"
    pad_order = order[..., None] if input.ndim == 3 else order
    shuffled = torch.take_along_dim(input, pad_order, dim=1)
    return shuffled
