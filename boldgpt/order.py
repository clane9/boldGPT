from typing import Optional

import torch


class Order:
    """
    A token ordering.
    """

    def __init__(self, order: torch.Tensor):
        B, T = order.shape
        self.batch_size = B
        self.num_tokens = T

        device = order.device
        ranking = torch.argsort(order, dim=-1)
        extra = torch.full((B, 1), T, dtype=torch.int64, device=device)
        extended = torch.cat((order, extra), dim=1)
        next_indices = torch.take_along_dim(extended, ranking + 1, dim=1)
        attn_mask = ranking.unsqueeze(2) >= ranking.unsqueeze(1)

        self.order = order
        self.ranking = ranking
        self.next_indices = next_indices
        self.attn_mask = attn_mask

    @classmethod
    def shuffled(
        cls, batch_size: int, num_tokens: int, device: Optional[torch.device] = None
    ):
        priority = torch.rand((batch_size, num_tokens), device=device)
        order = torch.argsort(priority, dim=1)
        return cls(order)

    @classmethod
    def seq(
        cls, batch_size: int, num_tokens: int, device: Optional[torch.device] = None
    ):
        order = torch.arange(num_tokens, device=device)
        order = order.expand(batch_size, -1)
        return cls(order)

    @classmethod
    def fixed(
        cls, batch_size: int, order: torch.Tensor, device: Optional[torch.device] = None
    ):
        order = torch.as_tensor(order, device=device)
        order = order.expand(batch_size, -1)
        return cls(order)
