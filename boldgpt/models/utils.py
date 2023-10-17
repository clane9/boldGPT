from typing import List, Literal, Tuple, Union

import torch


def get_no_decay_keys(model: torch.nn.Module) -> List[str]:
    """
    Get all no decay keys for a model recursively.
    """
    func = getattr(model, "no_decay_keys", None)
    if func is not None:
        return func()

    keys = []
    for name, module in model.named_children():
        keys.extend([f"{name}.{key}" for key in get_no_decay_keys(module)])

    return keys


def r2_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    dim: Union[int, Tuple[int, ...]] = 1,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    var = torch.var(target, dim=dim, correction=0)
    res = torch.mean((target - pred) ** 2, dim=dim)
    score = 1.0 - res / var

    if reduction == "mean":
        score = score.mean()
    elif reduction == "sum":
        score = score.sum()
    elif reduction == "none":
        pass
    else:
        raise ValueError(f"Invalid reduction {reduction}")

    return score
