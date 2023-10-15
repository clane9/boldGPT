from typing import List

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
