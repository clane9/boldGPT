from typing import Callable, Dict, List

import torch

_MODEL_REGISTRY: Dict[str, Callable[..., torch.nn.Module]] = {}


def register_model(name: str):
    def _decorator(func):
        assert name not in _MODEL_REGISTRY, f"model {name} already registered"
        _MODEL_REGISTRY[name] = func
        return func

    return _decorator


def create_model(name: str, *args, **kwargs) -> torch.nn.Module:
    assert name in _MODEL_REGISTRY, f"model {name} not registered"
    model = _MODEL_REGISTRY[name](*args, **kwargs)
    return model


def list_models() -> List[str]:
    return list(_MODEL_REGISTRY.keys())
