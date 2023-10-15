from typing import Callable, Dict, List, Optional, Union

import torch

_MODEL_REGISTRY: Dict[str, Callable[..., torch.nn.Module]] = {}


def register_model(name_or_func: Union[Optional[str], Callable] = None):
    def _decorator(func: Callable):
        name = name_or_func if isinstance(name_or_func, str) else func.__name__
        assert name not in _MODEL_REGISTRY, f"Model {name} already registered"
        _MODEL_REGISTRY[name] = func
        return func

    if isinstance(name_or_func, Callable):
        return _decorator(name_or_func)
    return _decorator


def create_model(name: str, **kwargs) -> torch.nn.Module:
    assert name in _MODEL_REGISTRY, f"model {name} not registered"
    model = _MODEL_REGISTRY[name](**kwargs)
    return model


def list_models() -> List[str]:
    return list(_MODEL_REGISTRY.keys())
