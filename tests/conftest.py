import pytest
import torch

HEIGHT = 211
WIDTH = 213
BATCH_SIZE = 4


@pytest.fixture()
def mask() -> torch.Tensor:
    y = torch.linspace(-1, 1, HEIGHT)
    x = torch.linspace(-1, 1, WIDTH)

    xy = torch.stack(torch.meshgrid(y, x, indexing="ij"), axis=-1)
    dist = torch.linalg.norm(xy, dim=-1)
    mask = dist <= 1
    return mask


@pytest.fixture()
def img() -> torch.Tensor:
    img = torch.randn(BATCH_SIZE, HEIGHT, WIDTH)
    return img
