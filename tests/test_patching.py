import logging

import pytest
import torch
from pytest import FixtureRequest

from boldgpt.patching import ORDERINGS, MaskedPatchify

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


@pytest.fixture(params=[(), (3,)])
def img(request: FixtureRequest) -> torch.Tensor:
    shape = (BATCH_SIZE, *request.param, HEIGHT, WIDTH)
    img = torch.randn(shape)
    return img


@pytest.mark.parametrize("ordering", ORDERINGS)
def test_patchify(mask: torch.Tensor, img: torch.Tensor, ordering: str):
    img = mask * img
    num_channels = 1 if img.ndim == 3 else img.shape[1]

    patchify = MaskedPatchify(
        mask, patch_size=8, num_channels=num_channels, ordering=ordering
    )
    logging.info("%s", patchify)

    patches = patchify(img)
    assert patches.shape == (img.shape[0], 593, num_channels * 64)

    img2 = patchify.inverse(patches)
    assert torch.allclose(img, img2)

    mask2 = patchify.inverse(patchify.patch_mask[None, ...]).squeeze(0)
    assert torch.allclose(mask, mask2)


if __name__ == "__main__":
    pytest.main([__file__])
