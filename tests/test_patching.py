import logging

import numpy as np
import pytest
import torch

from boldgpt.patching import MaskedPatchify

HEIGHT = 211
WIDTH = 213


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
    img = torch.randn(4, HEIGHT, WIDTH)
    return img


def test_patchify(mask: np.ndarray, img: torch.Tensor):
    img = mask * img

    patchify = MaskedPatchify(mask, patch_size=8)
    logging.info("%s", patchify)

    patches = patchify(img)
    assert patches.shape == (4, 593, 64)

    img2 = patchify.inverse(patches)
    assert torch.allclose(img, img2)


if __name__ == "__main__":
    pytest.main([__file__])
