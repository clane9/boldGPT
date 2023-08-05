import logging

import pytest
import torch

from boldgpt.patching import MaskedPatchify


def test_patchify(mask: torch.Tensor, img: torch.Tensor):
    img = mask * img

    patchify = MaskedPatchify(mask, patch_size=8)
    logging.info("%s", patchify)

    patches = patchify(img)
    assert patches.shape == (img.shape[0], 593, 64)

    img2 = patchify.inverse(patches)
    assert torch.allclose(img, img2)


if __name__ == "__main__":
    pytest.main([__file__])
