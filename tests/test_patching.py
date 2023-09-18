import logging

import pytest
import torch

from boldgpt.patching import ORDERINGS, MaskedPatchify


@pytest.mark.parametrize("ordering", ORDERINGS)
def test_patchify(mask: torch.Tensor, img: torch.Tensor, ordering: str):
    img = mask * img

    patchify = MaskedPatchify(mask, patch_size=8, ordering=ordering)
    logging.info("%s", patchify)

    patches = patchify(img)
    assert patches.shape == (img.shape[0], 593, 64)

    img2 = patchify.inverse(patches)
    assert torch.allclose(img, img2)

    mask2 = patchify.inverse(patchify.patch_mask[None, ...]).squeeze(0)
    assert torch.allclose(mask, mask2)


if __name__ == "__main__":
    pytest.main([__file__])
