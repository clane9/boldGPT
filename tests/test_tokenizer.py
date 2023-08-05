import logging

import pytest
import torch

from boldgpt.tokenizer import ORDERINGS, BoldTokenizer


@pytest.mark.parametrize("ordering", ORDERINGS)
def test_bold_tokenizer(mask: torch.Tensor, img: torch.Tensor, ordering: str):
    img = mask * img

    tokenizer = BoldTokenizer(
        mask=mask,
        patch_size=8,
        vocab_size=1024,
        ordering=ordering,
    )
    logging.info("%s", tokenizer)

    tokenizer.fit(img)

    patches, tokens = tokenizer.forward(img)

    img2 = tokenizer.inverse_patches(patches)
    assert torch.allclose(img, img2)

    _, tokens2 = tokenizer.forward(img2)
    assert torch.all(tokens == tokens2)


if __name__ == "__main__":
    pytest.main([__file__])
