import logging

import pytest
import torch

from boldgpt.model import BoldGPT


@pytest.mark.parametrize("is_decoder", [True, False])
@pytest.mark.parametrize("with_cross", [True, False])
@pytest.mark.parametrize("shuffle", [True, False])
def test_boldgpt(is_decoder: bool, with_cross: bool, shuffle: bool):
    model = BoldGPT(
        num_patches=100,
        in_features=64,
        num_subs=1024,
        num_classes=512,
        embed_dim=192,
        depth=4,
        num_heads=3,
        is_decoder=is_decoder,
        with_cross=with_cross,
        drop_rate=0.2,
        sub_drop_rate=0.2,
        proj_drop_rate=0.2,
        attn_drop_rate=0.2,
        drop_path_rate=0.2,
    )

    if is_decoder and with_cross and shuffle:
        logging.info("%s", model)

    x = torch.randn(16, 100, 64)
    sub_indices = torch.arange(16)

    context = torch.randn(16, 50, 192) if with_cross else None
    order = torch.randperm(100) if shuffle else None

    y = model.forward(x, sub_indices, context=context, order=order)
    assert y.shape == (16, 101, 512)

    if is_decoder and with_cross and shuffle:
        no_decay_params = model.no_decay_named_parameters()
        logging.info("No decay params:\n%s", list(no_decay_params.keys()))


if __name__ == "__main__":
    pytest.main([__file__])
