import logging

import pytest
import torch

from boldgpt.model import BoldGPT


def test_boldgpt():
    model = BoldGPT(
        num_patches=100,
        in_features=64,
        num_subs=1024,
        num_classes=512,
        embed_dim=192,
        depth=4,
        num_heads=3,
        is_decoder=True,
        with_cross=True,
        drop_rate=0.2,
        sub_drop_rate=0.2,
        proj_drop_rate=0.2,
        attn_drop_rate=0.2,
        drop_path_rate=0.2,
    )

    logging.info("%s", model)

    x = torch.randn(16, 100, 64)
    sub_indices = torch.arange(16)

    context = torch.randn(16, 50, 192)
    order = torch.argsort(torch.rand(16, 100), dim=1)
    bool_masked_pos = torch.rand(16, 100) > 0.5

    y = model.forward(
        x, sub_indices, context=context, order=order, bool_masked_pos=bool_masked_pos
    )
    assert y.shape == (16, 101, 512)

    no_decay_params = model.no_decay_named_parameters()
    logging.info("No decay params:\n%s", list(no_decay_params.keys()))


if __name__ == "__main__":
    pytest.main([__file__])
