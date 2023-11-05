import logging

import pytest
import torch

from boldgpt.models.transformer import Attention, Transformer


def test_transformer():
    # Small model with all bells and whistles
    model = Transformer(
        num_patches=100,
        in_features=64,
        num_subs=1024,
        num_registers=4,
        num_classes=512,
        embed_dim=192,
        context_dim=256,
        depth=4,
        num_heads=3,
        with_next_pos=True,
        with_cross=True,
        is_causal=True,
        is_masked=True,
        drop_rate=0.2,
        sub_drop_rate=0.2,
        proj_drop_rate=0.2,
        attn_drop_rate=0.2,
        drop_path_rate=0.2,
    )

    logging.info("%s", model)

    x = torch.randn(16, 100, 64)
    sub_indices = torch.arange(16)

    context = torch.randn(16, 50, 256)
    order = torch.argsort(torch.rand(16, 100), dim=1)
    bool_masked_pos = torch.rand(16, 100) > 0.5

    y = model.forward(
        x, sub_indices, context=context, order=order, bool_masked_pos=bool_masked_pos
    )
    assert y.shape == (16, 105, 512)

    no_decay_keys = set(model.no_decay_keys())
    decay_keys = [
        name for name, _ in model.named_parameters() if name not in no_decay_keys
    ]
    logging.info("No decay keys:\n%s", list(no_decay_keys))
    logging.info("Decay keys:\n%s", list(decay_keys))


def test_attention_kv_cache():
    attn = Attention(dim=192, num_heads=3)
    x = torch.randn(4, 8, 192)
    with torch.no_grad():
        y = attn.forward(x, is_causal=True)
        y1, kv = attn.forward(x[:, :7], is_causal=True, return_kv=True)
        y2 = attn.forward(x[:, 7:], is_causal=False, kv_cache=kv)
    y12 = torch.cat([y1, y2], dim=1)
    assert torch.allclose(y, y12, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-k", "test_attention_kv_cache"])
    # pytest.main([__file__])
