import logging
from pathlib import Path

import matplotlib as mpl
import pytest
import torch
import torch.nn.functional as F

from boldgpt.models import CBIP, create_model
from boldgpt.models.cbip import CLIPLoss, SigLIPLoss

mpl.use("Agg")

RESULT_DIR = Path(__file__).parent / "results"
RESULT_DIR.mkdir(exist_ok=True)


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("mask_ratio", [None, 0.5])
def test_cbip(training: bool, mask_ratio: float):
    torch.manual_seed(42)

    model: CBIP = create_model(
        "cbip_tiny_patch10",
        mask_ratio=mask_ratio,
        encoder_name="vit_base_patch16_clip_224.openai",
    )
    logging.info("Model:\n%s", model)

    batch = {
        "image": torch.randn(4, 3, 224, 224),
        "activity": model.patchify.mask * torch.randn(4, 215, 200),
        "subject_id": torch.arange(4),
        "nsd_id": torch.arange(4),
    }

    model.train(training)
    output, state = model.forward(batch)
    loss = model.loss_fn(batch, output, state)

    logging.info("Loss: %.3e", loss.item())
    logging.info("State:\n%s", {k: get_shape(v) for k, v in state.items()})

    examples = model.prepare_examples(batch, state)
    model.plot_examples(
        examples,
        num_examples=3,
        fname=RESULT_DIR / f"examples_cbip_mr-{mask_ratio}_train-{training}.png",
    )


@pytest.mark.parametrize("siglip", [False, True])
def test_loss(siglip: bool):
    torch.manual_seed(42)
    x = F.normalize(torch.randn(10, 32), dim=-1)
    y = F.normalize(torch.randn(10, 32), dim=-1)
    sample_ids = torch.arange(5).expand(2, -1).flatten()

    loss_mod = SigLIPLoss() if siglip else CLIPLoss()
    with torch.no_grad():
        loss = loss_mod.forward(x, y, sample_ids)
    logging.info(f"Loss (siglip={siglip}): {loss.item():.3f}")


def get_shape(v):
    if isinstance(v, torch.Tensor):
        return tuple(v.shape)
    return v


if __name__ == "__main__":
    pytest.main([__file__, "-k", "test_cbip"])
    # pytest.main([__file__])
