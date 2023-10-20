import logging
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import pytest
import torch

from boldgpt.models import MAE, create_model

mpl.use("Agg")

RESULT_DIR = Path(__file__).parent / "results"
RESULT_DIR.mkdir(exist_ok=True)


@pytest.mark.parametrize("mask_ratio", [0.75, None])
def test_boldmae(mask_ratio: Optional[float]):
    torch.manual_seed(42)

    model: MAE = create_model("boldmae_tiny_patch10", mask_ratio=mask_ratio)
    logging.info("Model:\n%s", model)

    batch = {
        "activity": torch.randn(4, 215, 200),
        "subject_id": torch.arange(4),
        "nsd_id": torch.arange(4),
    }

    model.train()
    output, state = model.forward(batch)
    loss = model.loss_fn(batch, output, state)
    logging.info("Loss: %.3e", loss.item())

    def get_shape(v):
        if isinstance(v, torch.Tensor):
            return tuple(v.shape)
        return v

    logging.info("State:\n%s", {k: get_shape(v) for k, v in state.items()})

    examples = model.prepare_examples(batch, state)
    model.plot_examples(
        examples,
        num_examples=3,
        fname=RESULT_DIR / f"examples_mr-{mask_ratio}.png",
    )


if __name__ == "__main__":
    pytest.main([__file__])
