import logging
from pathlib import Path

import matplotlib as mpl
import pytest
import torch

from boldgpt.models import BoldGPT, create_model
from boldgpt.models.utils import get_no_decay_keys

mpl.use("Agg")

RESULT_DIR = Path(__file__).parent / "results"
RESULT_DIR.mkdir(exist_ok=True)


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("categorical", [True, False])
def test_boldgpt(categorical: bool, training: bool):
    torch.manual_seed(42)

    model: BoldGPT = create_model("boldgpt_tiny_patch10", categorical=categorical)
    logging.info("Model:\n%s", model)

    no_decay_keys = get_no_decay_keys(model)
    logging.info("No decay keys[:10]:\n%s", no_decay_keys[:10])

    batch = {
        "activity": torch.randn(4, 215, 200),
        "subject_id": torch.arange(4),
        "nsd_id": torch.arange(4),
    }

    model.train(training)
    loss, state = model.forward(batch)
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
        fname=RESULT_DIR / f"examples_cat-{categorical}_train-{training}.png",
    )


if __name__ == "__main__":
    pytest.main([__file__])
