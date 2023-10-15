import logging
from contextlib import suppress
from pathlib import Path

import matplotlib as mpl
import pytest
import torch

from boldgpt.models import BoldGPT, create_model

mpl.use("Agg")

RESULT_DIR = Path(__file__).parent / "results"
RESULT_DIR.mkdir(exist_ok=True)


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("categorical", [True, False])
def test_boldgpt(categorical: bool, training: bool):
    torch.manual_seed(42)

    model: BoldGPT = create_model("boldgpt_tiny_patch10", categorical=categorical)
    logging.info("Model:\n%s", model)

    batch = {
        "activity": torch.randn(4, 215, 200),
        "subject_id": torch.arange(4),
        "nsd_id": torch.arange(4),
    }

    loss, state = model.step(
        batch=batch,
        batch_idx=0,
        device=torch.device("cpu"),
        autocast=suppress,
        training=training,
    )

    logging.info("Loss: %.3e", loss.item())

    def get_shape(v):
        if isinstance(v, torch.Tensor):
            return tuple(v.shape)
        return v

    logging.info("State:\n%s", {k: get_shape(v) for k, v in state.items()})

    model.plot_examples(
        num_examples=3,
        fname=RESULT_DIR / f"examples_cat-{categorical}_train-{training}.png",
    )


if __name__ == "__main__":
    pytest.main([__file__])
