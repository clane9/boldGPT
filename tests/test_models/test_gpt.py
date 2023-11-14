import logging
from pathlib import Path

import matplotlib as mpl
import pytest
import torch

from boldgpt.models import IGPT, create_model
from boldgpt.models.utils import get_no_decay_keys

mpl.use("Agg")

RESULT_DIR = Path(__file__).parent / "results"
RESULT_DIR.mkdir(exist_ok=True)


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("categorical", [True, False])
def test_boldgpt(categorical: bool, training: bool):
    torch.manual_seed(42)

    model: IGPT = create_model("boldgpt_tiny_patch10", categorical=categorical)
    logging.info("Model:\n%s", model)

    no_decay_keys = get_no_decay_keys(model)
    logging.info("No decay keys[:10]:\n%s", no_decay_keys[:10])

    batch = {
        "activity": torch.randn(4, 215, 200),
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
        fname=RESULT_DIR / f"examples_boldgpt_cat-{categorical}_train-{training}.png",
    )


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("categorical", [True, False])
def test_imagegpt(categorical: bool, training: bool):
    torch.manual_seed(42)

    model: IGPT = create_model("imagegpt_tiny_patch16", categorical=categorical)
    logging.info("Model:\n%s", model)

    no_decay_keys = get_no_decay_keys(model)
    logging.info("No decay keys[:10]:\n%s", no_decay_keys[:10])

    batch = {
        "image": torch.randn(4, 3, 224, 224),
        "image_id": torch.arange(4),
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
        fname=RESULT_DIR / f"examples_imagegpt_cat-{categorical}_train-{training}.png",
    )


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("categorical", [True, False])
def test_image2bold(categorical: bool, training: bool):
    torch.manual_seed(42)

    model: IGPT = create_model("image2bold_tiny_patch10", categorical=categorical)
    logging.info("Model:\n%s", model)

    batch = {
        "image": torch.randn(4, 3, 224, 224),
        "activity": torch.randn(4, 215, 200),
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
        fname=RESULT_DIR
        / f"examples_image2bold_cat-{categorical}_train-{training}.png",
    )


@pytest.mark.parametrize(
    "prompt_fraction,shuffle",
    [(0.25, False), (0.25, True), (0.0, True)],
)
@pytest.mark.parametrize("categorical", [False])
def test_boldgpt_generate(categorical: bool, shuffle: bool, prompt_fraction: float):
    torch.manual_seed(42)

    model: IGPT = create_model("boldgpt_tiny_patch10", categorical=categorical)

    batch = {
        "activity": torch.randn(4, 215, 200),
        "subject_id": torch.arange(4),
        "nsd_id": torch.arange(4),
    }

    logging.info("Generating (no cache)")
    samples_no_cache, state = model.generate(
        batch,
        prompt_fraction=prompt_fraction,
        shuffle=shuffle,
        use_cache=False,
    )

    logging.info("Generating (cache)")
    order = state["order"]
    samples, state = model.generate(
        batch,
        prompt_fraction=prompt_fraction,
        order=order,
        use_cache=True,
    )

    assert torch.allclose(samples, samples_no_cache, rtol=1e-5, atol=1e-5)
    logging.info("Samples: %s", samples.shape)
    logging.info("State:\n%s", {k: get_shape(v) for k, v in state.items()})


def get_shape(v):
    if isinstance(v, torch.Tensor):
        return tuple(v.shape)
    return v


if __name__ == "__main__":
    pytest.main([__file__, "-k", "test_image2bold_compile"])
    # pytest.main([__file__, "-k", "test_boldgpt_generate"])
    # pytest.main([__file__])
