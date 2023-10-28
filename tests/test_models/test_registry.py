import logging

import pytest

from boldgpt.models import create_model, list_models, list_pretrained


def test_list_models():
    logging.info("Models:\n%s", list_models())
    logging.info("Pretrained models:\n%s", list_pretrained())


@pytest.mark.parametrize(
    "name,pretrained",
    [
        ("boldgpt_small_patch10", False),
        ("boldgpt_small_patch10.kmq", True),
    ],
)
def test_create_model(name: str, pretrained: bool):
    model = create_model(name, pretrained=pretrained)
    logging.info("Model: %s\n%s", name, model)


if __name__ == "__main__":
    pytest.main([__file__])
