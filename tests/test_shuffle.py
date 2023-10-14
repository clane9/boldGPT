from typing import Tuple

import pytest
import torch

from boldgpt.shuffle import random_order, shuffle


@pytest.mark.parametrize("shape", [(), (64,)])
def test_shuffle(shape: Tuple[int]):
    order, ranking = random_order(8, 16)
    input = torch.randn((8, 16, *shape))

    shuffled = shuffle(input, order)
    unshuffled = shuffle(shuffled, ranking)
    assert torch.all(unshuffled == input)


if __name__ == "__main__":
    pytest.main([__file__])
