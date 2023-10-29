from typing import Tuple

import pytest
import torch

from boldgpt.shuffle import permute, random_order


@pytest.mark.parametrize("shape", [(), (64,)])
def test_shuffle(shape: Tuple[int]):
    order, ranking = random_order(8, 16)
    input = torch.randn((8, 16, *shape))

    shuffled = permute(input, order)
    unshuffled = permute(shuffled, ranking)
    assert torch.all(unshuffled == input)


if __name__ == "__main__":
    pytest.main([__file__])
