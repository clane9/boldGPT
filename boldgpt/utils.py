"""
Misc utils.
"""

import hashlib
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .slug import random_slug


class ClusterInfo:
    """
    Encapsulates information about the current training cluster environment.
    """

    def __init__(self, use_cuda: bool):
        use_cuda = use_cuda and torch.cuda.is_available()
        ddp = int(os.environ.get("RANK", -1)) != -1

        if ddp:
            assert use_cuda, "Distributed training requires CUDA"
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            device = torch.device(f"cuda:{local_rank}")
        else:
            rank = local_rank = 0
            world_size = 1
            device = torch.device("cuda" if use_cuda else "cpu")

        self.use_cuda = use_cuda
        self.ddp = ddp
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = device

    @property
    def master_process(self) -> bool:
        return self.rank == 0

    def __repr__(self):
        return (
            f"ClusterInfo(use_cuda={self.use_cuda}, ddp={self.ddp}, "
            f"rank={self.rank}, local_rank={self.local_rank}, "
            f"world_size={self.world_size}, device={self.device})"
        )


def generate_splits(
    num_samples: int, split_sizes: List[float], seed: int
) -> List[np.ndarray]:
    """
    Generate reproducible data splits.

    Args:
        num_samples: number of samples
        split_sizes: fractional split sizes summing to one
        seed: random seed

    Returns:
        A list of split indices arrays
    """
    assert sum(split_sizes) == 1.0, "split_sizes must sum to 1"

    split_lengths = np.asarray(split_sizes) * num_samples
    split_ends = np.round(np.cumsum(split_lengths)).astype(int)
    split_starts = np.concatenate([[0], split_ends[:-1]])

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)

    splits = [
        np.sort(indices[start:end]) for start, end in zip(split_starts, split_ends)
    ]
    return splits


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.

    Copied from:
    https://github.com/facebookresearch/DomainBed/blob/main/domainbed/lib/misc.py
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


def setup_logging(
    level: str = "INFO", path: Optional[Path] = None, rank: Optional[int] = None
):
    """
    Setup root logger.
    """
    if rank is not None:
        fmt = f"[%(levelname)s %(asctime)s {rank:>3d}]: %(message)s"
    else:
        fmt = "[%(levelname)s %(asctime)s]: %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%y-%m-%d %H:%M:%S")

    logger = logging.getLogger()
    logger.setLevel(level)
    # clean up any pre-existing handlers
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    logger.addHandler(stream_handler)

    if path:
        file_handler = logging.FileHandler(path, mode="a")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    # Redefining the root logger is not strictly best practice.
    # https://stackoverflow.com/a/7430495
    # But I want the convenience to just call e.g. `logging.info()`.
    logging.root = logger  # type: ignore


def get_sha():
    """
    Get the current commit hash
    """
    # Copied from: https://github.com/facebookresearch/dino/blob/main/utils.py
    cwd = Path(__file__).parent.parent.absolute()

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def get_exp_name(prefix: str, seed: int):
    """
    Generate a unique experiment name based on a prefix and a random seed.

    Example::
        >>> get_exp_name("my-experiment", 123)
        >>> "202309011000-my-experiment-clumsy-cricket"
    """
    name = datetime.now().strftime("%y%m%d%H%M%S")
    if prefix:
        name = name + "-" + prefix
    name = name + "-" + random_slug(seed=seed)
    return name
