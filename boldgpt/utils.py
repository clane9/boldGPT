"""
Misc utils.
"""
import hashlib
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


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


def setup_logging(out_dir: Optional[Path] = None, level: str = "INFO"):
    """
    Setup root logger.
    """
    fmt = "[%(levelname)s %(asctime)s %(lineno)4d]: %(message)s"
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

    if out_dir:
        log_path = out_dir / "log.txt"
        file_handler = logging.FileHandler(log_path, mode="a")
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
