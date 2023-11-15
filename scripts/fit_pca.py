import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from hf_argparser import HfArg, HfArgumentParser
from timm.utils import random_seed

from boldgpt.data import ActivityTransform, load_nsd_flat, load_nsd_flat_mask
from boldgpt.pca import MaskedPCA

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)


@dataclass
class Args:
    out_dir: str = HfArg(aliases=["-o"], help="path to output directory")
    dim: int = HfArg(aliases=["-d"], default=1024, help="pca dimension")
    num_samples: Optional[int] = HfArg(
        aliases=["-n"], default=None, help="number of training samples"
    )
    batch_size: Optional[int] = HfArg(
        aliases=["--bs"], default=None, help="incremental pca batch size"
    )
    subid: Optional[int] = HfArg(aliases=["-s"], default=None, help="subject ID")
    seed: int = HfArg(default=42, help="random seed")


def main(args: Args):
    start_time = time.monotonic()
    random_seed(args.seed)

    outprefix = (
        f"pca_d-{args.dim}_n-{args.num_samples}_"
        f"bs-{args.batch_size}_s-{args.subid}_seed-{args.seed}"
    )
    out_dir = Path(args.out_dir) / outprefix
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    logging.info("Loading data")
    dsets = load_nsd_flat(keep_in_memory=False)
    mask = load_nsd_flat_mask()

    transform = ActivityTransform()
    if args.num_samples:
        train_indices = torch.randperm(len(dsets["train"]))[: args.num_samples]
        train_activity = dsets["train"].select(train_indices)["activity"]
    else:
        train_activity = dsets["train"]["activity"]
    train_activity = transform(train_activity)
    logging.info("Activity: %s", train_activity.shape)

    logging.info("Fitting PCA")
    pca = MaskedPCA(mask, dim=args.dim, batch_size=args.batch_size)
    pca.fit(train_activity)

    logging.info("Saving PCA state")
    torch.save(pca.state_dict(), out_dir / "pca_state.pt")
    logging.info("Done! Run time: %.0fs", time.monotonic() - start_time)


if __name__ == "__main__":
    args: Args
    parser = HfArgumentParser(Args)
    (args,) = parser.parse_args_into_dataclasses()
    main(args)
