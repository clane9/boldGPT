import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
from hf_argparser import HfArg, HfArgumentParser
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from timm.utils import random_seed
from torchvision.utils import make_grid

from boldgpt.data import ActivityTransform, load_nsd_flat
from boldgpt.tokenizer import BoldTokenizer

plt.style.use("ggplot")
plt.rcParams["figure.dpi"] = 150

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)

NUM_EXAMPLES = 1024


@dataclass
class Args:
    out_dir: str = HfArg(aliases=["-o"], help="path to output directory")
    patch_size: int = HfArg(aliases=["--ps"], default=10, help="input patch size")
    vocab_size: int = HfArg(
        aliases=["--vs"], default=1024, help="visual token vocab size"
    )
    vocab_samples: int = HfArg(
        aliases=["--vss"], default=4000, help="num vocab training samples"
    )
    seed: int = HfArg(default=42, help="random seed")


def main(args: Args):
    start_time = time.monotonic()
    random_seed(args.seed)

    outprefix = (
        f"ps-{args.patch_size}_vs-{args.vocab_size}_"
        f"vss-{args.vocab_samples}_seed-{args.seed}"
    )
    out_dir = Path(args.out_dir) / outprefix
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    logging.info("Loading data")
    dsets, mask = load_nsd_flat(keep_in_memory=False)

    transform = ActivityTransform()
    sample_indices = torch.randperm(len(dsets["train"]))[: args.vocab_samples]
    sample_activity = dsets["train"].select(sample_indices)["activity"]
    sample_activity = transform(sample_activity)
    logging.info("Sample activity shape: %s", sample_activity.shape)
    logging.info(
        "Sample activity min: %.3f, mean: %.3f, max: %.3f",
        sample_activity.min(),
        sample_activity.mean(),
        sample_activity.max(),
    )

    logging.info("Creating tokenizer")
    tokenizer = BoldTokenizer(
        mask, patch_size=args.patch_size, vocab_size=args.vocab_size
    )
    logging.info(
        "Patch size: %d, num patches: %d, dim: %d",
        args.patch_size,
        tokenizer.num_patches,
        tokenizer.dim,
    )

    logging.info("Fitting tokenizer")
    tokenizer.fit(sample_activity)
    logging.info("Vocab shape: %s", tokenizer.vocab.shape)
    logging.info(
        "Vocab min: %.3f, mean: %.3f, max: %.3f",
        tokenizer.vocab.min(),
        tokenizer.vocab.mean(),
        tokenizer.vocab.max(),
    )
    logging.info("Vocab[:3, :5]: %s", tokenizer.vocab[:3, :5])

    logging.info("Fitting vocab embedding")
    vocab_examples = tokenizer.vocab[:NUM_EXAMPLES]
    pca = PCA(n_components=2).fit(vocab_examples)
    vocab_emb = pca.transform(vocab_examples)

    logging.info("Saving embedded examples image")
    nrow = math.isqrt(NUM_EXAMPLES)
    xmin, xmax = np.quantile(vocab_emb[:, 0], [0.05, 0.95])
    ymin, ymax = np.quantile(vocab_emb[:, 1], [0.05, 0.95])
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, nrow), np.linspace(ymin, ymax, nrow), indexing="xy"
    )
    grid_points = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    _, cind = linear_sum_assignment(cdist(grid_points, vocab_emb))
    vocab_examples = vocab_examples[cind]

    vocab_examples = vocab_examples.reshape(-1, 1, args.patch_size, args.patch_size)
    vocab_grid = make_grid(
        vocab_examples, nrow=math.isqrt(NUM_EXAMPLES), padding=1, normalize=True
    )
    vocab_img = F.to_pil_image(vocab_grid.squeeze(1))
    vocab_img.save(out_dir / "vocab.png")

    logging.info("Saving tokenizer state")
    torch.save(tokenizer.state_dict(), out_dir / "tok_state.pt")

    logging.info("Done! Run time: %.0fs", time.monotonic() - start_time)


if __name__ == "__main__":
    args: Args
    parser = HfArgumentParser(Args)
    (args,) = parser.parse_args_into_dataclasses()
    main(args)
