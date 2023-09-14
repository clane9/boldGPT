"""
Training script for BoldGPT.
"""

import json
import logging
import math
import shutil
import sys
import time
from argparse import Namespace
from collections import defaultdict
from contextlib import suppress
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from hf_argparser import HfArg, HfArgumentParser
from matplotlib import pyplot as plt
from timm.utils import AverageMeter, random_seed, reduce_tensor
from torch.cuda.amp import GradScaler
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

import wandb
from boldgpt import create_model, list_models
from boldgpt.data import ActivityTransform, load_nsd_flat
from boldgpt.model import BoldGPT
from boldgpt.tokenizer import BoldShuffle, BoldTokenizer
from boldgpt.utils import ClusterInfo, get_exp_name, get_sha, seed_hash, setup_logging
from boldgpt.visualization import plot_examples

np.set_printoptions(precision=3)

Criterion = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

PROJECT = "boldgpt"

LOG_INTERVAL = 10
NUM_EXAMPLES = 10
CKPT_INTERVAL = 5
MAX_CKPTS = 5


@dataclass
class Args:
    # Architecture
    model: str = HfArg(
        default="boldgpt_base", help=f"model ({', '.join(list_models())})"
    )
    patch_size: int = HfArg(aliases=["--ps"], default=10, help="input patch size")
    vocab_size: int = HfArg(
        aliases=["--vs"], default=1024, help="visual token vocab size"
    )
    vocab_samples: int = HfArg(
        aliases=["--vss"], default=4000, help="num vocab training samples"
    )
    vocab_state: Optional[str] = HfArg(
        aliases=["--vst"], default=None, help="tokenizer vocab state to load"
    )
    ordering: Optional[str] = HfArg(default="radial", help="default patch ordering")
    # Paths
    out_dir: str = HfArg(default="results", help="path to root output directory")
    prefix: Optional[str] = HfArg(default=None, help="experiment name prefix")
    name: Optional[str] = HfArg(default=None, help="full experiment name")
    desc: Optional[str] = HfArg(default=None, help="description to attach to run")
    # Regularization and augmentation
    shuffle: Optional[bool] = HfArg(default=False, help="shuffle patch ordering")
    drop_rate: float = HfArg(aliases=["--dr"], default=0.0, help="head dropout rate")
    sub_drop_rate: float = HfArg(
        aliases=["--sdr"], default=0.0, help="subject ID dropout rate"
    )
    proj_drop_rate: float = HfArg(
        aliases=["--pdr"], default=0.0, help="projection dropout rate"
    )
    attn_drop_rate: float = HfArg(
        aliases=["--adr"], default=0.0, help="attention dropout rate"
    )
    drop_path_rate: float = HfArg(aliases=["--dpr"], default=0.0, help="drop path rate")
    # Optimization
    epochs: int = HfArg(default=1000, help="number of epochs")
    batch_size: int = HfArg(
        aliases=["--bs"], default=256, help="batch size per replica"
    )
    lr: float = HfArg(default=3e-4, help="learning rate")
    decay_lr: bool = HfArg(default=True, help="decay learning rate")
    warmup_steps: float = HfArg(default=2000, help="number of warmup steps")
    min_lr_fraction: float = HfArg(
        default=0.05, help="minimum lr as a fraction of max lr"
    )
    weight_decay: float = HfArg(aliases=["--wd"], default=0.05, help="weight decay")
    beta1: float = HfArg(default=0.9, help="AdamW beta1")
    beta2: float = HfArg(default=0.95, help="AdamW beta2")
    grad_accum_steps: int = HfArg(
        aliases=["--accum"], default=1, help="number of gradient accumulation steps"
    )
    clip_grad: Optional[float] = HfArg(default=1.0, help="gradient norm clipping")
    # Logistics
    checkpoint: Optional[str] = HfArg(
        aliases=["--ckpt"], default=None, help="checkpoint to load"
    )
    restart: bool = HfArg(
        default=False, help="Restart training rather than resume from checkpoint"
    )
    cuda: bool = HfArg(default=True, help="use cuda")
    amp: bool = HfArg(default=False, help="use AMP")
    compile: bool = HfArg(default=False, help="use torch compile")
    workers: int = HfArg(aliases=["-j"], default=4, help="data loading workers")
    overwrite: bool = HfArg(default=False, help="overwrite pre-existing results")
    wandb: bool = HfArg(default=False, help="log to wandb")
    figures: bool = HfArg(default=True, help="generate figures")
    sweep: bool = HfArg(default=False, help="whether we're in a wandb sweep")
    debug: bool = HfArg(default=False, help="quick debug mode")
    seed: int = HfArg(default=42, help="random seed")


def main(args: Args):
    start_time = time.monotonic()
    random_seed(args.seed)

    # Device and distributed training setup
    clust = ClusterInfo(args.cuda)
    if clust.ddp:
        init_process_group(backend="nccl")
        torch.cuda.set_device(clust.device)

    # Output naming
    commit_sha = get_sha()
    if args.name is None:
        name_seed = seed_hash(commit_sha, json.dumps(args.__dict__))
        name = get_exp_name(args.prefix, seed=name_seed)
    else:
        name = args.name
    out_dir = Path(args.out_dir) / PROJECT
    if args.sweep:
        out_dir = out_dir / "sweeps"
    out_dir = out_dir / name

    # Creating output dir
    overwritten = False
    if clust.master_process and out_dir.exists():
        if args.overwrite:
            overwritten = True
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(f"Output directory {out_dir} already exists")
    if clust.master_process:
        out_dir.mkdir(parents=True)
    else:
        while not out_dir.exists():
            time.sleep(0.1)

    log_path = out_dir / "logs" / f"log-{clust.rank:02d}.txt"
    log_path.parent.mkdir(exist_ok=True)
    setup_logging(path=log_path, stdout=clust.master_process, rank=clust.rank)

    # Wandb setup
    if clust.master_process and args.wandb and not args.sweep:
        wandb.init(project=PROJECT, name=name, config=args.__dict__)

    # Initial logging
    logging.info("Starting training: %s/%s", PROJECT, name)
    logging.info("Args:\n%s", yaml.safe_dump(args.__dict__, sort_keys=False))
    logging.info(commit_sha)

    logging.info("Writing to %s", out_dir)
    if overwritten:
        logging.warning("Overwriting previous results")
    if clust.master_process:
        with (out_dir / "args.yaml").open("w") as f:
            yaml.safe_dump(args.__dict__, f, sort_keys=False)

    logging.info("Running on: %s", clust)

    # AMP setup
    if args.amp:
        logging.info("Running in mixed precision with native PyTorch AMP")
        autocast = partial(
            torch.autocast, device_type=clust.device.type, dtype=torch.float16
        )
        scaler = GradScaler()
    else:
        autocast = suppress
        scaler = None

    # Datasets and loaders
    logging.info("Loading NSD-Flat dataset")
    dsets, mask = load_nsd_flat(args)
    for split, ds in dsets.items():
        logging.info("%s samples: %d", split.capitalize(), len(ds))
    logging.info("Activity shape: %s", mask.shape)
    logging.info(
        "Mask size: %d/%d (%.3f)", mask.sum(), mask.numel(), mask.float().mean()
    )

    loaders = {}
    transform = ActivityTransform()
    if clust.ddp:
        samplers = {
            "train": DistributedSampler(dsets["train"], shuffle=True),
            "val": DistributedSampler(dsets["val"], shuffle=False),
        }
    else:
        samplers = {"train": RandomSampler(dsets["train"]), "val": None}

    for split, ds in dsets.items():
        loaders[split] = DataLoader(
            ds,
            batch_size=args.batch_size,
            sampler=samplers[split],
            num_workers=args.workers,
            pin_memory=clust.use_cuda,
            collate_fn=Collate(transform),
            drop_last=True,  # helpful for torch.compile
        )

    # Tokenizer
    logging.info("Creating tokenizer")
    tokenizer = BoldTokenizer(
        mask,
        patch_size=args.patch_size,
        vocab_size=args.vocab_size,
        ordering=args.ordering,
    )
    logging.info(
        "Patch size: %d, Num patches: %d, Dim: %d",
        args.patch_size,
        tokenizer.num_patches,
        tokenizer.dim,
    )

    if args.vocab_state:
        logging.info("Loading tokenizer vocab state: %s", args.vocab_state)
        state_dict = torch.load(args.vocab_state, map_location="cpu")
        tokenizer.load_vocab(state_dict["vocab"])
    else:
        logging.info("Fitting tokenizer vocab")
        sample_indices = torch.randperm(len(dsets["train"]))[: args.vocab_samples]
        sample_activity = dsets["train"].select(sample_indices)["activity"]
        sample_activity = transform(sample_activity)
        tokenizer.fit(sample_activity)
        logging.info("Vocab[:3, :5]: %s", tokenizer.vocab[:3, :5])

    if clust.master_process:
        torch.save(tokenizer.state_dict(), out_dir / "tok_state.pt")
    tokenizer = tokenizer.to(clust.device)

    # Model and optimizer
    logging.info("Creating model: %s", args.model)
    model = create_model(
        args.model,
        num_patches=tokenizer.num_patches,
        in_features=tokenizer.dim,
        num_classes=args.vocab_size,
        drop_rate=args.drop_rate,
        sub_drop_rate=args.sub_drop_rate,
        proj_drop_rate=args.proj_drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
    )
    model = model.to(clust.device)
    logging.info("%s", model)
    logging.info("Params: %.0fM", sum(p.numel() for p in model.parameters()) / 1e6)

    optimizer = create_optimizer(args, model)

    # Load checkpoint
    if args.checkpoint:
        logging.info("Loading checkpoint: %s", args.checkpoint)
        start_epoch, best_metric = load_checkpoint(args, model, optimizer, clust.device)
    else:
        start_epoch = 0
        best_metric = float("inf")

    if clust.ddp:
        model = DDP(model, device_ids=[clust.local_rank])

    # Compile after ddp for more optimizations
    if args.compile:
        assert hasattr(torch, "compile"), "PyTorch >= 2.0 required for torch.compile"
        logging.info("Compiling the model")
        model = torch.compile(model)

    best_epoch = start_epoch
    epoch_steps = math.ceil(len(loaders["train"]) / args.grad_accum_steps)
    logging.info("Steps per epoch: %d", epoch_steps)

    for epoch in range(start_epoch, args.epochs):
        logging.info("Starting epoch %d", epoch)
        if clust.ddp:
            samplers["train"].set_epoch(epoch)

        train(
            args=args,
            epoch=epoch,
            tokenizer=tokenizer,
            model=model,
            train_loader=loaders["train"],
            optimizer=optimizer,
            clust=clust,
            autocast=autocast,
            scaler=scaler,
            out_dir=out_dir,
        )

        metric = validate(
            args=args,
            epoch=epoch,
            step=(epoch + 1) * epoch_steps,
            tokenizer=tokenizer,
            model=model,
            val_loader=loaders["val"],
            clust=clust,
            out_dir=out_dir,
        )

        if clust.master_process:
            save_checkpoint(
                epoch=epoch,
                metric=metric,
                is_best=metric < best_metric,
                model=model,
                optimizer=optimizer,
                out_dir=out_dir,
            )

        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch

        if args.debug:
            break

    if clust.master_process and args.wandb:
        wandb.log(
            {"score_last": metric, "score_best": best_metric},
            step=args.epochs * epoch_steps,
        )

    logging.info("Done! Run time: %.0fs", time.monotonic() - start_time)
    logging.info("*** Best metric: %.3f (epoch %d)", best_metric, best_epoch)

    if clust.ddp:
        destroy_process_group()


class Collate(torch.nn.Module):
    def __init__(self, transform: torch.nn.Module):
        super().__init__()
        self.transform = transform

    def forward(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        collated = defaultdict(list)
        for sample in batch:
            for k, v in sample.items():
                collated[k].append(v)

        collated["activity"] = [self.transform(act) for act in collated["activity"]]

        collated = {k: torch.stack(v) for k, v in collated.items()}
        return collated


def create_optimizer(args: Args, model: BoldGPT) -> torch.optim.Optimizer:
    named_params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    nodecay_keys = set(model.nodecay_keys())
    decay_params = [p for name, p in named_params.items() if name not in nodecay_keys]
    nodecay_params = [p for name, p in named_params.items() if name in nodecay_keys]

    optim_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups, lr=args.lr, betas=(args.beta1, args.beta2)
    )
    return optimizer


def train(
    args: Args,
    epoch: int,
    tokenizer: BoldTokenizer,
    model: BoldGPT,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    clust: ClusterInfo,
    autocast: Callable,
    scaler: Optional[GradScaler],
    out_dir: Path,
):
    if clust.use_cuda:
        torch.cuda.reset_peak_memory_stats()

    model.train()
    shuffle = BoldShuffle()

    params = [p for group in optimizer.param_groups for p in group["params"]]

    def clip_grad():
        if args.clip_grad is not None and args.clip_grad > 0:
            if scaler is not None:
                # unscale the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(params, args.clip_grad).item()
        else:
            total_norm = float("nan")
        return total_norm

    loss_m = AverageMeter()
    data_time_m = AverageMeter()
    step_time_m = AverageMeter()
    examples = {}

    epoch_batches = len(train_loader)
    accum_steps = args.grad_accum_steps
    epoch_steps = math.ceil(epoch_batches / accum_steps)
    first_step = epoch * epoch_steps
    last_accum_steps = epoch_batches % accum_steps
    last_batch_idx_to_accum = epoch_batches - last_accum_steps

    # Initialize LR
    lr = update_lr(args, optimizer, first_step, epoch_steps)
    optimizer.zero_grad()

    end = time.monotonic()
    for batch_idx, sample in enumerate(train_loader):
        step = first_step + batch_idx // accum_steps
        is_last_batch = batch_idx + 1 == epoch_batches
        need_update = is_last_batch or (batch_idx + 1) % accum_steps == 0
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        # Unpack and map data to cuda
        activity = sample["activity"].to(clust.device)
        sub_indices = sample["subject_id"].to(clust.device)
        batch_size = len(activity)
        data_time = time.monotonic() - end

        # Patchify, tokenize, and optionally shuffle
        patches, tokens = tokenizer(activity)
        if args.shuffle:
            patches, tokens, order = shuffle(patches, tokens)
        else:
            order = None

        # Predict and compute loss
        with autocast():
            logits = model(patches, sub_indices, order=order)
            loss = F.cross_entropy(logits.flatten(0, 1), tokens.flatten())
        if clust.ddp:
            loss_item = reduce_tensor(loss.detach(), clust.world_size).item()
        else:
            loss_item = loss.item()
        if accum_steps > 1:
            loss = loss / accum_steps

        if math.isnan(loss_item) or math.isinf(loss_item):
            raise RuntimeError("NaN/Inf loss encountered on step %d; exiting", step)

        # Update learning rate according to schedule
        if need_update:
            lr = update_lr(args, optimizer, step, epoch_steps)

        # Optimization step
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if need_update:
                total_norm = clip_grad()
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            if need_update:
                total_norm = clip_grad()
                optimizer.step()

        # End of iteration timing
        if clust.use_cuda:
            torch.cuda.synchronize()
        step_time = time.monotonic() - end

        loss_m.update(loss_item, batch_size)
        data_time_m.update(data_time, batch_size)
        step_time_m.update(step_time, batch_size)

        if clust.master_process and batch_idx == 0 and args.figures:
            pred = logits.detach().argmax(dim=-1)
            # Unshuffle tokens and predictions if necessary
            if args.shuffle:
                tokens = shuffle.inverse(tokens)
                pred = shuffle.inverse(pred)

            examples["subid"] = sub_indices[:NUM_EXAMPLES]
            examples["nsdid"] = sample["nsd_id"][:NUM_EXAMPLES]
            examples["activity"] = activity[:NUM_EXAMPLES]
            examples["tokens"] = tokens[:NUM_EXAMPLES]
            examples["pred"] = pred[:NUM_EXAMPLES]
            examples["order"] = order[:NUM_EXAMPLES] if args.shuffle else None

        if (step % LOG_INTERVAL == 0 and need_update) or is_last_batch or args.debug:
            tput = (clust.world_size * args.batch_size) / step_time_m.avg
            if clust.use_cuda:
                alloc_mem_gb = torch.cuda.max_memory_allocated() / 1e9
                res_mem_gb = torch.cuda.max_memory_reserved() / 1e9
            else:
                alloc_mem_gb = res_mem_gb = 0.0

            logging.info(
                f"Train: {epoch:>3d} [{batch_idx:>3d}/{epoch_batches}][{step:>6d}]"
                f"  Loss: {loss_m.val:#.3g} ({loss_m.avg:#.3g})"
                f"  LR: {lr:.3e}"
                f"  Grad: {total_norm:.3e}"
                f"  Time: {data_time_m.avg:.3f},{step_time_m.avg:.3f} {tput:.0f}/s"
                f"  Mem: {alloc_mem_gb:.2f},{res_mem_gb:.2f} GB"
            )

            if clust.master_process:
                record = {
                    "step": step,
                    "epoch": epoch,
                    "loss": loss_m.val,
                    "lr": lr,
                    "grad": total_norm,
                    "data_time": data_time_m.avg,
                    "step_time": step_time_m.avg,
                    "tput": tput,
                }

                with (out_dir / "train_log.json").open("a") as f:
                    print(json.dumps(record), file=f)

                if args.wandb:
                    wandb.log({"train": record}, step=step)

        # Restart timer for next iteration
        end = time.monotonic()

        if args.debug:
            break

    if clust.master_process and args.figures:
        example_path = out_dir / "figures" / f"train_examples-{epoch:04d}.png"
        example_path.parent.mkdir(exist_ok=True)
        plot_examples(tokenizer, fname=example_path, **examples)

        if args.wandb:
            wandb.log(
                {"figs": {"train_examples": wandb.Image(str(example_path))}}, step=step
            )

        plt.close("all")


@torch.no_grad()
def validate(
    args: Args,
    epoch: int,
    step: int,
    tokenizer: BoldTokenizer,
    model: BoldGPT,
    val_loader: DataLoader,
    clust: ClusterInfo,
    out_dir: Path,
) -> float:
    if clust.use_cuda:
        torch.cuda.reset_peak_memory_stats()

    model.eval()

    loss_m = AverageMeter()
    data_time_m = AverageMeter()
    step_time_m = AverageMeter()
    examples = {}

    epoch_batches = len(val_loader)
    end = time.monotonic()
    for batch_idx, sample in enumerate(val_loader):
        # Unpack and map data to cuda
        activity = sample["activity"].to(clust.device)
        sub_indices = sample["subject_id"].to(clust.device)
        batch_size = len(activity)
        data_time = time.monotonic() - end

        # Predict and compute loss
        patches, tokens = tokenizer(activity)
        logits = model(patches, sub_indices)
        loss = F.cross_entropy(logits.flatten(0, 1), tokens.flatten())
        if clust.ddp:
            loss_item = reduce_tensor(loss.detach(), clust.world_size).item()
        else:
            loss_item = loss.item()

        # End of iteration timing
        if clust.use_cuda:
            torch.cuda.synchronize()
        step_time = time.monotonic() - end

        loss_m.update(loss_item, batch_size)
        data_time_m.update(data_time, batch_size)
        step_time_m.update(step_time, batch_size)

        if clust.master_process and batch_idx == 0 and args.figures:
            examples["subid"] = sub_indices[:NUM_EXAMPLES]
            examples["nsdid"] = sample["nsd_id"][:NUM_EXAMPLES]
            examples["activity"] = activity[:NUM_EXAMPLES]
            examples["tokens"] = tokens[:NUM_EXAMPLES]
            examples["pred"] = logits.detach().argmax(-1)[:NUM_EXAMPLES]

        if (
            batch_idx % LOG_INTERVAL == 0
            or batch_idx + 1 == epoch_batches
            or args.debug
        ):
            tput = (clust.world_size * args.batch_size) / step_time_m.avg
            if clust.use_cuda:
                alloc_mem_gb = torch.cuda.max_memory_allocated() / 1e9
                res_mem_gb = torch.cuda.max_memory_reserved() / 1e9
            else:
                alloc_mem_gb = res_mem_gb = 0.0

            logging.info(
                f"Val: {epoch:>3d} [{batch_idx:>3d}/{epoch_batches}]"
                f"  Loss: {loss_m.val:#.3g} ({loss_m.avg:#.3g})"
                f"  Time: {data_time_m.avg:.3f},{step_time_m.avg:.3f} {tput:.0f}/s"
                f"  Mem: {alloc_mem_gb:.2f},{res_mem_gb:.2f} GB"
            )

        if args.debug:
            break

        # Reset timer
        end = time.monotonic()

    if clust.master_process:
        record = {
            "step": step,
            "epoch": epoch,
            "loss": loss_m.avg,
            "data_time": data_time_m.avg,
            "step_time": step_time_m.avg,
            "tput": tput,
        }

        with (out_dir / "val_log.json").open("a") as f:
            print(json.dumps(record), file=f)

        if args.wandb:
            wandb.log({"val": record}, step=step)

    if clust.master_process and args.figures:
        example_path = out_dir / "figures" / f"val_examples-{epoch:04d}.png"
        plot_examples(tokenizer, fname=example_path, **examples)

        if args.wandb:
            wandb.log(
                {"figs": {"val_examples": wandb.Image(str(example_path))}}, step=step
            )

        plt.close("all")

    return loss_m.avg


def load_checkpoint(
    args: Args,
    model: BoldGPT,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])

    if not args.restart:
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"]
        best_metric = state["metric"]
    else:
        start_epoch = 0
        best_metric = float("inf")
    return start_epoch, best_metric


def save_checkpoint(
    epoch: int,
    metric: float,
    is_best: bool,
    model: Union[DDP, BoldGPT],
    optimizer: torch.optim.Optimizer,
    out_dir: Path,
):
    if isinstance(model, DDP):
        model = model.module

    state = {
        "epoch": epoch,
        "metric": metric,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    ckpt_dir = out_dir / "checkpoints"
    ckpt_path = ckpt_dir / f"ckpt-{epoch:04d}.pt"
    ckpt_path.parent.mkdir(exist_ok=True)
    torch.save(state, ckpt_path)

    all_ckpts = sorted(ckpt_dir.glob("ckpt-[0-9]*.pt"))
    for p in all_ckpts[:-MAX_CKPTS]:
        p.unlink()

    ckpt_last = ckpt_dir / "ckpt-last.pt"
    ckpt_last.unlink(missing_ok=True)
    ckpt_last.symlink_to(ckpt_path.name)

    if is_best:
        ckpt_best = ckpt_dir / "ckpt-best.pt"
        torch.save(state, ckpt_best)


def update_lr(
    args: Args, optimizer: torch.optim.Optimizer, step: int, epoch_steps: int
):
    """
    Update optimizer lr according to a linear warmup + cosine decay schedule.

    Adapted from: https://github.com/karpathy/nanoGPT
    """
    total_steps = args.epochs * epoch_steps
    if step == 0 and args.warmup_steps > total_steps:
        logging.warning(
            f"warmup_steps ({args.warmup_steps}) greater than total steps {total_steps}"
        )
    warmup_steps = min(args.warmup_steps, total_steps)
    min_lr = args.min_lr_fraction * args.lr

    # Linear warmup
    if step < warmup_steps:
        lr = min_lr + (step / warmup_steps) * (args.lr - min_lr)
    # Cosine decay
    elif args.decay_lr:
        decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1

        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        lr = min_lr + coeff * (args.lr - min_lr)
    else:
        lr = args.lr

    # Update lr in place
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


if __name__ == "__main__":
    args: Args
    parser = HfArgumentParser(Args)
    if sys.argv[1].endswith(".yaml"):
        # If the first argument is a yaml file, parse it first to get default arguments.
        (args,) = parser.parse_yaml_file(yaml_file=sys.argv[1])

        # Treat any remaining args as overrides
        parsed = parser.parse_args(
            args=sys.argv[2:], namespace=Namespace(**asdict(args))
        )
        (args,) = parser.parse_dict(parsed.__dict__)
    else:
        (args,) = parser.parse_args_into_dataclasses()

    try:
        main(args)
    except Exception as exc:
        logging.error("Exited with exception", exc_info=exc)
        sys.exit(1)
