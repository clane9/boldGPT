#!/bin/bash

#SBATCH --job-name=train_boldgpt
#SBATCH --partition=GPU-shared
#SBATCH -N 1
#SBATCH --gpus=v100-32:4
#SBATCH --time=10:00:00
#SBATCH --account=med230001p

# Set some environment variables
ROOT="/ocean/projects/med230001p/clane2/code/boldGPT"
cd $ROOT

# Set up python environment
source .venv/bin/activate

# Setup wandb
source .env
wandb login

torchrun --standalone --nproc_per_node=4 \
    scripts/train_gpt.py --out_dir results \
    --model boldgpt_small \
    --ps 10 --vs 1024 --vocab_state checkpoints/ps-10_vs-1024_vss-4000_seed-42/tok_state.pt \
    --shuffle --epochs 1000 --bs 512 \
    --workers 0 --amp --compile --wandb
