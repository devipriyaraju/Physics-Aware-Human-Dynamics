#!/bin/bash
#SBATCH --job-name=hybrid-grf
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module purge
module load python
source .venv/bin/activate

python scripts/train_hybrid.py --config configs/default.yaml
