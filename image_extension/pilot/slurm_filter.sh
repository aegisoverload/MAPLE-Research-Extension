#!/bin/bash
#SBATCH --job-name=maple-filter
#SBATCH --account=mst115005
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=/work/%u/maple_pilot/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
set -euo pipefail
nvidia-smi -L
ml purge; ml load miniconda3/24.11.1
PY=/work/$USER/envs/maple-img/bin/python
export HF_HOME=/work/$USER/.cache/huggingface TORCH_HOME=/work/$USER/.cache/torch HF_HUB_OFFLINE=1
cd /home/$USER/MAPLE-Research-Extension/image_extension/pilot
$PY filter_eval.py
echo FILTER_DONE
