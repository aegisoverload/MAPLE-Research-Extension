#!/bin/bash
#SBATCH --job-name=maple-prep
#SBATCH --account=mst115005
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=/work/%u/maple_pilot/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
set -euo pipefail
ml purge; ml load miniconda3/24.11.1
PY=/work/$USER/envs/maple-img/bin/python
export HF_HOME=/work/$USER/.cache/huggingface TORCH_HOME=/work/$USER/.cache/torch HF_HUB_OFFLINE=1

cd /home/$USER/MAPLE-Research-Extension/image_extension/pilot
[ -f /work/$USER/maple_pilot/private.csv ] || $PY prepare_celeba.py
JAX_PLATFORMS=cpu $PY dp_metadata.py --epsilon 1 --num-gen 500
$PY generate_init.py --run smoke --n 4 --n-refs 10 --ref-area $((512*512))
