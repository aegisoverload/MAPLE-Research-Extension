#!/bin/bash
#SBATCH --job-name=maple-votesnr
#SBATCH --account=mst115005
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/%u/maple_pilot/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
# Vote SNR diagnostic (CPU numpy on cached embeddings; dev partition requires a GPU allocation).
set -euo pipefail
ml purge; ml load miniconda3/24.11.1
PY=/work/$USER/envs/maple-img/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /home/$USER/MAPLE-Research-Extension/image_extension/pilot
$PY vote_snr.py ${SNR_ARGS:-}
echo SNR_DONE
