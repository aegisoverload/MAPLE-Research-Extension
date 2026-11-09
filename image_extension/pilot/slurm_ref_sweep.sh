#!/bin/bash
#SBATCH --job-name=maple-refsweep
#SBATCH --account=mst115005
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:45:00
#SBATCH --output=/work/%u/maple_pilot/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
# Smoke sweep: 10 separate reference images broke the model (multi-person scenes); find a working ref setting.
set -euo pipefail
nvidia-smi -L; echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
ml purge; ml load miniconda3/24.11.1
PY=/work/$USER/envs/maple-img/bin/python
export HF_HOME=/work/$USER/.cache/huggingface TORCH_HOME=/work/$USER/.cache/torch HF_HUB_OFFLINE=1
cd /home/$USER/MAPLE-Research-Extension/image_extension/pilot
G=ref,meta_ref
$PY generate_init.py --run sweep_B_k1      --groups $G --n 4 --n-refs 1  --ref-area $((1024*1024))
$PY generate_init.py --run sweep_C_collage --groups $G --n 4 --n-refs 10 --ref-mode collage --ref-area $((1024*1024))
$PY generate_init.py --run sweep_A_k3      --groups $G --n 4 --n-refs 3  --ref-area $((1024*1024))
$PY generate_init.py --run sweep_D_k3_512  --groups $G --n 4 --n-refs 3  --ref-area $((512*512))
echo SWEEP_DONE
