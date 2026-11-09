#!/bin/bash
#SBATCH --job-name=maple-init
#SBATCH --account=mst115005
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=450G
#SBATCH --time=05:00:00
#SBATCH --output=/work/%u/maple_pilot/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
# Initialization pilot, N=500 per group, one process per GPU; 8 shards total.
# Submit twice: RANKS="0 1 2 3" and RANKS="4 5 6 7" (4 GPUs each, easier to schedule than a full node).
#   init_pilot     : random, meta, ref, meta_ref (refs = 10 donors as one collage image)
#   init_pilot_k1  : ref, meta_ref with a single reference image (copy-risk arm)
# Separate multi-image references (k=3, k=10) made the model compose group photos; see sweep_* runs.
set -euo pipefail
nvidia-smi -L; echo "host=$(hostname)"
ml purge; ml load miniconda3/24.11.1
PY=/work/$USER/envs/maple-img/bin/python
export HF_HOME=/work/$USER/.cache/huggingface TORCH_HOME=/work/$USER/.cache/torch HF_HUB_OFFLINE=1
cd /home/$USER/MAPLE-Research-Extension/image_extension/pilot
RANKS=${RANKS:-0 1 2 3}
IFS=, read -ra DEVS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"; echo "allocated GPUs: ${DEVS[*]} ranks: $RANKS"
pids=(); gpu=0
for r in $RANKS; do
  ( export CUDA_VISIBLE_DEVICES=${DEVS[$gpu]}
    $PY generate_init.py --run init_pilot --n 500 --n-refs 10 --ref-mode collage --ref-area $((1024*1024)) --rank $r --world 8
    $PY generate_init.py --run init_pilot_k1 --groups ref,meta_ref --n 500 --n-refs 1 --ref-area $((1024*1024)) --rank $r --world 8
  ) > /work/$USER/maple_pilot/logs/init_rank${r}_${SLURM_JOB_ID}.log 2>&1 &
  pids+=($!); gpu=$((gpu + 1))
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
echo "GENERATION_DONE fail=$fail"
exit $fail
