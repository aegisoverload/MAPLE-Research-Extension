#!/bin/bash
#SBATCH --job-name=maple-loop
#SBATCH --account=mst115005
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=450G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work/%u/maple_pilot/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
# Full Route-A MAPLE loop for one epsilon and a list of T values (see full_loop.py).
#   EPS=1 TS="1 2 3" sbatch slurm_loop.sh
# Every stage is resumable; resubmitting the same command continues where it stopped.
# Smoke test: EPS=1 TS=1 LOOP_ARGS="--root full_smoke --N 8 --L 1 --steps 8 --size 512" \
#             sbatch --partition=dev --time=02:00:00 slurm_loop.sh
# Arms (LOOP_ARGS --arm): maple (default) | random (needs maple's gen-init finished at this eps; submit with
#   --dependency=afterok:<maple job>, or after it) | pe (own init, no metadata).
set -euo pipefail
nvidia-smi -L; echo "host=$(hostname)"
ml purge; ml load miniconda3/24.11.1
PY=/work/$USER/envs/maple-img/bin/python
export HF_HOME=/work/$USER/.cache/huggingface TORCH_HOME=/work/$USER/.cache/torch HF_HUB_OFFLINE=1
cd /home/$USER/MAPLE-Research-Extension/image_extension/pilot
EPS=${EPS:?set EPS}; TS=${TS:-1 2 3}; LOOP_ARGS=${LOOP_ARGS:-}
read -r ROOT N L ARM DELTA AIMF < <($PY - $LOOP_ARGS <<'PYEOF'
import argparse, sys
ap = argparse.ArgumentParser(); ap.add_argument('--root', default='full'); ap.add_argument('--N', type=int, default=500)
ap.add_argument('--L', type=int, default=2); ap.add_argument('--arm', default='maple'); ap.add_argument('--delta', default='1/n')
ap.add_argument('--aim-fraction', default='0.1')
a, _ = ap.parse_known_args(sys.argv[1:]); print(a.root, a.N, a.L, a.arm, a.delta, a.aim_fraction)
PYEOF
)
WORKDIR=/work/$USER/maple_pilot/$ROOT; mkdir -p $WORKDIR /work/$USER/maple_pilot/logs
IFS=, read -ra DEVS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"; NG=${#DEVS[@]}
LOOP="$PY full_loop.py --eps $EPS $LOOP_ARGS"

gen() {  # gen <stage> [args]: one process per GPU
  local pids=() fail=0 tag="$*"
  tag=${tag// /_}  # ${*// /_} substitutes per-arg and rejoins with spaces -> ambiguous redirect
  for r in $(seq 0 $((NG - 1))); do
    CUDA_VISIBLE_DEVICES=${DEVS[$r]} $LOOP "$@" --rank $r --world $NG \
      > /work/$USER/maple_pilot/logs/loop_${ROOT}_${ARM}_eps${EPS}_${tag}_r${r}_${SLURM_JOB_ID}.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || fail=1; done
  [ $fail -eq 0 ] || { echo "FAILED: $*"; exit 1; }
  echo "done: $*"
}

echo "root=$ROOT arm=$ARM eps=$EPS delta=$DELTA Ts=$TS"
if [ "$ARM" = maple ]; then
  META=$WORKDIR/dp_metadata_eps${EPS}.csv
  [ -f "$META" ] || $PY dp_metadata.py --epsilon $EPS --num-gen $((N * (L + 1))) --delta "$DELTA" \
    --aim-fraction $AIMF --out $META
fi
[ "$ARM" = random ] || gen gen-init  # random reuses maple's init
for T in $TS; do
  CUDA_VISIBLE_DEVICES=${DEVS[0]} $LOOP vote --T $T --iter 0
  for t in $(seq 1 $T); do
    gen gen-var --T $T --iter $t
    CUDA_VISIBLE_DEVICES=${DEVS[0]} $LOOP vote --T $T --iter $t
  done
  CUDA_VISIBLE_DEVICES=${DEVS[0]} $LOOP evaluate --T $T
done
echo LOOP_DONE
