#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$ROOT/finallog/$TS"
mkdir -p "$OUTDIR/logs"

MASTER_LOG="$OUTDIR/master.log"
echo "Writing master log to: $MASTER_LOG"

run_one () {
  local name="$1"; shift
  echo "" | tee -a "$MASTER_LOG"
  echo "==================== RUN: $name ====================" | tee -a "$MASTER_LOG"
  echo "CMD: $*" | tee -a "$MASTER_LOG"
  echo "ENV: NS_CIM_INTERFACE_MODE=${NS_CIM_INTERFACE_MODE:-unset} EASCIM_TO_S=${EASCIM_TO_S:-unset} EASCIM_FC_HZ=${EASCIM_FC_HZ:-unset} EASCIM_SD_VDD_V=${EASCIM_SD_VDD_V:-unset}" | tee -a "$MASTER_LOG"
  echo "====================================================" | tee -a "$MASTER_LOG"
  # Run inside the NeuroSim conda environment for correct torch/CUDA setup.
  # Avoid `conda run` here (can hang due to base env write permissions); use an interactive bash that activates env.
  (
    bash -lc 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate neurosim && "$@"' _ "$@"
  ) 2>&1 | tee "$OUTDIR/logs/${name}.log" | tee -a "$MASTER_LOG"
}

# Common flags taken from VGGcifar10Compare.log (setup user requested)
COMMON_FLAGS=(
  --dataset cifar10
  --mode WAGE
  --batch_size 200
  --epochs 200
  --grad_scale 8
  --seed 117
  --log_interval 100
  --test_interval 1
  --lr 0.01
  --decreasing_lr 140,180
  --wl_weight 8
  --wl_grad 8
  --wl_activate 8
  --wl_error 8
  --inference 1
  --ADCprecision 6
  --cellBit 2
  --onoffratio 10
  --vari 0.0
  --t 0
  --v 0
  --detect 0
  --target 0
)

run_grid () {
  local model="$1"
  # User request: only subArray=128 and parallelRead=128 for now
  for sub in 128; do
    local pr="128"

    # Baseline (ADC)
    NS_CIM_INTERFACE_MODE=0 run_one "${model}_baseline_sub${sub}" \
      python3 -u inference.py "${COMMON_FLAGS[@]}" --model "$model" --subArray "$sub" --parallelRead "$pr" \
      --logdir "project/$TS/${model}/baseline/sub${sub}"

    # Sigma-delta: choose To such that ~3 pulses are observed at p≈0 (f≈fc): To ≈ 3/fc
    for fc in 50e6 100e6 200e6; do
      # Compute To in seconds (scientific notation) using python for robustness.
      local to
      to="$(FC_HZ="$fc" python3 - <<'PY'
import os
fc = float(os.environ["FC_HZ"])
print("{:.12g}".format(3.0/fc))
PY
      )"
      NS_CIM_INTERFACE_MODE=1 EASCIM_TO_S="$to" EASCIM_FC_HZ="$fc" EASCIM_SD_VDD_V=0.4 run_one \
        "${model}_sigmadelta_sub${sub}_To${to}_fc${fc}" \
        python3 -u inference.py "${COMMON_FLAGS[@]}" --model "$model" --subArray "$sub" --parallelRead "$pr" \
        --logdir "project/$TS/${model}/sigmadelta/sub${sub}/To${to}_fc${fc}"
    done
  done
}

run_one "build_info" bash -lc "uname -a && python3 -V && (cd NeuroSIM && ./main --help 2>/dev/null || true)"

run_grid "VGG8"

echo "" | tee -a "$MASTER_LOG"
echo "DONE. Results in: $OUTDIR" | tee -a "$MASTER_LOG"

