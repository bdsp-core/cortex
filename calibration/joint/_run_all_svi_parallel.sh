#!/bin/zsh
# Phase 3.5: 6 IIIC SVI fits CONCURRENTLY (independent tasks). Each process
# capped to ~2 threads to avoid the N×cores oversubscription that runs
# SLOWER (scripts/_parallel.py philosophy). Calibration stage only — the
# engine's numpy single-thread-BLAS bit-exact contract is unaffected.
cd "$(dirname "$0")/../.."
TPP=2   # threads per process; 6 procs x 2 = 12 of 16 cores
export OMP_NUM_THREADS=$TPP OPENBLAS_NUM_THREADS=$TPP MKL_NUM_THREADS=$TPP
export NUMEXPR_NUM_THREADS=$TPP VECLIB_MAXIMUM_THREADS=$TPP
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
mkdir -p calibration/joint/logs
t0=$(date +%s)
for t in sz lpd gpd lrda grda iic; do
  ( echo "=== $(date +%H:%M:%S) START $t ===" >> calibration/joint/logs/svi_${t}.log
    .venv/bin/python -m pipeline.joint_calibration.fit_joint_iiic \
       --task $t --method svi --svi-steps 40000 --post-draws 500 --seed 0 \
       >> calibration/joint/logs/svi_${t}.log 2>&1
    echo "=== $(date +%H:%M:%S) DONE $t (rc=$?) ===" >> calibration/joint/logs/svi_${t}.log
  ) &
done
wait
echo "ALL 6 IIIC SVI FITS COMPLETE in $(( ($(date +%s)-t0)/60 )) min"
