#!/bin/zsh
# NUTS-subsample VI-vs-NUTS s_sd variance cross-check (the SVI-defensibility
# gate item). 6 tasks concurrently, 2 threads/proc; 80k stratified subsample.
cd "$(dirname "$0")/../.."
TPP=2
export OMP_NUM_THREADS=$TPP OPENBLAS_NUM_THREADS=$TPP MKL_NUM_THREADS=$TPP
export NUMEXPR_NUM_THREADS=$TPP VECLIB_MAXIMUM_THREADS=$TPP
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
mkdir -p calibration/joint/logs
t0=$(date +%s)
for t in sz lpd gpd lrda grda iic; do
  ( .venv/bin/python -m pipeline.joint_calibration.fit_joint_iiic \
      --task $t --method nuts --subsample 80000 --warmup 400 --samples 400 \
      --chains 2 --seed 0 > calibration/joint/logs/nuts_${t}.log 2>&1
    mv -f calibration/joint/${t}_posterior.npz calibration/joint/${t}_nuts_posterior.npz 2>/dev/null
    mv -f calibration/joint/${t}_summary.json calibration/joint/${t}_nuts_summary.json 2>/dev/null
  ) &
done
wait
echo "NUTS x-check done in $(( ($(date +%s)-t0)/60 )) min"
