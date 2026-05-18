#!/bin/zsh
set -e
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for t in sz lpd gpd lrda grda iic; do
  echo "=== $(date +%H:%M:%S) START $t ==="
  .venv/bin/python -m pipeline.joint_calibration.fit_joint_iiic \
     --task $t --method svi --svi-steps 40000 --post-draws 500 --seed 0 \
     >> calibration/joint/logs/svi_${t}.log 2>&1
  echo "=== $(date +%H:%M:%S) DONE  $t ==="
done
echo "ALL 6 IIIC SVI FITS COMPLETE"
