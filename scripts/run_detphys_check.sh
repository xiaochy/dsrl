#!/bin/bash
# Does PhysX enable_enhanced_determinism remove the step-1 divergence?
# Baseline (without it, measured 2026-10-19): identical noise, same process,
# same scene -> obs and action chunk bit-identical at step 0, diverging at
# step 1; outcomes False/True at steps=1 and episode length 13 vs 10 at
# steps=10. If determinism works, both repeats must agree exactly and
# first_action_divergence must be None.
set -o pipefail
cd /data/vxiao/bimanual_manipulation/RoboTwin
PY=/home/veraxiao/miniconda3/envs/RoboTwin/bin/python3
S=/data/vxiao/bimanual_manipulation/dsrl-scan-object/tests/diagnose_nondeterminism.py
OUT=/data/vxiao/bimanual_manipulation/dsrl-scan-object/outputs
CK=/data/vxiao/bimanual_manipulation/GAP-tte-flattenmlp-concatdit-actionrope/data/outputs/2026.09.13/17.24.29_GAP_scan_object_actionrope-baseline/checkpoints/200.ckpt

CUDA_VISIBLE_DEVICES=0 $PY $S --gap_ckpt $CK --scene 100200009 --n_repeat 2 \
    --step_lim 500 --deterministic_physics \
    2>&1 | tee $OUT/detphys_steps10.out &
p1=$!
CUDA_VISIBLE_DEVICES=1 $PY $S --gap_ckpt $CK --scene 100200009 --n_repeat 2 \
    --step_lim 500 --num_inference_steps 1 --deterministic_physics \
    2>&1 | tee $OUT/detphys_steps1.out &
p2=$!
wait $p1; r1=$?
wait $p2; r2=$?
echo "=== rc_steps10=$r1 rc_steps1=$r2 ==="
