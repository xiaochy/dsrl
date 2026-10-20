#!/bin/bash
# Separates POLICY-side float nondeterminism from SAPIEN physics
# nondeterminism, after the --fixed_noise control showed flip rates of 0.10
# (steps=10) and 0.33 (steps=1) with IDENTICAL action noise and
# fixed_track_noise=True -- i.e. something other than the noise is deciding
# outcomes, which invalidates attributing any flip rate to the noise.
set -o pipefail
cd /data/vxiao/bimanual_manipulation/RoboTwin
PY=/home/veraxiao/miniconda3/envs/RoboTwin/bin/python3
S=/data/vxiao/bimanual_manipulation/dsrl-scan-object/tests/diagnose_nondeterminism.py
OUT=/data/vxiao/bimanual_manipulation/dsrl-scan-object/outputs
CK=/data/vxiao/bimanual_manipulation/GAP-tte-flattenmlp-concatdit-actionrope/data/outputs/2026.09.13/17.24.29_GAP_scan_object_actionrope-baseline/checkpoints/200.ckpt

CUDA_VISIBLE_DEVICES=0 $PY $S --gap_ckpt $CK --scene 100200009 --n_repeat 2 \
    --step_lim 500 2>&1 | tee $OUT/nondeterminism_steps10.out &
p1=$!
CUDA_VISIBLE_DEVICES=1 $PY $S --gap_ckpt $CK --scene 100200009 --n_repeat 2 \
    --step_lim 500 --num_inference_steps 1 2>&1 | tee $OUT/nondeterminism_steps1.out &
p2=$!
wait $p1; r1=$?
wait $p2; r2=$?
echo "=== rc_steps10=$r1 rc_steps1=$r2 ==="
