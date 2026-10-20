#!/bin/bash
# Success rate of the unsteered frozen GAP policy at num_inference_steps=1.
#
# WHY 1 AND NOTHING ELSE: tests/sweep_inference_steps_authority.py measured
# noise authority against step count on one fixed set of states --
#   steps 10 / 5 / 3 / 2 -> 0.00345 / 0.00263 / 0.00252 / 0.00202  (path A)
#   steps 1              -> 0.04626                               (13.4x of 10)
# so 1 is the ONLY step count that changes anything; 2-5 are strictly worse
# than the deployed 10. Each DDIM step is a contraction toward the
# conditional mean under prediction_type='sample', so more steps converge
# harder onto a fixed point independent of x_T; at 1 step the output is just
# net(x_T, T, obs) with nothing to re-anchor it.
#
# But 1 step is also a much weaker policy, and authority is worthless if the
# policy cannot do the task. Both arms, same batch as everything else:
#   raw          -> pairs directly against the steps=10 unconditioned number
#   expert_check -> pairs against the 75.33% conditioned baseline
#
# Waits for the in-flight steps=10 runs to exit so the two measurements do
# not contend for the GPUs (and so timings stay comparable).
set -o pipefail
while pgrep -f "eval_unsteered_baseline.py" > /dev/null; do sleep 30; done
cd /data/vxiao/bimanual_manipulation/RoboTwin
PY=/home/veraxiao/miniconda3/envs/RoboTwin/bin/python3
SCRIPT=/data/vxiao/bimanual_manipulation/dsrl-scan-object/tests/eval_unsteered_baseline.py
OUT=/data/vxiao/bimanual_manipulation/dsrl-scan-object/outputs
GAP_CKPT=/data/vxiao/bimanual_manipulation/GAP-tte-flattenmlp-concatdit-actionrope/data/outputs/2026.09.13/17.24.29_GAP_scan_object_actionrope-baseline/checkpoints/200.ckpt

CUDA_VISIBLE_DEVICES=0 $PY $SCRIPT --gap_ckpt $GAP_CKPT --env_seed 1001 --n 30 \
    --step_lim 500 --num_inference_steps 1 \
    2>&1 | tee $OUT/unsteered_steps1_raw.out &
pid_a=$!
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT --gap_ckpt $GAP_CKPT --env_seed 1001 --n 30 \
    --step_lim 500 --num_inference_steps 1 --expert_check \
    2>&1 | tee $OUT/unsteered_steps1_expertcheck.out &
pid_b=$!
wait $pid_a; rc_a=$?
wait $pid_b; rc_b=$?
echo "=== rc_steps1_raw=$rc_a rc_steps1_expertcheck=$rc_b ==="
