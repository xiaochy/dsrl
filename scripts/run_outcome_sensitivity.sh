#!/bin/bash
# Does the DDIM noise change the OUTCOME, not just the action?
#
# Nothing in this project has measured this. Authority (how far the noise
# moves the action chunk) is 0.0035 at the deployed 10 steps and 0.046 at 1,
# but DSRL learns from REWARD, and reward here is terminal task success. If
# every noise draw on a fixed scene yields the same outcome, the return is a
# constant function of the action and there is no gradient to climb at any
# authority. This replays each scene K times from an identical SAPIEN initial
# state with a different noise stream and counts disagreements -- no reward
# model involved, so the action-SVM's own candidate-insensitivity cannot
# contaminate the answer.
#
# Both step counts, in parallel, expert_check on so the scenes are the same
# distribution the published baselines are scored on:
#   GPU 0  steps=10 (deployed)  -- could DSRL ever have worked here
#   GPU 1  steps=1              -- does the 13.4x authority buy real headroom
#
# 15 scenes x 3 draws = 45 episodes per arm, ~1.2h.
set -o pipefail
while pgrep -f "eval_unsteered_baseline.py" > /dev/null; do sleep 30; done
cd /data/vxiao/bimanual_manipulation/RoboTwin
PY=/home/veraxiao/miniconda3/envs/RoboTwin/bin/python3
SCRIPT=/data/vxiao/bimanual_manipulation/dsrl-scan-object/tests/measure_outcome_sensitivity.py
OUT=/data/vxiao/bimanual_manipulation/dsrl-scan-object/outputs
GAP_CKPT=/data/vxiao/bimanual_manipulation/GAP-tte-flattenmlp-concatdit-actionrope/data/outputs/2026.09.13/17.24.29_GAP_scan_object_actionrope-baseline/checkpoints/200.ckpt

CUDA_VISIBLE_DEVICES=0 $PY $SCRIPT --gap_ckpt $GAP_CKPT --env_seed 1001 \
    --n_scenes 15 --n_noise 3 --step_lim 500 --expert_check \
    2>&1 | tee $OUT/outcome_sensitivity_steps10.out &
pid_a=$!
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT --gap_ckpt $GAP_CKPT --env_seed 1001 \
    --n_scenes 15 --n_noise 3 --step_lim 500 --expert_check \
    --num_inference_steps 1 \
    2>&1 | tee $OUT/outcome_sensitivity_steps1.out &
pid_b=$!
wait $pid_a; rc_a=$?
wait $pid_b; rc_b=$?
echo "=== rc_steps10=$rc_a rc_steps1=$rc_b ==="
