#!/bin/bash
# The comparison the project never actually ran: the UNSTEERED frozen GAP
# policy, measured through the DSRL env, both ways.
#
#   arm A  expert_check=True   -> directly comparable to the 75.33% number
#                                 (both conditioned on expert-solvable seeds)
#   arm B  expert_check=False  -> must reproduce the 0.400 that every DSRL
#                                 eval reported, and its per-seed outcomes
#                                 can be conditioned POST-HOC on the already
#                                 measured expert mask for seeds
#                                 100200000..100200029 (13/30 pass) -- a
#                                 second, exactly paired route to the same
#                                 conditioned number, for free.
#
# Both use reward_source="none": no dynamics model, no SVM, no scorer
# subprocess. That also makes arm B a free test of the leftover hypothesis
# that the reward path (_collect_multiview_obs / _verify_action_track_setup
# inside step()) was perturbing the rollout -- if B != 0.400, it was.
set -o pipefail
cd /data/vxiao/bimanual_manipulation/RoboTwin
PY=/home/veraxiao/miniconda3/envs/RoboTwin/bin/python3
SCRIPT=/data/vxiao/bimanual_manipulation/dsrl-scan-object/tests/eval_unsteered_baseline.py
OUT=/data/vxiao/bimanual_manipulation/dsrl-scan-object/outputs
GAP_CKPT=/data/vxiao/bimanual_manipulation/GAP-tte-flattenmlp-concatdit-actionrope/data/outputs/2026.09.13/17.24.29_GAP_scan_object_actionrope-baseline/checkpoints/200.ckpt

CUDA_VISIBLE_DEVICES=0 $PY $SCRIPT --gap_ckpt $GAP_CKPT --env_seed 1001 --n 30 \
    --step_lim 500 --expert_check 2>&1 | tee $OUT/unsteered_expertcheck.out &
pid_a=$!
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT --gap_ckpt $GAP_CKPT --env_seed 1001 --n 30 \
    --step_lim 500 2>&1 | tee $OUT/unsteered_raw.out &
pid_b=$!
wait $pid_a; rc_a=$?
wait $pid_b; rc_b=$?
echo "=== rc_expertcheck=$rc_a rc_raw=$rc_b ==="
