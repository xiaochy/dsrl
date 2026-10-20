#!/bin/bash
# Re-does the outcome-sensitivity measurement properly, after a recorded
# rollout disagreed with the first run on the same scene and nominally the
# same noise draw (2026-10-19).
#
# Cause: gap_base_policy passed generator=None to
# predict_action_from_noise, so with joint_diffusion the TRACK branch drew
# from the GLOBAL torch RNG. The track influences the action, so the policy
# output depended on process history -- uncontrolled, and not something the
# DSRL agent can steer. GAP's own eval path never had this problem
# (deploy_policy.py:507/:561 pass a seeded generator). Now fixed:
# fixed_track_noise=True re-seeds the track generator from the episode seed
# at every reset.
#
#   GPU 0  DETERMINISM CONTROL, steps=10: identical action noise on every
#          draw. MUST come back 0/N flips. If it does not, flip counts
#          cannot be attributed to the noise at all and the previous
#          headroom numbers mean nothing.
#   GPU 1  steps=10, varying action noise, track noise now held fixed per
#          scene -- so flips are attributable to the DSRL ACTION alone,
#          which is the quantity DSRL actually needs. The first run
#          conflated this with track noise.
set -o pipefail
while pgrep -f "record_rollout_videos.py" > /dev/null; do sleep 20; done
cd /data/vxiao/bimanual_manipulation/RoboTwin
PY=/home/veraxiao/miniconda3/envs/RoboTwin/bin/python3
SCRIPT=/data/vxiao/bimanual_manipulation/dsrl-scan-object/tests/measure_outcome_sensitivity.py
OUT=/data/vxiao/bimanual_manipulation/dsrl-scan-object/outputs
GAP_CKPT=/data/vxiao/bimanual_manipulation/GAP-tte-flattenmlp-concatdit-actionrope/data/outputs/2026.09.13/17.24.29_GAP_scan_object_actionrope-baseline/checkpoints/200.ckpt

CUDA_VISIBLE_DEVICES=0 $PY $SCRIPT --gap_ckpt $GAP_CKPT --env_seed 1001 \
    --n_scenes 10 --n_noise 3 --step_lim 500 --expert_check --fixed_noise \
    2>&1 | tee $OUT/sens_control_fixednoise_steps10.out &
pid_a=$!
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT --gap_ckpt $GAP_CKPT --env_seed 1001 \
    --n_scenes 15 --n_noise 3 --step_lim 500 --expert_check \
    2>&1 | tee $OUT/sens_fixedtrack_steps10.out &
pid_b=$!
wait $pid_a; rc_a=$?
wait $pid_b; rc_b=$?
echo "=== rc_control=$rc_a rc_fixedtrack=$rc_b ==="
