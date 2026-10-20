#!/bin/bash
# steps=1 counterpart of run_sensitivity_control.sh (2026-10-19).
#
# The steps=1 arm is the only one whose flip rate is nonzero (0.6, headroom
# +0.289), so it is the only one whose attribution to the noise actually
# needs defending -- and it was measured at 15:57, before the track branch
# was pinned (gap_base_policy fixed_track_noise) and before --fixed_noise
# existed. The steps=10 control is near-vacuous by comparison: 45/45
# identical outcomes already prove determinism there.
#
#   GPU 0  DETERMINISM CONTROL: identical action noise on every draw.
#          MUST come back 0/15 flips. If it does not, the 0.6 flip rate and
#          the +0.289 headroom cannot be attributed to the noise and the
#          "1 step has real headroom" claim is dead.
#   GPU 1  varying action noise, track noise now held fixed per scene --
#          re-measures the 0.6 with the track confound removed, so flips are
#          attributable to the DSRL ACTION alone. Replaces the 15:57 number.
set -o pipefail
while pgrep -f "measure_outcome_sensitivity.py" > /dev/null; do sleep 30; done
cd /data/vxiao/bimanual_manipulation/RoboTwin
PY=/home/veraxiao/miniconda3/envs/RoboTwin/bin/python3
SCRIPT=/data/vxiao/bimanual_manipulation/dsrl-scan-object/tests/measure_outcome_sensitivity.py
OUT=/data/vxiao/bimanual_manipulation/dsrl-scan-object/outputs
GAP_CKPT=/data/vxiao/bimanual_manipulation/GAP-tte-flattenmlp-concatdit-actionrope/data/outputs/2026.09.13/17.24.29_GAP_scan_object_actionrope-baseline/checkpoints/200.ckpt

CUDA_VISIBLE_DEVICES=0 $PY $SCRIPT --gap_ckpt $GAP_CKPT --env_seed 1001 \
    --n_scenes 15 --n_noise 3 --step_lim 500 --expert_check \
    --num_inference_steps 1 --fixed_noise \
    2>&1 | tee $OUT/sens_control_fixednoise_steps1.out &
pid_a=$!
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT --gap_ckpt $GAP_CKPT --env_seed 1001 \
    --n_scenes 15 --n_noise 3 --step_lim 500 --expert_check \
    --num_inference_steps 1 \
    2>&1 | tee $OUT/sens_fixedtrack_steps1.out &
pid_b=$!
wait $pid_a; rc_a=$?
wait $pid_b; rc_b=$?
echo "=== rc_control=$rc_a rc_fixedtrack=$rc_b ==="
