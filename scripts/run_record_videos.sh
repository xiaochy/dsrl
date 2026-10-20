#!/bin/bash
# Head-camera video of the flipped rollouts, so the flip can be WATCHED and
# not just counted. The numbers (9/15 scenes flip at 1 step, 0/15 at 10) say
# the noise decides the outcome at 1 step; they say nothing about what it
# changes behaviourally. Three flipped scenes at 1 step, plus the same scene
# at the deployed 10 steps as a contrast where all three draws should look
# identical.
set -o pipefail
cd /data/vxiao/bimanual_manipulation/RoboTwin
PY=/home/veraxiao/miniconda3/envs/RoboTwin/bin/python3
SCRIPT=/data/vxiao/bimanual_manipulation/dsrl-scan-object/tests/record_rollout_videos.py
OUT=/data/vxiao/bimanual_manipulation/dsrl-scan-object/outputs
VID=/data/vxiao/bimanual_manipulation/dsrl-scan-object/results/videos
GAP_CKPT=/data/vxiao/bimanual_manipulation/GAP-tte-flattenmlp-concatdit-actionrope/data/outputs/2026.09.13/17.24.29_GAP_scan_object_actionrope-baseline/checkpoints/200.ckpt

# steps=1, the three scenes whose measured outcomes were [1,0,1] / [1,0,0] / [1,1,0]
CUDA_VISIBLE_DEVICES=0 $PY $SCRIPT --gap_ckpt $GAP_CKPT \
    --scenes 100200009 100200025 100200034 --n_noise 3 \
    --num_inference_steps 1 --step_lim 500 --video_dir $VID \
    2>&1 | tee $OUT/record_videos_steps1.out &
pid_a=$!
# steps=10 contrast on one of the same scenes (measured 0/15 flips)
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT --gap_ckpt $GAP_CKPT \
    --scenes 100200009 --n_noise 3 --step_lim 500 --video_dir $VID \
    2>&1 | tee $OUT/record_videos_steps10.out &
pid_b=$!
wait $pid_a; rc_a=$?
wait $pid_b; rc_b=$?
echo "=== rc_steps1=$rc_a rc_steps10=$rc_b ==="
ls -la $VID
