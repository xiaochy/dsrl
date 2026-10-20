#!/bin/bash
# One-episode check of the sub-step video hook before committing to a batch.
# Baseline for comparison: the first recordings measured 15.2-16.1 frames per
# chunk (one frame per take_action, written BEFORE that action executes), which
# is why the arm teleported between waypoints.
set -o pipefail
cd /data/vxiao/bimanual_manipulation/RoboTwin
CUDA_VISIBLE_DEVICES=0 /home/veraxiao/miniconda3/envs/RoboTwin/bin/python3 \
    /data/vxiao/bimanual_manipulation/dsrl-scan-object/tests/record_rollout_videos.py \
    --gap_ckpt /data/vxiao/bimanual_manipulation/GAP-tte-flattenmlp-concatdit-actionrope/data/outputs/2026.09.13/17.24.29_GAP_scan_object_actionrope-baseline/checkpoints/200.ckpt \
    --scenes 100200009 --n_noise 1 --num_inference_steps 1 --step_lim 500 \
    --video_substep_freq 10 --video_fps 30 \
    --video_dir /data/vxiao/bimanual_manipulation/dsrl-scan-object/results/videos_smooth \
    2>&1 | tee /data/vxiao/bimanual_manipulation/dsrl-scan-object/outputs/smooth_video_check.out
echo "=== rc=$? ==="
