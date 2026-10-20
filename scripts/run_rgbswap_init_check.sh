#!/bin/bash
# Step-0 eval of a FRESH, untrained actor with the RGB channel-swap bug fixed
# (env.py passed swap_rgb_channels=False and never swapped, so every DSRL run
# so far fed true-colour RGB into DinoV2 while this GAP checkpoint was trained
# on channel-SWAPPED features -- docs/color-channel-issue.md).
#
# Deliberately the ORIGINAL squashed parameterization (action_magnitude=1.5,
# target_ent=174.5), so this is a one-variable change against 2026-10-16's
# 40.0% point: same protocol, same eval seed, n=30. If the channel swap is
# the ~35pp gap, this scores ~70-75%.
#
# No training: DSRL_TOTAL_TIMESTEPS=1, prefill skipped. ~40 min.
set -o pipefail
cd /data/vxiao/bimanual_manipulation/RoboTwin

CUDA_VISIBLE_DEVICES=1 DSRL_REWARD_SOURCE=track DSRL_INIT_ROLLOUT_STEPS=0 \
    DSRL_TOTAL_TIMESTEPS=1 DSRL_LEARNING_STARTS=1 \
    DSRL_EVAL_FREQ=100000 DSRL_EVAL_EPISODES=30 \
    /home/veraxiao/miniconda3/envs/RoboTwin/bin/python3 \
    /data/vxiao/bimanual_manipulation/dsrl-scan-object/train_dsrl_scan_object.py \
    train.squash_actions=True \
    train.action_magnitude=1.5 train.target_ent=174.5 \
    env.max_episode_steps=500 \
    name=scan_object_rgbswap_init_check \
    log_dir=/data/vxiao/bimanual_manipulation/dsrl-scan-object/logs \
    2>&1 | tee /data/vxiao/bimanual_manipulation/dsrl-scan-object/outputs/rgbswap_init_check.out
echo "=== rc=$? ==="
