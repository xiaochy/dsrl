#!/bin/bash
# Decisive cheap gate for the unsquashed-noise fix: a FRESH, untrained actor
# with squash_actions=False + standard_gauss_init=True is EXACTLY N(0,1)
# (verified to 4 digits on std/kurtosis/tail mass vs a true-normal control,
# tests/compare_noise_distribution.py). So if the 35pp gap to the plain-policy
# baseline really is caused by tanh squashing the DDIM init noise, this run
# must score ~70-75%, not the 40.0% the SQUASHED untrained actor scored under
# the identical protocol (n=30, same eval seed, 2026-10-16's am sweep).
#
# No training happens: DSRL_TOTAL_TIMESTEPS=1, prefill skipped (it only fills
# the replay buffer and cannot touch actor weights). ~40 min.
set -o pipefail
cd /data/vxiao/bimanual_manipulation/RoboTwin

CUDA_VISIBLE_DEVICES=1 DSRL_REWARD_SOURCE=track DSRL_INIT_ROLLOUT_STEPS=0 \
    DSRL_TOTAL_TIMESTEPS=1 DSRL_LEARNING_STARTS=1 \
    DSRL_EVAL_FREQ=100000 DSRL_EVAL_EPISODES=30 \
    /home/veraxiao/miniconda3/envs/RoboTwin/bin/python3 \
    /data/vxiao/bimanual_manipulation/dsrl-scan-object/train_dsrl_scan_object.py \
    train.squash_actions=False train.standard_gauss_init=True \
    train.action_magnitude=1.0 train.target_ent=363.2 \
    env.max_episode_steps=500 \
    name=scan_object_unsquashed_init_check \
    log_dir=/data/vxiao/bimanual_manipulation/dsrl-scan-object/logs \
    2>&1 | tee /data/vxiao/bimanual_manipulation/dsrl-scan-object/outputs/unsquashed_init_check.out
echo "=== rc=$? ==="
