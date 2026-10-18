#!/bin/bash
#SBATCH --job-name=dsrl_train_continue_10k
#SBATCH --partition=dheld
#SBATCH --exclude=grogu-4-3,grogu-4-23,grogu-4-18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=/project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/outputs/dsrl_train_continue_10k-%j.out

# Continues the 2000-step exploratory run (job 4028647,
# logs/scan_object-dsrl/scan_object_explore_2k/checkpoint/final.zip) up to
# 10,000 total timesteps -- matching the scale Robometer (arXiv:2603.02115,
# Appendix E-2) reports for DSRL+learned-reward-model training (10k
# timesteps, 20%->85% single-stage / 20%->70% two-stage). Their wall-clock
# was <=45 min; ours is ~9-10h at ~3.5-4s/chunk-step (real SAPIEN sim +
# dynamics-model + SVM reward subprocess, not a fast sim) -- user's own
# explicit choice (2026-10-13) to spend this budget after validating
# DeterministicEvalCallback + DSRL_RESUME_FROM in a 20-step smoke test
# (job 4029042, rc=0 -- resumed correctly from step 2000, eval callback
# ran clean against the held-out-seed eval env).
#
# Checkpoints saved every 1000 steps (not just at the end) in case this
# 12h job times out or dies partway -- can resume again from whichever
# checkpoint is most recent via DSRL_RESUME_FROM.

set -o pipefail
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh
conda activate RoboTwin

cd /project_data/held/vxiao/Bimanual_Manipulation_Project/RoboTwin

RESUME_CKPT=/project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/logs/scan_object-dsrl/scan_object_explore_2k/checkpoint/final.zip

DSRL_RESUME_FROM="$RESUME_CKPT" DSRL_TOTAL_TIMESTEPS=8000 DSRL_LEARNING_STARTS=1 \
    DSRL_LOG_FREQ=10 DSRL_EVAL_FREQ=500 DSRL_EVAL_EPISODES=3 \
    python3 /project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/train_dsrl_scan_object.py \
    use_wandb=False env.max_episode_steps=500 save_model_interval=1000 save_replay_buffer=true \
    name=scan_object_explore_10k_continued \
    log_dir=/project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/logs
rc=$?
echo "=== train_dsrl_scan_object.py rc=$rc ==="
