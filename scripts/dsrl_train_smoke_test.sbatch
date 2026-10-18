#!/bin/bash
#SBATCH --job-name=dsrl_train_smoke_test
#SBATCH --partition=dheld
#SBATCH --exclude=grogu-4-3,grogu-4-23,grogu-4-18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/data/vxiao/bimanual_manipulation/dsrl-scan-object/outputs/dsrl_train_smoke_test-%j.out

# Validates train_dsrl_scan_object.py end to end: SAC construction, the
# replay buffer filling from real env.step() transitions, and at least one
# gradient update actually running -- against real SAPIEN, not a mock env.
# Tiny budget (40 chunk-steps, learning_starts=1, short episodes,
# wandb disabled) -- a wiring check, not a real training run.

set -o pipefail
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh
conda activate RoboTwin

# Same reason as dsrl_env_pilot.sbatch: RoboTwin's envs module needs
# cwd == ROBOTWIN_ROOT for its relative asset paths; hydra.job.chdir=False
# in cfg/dsrl_scan_object.yaml keeps it that way through training too.
cd /data/vxiao/bimanual_manipulation/RoboTwin

DSRL_TOTAL_TIMESTEPS=40 DSRL_LEARNING_STARTS=1 DSRL_LOG_FREQ=1 \
    python3 /data/vxiao/bimanual_manipulation/dsrl-scan-object/train_dsrl_scan_object.py \
    use_wandb=False env.max_episode_steps=40 log_dir=/data/vxiao/bimanual_manipulation/dsrl-scan-object/logs
rc=$?
echo "=== train_dsrl_scan_object.py rc=$rc ==="
