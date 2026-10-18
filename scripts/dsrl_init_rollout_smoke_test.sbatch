#!/bin/bash
#SBATCH --job-name=dsrl_init_rollout_smoke
#SBATCH --partition=dheld
#SBATCH --exclude=grogu-4-3,grogu-4-23,grogu-4-18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/data/vxiao/bimanual_manipulation/dsrl-scan-object/outputs/dsrl_init_rollout_smoke-%j.out

# Validates collect_init_rollouts (the init_rollout_steps pre-fill fix,
# 2026-10-15) before committing to a real 1500-step prefill: small budget
# (20 prefill steps + 20 training steps), checks the replay_buffer.add()
# shapes/scale_action call don't error and that SAC actually trains after
# the prefill (not just during it).

set -o pipefail
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh
conda activate RoboTwin

cd /data/vxiao/bimanual_manipulation/RoboTwin

DSRL_INIT_ROLLOUT_STEPS=20 DSRL_TOTAL_TIMESTEPS=20 DSRL_LEARNING_STARTS=1 \
    DSRL_LOG_FREQ=5 DSRL_EVAL_FREQ=10 DSRL_EVAL_EPISODES=2 \
    python3 /data/vxiao/bimanual_manipulation/dsrl-scan-object/train_dsrl_scan_object.py \
    use_wandb=False env.max_episode_steps=40 \
    name=scan_object_init_rollout_smoke \
    log_dir=/data/vxiao/bimanual_manipulation/dsrl-scan-object/logs
rc=$?
echo "=== train_dsrl_scan_object.py rc=$rc ==="
