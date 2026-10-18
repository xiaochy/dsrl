#!/bin/bash
#SBATCH --job-name=dsrl_eval_resume_smoke
#SBATCH --partition=dheld
#SBATCH --exclude=grogu-4-3,grogu-4-23,grogu-4-18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=00:45:00
#SBATCH --output=/project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/outputs/dsrl_eval_resume_smoke-%j.out

# Validates two new pieces before committing to a long run: (1)
# DeterministicEvalCallback actually runs against a separate held-out-seed
# eval_env and reports sane success_rate/reward without crashing or
# interfering with the training env's own rollout state, (2)
# DSRL_RESUME_FROM correctly resumes from the 2000-step exploratory run's
# checkpoint (num_timesteps continues from 2000, not reset to 0). Tiny
# budget (20 additional steps, eval every 10, 2 eval episodes) -- a
# wiring check, not a real training step.

set -o pipefail
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh
conda activate RoboTwin

cd /project_data/held/vxiao/Bimanual_Manipulation_Project/RoboTwin

RESUME_CKPT=/project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/logs/scan_object-dsrl/scan_object_explore_2k/checkpoint/final.zip

DSRL_RESUME_FROM="$RESUME_CKPT" DSRL_TOTAL_TIMESTEPS=20 DSRL_LEARNING_STARTS=1 \
    DSRL_LOG_FREQ=10 DSRL_EVAL_FREQ=10 DSRL_EVAL_EPISODES=2 \
    python3 /project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/train_dsrl_scan_object.py \
    use_wandb=False env.max_episode_steps=500 \
    name=scan_object_eval_resume_smoke \
    log_dir=/project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/logs
rc=$?
echo "=== train_dsrl_scan_object.py rc=$rc ==="
