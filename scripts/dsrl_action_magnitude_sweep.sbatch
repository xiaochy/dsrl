#!/bin/bash
#SBATCH --job-name=dsrl_action_magnitude_sweep
#SBATCH --partition=all
#SBATCH --exclude=grogu-4-3,grogu-4-23,grogu-4-18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/outputs/dsrl_action_magnitude_sweep-%j.out

# Sweeps action_magnitude for the FRESH/untrained-actor step-0 check
# (dsrl_initial_policy_check.sbatch's pattern): is the ~40-50% starting
# point found at action_magnitude=1.5 (2026-10-15) something a different
# noise-box size can move closer to the ~75% plain-policy baseline,
# independent of the (separately broken) training dynamics? No training
# here at all -- same reasoning as dsrl_initial_policy_check.sbatch:
# DSRL_INIT_ROLLOUT_STEPS=0 (irrelevant, doesn't touch actor weights),
# DSRL_TOTAL_TIMESTEPS=1 (minimum to exercise the code path, no real
# training), 30 eval episodes. Requires ACTION_MAGNITUDE via
# --export=ALL,ACTION_MAGNITUDE=<float>.

set -o pipefail
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh
conda activate RoboTwin

: "${ACTION_MAGNITUDE:?must pass --export=ALL,ACTION_MAGNITUDE=<float>}"

cd /project_data/held/vxiao/Bimanual_Manipulation_Project/RoboTwin

DSRL_REWARD_SOURCE=track DSRL_INIT_ROLLOUT_STEPS=0 DSRL_TOTAL_TIMESTEPS=1 \
    DSRL_LEARNING_STARTS=1 DSRL_EVAL_FREQ=100000 DSRL_EVAL_EPISODES=30 \
    python3 /project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/train_dsrl_scan_object.py \
    use_wandb=False env.max_episode_steps=500 train.action_magnitude="${ACTION_MAGNITUDE}" \
    name="scan_object_initial_policy_check_am${ACTION_MAGNITUDE}" \
    log_dir=/project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/logs
rc=$?
echo "=== train_dsrl_scan_object.py rc=$rc action_magnitude=${ACTION_MAGNITUDE} ==="
