#!/bin/bash
#SBATCH --job-name=dsrl_initial_policy_check
#SBATCH --partition=all
#SBATCH --exclude=grogu-4-3,grogu-4-23,grogu-4-18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/outputs/dsrl_initial_policy_check-%j.out

# Standalone check: does a FRESH, completely untrained SAC actor's noise
# (fed through the real, frozen GAP policy) give a success rate anywhere
# close to the plain-policy baseline (70-75%)? This is the "step=0" sanity
# point the entropy-collapse investigation (2026-10-15) wanted but never
# had cleanly, run with enough episodes (30, not 3 or 10) to actually trust
# the number -- this job does nothing BUT that one eval round:
# DSRL_INIT_ROLLOUT_STEPS=0 (irrelevant here -- prefill only touches the
# replay buffer, never the actor's weights, so it can't affect this result,
# and skipping it saves ~1.5-1.75h), DSRL_TOTAL_TIMESTEPS=1 (SAC must still
# be constructed via the normal code path, but we don't want any real
# training -- 1 step is the minimum that exercises .learn() without a
# second eval round firing, since eval_freq is set far above it).

set -o pipefail
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh
conda activate RoboTwin

cd /project_data/held/vxiao/Bimanual_Manipulation_Project/RoboTwin

DSRL_REWARD_SOURCE=track DSRL_INIT_ROLLOUT_STEPS=0 DSRL_TOTAL_TIMESTEPS=1 \
    DSRL_LEARNING_STARTS=1 DSRL_EVAL_FREQ=100000 DSRL_EVAL_EPISODES=30 \
    python3 /project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/train_dsrl_scan_object.py \
    use_wandb=False env.max_episode_steps=500 \
    name=scan_object_initial_policy_check \
    log_dir=/project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/logs
rc=$?
echo "=== train_dsrl_scan_object.py rc=$rc ==="
