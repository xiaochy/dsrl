#!/bin/bash
#SBATCH --job-name=dsrl_stochastic_eval_check
#SBATCH --partition=all
#SBATCH --exclude=grogu-4-3,grogu-4-23,grogu-4-18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/outputs/dsrl_stochastic_eval_check-%j.out

# Tests whether DeterministicEvalCallback's deterministic=True (the
# actor's MEAN noise, std~0.49, zero sampling entropy -- confirmed via a
# direct pre-tanh probe) is itself an unfair way to measure a noise-
# steering policy, vs. deterministic=False (a real stochastic sample,
# std~exp(log_std)~1, much closer to the i.i.d. N(0,1) draw GAP's
# conditional_sample_from_noise is actually valid for). Same fresh/
# untrained-actor, no-training, 30-episode pattern as
# dsrl_initial_policy_check.sbatch / dsrl_action_magnitude_sweep.sbatch --
# only difference is DSRL_EVAL_DETERMINISTIC=0.
# action_magnitude=1.5 landed at 40.0% deterministic (job 4031202) --
# this checks the same policy init class, same magnitude, sampled instead.

set -o pipefail
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh
conda activate RoboTwin

cd /project_data/held/vxiao/Bimanual_Manipulation_Project/RoboTwin

DSRL_REWARD_SOURCE=track DSRL_INIT_ROLLOUT_STEPS=0 DSRL_TOTAL_TIMESTEPS=1 \
    DSRL_LEARNING_STARTS=1 DSRL_EVAL_FREQ=100000 DSRL_EVAL_EPISODES=30 \
    DSRL_EVAL_DETERMINISTIC=0 \
    python3 /project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/train_dsrl_scan_object.py \
    use_wandb=False env.max_episode_steps=500 train.action_magnitude=1.5 \
    name=scan_object_stochastic_eval_check \
    log_dir=/project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/logs
rc=$?
echo "=== train_dsrl_scan_object.py rc=$rc deterministic=False ==="
