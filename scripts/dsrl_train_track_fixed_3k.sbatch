#!/bin/bash
#SBATCH --job-name=dsrl_train_track_fixed_3k
#SBATCH --partition=all
#SBATCH --exclude=grogu-4-3,grogu-4-23,grogu-4-18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --output=/project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/outputs/dsrl_train_track_fixed_3k-%j.out

# First real test of the 2026-10-15 entropy-collapse fixes (init_rollout_steps
# prefill, fixed eval seeds, num_timesteps logging, pre-training baseline
# eval, n_eval_episodes=10) -- FRESH run (not resumed), real step_lim=500,
# track reward (same reward source as the original 10k run that showed the
# collapse, for a clean before/after comparison). Moderate budget (3000
# training steps after the 1500-step prefill) -- a check that the fix
# actually keeps ent_coef/entropy healthy and the deterministic eval trend
# recovers, before committing to a full 10k-scale run.
#
# Cost estimate: ~1.5-1.75h prefill (1500 steps @ ~3.5-4s) + ~3.3-3.6h
# training (3000 steps) + ~1.5-2h eval overhead (6 rounds @ 10 episodes,
# eval_freq=500) -- roughly 7-8h total.

set -o pipefail
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh
conda activate RoboTwin

cd /project_data/held/vxiao/Bimanual_Manipulation_Project/RoboTwin

DSRL_REWARD_SOURCE=track DSRL_TOTAL_TIMESTEPS=3000 DSRL_LEARNING_STARTS=1 \
    DSRL_LOG_FREQ=10 DSRL_EVAL_FREQ=500 \
    python3 /project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/train_dsrl_scan_object.py \
    use_wandb=False env.max_episode_steps=500 \
    name=scan_object_track_fixed_3k \
    log_dir=/project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/logs
rc=$?
echo "=== train_dsrl_scan_object.py rc=$rc ==="
