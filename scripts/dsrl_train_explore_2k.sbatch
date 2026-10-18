#!/bin/bash
#SBATCH --job-name=dsrl_train_explore_2k
#SBATCH --partition=dheld
#SBATCH --exclude=grogu-4-3,grogu-4-23,grogu-4-18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/outputs/dsrl_train_explore_2k-%j.out

# Medium-scale exploratory DSRL training run (user's own choice, 2026-10-13):
# NOT a real/final training run -- 2000 chunk-steps (~3-4s/step incl. SAC
# gradient updates, utd=20) just to see whether ep_rew_mean/success_rate
# show any upward trend at all before committing to a longer (12h+) run.
# env.max_episode_steps=500 matches scan_object's real eval step_lim (the
# dsrl_train_smoke_test.sbatch pilot used an artificial 40 for speed only).

set -o pipefail
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh
conda activate RoboTwin

cd /project_data/held/vxiao/Bimanual_Manipulation_Project/RoboTwin

DSRL_TOTAL_TIMESTEPS=2000 DSRL_LEARNING_STARTS=1 DSRL_LOG_FREQ=10 \
    python3 /project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/train_dsrl_scan_object.py \
    use_wandb=False env.max_episode_steps=500 \
    name=scan_object_explore_2k \
    log_dir=/project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/logs
rc=$?
echo "=== train_dsrl_scan_object.py rc=$rc ==="
