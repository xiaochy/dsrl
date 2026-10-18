#!/bin/bash
#SBATCH --job-name=dsrl_train_track_lowutd_3k
#SBATCH --partition=all
#SBATCH --exclude=grogu-4-3,grogu-4-23,grogu-4-18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --output=/project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/outputs/dsrl_train_track_lowutd_3k-%j.out

# Follow-up to job 4031196 (init_rollout_steps=1500 prefill alone did NOT
# stop the entropy collapse -- ent_coef still crashed 0.7->~0.01-0.02).
# This time also drop utd 20->5: DSRL-reference uses utd=20 WITH n_envs=4
# (4 parallel envs per gradient-update cycle), so their effective
# updates-per-unique-env-step ratio is 20/4=5. We only have n_envs=1, so
# utd=20 was giving us 4x the per-sample gradient pressure their own
# config implies -- this matches that ratio instead of blindly copying
# their raw utd value. Keeps the 1500-step prefill fix too (both pieces
# together, not isolating further yet -- isolate only if this also fails).

set -o pipefail
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh
conda activate RoboTwin

cd /project_data/held/vxiao/Bimanual_Manipulation_Project/RoboTwin

DSRL_REWARD_SOURCE=track DSRL_TOTAL_TIMESTEPS=3000 DSRL_LEARNING_STARTS=1 \
    DSRL_LOG_FREQ=10 DSRL_EVAL_FREQ=500 \
    python3 /project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/train_dsrl_scan_object.py \
    use_wandb=False env.max_episode_steps=500 train.utd=5 \
    name=scan_object_track_lowutd_3k \
    log_dir=/project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/logs
rc=$?
echo "=== train_dsrl_scan_object.py rc=$rc ==="
