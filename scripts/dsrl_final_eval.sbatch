#!/bin/bash
#SBATCH --job-name=dsrl_final_eval
#SBATCH --partition=dheld
#SBATCH --exclude=grogu-4-3,grogu-4-23,grogu-4-18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=/data/vxiao/bimanual_manipulation/dsrl-scan-object/outputs/dsrl_final_eval-%j.out

# Final deterministic 3-seed x 50-episode eval of a trained DSRL checkpoint
# -- replaces the noisy n=3/point running estimate logged during training.
# Requires CKPT, REWARD_SOURCE, SEED via --export=ALL,CKPT=...,REWARD_SOURCE=...,SEED=<n>

set -o pipefail
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh
conda activate RoboTwin

: "${CKPT:?must pass --export=ALL,CKPT=<path>}"
: "${REWARD_SOURCE:?must pass --export=ALL,REWARD_SOURCE=track|action}"
: "${SEED:?must pass --export=ALL,SEED=<n>}"

GAP_CKPT=/data/vxiao/bimanual_manipulation/GAP-tte-flattenmlp-concatdit-actionrope/data/outputs/2026.09.13/17.24.29_GAP_scan_object_actionrope-baseline/checkpoints/200.ckpt
DYNAMICS_CKPT=/data/vxiao/bimanual_manipulation/dynamics-model-rollout-worldmodel/data/outputs/2026.09.28/19.53.05_GAP_scan_object_rollout-worldmodel/checkpoints/200.ckpt

cd /data/vxiao/bimanual_manipulation/RoboTwin
python3 /data/vxiao/bimanual_manipulation/dsrl-scan-object/dsrl_final_eval.py \
    --ckpt "$CKPT" --gap_ckpt "$GAP_CKPT" --dynamics_ckpt "$DYNAMICS_CKPT" \
    --reward_source "$REWARD_SOURCE" --seed "$SEED" --n_episodes 50
rc=$?
echo "=== dsrl_final_eval.py rc=$rc seed=$SEED reward_source=$REWARD_SOURCE ==="
