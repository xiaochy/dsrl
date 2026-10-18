#!/bin/bash
#SBATCH --job-name=score_episode_viz
#SBATCH --partition=dheld
#SBATCH --exclude=grogu-4-3,grogu-4-23,grogu-4-18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/data/vxiao/bimanual_manipulation/dsrl-scan-object/outputs/score_episode_viz-%j.out

set -o pipefail
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh
conda activate RoboTwin

GAP_DIR=/data/vxiao/bimanual_manipulation/GAP-tte-flattenmlp-concatdit-actionrope
cd /data/vxiao/bimanual_manipulation/dsrl-scan-object

python3 score_episode_both_rewards.py \
    --chain_pkl "${GAP_DIR}/chain_results/scan_object_svm_sanity_large/episode0_chain.pkl" \
    --out viz_episode0_success_scores.json
python3 score_episode_both_rewards.py \
    --chain_pkl "${GAP_DIR}/chain_results/scan_object_svm_sanity_large/episode5_chain.pkl" \
    --out viz_episode5_failure_scores.json
rc=$?
echo "=== score_episode_both_rewards.py rc=$rc ==="
