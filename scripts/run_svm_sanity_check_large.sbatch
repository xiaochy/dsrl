#!/bin/bash
#SBATCH --job-name=svm_sanity_check_large
#SBATCH --partition=dheld
#SBATCH --exclude=grogu-4-3,grogu-4-23,grogu-4-18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/data/vxiao/bimanual_manipulation/dsrl-scan-object/outputs/svm_sanity_check_large-%j.out

# Re-runs svm_sanity_check.py against the larger 50-episode pilot
# (svm_reward_sanity_pilot_large.sbatch, job 4028506) to confirm the
# AUC~0.99 finding from the 15-episode pilot (only 2 failure episodes)
# holds with a less failure-thin sample (9 failures here).

set -o pipefail
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh
conda activate RoboTwin

GAP_DIR=/data/vxiao/bimanual_manipulation/GAP-tte-flattenmlp-concatdit-actionrope
cd /data/vxiao/bimanual_manipulation/dsrl-scan-object
python3 svm_sanity_check.py \
    --chain_dir "${GAP_DIR}/chain_results/scan_object_svm_sanity_large" \
    --rollouts_json "${GAP_DIR}/results_svm_sanity_pilot_large/scan_object/GAP/demo_clean_depth/demo_clean/seed_0/200/_rollouts.json" \
    --dynamics_ckpt /data/vxiao/bimanual_manipulation/dynamics-model-rollout-worldmodel/data/outputs/2026.09.28/19.53.05_GAP_scan_object_rollout-worldmodel/checkpoints/200.ckpt
rc=$?
echo "=== svm_sanity_check.py rc=$rc ==="
