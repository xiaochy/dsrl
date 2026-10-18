#!/bin/bash
#SBATCH --job-name=action_svm_sanity_check
#SBATCH --partition=dheld
#SBATCH --exclude=grogu-4-3,grogu-4-23,grogu-4-18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/outputs/action_svm_sanity-%j.out

set -o pipefail
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh
conda activate RoboTwin

GAP_DIR=/project_data/held/vxiao/Bimanual_Manipulation_Project/GAP-tte-flattenmlp-concatdit-actionrope
cd /project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object
python3 action_svm_sanity_check.py \
    --chain_dir "${GAP_DIR}/chain_results/scan_object_svm_sanity_large" \
    --rollouts_json "${GAP_DIR}/results_svm_sanity_pilot_large/scan_object/GAP/demo_clean_depth/demo_clean/seed_0/200/_rollouts.json"
rc=$?
echo "=== action_svm_sanity_check.py rc=$rc ==="
