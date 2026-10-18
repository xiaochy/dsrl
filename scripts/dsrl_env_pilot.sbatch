#!/bin/bash
#SBATCH --job-name=dsrl_env_pilot
#SBATCH --partition=dheld
#SBATCH --exclude=grogu-4-3,grogu-4-23,grogu-4-18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/data/vxiao/bimanual_manipulation/dsrl-scan-object/outputs/dsrl_env_pilot-%j.out

# Validates env.py's DSRLScanObjectEnv against real SAPIEN before DSRL
# training is trusted to depend on it: reset()/step() run without error,
# obs/reward shapes are sane, episodes terminate. See dsrl_env_smoke_test.py.

set -o pipefail
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh
conda activate RoboTwin

# RoboTwin's envs module loads assets via paths relative to cwd (e.g.
# "./assets/objects/objaverse/list.json") -- eval.sh itself `cd`s into
# ROBOTWIN_ROOT before running anything for the same reason (see its line
# "cd ${ROBOTWIN_ROOT}"). Must do the same here, not cd into dsrl-scan-object.
cd /data/vxiao/bimanual_manipulation/RoboTwin
ROBOTWIN_ROOT=/data/vxiao/bimanual_manipulation/RoboTwin \
    python3 /data/vxiao/bimanual_manipulation/dsrl-scan-object/dsrl_env_smoke_test.py
rc=$?
echo "=== dsrl_env_smoke_test.py rc=$rc ==="
