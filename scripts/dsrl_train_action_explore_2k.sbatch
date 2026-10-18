#!/bin/bash
#SBATCH --job-name=dsrl_train_action_explore_2k
#SBATCH --partition=dheld
#SBATCH --exclude=grogu-4-3,grogu-4-23,grogu-4-18,grogu-4-13
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/data/vxiao/bimanual_manipulation/dsrl-scan-object/outputs/dsrl_train_action_explore_2k-%j.out

# Cautious first step for the NEW action-SVM reward (DSRL_REWARD_SOURCE=
# action, no dynamics model -- see env.py's DSRLScanObjectEnv docstring):
# same 2000-chunk-step exploratory budget as the first track-reward run
# (job 4028647), but with DeterministicEvalCallback wired in from the
# START this time (that run initially lacked it, which is exactly what
# made its training-rollout success_rate number look misleadingly low --
# see daily_logs/2026-10-13.md). action_svm_sanity_check.py already
# confirmed AUC=0.9693 on real executed actions (job 4029228) before
# spending this budget.
#
# First attempt (job 4029232, grogu-4-13) OOM'd: train_env + eval_env each
# held a full GAPPolicyWrapper+DINOv2 AND an independent MVActionSVMReward
# DINOv2 copy -- 4 DINOv2 instances on a 12GB card. Fixed by reusing GAP's
# own DINOv2 for the action-SVM (env.py's DSRLScanObjectEnv now passes
# dino=self.model.dino_model.dinov2); also excluding grogu-4-13 (smallest
# GPU in the partition) here for extra margin.

set -o pipefail
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh
conda activate RoboTwin

cd /data/vxiao/bimanual_manipulation/RoboTwin

DSRL_REWARD_SOURCE=action DSRL_TOTAL_TIMESTEPS=2000 DSRL_LEARNING_STARTS=1 \
    DSRL_LOG_FREQ=10 DSRL_EVAL_FREQ=200 DSRL_EVAL_EPISODES=3 \
    python3 /data/vxiao/bimanual_manipulation/dsrl-scan-object/train_dsrl_scan_object.py \
    use_wandb=False env.max_episode_steps=500 \
    name=scan_object_action_explore_2k \
    log_dir=/data/vxiao/bimanual_manipulation/dsrl-scan-object/logs
rc=$?
echo "=== train_dsrl_scan_object.py rc=$rc ==="
