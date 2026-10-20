#!/bin/bash
# Unsquashed-noise DSRL training, 10k steps. Gated on
# run_unsquashed_init_check.sh having reproduced the ~70-75% plain-policy
# baseline at step 0 -- without that, the parameterization change is not
# established and this run would just be a third way to sit at 40%.
#
# Config differences from the 2026-10-18 pair, and why each is forced:
#   squash_actions=False      the fix under test
#   standard_gauss_init=True  REQUIRED, not optional: with the default random
#                             Linear init on the log_std head the untrained
#                             actor measures std=1.20 / kurtosis=4.87, i.e.
#                             still not N(0,1) (verified in-process)
#   action_magnitude=1.0      makes predict()'s unscale_action the identity;
#                             train_dsrl_scan_object.py raises on anything else
#   target_ent=363.2          = 256 * 0.5*log(2*pi*e), the entropy of N(0,1)^256
#                             and the MEASURED log_prob of this actor (-362.9).
#                             The squashed arms' 174.5 is wrong here.
#   utd=5                     carried over from the 2026-10-19 arm. utd=20 and
#                             utd=5 both diverged the critic under the squashed
#                             parameterization, so this is not yet a settled
#                             choice -- see daily_logs/2026-10-19.md.
set -o pipefail
cd /data/vxiao/bimanual_manipulation/RoboTwin

CUDA_VISIBLE_DEVICES=${GPU:-1} DSRL_REWARD_SOURCE=track \
    DSRL_TOTAL_TIMESTEPS=10000 DSRL_EVAL_FREQ=2500 DSRL_EVAL_EPISODES=10 \
    DSRL_SB3_LOG_INTERVAL=1 \
    /home/veraxiao/miniconda3/envs/RoboTwin/bin/python3 \
    /data/vxiao/bimanual_manipulation/dsrl-scan-object/train_dsrl_scan_object.py \
    train.squash_actions=False train.standard_gauss_init=True \
    train.action_magnitude=1.0 train.target_ent=363.2 train.utd=5 \
    env.max_episode_steps=500 \
    name=scan_object_unsquashed_10k \
    log_dir=/data/vxiao/bimanual_manipulation/dsrl-scan-object/logs \
    2>&1 | tee /data/vxiao/bimanual_manipulation/dsrl-scan-object/outputs/unsquashed_10k.out
echo "=== rc=$? ==="
