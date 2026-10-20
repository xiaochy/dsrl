#!/bin/bash
# Measures RoboTwin's expert-check pass rate (eval_policy.py:200's hardcoded
# expert_check=True) on two seed batches:
#   100200000+30 -- the EXACT batch every DSRL step-0/eval number used
#                   (eval_seed = cfg.seed+1000 = 1001 -> 100000*(1+1001))
#   500100000+30 -- the batch the 75.33% baseline was drawn from (--seed 5000)
# env.py:21 skips that filter, so all DSRL evals are UNCONDITIONED rates.
# Expert planning only: no GAP ckpt, no DinoV2, no diffusion.
set -o pipefail
cd /data/vxiao/bimanual_manipulation/RoboTwin
PY=/home/veraxiao/miniconda3/envs/RoboTwin/bin/python3
SCRIPT=/data/vxiao/bimanual_manipulation/dsrl-scan-object/tests/measure_expert_pass_rate.py
OUT=/data/vxiao/bimanual_manipulation/dsrl-scan-object/outputs

CUDA_VISIBLE_DEVICES=0 $PY $SCRIPT --start_seed 100200000 --n 30 \
    2>&1 | tee $OUT/expert_pass_dsrlbatch.out &
pid_a=$!
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT --start_seed 500100000 --n 30 \
    2>&1 | tee $OUT/expert_pass_baselinebatch.out &
pid_b=$!
wait $pid_a; rc_a=$?
wait $pid_b; rc_b=$?
echo "=== rc_dsrlbatch=$rc_a rc_baselinebatch=$rc_b ==="
