#!/bin/bash
# Authority ratio vs num_inference_steps, paired on one fixed set of real
# observations. Attacks the ACTUAL blocker (noise authority ~1/370 at the
# deployed 10 steps) rather than a suspected wiring fault -- under
# prediction_type='sample', fewer denoising steps leave x_T less washed out.
# Free: no retraining, num_inference_steps is a plain attribute read by
# conditional_sample_from_noise's set_timesteps (gap.py:1851).
set -o pipefail
cd /data/vxiao/bimanual_manipulation/RoboTwin
CUDA_VISIBLE_DEVICES=0 /home/veraxiao/miniconda3/envs/RoboTwin/bin/python3 \
    /data/vxiao/bimanual_manipulation/dsrl-scan-object/tests/sweep_inference_steps_authority.py \
    --steps 10 5 3 2 1 --n_draws 32 --n_states 6 \
    2>&1 | tee /data/vxiao/bimanual_manipulation/dsrl-scan-object/outputs/inference_steps_sweep.out
echo "=== rc=$? ==="
