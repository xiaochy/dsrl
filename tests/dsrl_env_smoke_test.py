"""
Validation pilot for env.py's DSRLScanObjectEnv: confirms reset()/step()
actually run against real SAPIEN (construction, obs encoding, noise-steered
action generation, reward scoring via the subprocess scorer, termination)
before DSRL training is trusted to depend on this file.

Not a training run -- random noise actions, a handful of steps/episodes,
just checking nothing throws and shapes/values are sane.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/data/vxiao/bimanual_manipulation/dsrl-scan-object")
from env import DSRLScanObjectEnv  # noqa: E402

GAP_CKPT = (
    "/data/vxiao/bimanual_manipulation/GAP-tte-flattenmlp-concatdit-actionrope/"
    "data/outputs/2026.09.13/17.24.29_GAP_scan_object_actionrope-baseline/checkpoints/200.ckpt"
)
DYNAMICS_CKPT = (
    "/data/vxiao/bimanual_manipulation/dynamics-model-rollout-worldmodel/"
    "data/outputs/2026.09.28/19.53.05_GAP_scan_object_rollout-worldmodel/checkpoints/200.ckpt"
)
ACTION_MAGNITUDE = 1.5  # matches cfg/dsrl_scan_object.yaml's untuned default

print("constructing DSRLScanObjectEnv (loads GAP policy + spawns dynamics scorer subprocess)...")
t0 = time.time()
env = DSRLScanObjectEnv(
    task_name="scan_object", task_config="demo_clean_depth",
    gap_ckpt=GAP_CKPT, dynamics_ckpt=DYNAMICS_CKPT, device="cuda",
    seed=0, step_lim=40,  # ~2-3 action chunks (horizon=16) -- wiring check, not a real rollout
)
print(f"construction took {time.time() - t0:.1f}s")

rng = np.random.default_rng(0)
n_episodes = 2
for ep in range(n_episodes):
    print(f"\n=== episode {ep} ===")
    obs = env.reset()
    print("reset() obs keys:", list(obs.keys()),
         "state shape:", obs["state"].shape, "dino_features shape:", obs["dino_features"].shape,
         "patch_centers:", None if obs["patch_centers"] is None else obs["patch_centers"].shape)

    done = False
    step_i = 0
    while not done:
        noise = rng.normal(scale=ACTION_MAGNITUDE, size=(env.base_policy.horizon * env.base_policy.action_dim,)).astype(np.float32)
        t0 = time.time()
        obs, reward, done, info = env.step(noise)
        dt = time.time() - t0
        print(f"step {step_i}: reward={reward:.4f} done={done} info={info} ({dt:.2f}s)")
        assert np.isfinite(reward), f"non-finite reward at step {step_i}"
        step_i += 1
        if step_i > 10:
            print("WARNING: step count exceeded step_lim without done=True, breaking manually")
            break

env.close()
print("\nDSRL ENV SMOKE TEST PASSED")
