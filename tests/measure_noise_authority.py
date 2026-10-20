"""SUPERSEDED by tests/compare_policy_paths.py (2026-10-19), which measures
the same authority ratio AND checks path equivalence at the same time, and by
tests/sweep_inference_steps_authority.py, which measures it against step
count on one paired set of states. Kept only so the numbers in
daily_logs/2026-10-19.md can be traced back to the code that produced them;
do not extend this file.

Measures how much the DDIM initial noise x_T actually moves GAP's output
action chunk -- i.e. whether DSRL has a lever to pull at all.

Motivation: with squash_actions=False the SAC actor's noise is now EXACTLY
N(0,1) (verified to 4 digits), yet the paired step-0 eval scored 11/26
against the squashed actor's 11/26 on the same scenes, with 22/26 episodes
agreeing in outcome (vs 0.51 expected under independence) and the episodes
both arms solved differing in length by only ~2.6 steps. So the noise
perturbs the trajectory but never flips the result.

The suspected mechanism is the checkpoint's own diffusion parameterization:

    prediction_type: 'sample'    the network maps x_t -> x_0 DIRECTLY, rather
                                 than predicting the noise residual. It is
                                 trained to map ANY noisy input to the clean
                                 action, so x_T is largely washed out at the
                                 very first of only 10 DDIM steps.
    clip_sample: True            each step clips the predicted sample, further
                                 contracting whatever spread survives.

If that is right, DSRL cannot work on THIS policy no matter how SAC is
tuned: the steering variable barely reaches the objective. The number that
decides it is the ratio

    (spread of actions across noise draws at a FIXED obs)
  / (spread of actions across observations)

A ratio near 1 means noise has real authority. Near 0 means the policy is
effectively a deterministic function of the observation. For reference, the
project already recorded an adjacent measurement without naming it as this:
the action-SVM's candidate spread was ~0.004 across noise draws vs ~3 across
states (daily_logs/2026-10-18.md), a ratio of ~1/750.

Usage:
    python3 tests/measure_noise_authority.py --n_obs 8 --n_draws 32
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from gym_env import DSRLScanObjectGymEnv  # noqa: E402


def main():
    with open(os.path.join(_ROOT, "cfg", "dsrl_scan_object.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_obs", type=int, default=8, help="distinct observations")
    ap.add_argument("--n_draws", type=int, default=32, help="noise draws per obs")
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.join(_ROOT, "results", "noise_authority.json"))
    args = ap.parse_args()

    env = DSRLScanObjectGymEnv(
        task_name="scan_object", task_config="demo_clean_depth",
        gap_ckpt=cfg["gap_ckpt_path"], dynamics_ckpt=cfg["dynamics_ckpt_path"],
        device=args.device, seed=args.seed, step_lim=500,
        action_magnitude=1.0, reward_source="track",
    )
    inner = env.env  # DSRLScanObjectEnv

    # Walk one episode with unsteered noise, and at each of n_obs visited
    # states re-query the policy n_draws times from DIFFERENT x_T. The env is
    # never stepped by these extra queries -- base_policy() only runs the
    # frozen DDIM reverse process, so the rollout stays on one trajectory and
    # every draw at a given state sees the identical observation.
    import torch

    def _obs_t(raw):
        # Same construction as DSRLScanObjectEnv.step, so the policy sees
        # byte-identical input to a real rollout query.
        o = inner._encode(raw)
        d = {"agent_pos": torch.as_tensor(o["state"], dtype=torch.float32).unsqueeze(0),
             "dino_features": torch.as_tensor(o["dino_features"], dtype=torch.float32).unsqueeze(0)}
        if o.get("patch_centers") is not None:
            d["patch_centers"] = torch.as_tensor(o["patch_centers"],
                                                 dtype=torch.float32).unsqueeze(0)
        return d

    per_obs = []           # [n_obs][n_draws, H, action_dim]
    env.reset()
    rng = np.random.default_rng(0)
    D = env.action_space.shape[0]
    for t in range(args.n_obs):
        obs_t = _obs_t(inner._last_observation)
        chunks = [np.asarray(inner.base_policy(
                      obs_t, rng.standard_normal(D).astype(np.float32)[None])[0])
                  for _ in range(args.n_draws)]
        per_obs.append(np.stack(chunks))
        # advance the trajectory one chunk with a fresh unsteered draw
        env.step(rng.standard_normal(D).astype(np.float32))
        print(f"[authority] state {t + 1}/{args.n_obs} done")
    env.close()

    A = np.stack(per_obs)                      # [n_obs, n_draws, H, D]
    within = A.std(axis=1).mean()              # spread across NOISE, per state
    means = A.mean(axis=1)                     # [n_obs, H, D]
    across = means.std(axis=0).mean()          # spread across STATES
    scale = A.reshape(-1, A.shape[-1]).std(axis=0).mean()

    print(f"\naction chunk {A.shape[2]}x{A.shape[3]}   "
          f"{A.shape[0]} states x {A.shape[1]} noise draws")
    print(f"  spread across NOISE draws (fixed state) : {within:.6f}")
    print(f"  spread across STATES (noise-averaged)   : {across:.6f}")
    print(f"  overall action scale                    : {scale:.6f}")
    print(f"\n  AUTHORITY RATIO  noise / state = {within / across:.5f}")
    print("  ~1 means the noise genuinely selects the action; near 0 means the")
    print("  policy is a deterministic function of the observation and DSRL has")
    print("  nothing to steer.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import json
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"n_obs": args.n_obs, "n_draws": args.n_draws, "seed": args.seed,
                   "within_noise_std": float(within), "across_state_std": float(across),
                   "action_scale": float(scale),
                   "authority_ratio": float(within / across)}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
