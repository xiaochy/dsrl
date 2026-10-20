"""Collects a batch of real scan_object observations to an .npz so that
inspect_actor.py can diagnose any number of SAC checkpoints without paying
the env-construction cost (GAP + dynamics model + SAPIEN, ~4 min) every time.

Observations are taken along UNSTEERED-diffusion rollouts (noise drawn the way
collect_init_rollouts in train_dsrl_scan_object.py draws it), so the batch
covers both reset states and mid-episode states -- the actor's saturation
behaviour can differ between the two, and a reset-only batch would hide that.

Usage (from ROBOTWIN_ROOT, see dsrl_train_smoke_test.sbatch for why cwd matters):
    python3 tests/collect_obs_cache.py --out outputs/obs_cache.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from gym_env import DSRLScanObjectGymEnv  # noqa: E402


def _default_ckpts():
    cfg_path = os.path.join(os.path.dirname(_HERE), "cfg", "dsrl_scan_object.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["gap_ckpt_path"], cfg["dynamics_ckpt_path"]


def main():
    gap_default, dyn_default = _default_ckpts()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--steps_per_episode", type=int, default=8)
    ap.add_argument("--gap_ckpt", default=gap_default)
    ap.add_argument("--dynamics_ckpt", default=dyn_default)
    ap.add_argument("--action_magnitude", type=float, default=1.5)
    ap.add_argument("--reward_source", default="track")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=7000)  # distinct from training (0)
                                                       # and eval (seed+1000) batches
    args = ap.parse_args()

    env = DSRLScanObjectGymEnv(
        gap_ckpt=args.gap_ckpt, dynamics_ckpt=args.dynamics_ckpt,
        device=args.device, seed=args.seed, step_lim=500,
        action_magnitude=args.action_magnitude, reward_source=args.reward_source,
    )
    act_dim = env.action_space.shape[0]
    rng = np.random.default_rng(args.seed)

    obs_list = []
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            obs_list.append(obs.copy())
            for t in range(args.steps_per_episode):
                # Same clipped-N(0,1) noise collect_init_rollouts uses, so the
                # states visited are the ones unsteered GAP actually produces.
                noise = np.clip(rng.standard_normal(act_dim),
                                -args.action_magnitude, args.action_magnitude)
                obs, _, terminated, truncated, _ = env.step(noise.astype(np.float32))
                obs_list.append(obs.copy())
                if terminated or truncated:
                    break
            print(f"[collect] episode {ep + 1}/{args.episodes}: {len(obs_list)} obs so far")
    finally:
        env.close()

    obs_arr = np.stack(obs_list).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(args.out, obs=obs_arr, action_magnitude=args.action_magnitude)
    print(f"[collect] saved {obs_arr.shape} to {args.out}")


if __name__ == "__main__":
    main()
