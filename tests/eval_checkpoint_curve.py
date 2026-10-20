"""Builds a deterministic-eval learning curve by sweeping a run's periodic
checkpoints against ONE env instance.

Why not just call dsrl_final_eval.py once per checkpoint: that pays the
~4-minute env construction (GAP + dynamics model + SAPIEN) for every point,
and -- more importantly -- it gives each point a fresh, never-repeated batch
of episodes, so the resulting curve mixes "did the policy improve" with "was
this batch easier". Here the env is built once and
DSRLScanObjectEnv.reset_seed_counter() re-syncs it before every checkpoint,
so every point is scored on the identical episodes.

Per-episode outcomes (and their episode seeds) are written to JSON, which is
what makes a PAIRED comparison between two runs possible afterwards: the
experiment and control runs share eval_seed, so at matched training steps
they see the same scenes, and a paired test is far more sensitive than
comparing two independent proportions.

Usage:
    python3 tests/eval_checkpoint_curve.py \\
        --ckpt_dir logs/scan_object-dsrl/scan_object_targetent_3k/checkpoint \\
        --seed 5000 --n_episodes 20 \\
        --out results/curve_targetent_3k.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "..", "DSRL-reference", "stable-baselines3"))

from stable_baselines3 import SAC  # noqa: E402

from gym_env import DSRLScanObjectGymEnv  # noqa: E402


def _wilson(n_succ: int, n: int, z: float = 1.96):
    """Wilson score interval -- behaves sensibly at small n and near 0/1,
    where the normal approximation can run past [0,1]."""
    if n == 0:
        return float("nan"), float("nan")
    p = n_succ / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def _step_of(path: str) -> int:
    m = re.search(r"_(\d+)_steps\.zip$", os.path.basename(path))
    return int(m.group(1)) if m else 1 << 30  # final.zip sorts last


def _default_ckpts():
    with open(os.path.join(_ROOT, "cfg", "dsrl_scan_object.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["gap_ckpt_path"], cfg["dynamics_ckpt_path"]


def main():
    gap_default, dyn_default = _default_ckpts()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", help="sweeps dsrl_policy_*_steps.zip + final.zip")
    ap.add_argument("--ckpt", action="append", default=[], help="explicit, repeatable")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n_episodes", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gap_ckpt", default=gap_default)
    ap.add_argument("--dynamics_ckpt", default=dyn_default)
    ap.add_argument("--reward_source", default="track", choices=["track", "action"])
    ap.add_argument("--step_lim", type=int, default=500)
    ap.add_argument("--action_magnitude", type=float, default=1.5)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    ckpts = list(args.ckpt)
    if args.ckpt_dir:
        ckpts += sorted(glob.glob(os.path.join(args.ckpt_dir, "dsrl_policy_*_steps.zip")),
                        key=_step_of)
        final = os.path.join(args.ckpt_dir, "final.zip")
        if os.path.exists(final):
            ckpts.append(final)
    if not ckpts:
        ap.error("no checkpoints found -- pass --ckpt_dir or --ckpt")
    print(f"[curve] {len(ckpts)} checkpoints, {args.n_episodes} episodes each, "
          f"seed={args.seed}")

    env = DSRLScanObjectGymEnv(
        task_name="scan_object", task_config="demo_clean_depth",
        gap_ckpt=args.gap_ckpt, dynamics_ckpt=args.dynamics_ckpt, device=args.device,
        seed=args.seed, step_lim=args.step_lim,
        action_magnitude=args.action_magnitude, reward_source=args.reward_source,
    )

    points = []
    try:
        for ckpt in ckpts:
            model = SAC.load(ckpt, device=args.device)
            # Identical episode batch for every checkpoint -- see module docstring.
            env.reset_seed_counter(args.seed)
            successes, ep_seeds = [], []
            for ep in range(args.n_episodes):
                ep_seeds.append(env.env._now_seed)
                obs, _ = env.reset()
                done, ok = False, False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    ok = ok or bool(info.get("success", False))
                successes.append(ok)
            n_succ = sum(successes)
            lo, hi = _wilson(n_succ, len(successes))
            pt = {
                "ckpt": ckpt,
                "num_timesteps": int(model.num_timesteps),
                "n_updates": int(getattr(model, "_n_updates", -1)),
                "n_episodes": len(successes),
                "n_success": n_succ,
                "success_rate": n_succ / len(successes),
                "wilson95": [lo, hi],
                "successes": successes,
                "episode_seeds": ep_seeds,
            }
            points.append(pt)
            print(f"[curve] step={pt['num_timesteps']:>5}  "
                  f"{n_succ}/{len(successes)} = {pt['success_rate']:.3f}  "
                  f"95% CI [{lo:.3f}, {hi:.3f}]  {os.path.basename(ckpt)}")
            del model
    finally:
        env.close()

    points.sort(key=lambda p: p["num_timesteps"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "seed": args.seed,
            "n_episodes": args.n_episodes,
            "reward_source": args.reward_source,
            "step_lim": args.step_lim,
            "action_magnitude": args.action_magnitude,
            "points": points,
        }, f, indent=2)

    print(f"\n{'step':>8}{'success':>10}{'95% CI':>20}")
    print("-" * 38)
    for p in points:
        lo, hi = p["wilson95"]
        print(f"{p['num_timesteps']:>8}{p['success_rate']:>10.3f}"
              f"{f'[{lo:.2f}, {hi:.2f}]':>20}")
    print(f"\nwrote {args.out}")
    print("references: ~0.40-0.50 = untrained SAC actor, 0.7533 = plain-policy "
          "baseline (daily_logs/2026-10-13.md)")


if __name__ == "__main__":
    main()
