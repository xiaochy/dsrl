"""
Does the DDIM initial noise gain authority over the action when the policy
is run with FEWER denoising steps?

Background. tests/compare_policy_paths.py measured an authority ratio of
0.0028 at the deployed num_inference_steps=10 -- the action chunk moves
0.00038 across noise draws at a fixed state vs 0.135 across states, ~1/370.
The frozen policy is for practical purposes a deterministic function of the
observation, which is why DSRL has nothing to steer.

The mechanism is the checkpoint's `prediction_type='sample'`: the network
emits x_hat_0 = f(x_t, t, obs) directly and was trained to map ANY noisy
input to the clean action, so each step pulls the sample toward the
conditional mean and x_T is progressively washed out. Fewer steps means
less washing: at 1 step the output is literally net(x_T, T, obs), where x_T
is the only input besides the observation. (Under 'epsilon' -- what
DSRL-reference's base policy uses, cfg/robomimic/dsrl_square.yaml:81 --
x_T -> x_0 is an approximately invertible learned flow and this problem
does not arise.)

This is free: no retraining, and `num_inference_steps` is a plain attribute
read by conditional_sample_from_noise's set_timesteps call (gap.py:1851).

Design note. The expensive part is building real observations in SAPIEN, so
states are collected ONCE and every step count is measured on the same
states -- the comparison across step counts is paired, and the across-state
denominator of the authority ratio is recomputed per step count because the
policy's mean prediction itself changes with the step count.

Both paths are measured at every step count: B (externally supplied N(0,1),
the DSRL env's path) and A (GAPPolicyWrapper.get_action, GAP's own eval
entry point with its own torch.randn). A matters because it makes the
result a statement about the policy, not about DSRL's plumbing.

A ratio that stays ~0.003 at 1-2 steps kills noise-steering on this
checkpoint outright. A ratio that climbs toward 0.1-1 says DSRL is viable
at a reduced step count -- and then success rate at that step count has to
be measured before anything is built on it, since fewer steps also means a
worse policy.

Usage (from ROBOTWIN_ROOT, RoboTwin conda env):
    python3 tests/sweep_inference_steps_authority.py --steps 10 5 3 2 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from env import DSRLScanObjectEnv  # noqa: E402


def main():
    with open(os.path.join(_ROOT, "cfg", "dsrl_scan_object.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[10, 5, 3, 2, 1])
    ap.add_argument("--n_draws", type=int, default=32)
    ap.add_argument("--n_states", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.join(_ROOT, "results",
                                                  "inference_steps_authority.json"))
    args = ap.parse_args()

    # reward_source="none": the reward models play no part in this
    # measurement, and skipping them avoids loading the dynamics model and
    # spawning the scorer subprocess.
    env = DSRLScanObjectEnv(
        task_name="scan_object", task_config="demo_clean_depth",
        gap_ckpt=cfg["gap_ckpt_path"], dynamics_ckpt="", device=args.device,
        seed=args.seed, step_lim=500, reward_source="none",
    )
    model = env.model
    policy = model.policy_model
    default_steps = policy.num_inference_steps
    D = policy.horizon * policy.action_dim
    rng = np.random.default_rng(0)

    from deploy_policy import encode_obs as gap_encode  # env.py put GAP_DIR on sys.path

    # --- collect the states once -------------------------------------------
    states = []
    env.reset()
    for t in range(args.n_states):
        raw = env._last_observation
        o = env._encode(raw)
        obs_t = {"agent_pos": torch.as_tensor(o["state"], dtype=torch.float32).unsqueeze(0),
                 "dino_features": torch.as_tensor(o["dino_features"],
                                                  dtype=torch.float32).unsqueeze(0)}
        if o.get("patch_centers") is not None:
            obs_t["patch_centers"] = torch.as_tensor(o["patch_centers"],
                                                     dtype=torch.float32).unsqueeze(0)
        states.append({"obs_t": obs_t, "gap_obs": gap_encode(raw, env.TASK_ENV)})
        env.step(rng.standard_normal(D).astype(np.float32))
        print(f"[states] collected {t + 1}/{args.n_states}", flush=True)
    env.close()
    print(f"[states] done; deployed num_inference_steps={default_steps}", flush=True)

    # --- sweep --------------------------------------------------------------
    rows = []
    for k in args.steps:
        policy.num_inference_steps = k
        A_states, B_states = [], []
        for i, st in enumerate(states):
            go = st["gap_obs"]
            A = np.stack([np.asarray(model.get_action(
                              go["rgb"], go["state"],
                              patch_centers=go.get("patch_centers"))[0])
                          for _ in range(args.n_draws)])
            B = np.stack([np.asarray(env.base_policy(
                              st["obs_t"],
                              rng.standard_normal(D).astype(np.float32)[None])[0])
                          for _ in range(args.n_draws)])
            A_states.append(A)
            B_states.append(B)
            print(f"[steps={k}] state {i + 1}/{args.n_states}", flush=True)

        A, B = np.stack(A_states), np.stack(B_states)        # [S, n, H, d]
        row = {"num_inference_steps": k}
        for tag, X in (("A", A), ("B", B)):
            within = float(X.std(axis=1).mean())             # across noise, fixed state
            across = float(X.mean(axis=1).std(axis=0).mean())  # across states
            row[f"within_noise_{tag}"] = within
            row[f"across_state_{tag}"] = across
            row[f"authority_{tag}"] = within / across if across else float("nan")
        rows.append(row)
        print(f"  >> steps={k:>3}  authority A={row['authority_A']:.5f} "
              f"B={row['authority_B']:.5f}  "
              f"(within A={row['within_noise_A']:.6f} across A={row['across_state_A']:.6f})",
              flush=True)

    policy.num_inference_steps = default_steps

    print(f"\n{'steps':>6}  {'authority_A':>12}  {'authority_B':>12}  "
          f"{'within_A':>10}  {'across_A':>10}")
    for r in rows:
        print(f"{r['num_inference_steps']:>6}  {r['authority_A']:>12.5f}  "
              f"{r['authority_B']:>12.5f}  {r['within_noise_A']:>10.6f}  "
              f"{r['across_state_A']:>10.6f}")
    print("\n>> near 1 the noise selects the action; near 0 the policy is a "
          "deterministic function of the observation")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"n_draws": args.n_draws, "n_states": args.n_states,
                   "seed": args.seed, "deployed_steps": default_steps,
                   "rows": rows}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
