"""Settles, by measurement rather than code reading, whether the DSRL env's
policy path produces the same actions as GAP's own eval path -- and how much
authority the DDIM initial noise has over the result at all.

Why: an UNTRAINED SAC actor emits noise indistinguishable from N(0,1) (verified
to 4 digits, tests/compare_noise_distribution.py), yet scores ~40% where GAP's
own eval_policy.py scores 75.33% on the same task and checkpoint. Three
hypotheses have now been eliminated by measurement:

    squashed vs unsquashed noise   0.400 vs 0.400, n=30 each, and 22/26 of the
                                   same scenes agree in outcome
    action_magnitude 0.3 - 1.5     40-50%, no trend (2026-10-16)
    RGB channel order              0.400 (false, correct) vs 0.238 (true);
                                   eval.sh:31 confirms false is what the
                                   baseline ran

So the difference is in the path, not the noise. This script compares the two
paths' OUTPUT DISTRIBUTIONS at one fixed real observation, which needs no
exact noise matching (the policy is joint-diffusion: the track branch draws
from the same generator, so reproducing one path's action noise in the other
still leaves the track noise different, and tracks influence actions):

    set A   GAPPolicyWrapper.get_action(...)        -- GAP's own eval path
    set B   GAPBasePolicyWrapper(obs_t, noise)      -- the DSRL env's path

If the two sets are draws from the same predictive distribution, the paths
agree and the 35pp is in scene/episode setup instead. If their MEANS differ
by much more than the standard error of either mean, the paths differ, and
the per-dimension difference says where to look.

The same draws give the authority ratio: spread across noise at a fixed
state, over spread across states. Near 0 means the policy is effectively a
deterministic function of the observation and noise-steering has no lever.
NOTE: a low ratio here is NOT evidence about `prediction_type='sample'` on
its own -- the action-SVM's documented 0.004-vs-3 candidate spread
(daily_logs/2026-10-13.md) is a property of that REWARD MODEL, not of this
policy, and must not be cited for this.

Usage:
    python3 tests/compare_policy_paths.py --n_draws 32 --n_states 6
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

from gym_env import DSRLScanObjectGymEnv  # noqa: E402


def main():
    with open(os.path.join(_ROOT, "cfg", "dsrl_scan_object.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_draws", type=int, default=32)
    ap.add_argument("--n_states", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.join(_ROOT, "results", "policy_path_compare.json"))
    args = ap.parse_args()

    env = DSRLScanObjectGymEnv(
        task_name="scan_object", task_config="demo_clean_depth",
        gap_ckpt=cfg["gap_ckpt_path"], dynamics_ckpt=cfg["dynamics_ckpt_path"],
        device=args.device, seed=args.seed, step_lim=500,
        action_magnitude=1.0, reward_source="track",
    )
    inner = env.env
    model = inner.model
    D = env.action_space.shape[0]
    rng = np.random.default_rng(0)

    A_states, B_states = [], []
    env.reset()
    for t in range(args.n_states):
        raw = inner._last_observation
        o = inner._encode(raw)
        obs_t = {"agent_pos": torch.as_tensor(o["state"], dtype=torch.float32).unsqueeze(0),
                 "dino_features": torch.as_tensor(o["dino_features"],
                                                  dtype=torch.float32).unsqueeze(0)}
        if o.get("patch_centers") is not None:
            obs_t["patch_centers"] = torch.as_tensor(o["patch_centers"],
                                                     dtype=torch.float32).unsqueeze(0)

        # Path A: GAP's own eval entry point, its own internal noise draw.
        # deploy_policy is importable because env.py put GAP_DIR on sys.path.
        from deploy_policy import encode_obs as gap_encode
        go = gap_encode(raw, inner.TASK_ENV)
        A = np.stack([np.asarray(model.get_action(
                          go["rgb"], go["state"],
                          patch_centers=go.get("patch_centers"))[0])
                      for _ in range(args.n_draws)])

        # Path B: the DSRL env's path, externally supplied N(0,1) noise.
        B = np.stack([np.asarray(inner.base_policy(
                          obs_t, rng.standard_normal(D).astype(np.float32)[None])[0])
                      for _ in range(args.n_draws)])

        A_states.append(A)
        B_states.append(B)
        env.step(rng.standard_normal(D).astype(np.float32))
        print(f"[paths] state {t + 1}/{args.n_states}: A{A.shape} B{B.shape}")
    env.close()

    A = np.stack(A_states)   # [S, n, H, d]
    B = np.stack(B_states)
    mA, mB = A.mean(axis=1), B.mean(axis=1)                 # [S, H, d]
    # Standard error of each mean, pooled over the chunk.
    seA = (A.std(axis=1) / np.sqrt(A.shape[1])).mean()
    seB = (B.std(axis=1) / np.sqrt(B.shape[1])).mean()
    gap = np.abs(mA - mB).mean()

    withinA, withinB = A.std(axis=1).mean(), B.std(axis=1).mean()
    acrossA, acrossB = mA.std(axis=0).mean(), mB.std(axis=0).mean()

    print(f"\naction chunk {A.shape[2]}x{A.shape[3]}   "
          f"{A.shape[0]} states x {A.shape[1]} draws per path")
    print(f"\n  PATH AGREEMENT")
    print(f"    mean |A_mean - B_mean|            : {gap:.6f}")
    print(f"    standard error of a mean (A / B)  : {seA:.6f} / {seB:.6f}")
    print(f"    ratio gap / max(se)               : {gap / max(seA, seB):.2f}")
    print("    >> ~1-3 means the two paths agree; >>3 means they differ")
    print(f"\n  NOISE AUTHORITY")
    print(f"    spread across noise, fixed state (A / B) : {withinA:.6f} / {withinB:.6f}")
    print(f"    spread across states                (A / B) : {acrossA:.6f} / {acrossB:.6f}")
    print(f"    authority ratio                     (A / B) : "
          f"{withinA / acrossA:.5f} / {withinB / acrossB:.5f}")
    print("    >> near 1 the noise selects the action; near 0 the policy is")
    print("       a deterministic function of the observation")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"n_draws": args.n_draws, "n_states": args.n_states,
                   "seed": args.seed,
                   "path_gap": float(gap), "se_A": float(seA), "se_B": float(seB),
                   "gap_over_se": float(gap / max(seA, seB)),
                   "within_noise_A": float(withinA), "within_noise_B": float(withinB),
                   "across_state_A": float(acrossA), "across_state_B": float(acrossB),
                   "authority_A": float(withinA / acrossA),
                   "authority_B": float(withinB / acrossB)}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
