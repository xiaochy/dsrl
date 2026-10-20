"""Level-3 diagnostic: is the SAC actor's output actually distributed like the
N(0,1) that GAP's DDIM reverse process was trained to start from?

Why this and not another success-rate eval: the success gap is already
established and paired -- 75.33% with true N(0,1) vs 44.67% with a trained
actor vs ~40% with an UNTRAINED one, on the same seeds 5000/5001/5002
(daily_logs/2026-10-13.md, 2026-10-15.md). What is not established is WHICH
distributional property causes it, and that determines the fix. The
2026-10-13 diagnosis only ever compared the output NORM (15.6 untrained vs
16.0 = sqrt(256) for true N(0,1)) and concluded the untrained actor was
"healthy". A matching norm says almost nothing: tanh squashing is a hard
bound, so the actor physically cannot emit the 13.4% of N(0,1) mass that
lies beyond +-action_magnitude, and it can hold the norm constant while
redistributing that mass anywhere inside the box.

Measured here, all against a true-N(0,1) reference run through the identical
measurement code so the reference line is a control, not an analytic claim:

    std / kurtosis    shape of the marginal. N(0,1) has std 1, kurtosis 3.
                      Sub-3 kurtosis with std ~1 means mass pushed OUT toward
                      the bound (shoulders), i.e. the box edge is binding.
    tail mass         P(|x| > t) at t = 1, 1.5, 2, 3. Everything past
                      action_magnitude is structurally unreachable, so the
                      t=1.5 and t=2,3 rows quantify exactly how much of
                      DDIM's expected input support is missing.
    KS                pooled one-sample KS against N(0,1).
    mean |corr|       mean |correlation| over off-diagonal dim pairs, across
                      stochastic draws at fixed observations. True DDIM noise
                      is i.i.d. across dims; a shared obs-dependent mean
                      shift would show up here and is NOT fixed by removing
                      squashing, so this separates two candidate causes.

Usage:
    python3 tests/compare_noise_distribution.py --obs outputs/obs_cache.npz \\
        --ckpt logs/scan_object-dsrl/scan_object_initial_policy_check/checkpoint/final.zip \\
        --ckpt logs/scan_object-dsrl/scan_object_targetent_utd5_10k/checkpoint/final.zip
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "..", "DSRL-reference", "stable-baselines3"))

from stable_baselines3 import SAC  # noqa: E402

_THRESH = (1.0, 1.5, 2.0, 3.0)


def _norm_cdf(x):
    return 0.5 * (1.0 + torch.erf(torch.as_tensor(x, dtype=torch.float64) / np.sqrt(2.0))).numpy()


def _describe(samples: np.ndarray, label: str, action_magnitude: float):
    """samples: (n_draws, batch, n_dims), already scaled to noise units."""
    flat = samples.reshape(-1).astype(np.float64)
    mu, sd = flat.mean(), flat.std()
    kurt = ((flat - mu) ** 4).mean() / sd ** 4

    # Pooled one-sample KS vs N(0,1). Subsampled: the full flat array is
    # ~1e7 values and the statistic is already stable at 1e5.
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(flat, size=min(100_000, flat.size), replace=False))
    emp = np.arange(1, sub.size + 1) / sub.size
    ks = float(np.abs(emp - _norm_cdf(sub)).max())

    tails = {t: float((np.abs(flat) > t).mean()) for t in _THRESH}

    # Cross-dim correlation across draws, at each observation separately, then
    # averaged -- this isolates the sampling noise's own dependence structure
    # from genuine obs-to-obs variation in the mean.
    cors = []
    for b in range(samples.shape[1]):
        x = samples[:, b, :]                       # (n_draws, n_dims)
        if x.shape[0] < 3:
            continue
        c = np.corrcoef(x, rowvar=False)
        iu = np.triu_indices_from(c, k=1)
        cors.append(np.nanmean(np.abs(c[iu])))
    mean_abs_corr = float(np.mean(cors)) if cors else float("nan")

    # Fraction of dims pinned against the box edge.
    at_bound = float((np.abs(flat) > 0.99 * action_magnitude).mean())

    return {"label": label, "mean": mu, "std": sd, "kurtosis": kurt, "ks": ks,
            "at_bound": at_bound, "mean_abs_corr": mean_abs_corr, **tails}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--ckpt", action="append", default=[], required=True)
    ap.add_argument("--n_draws", type=int, default=256)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cache = np.load(args.obs)
    obs = cache["obs"]
    action_magnitude = float(cache["action_magnitude"])
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=args.device)

    rows = []
    n_dims = None
    for ckpt in args.ckpt:
        model = SAC.load(ckpt, device=args.device)
        n_dims = model.action_space.shape[0]
        actor = model.policy.actor
        with torch.no_grad():
            draws = torch.stack([actor.action_log_prob(obs_t)[0]
                                 for _ in range(args.n_draws)])
        s = (draws * action_magnitude).cpu().numpy()
        name = (os.path.basename(os.path.dirname(os.path.dirname(ckpt)))
                + f"  (upd={int(getattr(model, '_n_updates', -1))})")
        rows.append(_describe(s, name, action_magnitude))
        del model

    # The control: true N(0,1), same shape, same measurement code.
    rng = np.random.default_rng(1)
    ref = rng.standard_normal((args.n_draws, obs.shape[0], n_dims))
    rows.append(_describe(ref, "TRUE N(0,1)  (what DDIM expects)", action_magnitude))

    cols = ["mean", "std", "kurtosis", "ks", "at_bound", "mean_abs_corr",
            *_THRESH]
    hdr = {1.0: "P|x|>1", 1.5: "P|x|>1.5", 2.0: "P|x|>2", 3.0: "P|x|>3"}
    print(f"\nobs batch {obs.shape}   n_draws={args.n_draws}   "
          f"n_dims={n_dims}   action_magnitude={action_magnitude}")
    head = f"{'policy':<44}" + "".join(
        f"{hdr.get(c, c):>13}" for c in cols)
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['label'][:44]:<44}" + "".join(f"{r[c]:>13.4f}" for c in cols))

    print(f"\nEverything beyond |x| > {action_magnitude} is structurally "
          f"unreachable for a tanh-squashed actor.")
    print(f"True N(0,1) puts {rows[-1][1.5]:.1%} of its mass there "
          f"(at action_magnitude={action_magnitude}).")


if __name__ == "__main__":
    main()
