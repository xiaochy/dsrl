"""Level-2 diagnostic: measures whether a trained SAC actor's noise output is
still distributionally usable as DDIM's initial x_T, without running a single
episode (a success-rate eval costs hours; this costs seconds).

The quantities and their reference values come from the 2026-10-13 collapse
diagnosis (daily_logs/2026-10-13.md):

    output norm   16.0 = true N(0,1) over 256 dims (sqrt(256))
                  24.0 = fully saturated at +-action_magnitude (16 * 1.5)
                  15.7 = a fresh, untrained actor

                  CORRECTED 2026-10-19. This block used to assert "22.6 = the
                  collapsed 10k-step actor -- 94% of the way to a hypercube
                  vertex". Re-measured across EVERY SAC checkpoint on disk,
                  that number does not reproduce: the highest norm found is
                  17.5 (scan_object_action_explore_2k/final, 39,980 updates),
                  and the collapsed runs sit at 14.1-17.5 -- i.e. at or BELOW
                  the healthy 16.0, not approaching 24.0. sat_frac is
                  0.003-0.030 where a true N(0,1) already has 0.136 of its
                  mass outside +-1.5, and draw_std is 0.53-0.59, not near
                  zero. So the actors are mildly UNDER-dispersed, not
                  saturated.

                  The entropy collapse itself is real and severe (entropy
                  174 -> 5.75 -> -39.98 nats), but it happens by `sigma`
                  shrinking (1.094 -> 0.705) with a modest mean drift, NOT by
                  tanh saturation. The "saturation -> Jacobian term inflates
                  log_prob -> fools the entropy tuner" chain cannot be the
                  driver at sat_frac ~0.02. The simpler mechanism, consistent
                  with the 2026-10-18 target_ent correction, is that
                  target_ent=0.0 is satisfied with enormous slack, so the
                  tuner never pushes back, alpha -> 0, and the actor freely
                  narrows itself to exploit the critic.

    |mean_z|      the pre-tanh Gaussian mean. |z| > 3 already gives
                  |tanh(z)| > 0.995, so a value well above ~3 would mean tanh
                  had destroyed the distinction between different z. Measured
                  range across all checkpoints: 0.03-0.93, i.e. nowhere near
                  it.

    draw_std      per-dim std across stochastic draws at a fixed obs. The
                  collapsed actor had near-zero variance here -- exploration
                  was dead even though SAC was still "sampling".

The healthy reference is not synthesized -- it already exists on disk as
`scan_object_initial_policy_check/checkpoint/final.zip`, the 2026-10-15
standalone job that ran only the step-0 eval and so never took a gradient
step. The `updates` column reports each checkpoint's own gradient-update
count, so an untrained reference is self-evident rather than asserted.

Usage:
    python3 tests/inspect_actor.py --obs outputs/obs_cache.npz --target_ent 174.5 \\
        --ckpt logs/scan_object-dsrl/scan_object_initial_policy_check/checkpoint/final.zip \\
        --ckpt logs/scan_object-dsrl/scan_object_explore_10k_continued/checkpoint/dsrl_policy_10000_steps.zip \\
        --ckpt logs/scan_object-dsrl/scan_object_targetent_diag/checkpoint/final.zip

Pass --target_ent to also get the entropy-tuner balance term
(log_prob + target_ent), which is ~0 exactly when the entropy constraint is
satisfied -- see daily_logs/2026-10-16.md.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "..", "DSRL-reference", "stable-baselines3"))

from stable_baselines3 import SAC  # noqa: E402


def _stats(actor, obs_t: torch.Tensor, action_magnitude: float, n_draws: int,
           target_ent: float | None):
    with torch.no_grad():
        mean_z, log_std, _ = actor.get_action_dist_params(obs_t)

        # Stochastic draws at the SAME observations, to measure whether the
        # policy still has any spread left to explore with.
        draws = torch.stack([actor.action_log_prob(obs_t)[0] for _ in range(n_draws)])
        draw_std = draws.std(dim=0).mean().item() * action_magnitude

        actions, log_prob = actor.action_log_prob(obs_t)
        scaled = actions * action_magnitude
        norm = scaled.norm(dim=1).mean().item()
        # Marginal stats of the EMITTED noise, i.e. of x_T itself. Reference
        # values for a true N(0,1) over 256 dims: mean 0, std 1, and min/max
        # around -+3.2 (the expected extreme of 256 draws). A squashed actor
        # cannot exceed +-action_magnitude by construction, so min/max pinned
        # at exactly that bound is saturation, not a heavy tail.
        sat_frac = (scaled.abs() > 0.99 * action_magnitude).float().mean().item()

        out = {
            "norm": norm,
            "mean": scaled.mean().item(),
            "std": scaled.std().item(),
            "min": scaled.min().item(),
            "max": scaled.max().item(),
            "sat_frac": sat_frac,
            "|mean_z|": mean_z.abs().mean().item(),
            "sigma": log_std.exp().mean().item(),
            "draw_std": draw_std,
            "log_prob": log_prob.mean().item(),
        }
    out["entropy"] = -out["log_prob"]
    if target_ent is not None:
        out["balance"] = out["log_prob"] + target_ent
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True, help="npz from collect_obs_cache.py")
    ap.add_argument("--ckpt", action="append", default=[], required=True,
                    help="repeatable; include scan_object_initial_policy_check's "
                         "final.zip as the untrained reference")
    ap.add_argument("--target_ent", type=float, default=None,
                    help="if given, also report (log_prob + target_ent); ~0 means "
                         "the entropy tuner is at equilibrium")
    ap.add_argument("--n_draws", type=int, default=16)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cache = np.load(args.obs)
    obs = cache["obs"]
    # action_magnitude is a PER-RUN quantity, so it cannot be taken from the
    # obs cache and applied to every checkpoint: doing that reported
    # scan_object_unsquashed_init_check (which ran action_magnitude=1.0) with
    # the cache's 1.5, i.e. std 1.500 / norm 23.98 / max 5.64 instead of its
    # true std 1.000 / norm 15.99 / max 3.76 -- a row that looks like gross
    # saturation but is purely a scale error (found 2026-10-19). Each
    # checkpoint's own resolved hydra config is the authority; the cache value
    # is only a fallback for checkpoints whose run dir is gone.
    cache_magnitude = float(cache["action_magnitude"])
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=args.device)
    n_dims = None

    def run_magnitude(ckpt: str) -> float:
        """action_magnitude from this checkpoint's own run config."""
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
        cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
        try:
            with open(cfg_path, encoding="utf-8") as f:
                return float(yaml.safe_load(f)["train"]["action_magnitude"])
        except Exception:
            print(f"  [warn] no action_magnitude in {cfg_path}; "
                  f"falling back to the obs cache's {cache_magnitude}")
            return cache_magnitude

    rows = []
    magnitudes = {}
    for ckpt in args.ckpt:
        model = SAC.load(ckpt, device=args.device)
        n_dims = model.action_space.shape[0]
        am = run_magnitude(ckpt)
        magnitudes[ckpt] = am
        st = _stats(model.policy.actor, obs_t, am,
                    args.n_draws, args.target_ent)
        st["updates"] = float(getattr(model, "_n_updates", float("nan")))
        st["a_mag"] = am
        rows.append((os.path.basename(os.path.dirname(os.path.dirname(ckpt)))
                     + "/" + os.path.basename(ckpt), st))
        del model

    _common_magnitude = (max(set(magnitudes.values()),
                             key=list(magnitudes.values()).count)
                         if magnitudes else cache_magnitude)

    # True-N(0,1) control, pushed through the SAME stat expressions as every
    # actor row -- so any difference below is the actor's, not the
    # measurement's. This is the distribution GAP's DDIM was trained to start
    # from, and therefore the only meaningful reference for all of these
    # columns (not just the norm).
    if n_dims:
        g = torch.Generator().manual_seed(0)
        ctrl = torch.randn((obs_t.shape[0], n_dims), generator=g)
        rows.insert(0, ("[control] true N(0,1), same code path", {
            "updates": float("nan"),
            "norm": ctrl.norm(dim=1).mean().item(),
            "mean": ctrl.mean().item(),
            "std": ctrl.std().item(),
            "min": ctrl.min().item(),
            "max": ctrl.max().item(),
            "a_mag": float("nan"),
            # The control is a true N(0,1); the fraction of its mass outside
            # any given bound is a property of N(0,1), so report it against
            # the most common magnitude among the rows.
            "sat_frac": (ctrl.abs() > 0.99 * _common_magnitude).float().mean().item(),
            "|mean_z|": float("nan"), "sigma": float("nan"),
            "draw_std": float("nan"), "log_prob": float("nan"),
            "entropy": float("nan"), "balance": float("nan"),
        }))

    ref_clean = float(np.sqrt(n_dims)) if n_dims else float("nan")
    ref_sat = ref_clean * _common_magnitude
    print(f"\nobs batch: {obs.shape}   n_dims={n_dims}   "
          f"action_magnitude per run (a_mag column); control uses "
          f"{_common_magnitude}")
    print(f"reference norms: {ref_clean:.1f} = true N(0,1)   "
          f"{ref_sat:.1f} = fully saturated\n")

    cols = ["updates", "a_mag", "mean", "std", "min", "max", "norm", "sat_frac",
            "|mean_z|", "sigma", "draw_std",
            "log_prob", "entropy"]
    if args.target_ent is not None:
        cols.append("balance")
    head = f"{'checkpoint':<52}" + "".join(f"{c:>11}" for c in cols)
    print(head)
    print("-" * len(head))
    for name, st in rows:
        print(f"{name[:52]:<52}" + "".join(f"{st[c]:>11.3f}" for c in cols))

    print(f"\nnorm near {ref_clean:.1f} = healthy noise distribution; "
          f"near {ref_sat:.1f} = collapsed to a hypercube vertex.")
    print("sat_frac near 1.0, draw_std near 0, or |mean_z| >> 3 each independently "
          "indicate tanh saturation.")


if __name__ == "__main__":
    main()
