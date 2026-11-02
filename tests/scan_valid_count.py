"""Is the number of patches carrying a track the bottleneck?

`measure_track_sensitivity.py` showed the track-SVM barely responds to its
track, and `diagnose_track_pathway.py` showed only ~24 of 391 patches ever
carry one (`valid_point_mask = depth_valid & is_object`). This script controls
that count directly: for k = 24, 50, 100, 200, 391 it picks k patches at
random, fills them with a random track of REALISTIC magnitude, marks exactly
those k valid, and measures |logit - logit(zero track)|.

    rising ~linearly in k -> coverage really is a binding constraint, so
                             widening it (is_object | is_robot, or a real
                             tracker) is worth doing
    saturating low        -> coverage is not the bottleneck; only the training
                             target can fix the insensitivity

Two arms, so "more valid tokens" is separated from "more content":

    content   k patches get a random track at the chunk's own GT rms
    marked    k patches get a ZERO track but are still marked valid
              (the `gt_allvalid` control, restricted to k patches: it changes
              which constant a token is, not whether it carries information)

The observation is held fixed per chunk exactly as in the other scripts: DINO
features and patch centres are computed once, only track/valid change.

    python tests/scan_valid_count.py \
        --chain_dir .../chain_results/scan_object_track_sensitivity \
        --out .../outputs/track_sens
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

SVM_REFERENCE = "/data/vxiao/bimanual_manipulation/SVM-reference"
sys.path.insert(0, SVM_REFERENCE)

from svm import MVTrackSVMReward  # noqa: E402
from svm.preprocess import build_batch, dino_features  # noqa: E402

CAMERAS = ("head", "left_wrist", "right_wrist")
CKPT = f"{SVM_REFERENCE}/checkpoints/mv-track-svm-scan-object"
N_PATCH = 391
KS = (24, 50, 100, 200, 391)
N_DRAWS = 3          # independent random patch subsets per (chunk, k)


def obs_from_record(rec: dict) -> dict[str, np.ndarray]:
    mv = rec["multiview"]
    obs = {}
    for cam in CAMERAS:
        for key in ("rgb", "depth", "K", "E"):
            obs[f"{cam}_{key}"] = np.asarray(mv[cam][key])[None]
    obs["state"] = np.asarray(rec["agent_pos"], np.float32)[None]
    obs["track"] = np.asarray(rec["action_track_gt"], np.float32)[None]
    obs["track_valid"] = np.asarray(rec["valid_mask"], bool)[None]
    return obs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chain_dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    scorer = MVTrackSVMReward(args.ckpt, device=device)
    model, norm = scorer.model, scorer.norm
    model.float()
    print(f"normalizer track_rms = {norm.track_rms * 1000:.2f} mm")

    episodes = sorted(args.chain_dir.glob("episode*_chain.pkl"),
                      key=lambda q: int(q.stem.split("_")[0].replace("episode", "")))
    if not episodes:
        raise SystemExit(f"no episode*_chain.pkl in {args.chain_dir}")

    rows = []
    for ep_path in episodes:
        with open(ep_path, "rb") as f:
            chunks = [c for c in pickle.load(f) if c.get("multiview") is not None]
        ep = int(ep_path.stem.split("_")[0].replace("episode", ""))
        for ci, rec in enumerate(chunks):
            obs = obs_from_record(rec)
            feats = torch.stack([dino_features(scorer.dino, obs[f"{c}_rgb"], device) for c in CAMERAS], 1)
            base = build_batch(obs, feats, norm, device, "track")
            gt = np.asarray(rec["action_track_gt"], np.float32)
            v0 = np.asarray(rec["valid_mask"], bool)
            # realistic magnitude: this chunk's own GT rms over its valid patches,
            # falling back to the dataset-wide rms for a chunk with no motion
            rms = float(np.sqrt((gt[v0] ** 2).mean())) if v0.any() else float(norm.track_rms)
            rms = max(rms, 1e-4)

            def score(track: np.ndarray, valid: np.ndarray) -> float:
                batch = dict(base)
                batch["track"] = torch.as_tensor(norm.track_norm(track)[None], dtype=torch.float32, device=device)
                batch["valid"] = torch.as_tensor(valid[None], dtype=torch.bool, device=device)
                with torch.no_grad():
                    return float(model(batch).float().item())

            zero = score(np.zeros_like(gt), np.zeros(N_PATCH, bool))
            row = {"episode": ep, "chunk": ci, "n_valid_real": int(v0.sum()),
                   "gt_rms": rms, "logit_zero": zero,
                   "logit_gt": score(gt, v0)}
            for k in KS:
                d_content, d_marked = [], []
                for _ in range(N_DRAWS):
                    sel = rng.choice(N_PATCH, size=k, replace=False)
                    m = np.zeros(N_PATCH, bool)
                    m[sel] = True
                    t = np.zeros_like(gt)
                    t[sel] = rng.normal(scale=rms, size=(k, *gt.shape[1:])).astype(np.float32)
                    d_content.append(abs(score(t, m) - zero))
                    d_marked.append(abs(score(np.zeros_like(gt), m) - zero))
                row[f"content_k{k}"] = float(np.mean(d_content))
                row[f"marked_k{k}"] = float(np.mean(d_marked))
            rows.append(row)
            print(f"ep{ep} chunk{ci:02d} nv={row['n_valid_real']:3d} rms={rms:.4f}  "
                  + "  ".join(f"k{k}={row[f'content_k{k}']:.4f}" for k in KS), flush=True)

    with open(args.out / "valid_count_scan.json", "w") as f:
        json.dump({"ks": list(KS), "n_draws": N_DRAWS, "rows": rows}, f, indent=2)

    across = float(np.std([r["logit_zero"] for r in rows]))
    gt_med = float(np.median([abs(r["logit_gt"] - r["logit_zero"]) for r in rows]))
    print(f"\n{len(rows)} chunks; across-state logit std = {across:.4f}")
    print(f"real GT track (k~24, real content): median |d| = {gt_med:.5f}\n")
    print(f"{'k':>5} {'content median':>15} {'p90':>9} {'marked-only median':>20} {'vs across':>10}")
    for k in KS:
        c = np.array([r[f"content_k{k}"] for r in rows])
        m = np.array([r[f"marked_k{k}"] for r in rows])
        print(f"{k:>5} {np.median(c):>15.5f} {np.percentile(c, 90):>9.4f} "
              f"{np.median(m):>20.5f} {np.median(c)/across:>10.2}")
    c24, c391 = (np.median([r[f"content_k{k}"] for r in rows]) for k in (24, 391))
    print(f"\n391/24 content ratio = {c391 / max(c24, 1e-12):.2f}x   "
          f"(linear in k would be {391/24:.1f}x)")


if __name__ == "__main__":
    main()
