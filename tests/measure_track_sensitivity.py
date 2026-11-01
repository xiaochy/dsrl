"""Does the track-SVM actually read its track input?

For every chunk of a CHAIN_COLLECT_MULTIVIEW rollout the observation is held
FIXED and only the track is swapped, so every number below is a within-state
quantity. The contrast that matters is

    within-state spread (swapping the track)   vs   across-state spread (the
                                                    same variant over chunks)

which is the same contrast the action-SVM's README reports as ~0.004 vs ~3.

Variants (all [391, 16, 3] world-frame displacement in metres):

    zero            zeros -- V(s, 0), what the student recipe subtracts
    gt              the simulator's true rigid-body track of the executed chunk
    pred            GAP's own predicted track for that chunk
    gt_x{0.5,2,5,10}  GT scaled: pure motion-magnitude perturbation
    gt_neg          -GT: same magnitude, exactly reversed motion
    gt_shuffle      GT vectors permuted AMONG the valid patches: same
                    magnitudes, wrong spatial assignment
    gt_randdir      per-patch/per-step magnitude of GT kept, direction random:
                    isolates "is it only reading |track|?"
    rand_1x/_10x    gaussian at 1x / 10x GT's RMS
    gt_other        the GT track of a DIFFERENT chunk of the same episode
    gt_allvalid     GT with valid=all True (mask ablation)
    gt_allinvalid   GT with valid=all False (the track is fully masked out)

Writes <out>/track_sensitivity.npz (logits [n_chunks, n_variants]) and a
summary table to stdout.

    python tests/measure_track_sensitivity.py \
        --chain_dir .../chain_results/scan_object_track_sensitivity \
        --out .../outputs/track_sens
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

SVM_REFERENCE = "/data/vxiao/bimanual_manipulation/SVM-reference"
sys.path.insert(0, SVM_REFERENCE)

from svm import MVTrackSVMReward, svm_reward  # noqa: E402
from svm.preprocess import build_batch, dino_features  # noqa: E402

CAMERAS = ("head", "left_wrist", "right_wrist")
CKPT = f"{SVM_REFERENCE}/checkpoints/mv-track-svm-scan-object"


def obs_from_record(rec: dict) -> dict[str, np.ndarray]:
    """One chunk's record -> the SVM's obs dict with a leading batch dim of 1."""
    mv = rec["multiview"]
    obs = {}
    for cam in CAMERAS:
        for key, name in (("rgb", "rgb"), ("depth", "depth"), ("K", "K"), ("E", "E")):
            obs[f"{cam}_{key}"] = np.asarray(mv[cam][name])[None]
    obs["state"] = np.asarray(rec["agent_pos"], np.float32)[None]
    obs["track"] = np.asarray(rec["action_track_gt"], np.float32)[None]
    obs["track_valid"] = np.asarray(rec["valid_mask"], bool)[None]
    return obs


def track_variants(rec: dict, other: dict | None, rng: np.random.Generator) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """name -> (track [391,16,3], valid [391]). The observation never changes."""
    gt = np.asarray(rec["action_track_gt"], np.float32)
    pred = np.asarray(rec["pred_track_policy"], np.float32)
    valid = np.asarray(rec["valid_mask"], bool)
    # GT's own scale, measured on the patches that are actually tracked
    rms = float(np.sqrt((gt[valid] ** 2).mean())) if valid.any() else 0.0

    idx = np.where(valid)[0]
    shuffled = gt.copy()
    if len(idx) > 1:
        shuffled[idx] = gt[rng.permutation(idx)]

    # same |displacement| per patch and step, random direction
    mag = np.linalg.norm(gt, axis=-1, keepdims=True)              # [391,16,1]
    d = rng.normal(size=gt.shape).astype(np.float32)
    randdir = (d / np.clip(np.linalg.norm(d, axis=-1, keepdims=True), 1e-8, None)) * mag

    v = {
        "zero": np.zeros_like(gt),
        "gt": gt,
        "pred": pred,
        "gt_x0.5": 0.5 * gt,
        "gt_x2": 2.0 * gt,
        "gt_x5": 5.0 * gt,
        "gt_x10": 10.0 * gt,
        "gt_neg": -gt,
        "gt_shuffle": shuffled,
        "gt_randdir": randdir,
        "rand_1x": rng.normal(scale=max(rms, 1e-6), size=gt.shape).astype(np.float32),
        "rand_10x": rng.normal(scale=max(10 * rms, 1e-6), size=gt.shape).astype(np.float32),
    }
    out = {k: (t, valid) for k, t in v.items()}
    if other is not None:
        out["gt_other"] = (np.asarray(other["action_track_gt"], np.float32), valid)
    out["gt_allvalid"] = (gt, np.ones_like(valid))
    out["gt_allinvalid"] = (gt, np.zeros_like(valid))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chain_dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32",
                   help="bf16 reproduces the deployed autocast path, whose resolution at logit~3 is "
                        "~0.016 -- coarser than the effect being measured, so fp32 is the default here")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    scorer = MVTrackSVMReward(args.ckpt, device=device)
    model, norm = scorer.model, scorer.norm
    if args.dtype == "fp32":
        model.float()
    use_bf16 = args.dtype == "bf16" and device.startswith("cuda")

    episodes = sorted(args.chain_dir.glob("episode*_chain.pkl"),
                      key=lambda q: int(q.stem.split("_")[0].replace("episode", "")))
    if not episodes:
        raise SystemExit(f"no episode*_chain.pkl in {args.chain_dir}")

    rows, names = [], None
    for ep_path in episodes:
        with open(ep_path, "rb") as f:
            chunks = pickle.load(f)
        chunks = [c for c in chunks if c.get("multiview") is not None]
        ep = int(ep_path.stem.split("_")[0].replace("episode", ""))
        for ci, rec in enumerate(chunks):
            obs = obs_from_record(rec)
            # the DINO features / patch centres are computed ONCE; only the
            # track tensor changes between variants
            feats = torch.stack([dino_features(scorer.dino, obs[f"{c}_rgb"], device) for c in CAMERAS], 1)
            base = build_batch(obs, feats, norm, device, "track")

            other = chunks[(ci + len(chunks) // 2) % len(chunks)] if len(chunks) > 1 else None
            variants = track_variants(rec, other, rng)
            if names is None:
                names = list(variants)
            logits = {}
            with torch.no_grad():
                for name in names:
                    track, valid = variants[name]
                    batch = dict(base)
                    batch["track"] = torch.as_tensor(norm.track_norm(track)[None], dtype=torch.float32, device=device)
                    batch["valid"] = torch.as_tensor(valid[None], dtype=torch.bool, device=device)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                        logits[name] = float(model(batch).float().item())
            v = np.asarray(rec["valid_mask"], bool)
            gt = np.asarray(rec["action_track_gt"], np.float32)
            rows.append({
                "episode": ep, "chunk": ci, "start_frame": int(rec["start_frame"]),
                "n_valid": int(v.sum()),
                "gt_rms": float(np.sqrt((gt[v] ** 2).mean())) if v.any() else 0.0,
                "gt_max": float(np.abs(gt[v]).max()) if v.any() else 0.0,
                **{f"logit_{k}": logits[k] for k in names},
            })
            print(f"ep{ep} chunk{ci:02d} valid={v.sum():3d} gt_rms={rows[-1]['gt_rms']:.4f}  "
                  + "  ".join(f"{k}={logits[k]:+.4f}" for k in ("zero", "gt", "gt_neg", "rand_10x")))

    L = np.array([[r[f"logit_{k}"] for k in names] for r in rows])          # [n_chunks, n_variants]
    np.savez(args.out / "track_sensitivity.npz", logits=L, variants=np.array(names),
             meta=json.dumps(rows))
    with open(args.out / "track_sensitivity.json", "w") as f:
        json.dump({"variants": names, "rows": rows}, f, indent=2)

    zi, gi = names.index("zero"), names.index("gt")
    print(f"\n{len(rows)} chunks from {len(episodes)} episodes\n")
    print(f"{'variant':<14} {'mean logit':>11} {'mean d vs zero':>15} {'max |d| vs zero':>16} {'mean d vs gt':>13}")
    for j, k in enumerate(names):
        d0, dg = L[:, j] - L[:, zi], L[:, j] - L[:, gi]
        print(f"{k:<14} {L[:, j].mean():>+11.4f} {d0.mean():>+15.4f} {np.abs(d0).max():>16.4f} {dg.mean():>+13.4f}")

    # The headline contrast. "Plausible" = tracks a policy could actually produce
    # for this state; the x5/x10/rand_10x variants are deliberately out of
    # distribution and are reported separately.
    plausible = [k for k in ("gt", "pred", "gt_x0.5", "gt_other", "gt_shuffle", "gt_randdir",
                             "gt_allvalid", "gt_allinvalid") if k in names]
    ood = [k for k in ("gt_x2", "gt_x5", "gt_x10", "gt_neg", "rand_1x", "rand_10x") if k in names]
    cols = lambda ks: [names.index(k) for k in ks]  # noqa: E731
    d_pl = np.abs(L[:, cols(plausible)] - L[:, zi : zi + 1]).max(axis=1)
    d_ood = np.abs(L[:, cols(ood)] - L[:, zi : zi + 1]).max(axis=1)
    across = L[:, zi].std()

    print(f"\nacross-state spread (std of the zero-track logit over chunks): {across:.4f}")
    print(f"across-state range:                                            {L[:, zi].ptp():.4f}")
    print(f"\nwithin-state, PLAUSIBLE tracks only ({', '.join(plausible)}):")
    print(f"  max |logit - logit(zero)| per chunk: mean {d_pl.mean():.4f}, max {d_pl.max():.4f}")
    print(f"  ratio to the across-state spread:    {d_pl.mean() / max(across, 1e-9):.4f}")
    print(f"\nwithin-state, OUT-OF-DISTRIBUTION tracks ({', '.join(ood)}):")
    print(f"  max |logit - logit(zero)| per chunk: mean {d_ood.mean():.4f}, max {d_ood.max():.4f}")
    print(f"  ratio to the across-state spread:    {d_ood.mean() / max(across, 1e-9):.4f}")
    print(f"\nwrote {args.out}/track_sensitivity.npz and .json")


if __name__ == "__main__":
    main()
