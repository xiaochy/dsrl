"""Does the action-SVM actually read its action input?

The action twin of `measure_track_sensitivity.py`. For every chunk of a
CHAIN_COLLECT_MULTIVIEW rollout the observation is held FIXED (three RGB-D
views + state) and only `batch["action"]` is swapped, so every number is a
within-state quantity. The reference scale is the across-state spread of the
logit, i.e. how much the score moves when the STATE changes.

An action chunk is [16, 16]: 16 commanded EEF poses, each
[L xyz, L quat(wxyz), L grip, R xyz, R quat(wxyz), R grip] in the world frame.
`build_batch` also derives `action_xyz` from step 0, which feeds the 3D RoPE,
so moving the arm moves the action tokens' positional encoding too.

Variants, roughly ordered from "a policy could have produced this" to absurd:

    hold          freeze at the current EEF pose for all 16 steps (do nothing)
    gt            the chunk actually executed                     (the reference)
    other         the chunk of a DIFFERENT decision step of this episode
    reverse       the same chunk played backwards in time
    mirror        2*p0 - a: same magnitude, exactly the opposite direction
    scale2/5      displacement from the current pose amplified 2x / 5x
    jitter1/5cm   gaussian noise on xyz
    gripflip      gripper commands inverted (semantically decisive, geometrically ~0)
    quatrand      random unit quaternions, xyz untouched
    swaparms      the left arm gets the right arm's trajectory and vice versa
    far50cm       every waypoint teleported +0.5 m in x (way outside the workspace)
    uniform       every dim drawn uniformly inside the normalizer's train range

    python tests/measure_action_sensitivity.py \
        --chain_dir .../chain_results/scan_object_track_sensitivity \
        --out .../outputs/action_sens
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

from svm import MVActionSVMReward  # noqa: E402
from svm.preprocess import build_batch, dino_features  # noqa: E402

CAMERAS = ("head", "left_wrist", "right_wrist")
CKPT = f"{SVM_REFERENCE}/checkpoints/mv-action-svm-scan-object"
ARM = 8          # dims per arm in the 16-D pose vector
XYZ = [slice(0, 3), slice(ARM, ARM + 3)]
QUAT = [slice(3, 7), slice(ARM + 3, ARM + 7)]
GRIP = [7, ARM + 7]


def obs_from_record(rec: dict) -> dict[str, np.ndarray]:
    """One chunk's record -> the SVM's obs dict with a leading batch dim of 1."""
    mv = rec["multiview"]
    obs = {}
    for cam in CAMERAS:
        for key in ("rgb", "depth", "K", "E"):
            obs[f"{cam}_{key}"] = np.asarray(mv[cam][key])[None]
    obs["state"] = np.asarray(rec["agent_pos"], np.float32)[None]
    obs["action"] = np.asarray(rec["action"], np.float32)[None]
    return obs


def action_variants(rec: dict, other: dict | None, lo: np.ndarray, hi: np.ndarray,
                    rng: np.random.Generator) -> dict[str, np.ndarray]:
    """[16, 16] action chunks. `p0` is the CURRENT pose, i.e. where the arms are
    before the chunk runs; displacements are taken relative to it."""
    a = np.asarray(rec["action"], np.float32)
    p0 = np.asarray(rec["agent_pos"], np.float32)[None]      # [1, 16]
    out: dict[str, np.ndarray] = {
        "hold": np.repeat(p0, len(a), 0),
        "gt": a,
        "reverse": a[::-1].copy(),
        "mirror": (2 * p0 - a).astype(np.float32),
        "scale2": (p0 + 2 * (a - p0)).astype(np.float32),
        "scale5": (p0 + 5 * (a - p0)).astype(np.float32),
    }
    if other is not None:
        out["other"] = np.asarray(other["action"], np.float32)

    for nm, sigma in (("jitter1cm", 0.01), ("jitter5cm", 0.05)):
        b = a.copy()
        for s in XYZ:
            b[:, s] += rng.normal(scale=sigma, size=b[:, s].shape).astype(np.float32)
        out[nm] = b

    b = a.copy()                                   # gripper open <-> closed
    for g in GRIP:
        b[:, g] = lo[g] + hi[g] - b[:, g]
    out["gripflip"] = b

    b = a.copy()                                   # random orientation, same path
    for s in QUAT:
        q = rng.normal(size=(len(a), 4)).astype(np.float32)
        b[:, s] = q / np.linalg.norm(q, axis=-1, keepdims=True)
    out["quatrand"] = b

    b = a.copy()                                   # left arm <-> right arm
    b[:, :ARM], b[:, ARM:] = a[:, ARM:].copy(), a[:, :ARM].copy()
    out["swaparms"] = b

    b = a.copy()                                   # teleport out of the workspace
    for s in XYZ:
        b[:, s.start] += 0.5
    out["far50cm"] = b

    out["uniform"] = rng.uniform(lo, hi, size=a.shape).astype(np.float32)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chain_dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    scorer = MVActionSVMReward(args.ckpt, device=device)
    model, norm = scorer.model, scorer.norm
    if args.dtype == "fp32":
        model.float()
    use_bf16 = args.dtype == "bf16" and device.startswith("cuda")
    lo, hi = norm.action_lo, norm.action_hi
    print(f"action range per dim: xyz_L [{lo[0]:.3f},{hi[0]:.3f}] grip_L [{lo[7]:.3f},{hi[7]:.3f}]")

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
            # DINO features and patch centres are computed ONCE per chunk;
            # only the action tensor changes between variants
            feats = torch.stack([dino_features(scorer.dino, obs[f"{c}_rgb"], device) for c in CAMERAS], 1)
            base = build_batch(obs, feats, norm, device, "action")

            other = chunks[(ci + len(chunks) // 2) % len(chunks)] if len(chunks) > 1 else None
            variants = action_variants(rec, other, lo, hi, rng)
            if names is None:
                names = list(variants)
            logits = {}
            with torch.no_grad():
                for name in names:
                    act = variants[name]
                    batch = dict(base)
                    batch["action"] = torch.as_tensor(norm.action_norm(act)[None],
                                                      dtype=torch.float32, device=device)
                    batch["action_xyz"] = torch.as_tensor(
                        np.stack([act[0, 0:3], act[0, ARM:ARM + 3]])[None],
                        dtype=torch.float32, device=device)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                        logits[name] = float(model(batch).float().item())
            a = np.asarray(rec["action"], np.float32)
            p0 = np.asarray(rec["agent_pos"], np.float32)
            disp = float(np.linalg.norm(a[-1, 0:3] - p0[0:3]) + np.linalg.norm(a[-1, ARM:ARM + 3] - p0[ARM:ARM + 3]))
            rows.append({"episode": ep, "chunk": ci, "start_frame": int(rec["start_frame"]),
                         "chunk_disp_m": disp,
                         **{f"logit_{k}": logits[k] for k in names}})
            print(f"ep{ep} chunk{ci:02d} disp={disp:.3f}m  "
                  + "  ".join(f"{k}={logits[k]:+.4f}" for k in ("hold", "gt", "far50cm", "uniform")))

    L = np.array([[r[f"logit_{k}"] for k in names] for r in rows])
    np.savez(args.out / "action_sensitivity.npz", logits=L, variants=np.array(names),
             meta=json.dumps(rows))
    with open(args.out / "action_sensitivity.json", "w") as f:
        json.dump({"variants": names, "rows": rows}, f, indent=2)

    gi = names.index("gt")
    across = L[:, gi].std()
    print(f"\n{len(rows)} chunks from {len(episodes)} episodes")
    print(f"across-state spread of logit(gt): std {across:.4f}, range {L[:, gi].ptp():.4f}\n")
    print(f"{'variant':<12} {'mean logit':>11} {'median |d vs gt|':>17} {'p90':>9} {'max':>9} {'vs across':>10}")
    for j, k in enumerate(names):
        d = np.abs(L[:, j] - L[:, gi])
        print(f"{k:<12} {L[:, j].mean():>+11.4f} {np.median(d):>17.5f} "
              f"{np.percentile(d, 90):>9.4f} {d.max():>9.4f} {np.median(d)/across:>10.2}")


if __name__ == "__main__":
    main()
