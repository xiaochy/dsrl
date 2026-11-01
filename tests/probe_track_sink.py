"""Is the invalid-track-token block CAUSALLY why the logit ignores the track,
or is the insensitivity upstream of attention?

diagnose_track_pathway.py shows CLS spending most of its self-attention mass on
the ~368 track tokens that carry no track (they are all the same constant
invalid_embed + track_type_embed), and well under 1% on the ~23 that do. That
is consistent with two different stories:

  (sink)      the constant block is an attention sink; CLS dumps mass there as
              a no-op, which starves the informative track tokens. Remove the
              sink and CLS should have to look at real track tokens, and the
              logit should become track-sensitive.
  (upstream)  the trunk simply never learned to use the track, and the sink is
              a symptom. Removing it changes where attention goes but not how
              much the track moves the logit.

So: per chunk, run the model under `valid` as collected and under valid=all
True (no token carries invalid_embed any more), each with the GT track and with
a zero track, and compare both the attention split and the gt-vs-zero gap.

    python tests/probe_track_sink.py \
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_track_pathway import AttnTap, I_DINO, I_TRACK, N_PATCH  # noqa: E402
from measure_track_sensitivity import CAMERAS, CKPT, obs_from_record  # noqa: E402

N_SELF = 1 + 2 + 2 * N_PATCH


def run(model, base, track, valid, device, want_attn: bool, ref_valid=None):
    """`ref_valid` is the mask used for ACCOUNTING -- always the real one, so
    "informative" means the same set of patches in both conditions (under
    valid=all True the model's own mask would call all 391 informative)."""
    batch = dict(base)
    batch["track"] = torch.as_tensor(track[None], dtype=torch.float32, device=device)
    batch["valid"] = torch.as_tensor(valid[None], dtype=torch.bool, device=device)
    if not want_attn:
        with torch.no_grad():
            return float(model(batch).item()), None
    with AttnTap() as tap, torch.no_grad():
        logit = float(model(batch).item())
    self_calls = [w for n_keys, w in tap.calls if n_keys == N_SELF]
    v = (valid if ref_valid is None else ref_valid).astype(bool)
    attn = [{"track_informative": float(w[I_TRACK:][v].sum()),
             "track_rest": float(w[I_TRACK:][~v].sum()),
             "track_all": float(w[I_TRACK:].sum()),
             "dino": float(w[I_DINO:I_TRACK].sum())} for w in self_calls]
    return logit, attn


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chain_dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_chunks_per_ep", type=int, default=10)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    scorer = MVTrackSVMReward(args.ckpt, device=device)
    model, norm = scorer.model, scorer.norm
    model.float()

    rows = []
    for ep_path in sorted(args.chain_dir.glob("episode*_chain.pkl")):
        with open(ep_path, "rb") as f:
            chunks = [c for c in pickle.load(f) if c.get("multiview") is not None]
        ep = int(ep_path.stem.split("_")[0].replace("episode", ""))
        for ci, rec in enumerate(chunks[: args.max_chunks_per_ep]):
            obs = obs_from_record(rec)
            feats = torch.stack([dino_features(scorer.dino, obs[f"{c}_rgb"], device) for c in CAMERAS], 1)
            base = build_batch(obs, feats, norm, device, "track")
            base = {k: (v.float() if v.dtype.is_floating_point else v) for k, v in base.items()}

            gt = norm.track_norm(np.asarray(rec["action_track_gt"], np.float32))
            zero = np.zeros_like(gt)
            v_real = np.asarray(rec["valid_mask"], bool)
            v_all = np.ones_like(v_real)

            l_gt, a_gt = run(model, base, gt, v_real, device, True, ref_valid=v_real)
            l_zero, _ = run(model, base, zero, v_real, device, False)
            l_gt_av, a_gt_av = run(model, base, gt, v_all, device, True, ref_valid=v_real)
            l_zero_av, _ = run(model, base, zero, v_all, device, False)

            # with valid=all True there is no invalid block, but only the ~23
            # genuinely tracked patches hold a nonzero track; the rest are
            # MLP(0). "informative" = the same real patches in both conditions.
            rows.append({
                "episode": ep, "chunk": ci, "n_valid": int(v_real.sum()),
                "logit_gt": l_gt, "logit_zero": l_zero, "gap_real_mask": l_gt - l_zero,
                "logit_gt_allvalid": l_gt_av, "logit_zero_allvalid": l_zero_av,
                "gap_allvalid": l_gt_av - l_zero_av,
                "attn_real": a_gt, "attn_allvalid": a_gt_av,
            })
            print(f"ep{ep} chunk{ci:02d} valid={v_real.sum():3d}  "
                  f"gap(real mask)={l_gt - l_zero:+.5f}  gap(all-valid)={l_gt_av - l_zero_av:+.5f}")

    with open(args.out / "track_sink.json", "w") as f:
        json.dump(rows, f, indent=2)

    nl = len(rows[0]["attn_real"])
    print(f"\n==== {len(rows)} chunks ====")
    print("\nCLS self-attention mass, GT track (mean over chunks)")
    print(f"  {'layer':<7}{'real: informative':>19}{'real: rest':>12}{'real: dino':>12}"
          f"{'allvalid: informative':>23}{'allvalid: dino':>16}")
    for li in range(nl):
        f = lambda key, cond: float(np.mean([r[cond][li][key] for r in rows]))  # noqa: E731
        print(f"  L{li:<6}{f('track_informative', 'attn_real'):>19.5f}{f('track_rest', 'attn_real'):>12.5f}"
              f"{f('dino', 'attn_real'):>12.5f}{f('track_informative', 'attn_allvalid'):>23.5f}"
              f"{f('dino', 'attn_allvalid'):>16.5f}")

    g_real = np.abs([r["gap_real_mask"] for r in rows])
    g_all = np.abs([r["gap_allvalid"] for r in rows])
    print(f"\n|logit(GT) - logit(zero)|  with the real mask: mean {g_real.mean():.5f}, max {g_real.max():.5f}")
    print(f"|logit(GT) - logit(zero)|  with valid=all True: mean {g_all.mean():.5f}, max {g_all.max():.5f}")
    print(f"ratio: {g_all.mean() / max(g_real.mean(), 1e-12):.3f}")
    print("\nIf removing the constant block does NOT raise the gap, the insensitivity is")
    print("upstream of attention: the trunk never learned to use the track, and the")
    print("attention sink is a symptom rather than the cause.")
    print(f"\nwrote {args.out}/track_sink.json")


if __name__ == "__main__":
    main()
