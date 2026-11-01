"""WHY the track-SVM barely moves when its track changes: measure the track
pathway itself, not just the output.

Four measurements per chunk, all on the real multiview observation:

1. Token budget. The self-attention sequence is
   [CLS | state L,R | head DINO x391 | track x391] = 785 tokens. `valid` marks
   the patches that actually have a track; every INVALID track token is the
   same constant vector (invalid_embed + track_type_embed) no matter what the
   track array holds, so the track can only enter through `n_valid` tokens.

2. Token norms. ||track token|| vs ||head DINO token||: if the track tokens
   enter the residual stream an order of magnitude smaller than the DINO
   tokens they are drowned before attention even runs.

3. CLS attention mass. The readout is the CLS token, so what reaches the logit
   is whatever CLS attends to. Per layer: the fraction of CLS's self-attention
   mass on the valid track tokens vs the head DINO tokens, plus the share of
   the layer's cross-attention (the wrist memory).

4. Input leverage (gradients). ||d logit / d x|| * (typical |x|) for the track,
   the head DINO features and the state, in the model's own normalized units --
   i.e. how much the logit moves per realistic-size perturbation of each input.

    python tests/diagnose_track_pathway.py \
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
import torch.nn.functional as F

SVM_REFERENCE = "/data/vxiao/bimanual_manipulation/SVM-reference"
sys.path.insert(0, SVM_REFERENCE)

from svm import MVTrackSVMReward  # noqa: E402
from svm.preprocess import build_batch, dino_features  # noqa: E402
import svm.model as svm_model  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_track_sensitivity import CAMERAS, CKPT, obs_from_record  # noqa: E402

N_PATCH = 391
# segment starts in the 785-token self-attention sequence
I_CLS, I_STATE, I_DINO, I_TRACK = 0, 1, 3, 3 + N_PATCH


class AttnTap:
    """Records the CLS row of every attention call (self: 785 keys, cross: 782)."""

    def __init__(self):
        self.calls: list[tuple[int, np.ndarray]] = []
        self._orig = svm_model.F.scaled_dot_product_attention

    def __enter__(self):
        def patched(q, k, v, *a, **kw):
            with torch.no_grad():
                w = torch.softmax(
                    (q[:, :, :1].float() @ k.float().transpose(-1, -2)) / q.shape[-1] ** 0.5, dim=-1)
                self.calls.append((k.shape[-2], w[0, :, 0].mean(0).cpu().numpy()))
            return self._orig(q, k, v, *a, **kw)

        svm_model.F.scaled_dot_product_attention = patched
        return self

    def __exit__(self, *exc):
        svm_model.F.scaled_dot_product_attention = self._orig


def token_norms(model, batch) -> dict[str, float]:
    """Norms of the tokens the trunk is actually given, before layer 1."""
    with torch.no_grad():
        B = batch["dino"].shape[0]
        head_dino = batch["dino"][:, 0]                                   # [B,391,1024]
        valid = batch["valid"]
        track_tok = model.track_temporal_encoder(batch["track"] * valid[..., None, None])
        track_tok = track_tok + (~valid)[..., None] * model.invalid_embed + model.track_type_embed
        state_tok = model.state_encoder(batch["state"].reshape(B, 2, model.arm_dim)) + model.arm_embed.weight
        v = valid[0].bool()
        n = lambda x: float(x.norm(dim=-1).mean())  # noqa: E731
        return {
            "dino_token_norm": n(head_dino),
            "track_token_norm_valid": n(track_tok[0][v]) if bool(v.any()) else float("nan"),
            "track_token_norm_invalid": n(track_tok[0][~v]) if bool((~v).any()) else float("nan"),
            # the part of the valid track token that carries the track at all
            "track_signal_norm": n(model.track_temporal_encoder(batch["track"] * valid[..., None, None])[0][v])
            if bool(v.any()) else float("nan"),
            "invalid_embed_norm": float((model.invalid_embed + model.track_type_embed).norm()),
            "state_token_norm": n(state_tok),
            "cls_norm": float(model.cls_token.norm()),
        }


def leverage(model, batch) -> dict[str, float]:
    """||d logit/d x|| and the logit change for a 1-RMS-size perturbation of x."""
    b = {k: (v.clone().float().requires_grad_(k in ("track", "dino", "state")) if v.dtype.is_floating_point
             else v.clone()) for k, v in batch.items()}
    logit = model(b).sum()
    logit.backward()
    v = b["valid"][0].bool()
    out = {"logit": float(logit.detach())}
    g_track = b["track"].grad[0][v]                       # [n_valid,16,3] normalized units
    x_track = b["track"].detach()[0][v]
    g_dino = b["dino"].grad[0, 0]
    x_dino = b["dino"].detach()[0, 0]
    g_state, x_state = b["state"].grad[0], b["state"].detach()[0]
    for name, g, x in (("track", g_track, x_track), ("dino", g_dino, x_dino), ("state", g_state, x_state)):
        gn, xr = float(g.norm()), float(x.pow(2).mean().sqrt())
        out[f"grad_{name}"] = gn
        out[f"rms_{name}"] = xr
        out[f"numel_{name}"] = int(x.numel())
        # per-element sensitivity, so the comparison is not just dimensionality
        out[f"grad_per_elem_{name}"] = gn / max(float(x.numel()) ** 0.5, 1.0)
        # logit change if every element of x moves by its own RMS, worst-case aligned
        out[f"leverage_{name}"] = gn * xr * float(x.numel()) ** 0.5
    # first-order prediction of the GT -> zero swap: g . (0 - x), the exact
    # quantity measure_track_sensitivity.py measures empirically as gt - zero
    out["dlogit_firstorder_zero_swap"] = float(-(g_track * x_track).sum())
    out["dlogit_firstorder_neg_swap"] = float(-2 * (g_track * x_track).sum())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chain_dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--device", default="cuda")
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
        for ci, rec in enumerate(chunks):
            obs = obs_from_record(rec)
            feats = torch.stack([dino_features(scorer.dino, obs[f"{c}_rgb"], device) for c in CAMERAS], 1)
            batch = build_batch(obs, feats, norm, device, "track")
            batch = {k: (v.float() if v.dtype.is_floating_point else v) for k, v in batch.items()}
            v = batch["valid"][0].bool().cpu().numpy()

            row = {"episode": ep, "chunk": ci, "n_valid": int(v.sum())}
            row |= token_norms(model, batch)
            row |= leverage(model, batch)

            with AttnTap() as tap, torch.no_grad():
                model(batch)
            self_calls = [w for n_keys, w in tap.calls if n_keys == 1 + 2 + 2 * N_PATCH]
            cross_calls = [w for n_keys, w in tap.calls if n_keys == 2 * N_PATCH]
            for li, w in enumerate(self_calls):
                row[f"cls_attn_L{li}_track_valid"] = float(w[I_TRACK:][v].sum())
                row[f"cls_attn_L{li}_track_all"] = float(w[I_TRACK:].sum())
                row[f"cls_attn_L{li}_dino"] = float(w[I_DINO:I_TRACK].sum())
                row[f"cls_attn_L{li}_state"] = float(w[I_STATE:I_DINO].sum())
                row[f"cls_attn_L{li}_cls"] = float(w[I_CLS])
            row["n_layers"] = len(self_calls)
            row["n_cross"] = len(cross_calls)
            rows.append(row)
            print(f"ep{ep} chunk{ci:02d} valid={row['n_valid']:3d} "
                  f"||dino||={row['dino_token_norm']:.2f} ||track_v||={row['track_token_norm_valid']:.2f} "
                  f"||track_inv||={row['track_token_norm_invalid']:.2f} "
                  f"CLS->track(valid) L0..L5=" + ",".join(
                      f"{row[f'cls_attn_L{i}_track_valid']:.4f}" for i in range(len(self_calls))))

    with open(args.out / "track_pathway.json", "w") as f:
        json.dump(rows, f, indent=2)

    def avg(key):
        vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    nl = rows[0]["n_layers"]
    print(f"\n==== {len(rows)} chunks ====")
    print(f"valid track tokens: mean {avg('n_valid'):.1f} / {N_PATCH} "
          f"({avg('n_valid') / N_PATCH * 100:.1f}% of patches, "
          f"{avg('n_valid') / (1 + 2 + 2 * N_PATCH) * 100:.2f}% of the 785-token sequence)")
    print("\ntoken norms (mean over chunks)")
    for k in ("dino_token_norm", "track_token_norm_valid", "track_token_norm_invalid",
              "track_signal_norm", "invalid_embed_norm", "state_token_norm", "cls_norm"):
        print(f"  {k:<26} {avg(k):10.4f}")
    print(f"  track/dino norm ratio       {avg('track_token_norm_valid') / avg('dino_token_norm'):10.4f}")
    print(f"  track SIGNAL / dino ratio   {avg('track_signal_norm') / avg('dino_token_norm'):10.4f}")

    print("\nCLS self-attention mass (mean over chunks)")
    print(f"  {'layer':<7}{'track(valid)':>14}{'track(all)':>12}{'head DINO':>11}{'state':>9}{'CLS':>9}")
    for li in range(nl):
        print(f"  L{li:<6}{avg(f'cls_attn_L{li}_track_valid'):>14.5f}"
              f"{avg(f'cls_attn_L{li}_track_all'):>12.5f}{avg(f'cls_attn_L{li}_dino'):>11.5f}"
              f"{avg(f'cls_attn_L{li}_state'):>9.5f}{avg(f'cls_attn_L{li}_cls'):>9.5f}")

    print("\ninput leverage (normalized units; leverage = logit change for a "
          "worst-case perturbation of relative size 1)")
    for k in ("track", "dino", "state"):
        print(f"  {k:<6} numel {avg(f'numel_{k}'):>8.0f}  |grad| {avg(f'grad_{k}'):8.4f}  "
              f"|grad|/elem {avg(f'grad_per_elem_{k}'):9.6f}  input rms {avg(f'rms_{k}'):7.4f}  "
              f"leverage {avg(f'leverage_{k}'):9.4f}")
    print(f"\nfirst-order prediction of the GT->zero swap: {avg('dlogit_firstorder_zero_swap'):+.4f}"
          f"   (GT->-GT: {avg('dlogit_firstorder_neg_swap'):+.4f})")
    print("  compare with the measured gt-vs-zero delta from measure_track_sensitivity.py")
    print(f"\nwrote {args.out}/track_pathway.json")


if __name__ == "__main__":
    main()
