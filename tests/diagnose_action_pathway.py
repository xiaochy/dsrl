"""Where does the action-SVM's CLS token actually look? The action twin of
`diagnose_track_pathway.py`, built to answer the same question panel (b) of
fig1 asks for the track variant.

The self-attention sequence of `MultiViewActionSVM` is

    [CLS | state L,R | action L,R | head DINO x391]  = 396 tokens

so it differs from the track variant in two ways that matter for reading the
figure:

  * the action occupies **2** tokens, not 391, and
  * there is **no constant block**: `act_ok = torch.ones(...)` (model.py:261),
    every action token always carries content, and there is no `invalid_embed`.

That second point is why this measurement is worth having. In the track
variant 367 of the 785 tokens are a fixed constant and CLS parks >50% of its
attention there from L1 on. If the same parking shows up here, where no such
block exists, then it is a generic attention-sink phenomenon and not something
the track encoding caused -- which is the control the track write-up has been
missing.

Raw attention mass is misleading with 2 tokens against 391, so every quantity
is also reported per token (mass / number of tokens in the segment), against
the uniform share 1/396.

    python tests/diagnose_action_pathway.py \
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
import svm.model as svm_model  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_action_sensitivity import CAMERAS, CKPT, obs_from_record  # noqa: E402
from diagnose_track_pathway import AttnTap  # noqa: E402

N_PATCH = 391
N_WRIST = 2
# segment starts in the 396-token self-attention sequence. NOTE the order:
# MultiViewActionSVM._head_side returns (action, dino), so the action tokens
# come BEFORE the DINO block -- the opposite of the track variant's layout.
I_CLS, I_STATE, I_ACT, I_DINO = 0, 1, 3, 5
N_SEQ = 1 + 2 + 2 + N_PATCH
SEG = {"cls": (I_CLS, I_STATE), "state": (I_STATE, I_ACT),
       "action": (I_ACT, I_DINO), "dino": (I_DINO, N_SEQ)}


def token_norms(model, batch) -> dict[str, float]:
    """Norms of the tokens the trunk is given, before layer 1."""
    with torch.no_grad():
        B = batch["dino"].shape[0]
        head_dino = batch["dino"][:, 0]
        chunk = batch["action"].reshape(B, model.horizon, 2, model.arm_dim).transpose(1, 2)
        act_tok = model.action_encoder(chunk.reshape(B, 2, model.horizon * model.arm_dim)) + model.arm_embed.weight
        state_tok = model.state_encoder(batch["state"].reshape(B, 2, model.arm_dim)) + model.arm_embed.weight
        n = lambda x: float(x.norm(dim=-1).mean())  # noqa: E731
        return {"dino_token_norm": n(head_dino), "action_token_norm": n(act_tok),
                "state_token_norm": n(state_tok), "cls_norm": float(model.cls_token.norm())}


def leverage(model, batch) -> dict[str, float]:
    """||d logit/d x|| and the logit change for a 1-RMS-size perturbation of x."""
    b = {k: (v.clone().float().requires_grad_(k in ("action", "dino", "state"))
             if v.dtype.is_floating_point else v.clone()) for k, v in batch.items()}
    logit = model(b).sum()
    logit.backward()
    out = {"logit": float(logit.detach())}
    for name, g, x in (("action", b["action"].grad[0], b["action"].detach()[0]),
                       ("dino", b["dino"].grad[0, 0], b["dino"].detach()[0, 0]),
                       ("state", b["state"].grad[0], b["state"].detach()[0])):
        gn, xr = float(g.norm()), float(x.pow(2).mean().sqrt())
        out[f"grad_{name}"] = gn
        out[f"rms_{name}"] = xr
        out[f"numel_{name}"] = int(x.numel())
        out[f"grad_per_elem_{name}"] = gn / max(float(x.numel()) ** 0.5, 1.0)
        out[f"leverage_{name}"] = gn * xr * float(x.numel()) ** 0.5
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

    scorer = MVActionSVMReward(args.ckpt, device=device)
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
            batch = build_batch(obs, feats, norm, device, "action")
            batch = {k: (v.float() if v.dtype.is_floating_point else v) for k, v in batch.items()}

            row = {"episode": ep, "chunk": ci}
            row |= token_norms(model, batch)
            row |= leverage(model, batch)

            with AttnTap() as tap, torch.no_grad():
                model(batch)
            self_calls = [w for n_keys, w in tap.calls if n_keys == N_SEQ]
            cross_calls = [w for n_keys, w in tap.calls if n_keys == N_WRIST * N_PATCH]
            for li, w in enumerate(self_calls):
                for nm, (a, b_) in SEG.items():
                    row[f"cls_attn_L{li}_{nm}"] = float(w[a:b_].sum())
            row["n_layers"] = len(self_calls)
            row["n_cross"] = len(cross_calls)
            rows.append(row)
            print(f"ep{ep} chunk{ci:02d} CLS->action L0..L{len(self_calls)-1}=" +
                  ",".join(f"{row[f'cls_attn_L{i}_action']:.4f}" for i in range(len(self_calls))))

    with open(args.out / "action_pathway.json", "w") as f:
        json.dump(rows, f, indent=2)

    def avg(key):
        vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    nl = rows[0]["n_layers"]
    ntok = {"cls": 1, "state": 2, "action": 2, "dino": N_PATCH}
    print(f"\n==== {len(rows)} chunks, {N_SEQ}-token sequence, no constant block ====")
    print("\ntoken norms (mean over chunks)")
    for k in ("dino_token_norm", "action_token_norm", "state_token_norm", "cls_norm"):
        print(f"  {k:<22} {avg(k):10.4f}")
    print(f"  action/dino norm ratio {avg('action_token_norm') / avg('dino_token_norm'):10.4f}")

    print("\nCLS self-attention MASS (mean over chunks)")
    print(f"  {'layer':<7}{'action':>10}{'head DINO':>12}{'state':>9}{'CLS':>9}")
    for li in range(nl):
        print(f"  L{li:<6}" + "".join(f"{avg(f'cls_attn_L{li}_{k}'):>{w}.5f}"
                                      for k, w in (("action", 10), ("dino", 12), ("state", 9), ("cls", 9))))
    print(f"\nCLS self-attention PER TOKEN / uniform share (uniform = 1/{N_SEQ})")
    print(f"  {'layer':<7}{'action':>10}{'head DINO':>12}{'state':>9}{'CLS':>9}")
    for li in range(nl):
        print(f"  L{li:<6}" + "".join(f"{avg(f'cls_attn_L{li}_{k}') / ntok[k] * N_SEQ:>{w}.3f}"
                                      for k, w in (("action", 10), ("dino", 12), ("state", 9), ("cls", 9))))

    print("\ninput leverage (normalized units)")
    for k in ("action", "dino", "state"):
        print(f"  {k:<7} numel {avg(f'numel_{k}'):>8.0f}  |grad| {avg(f'grad_{k}'):8.4f}  "
              f"|grad|/elem {avg(f'grad_per_elem_{k}'):9.6f}  rms {avg(f'rms_{k}'):7.4f}  "
              f"leverage {avg(f'leverage_{k}'):9.4f}")
    print(f"\nwrote {args.out}/action_pathway.json")


if __name__ == "__main__":
    main()
