"""fig3: where the action-SVM's CLS token looks, next to the track-SVM.

The control fig1(b) was missing. The track variant's sequence is half constant
tokens (367 of 785 have no GT track), and CLS puts >50% of its attention mass
there from L1 on; that was read as the constant block "stealing" attention.
The action variant has no constant block at all, so it says what the baseline
behaviour of this trunk is.

    python tests/plot_action_attention.py \
        --sens .../outputs --out .../docs/track-sens/figs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_track_findings import (  # noqa: E402
    SURF, INK, INK2, MUTED, AXIS, BLUE, ORANGE, CONST, style)

N_PATCH = 391
ACT_SEQ = 1 + 2 + 2 + N_PATCH            # 396
TRK_SEQ = 1 + 2 + 2 * N_PATCH            # 785


def _avg(rows, key):
    v = [r[key] for r in rows if key in r and np.isfinite(r[key])]
    return float(np.mean(v)) if v else float("nan")


def panel_mass(ax, act, nl):
    segs = [("head DINO (391 tokens)", "dino", BLUE),
            ("action tokens (2)", "action", ORANGE),
            ("state + CLS (3)", "state_cls", MUTED)]
    bot = np.zeros(nl)
    for label, key, c in segs:
        if key == "state_cls":
            v = np.array([_avg(act, f"cls_attn_L{i}_state") + _avg(act, f"cls_attn_L{i}_cls")
                          for i in range(nl)])
        else:
            v = np.array([_avg(act, f"cls_attn_L{i}_{key}") for i in range(nl)])
        ax.bar(np.arange(nl), v, 0.74, bottom=bot, color=c, label=label,
               edgecolor=SURF, linewidth=1.0, zorder=3)
        bot += v
    for i in range(nl):
        d = _avg(act, f"cls_attn_L{i}_dino")
        a = _avg(act, f"cls_attn_L{i}_action")
        ax.text(i, d / 2, f"{d:.0%}", ha="center", va="center", fontsize=7.5, color="white", zorder=5)
        # Only L0's action band is thick enough to carry a label; L1+ are all
        # ~0.5% and would collide with the title. Panel (b) is where the L1+
        # numbers are meant to be read anyway, per token rather than as mass.
        if a > 0.02:
            ax.text(i, d + a / 2, f"{a*100:.1f}%", ha="center", va="center",
                    fontsize=7, color="white", zorder=5)
    ax.set_xticks(np.arange(nl), [f"L{i}" for i in range(nl)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("CLS self-attention mass (sums to 1 per layer)")
    ax.set_title("(a) action-SVM: no constant block, so DINO holds ~99%", loc="left")
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="lower left", fontsize=6.5, ncol=1, framealpha=0.9)


def panel_pertoken(ax, act, nl):
    ntok = {"action": 2, "dino": N_PATCH, "state": 2, "cls": 1}
    series = [("action tokens", "action", ORANGE, "o"),
              ("head DINO", "dino", BLUE, "^"),
              ("state", "state", MUTED, "s")]
    for label, key, c, m in series:
        v = [_avg(act, f"cls_attn_L{i}_{key}") / ntok[key] * ACT_SEQ for i in range(nl)]
        ax.plot(range(nl), v, marker=m, color=c, lw=1.4, ms=5, label=label, zorder=4)
        if key == "action":
            ax.annotate(f"L0: {v[0]:.1f}x", (0, v[0]), textcoords="offset points",
                        xytext=(8, -4), fontsize=7, color=ORANGE)
    ax.axhline(1.0, color=INK, lw=1.0, ls="--", zorder=2)
    ax.text(nl - 1.1, 1.06, "uniform share = 1.0", fontsize=7, color=INK, ha="right")
    ax.set_xticks(range(nl), [f"L{i}" for i in range(nl)])
    ax.set_yscale("log")
    ax.set_ylabel(f"per-token attention / uniform share (1/{ACT_SEQ})")
    ax.set_title("(b) Read hard at L0 (12x), then exactly its fair share", loc="left")
    ax.grid(zorder=0)
    ax.legend(loc="upper right", fontsize=7)


def panel_compare(ax, act, trk, nl):
    """The quantity under test, per token, in both models."""
    a = [_avg(act, f"cls_attn_L{i}_action") / 2 * ACT_SEQ for i in range(nl)]
    tv = [_avg(trk, f"cls_attn_L{i}_track_valid") / _avg(trk, "n_valid") * TRK_SEQ for i in range(nl)]
    ti = [(_avg(trk, f"cls_attn_L{i}_track_all") - _avg(trk, f"cls_attn_L{i}_track_valid"))
          / (N_PATCH - _avg(trk, "n_valid")) * TRK_SEQ for i in range(nl)]
    td = [_avg(trk, f"cls_attn_L{i}_dino") / N_PATCH * TRK_SEQ for i in range(nl)]
    for label, v, c, m, ls in (("action-SVM: action token", a, ORANGE, "o", "-"),
                               ("track-SVM: valid track token", tv, "#c43b7a", "o", "--"),
                               ("track-SVM: constant invalid token", ti, "#8a8880", "s", "--"),
                               ("track-SVM: head DINO token", td, BLUE, "^", "--")):
        ax.plot(range(nl), v, marker=m, ls=ls, color=c, lw=1.4, ms=5, label=label, zorder=4)
    ax.axhline(1.0, color=INK, lw=1.0, ls="--", zorder=2)
    ax.set_xticks(range(nl), [f"L{i}" for i in range(nl)])
    ax.set_yscale("log")
    ax.set_ylabel("per-token attention / uniform share")
    ax.set_title("(c) The constant block is not a magnet: 1.1-1.3x, by headcount", loc="left")
    ax.grid(zorder=0)
    ax.legend(loc="lower left", fontsize=6.3)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sens", required=True, type=Path, help="outputs/ dir holding action_sens/ and track_sens/")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    style()
    act = json.loads((a.sens / "action_sens" / "action_pathway.json").read_text())
    trk = json.loads((a.sens / "track_sens" / "track_pathway.json").read_text())
    nl = act[0]["n_layers"]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.3))
    panel_mass(axes[0], act, nl)
    panel_pertoken(axes[1], act, nl)
    panel_compare(axes[2], act, trk, nl)
    fig.suptitle("Where CLS looks in the action-SVM, and what that says about the track-SVM's constant block",
                 x=0.013, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.013, 0.915,
             f"Same {len(act)} decision chunks as fig1. action-SVM sequence = "
             f"[CLS | state x2 | action x2 | head DINO x391] = {ACT_SEQ} tokens, no invalid mask and no constant "
             f"block (model.py:261 sets act_ok = ones). Per-token = mass / tokens in the segment.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = a.out / "fig3_action_attention.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
