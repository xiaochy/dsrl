"""Summary figures for the track-SVM insensitivity findings.

Reads what measure_track_sensitivity.py / diagnose_track_pathway.py already
wrote (plus the chain pkls for the spatial panels) and renders two figures:

  fig1_track_findings.png   the measurement and the pathway diagnosis
  fig2_valid_coverage.png   what `valid` actually covers, and GAP's weighting

    python tests/plot_track_findings.py \
        --sens   .../outputs/track_sens \
        --chain  .../chain_results/scan_object_track_sensitivity \
        --out    .../docs/track-sens/figs
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# ---- design tokens (dataviz reference palette, light mode) -------------------
SURF = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"      # categorical slot 1 -> head DINO / "the wide channel"
ORANGE = "#eb6834"    # categorical slot 2 -> track / "the narrow channel"
CONST = "#d8d7cf"     # chrome grey -> the constant (no-information) block
SEQ = ["#cde2fb", "#6da7ec", "#184f95"]   # blue sequential, low -> high

PATCH_GRID = (17, 23)
N_PATCH = 391
N_SEQ = 1 + 2 + 2 * N_PATCH
IMG_HW = (240, 320)


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "font.family": "sans-serif", "font.size": 8,
        "font.sans-serif": ["DejaVu Sans"],
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK,
        "axes.titlecolor": INK, "axes.titlesize": 9, "axes.titleweight": "bold",
        "axes.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
        "legend.fontsize": 7, "lines.linewidth": 1.2, "lines.markersize": 4,
    })


def patch_pixel_centres() -> np.ndarray:
    ph, pw = PATCH_GRID
    dino_hw = (ph * 14, pw * 14)
    gx, gy = np.meshgrid((np.arange(pw) + 0.5) * 14, (np.arange(ph) + 0.5) * 14)
    scale = np.array([IMG_HW[1] / dino_hw[1], IMG_HW[0] / dino_hw[0]])
    return np.stack([gx.ravel(), gy.ravel()], -1) * scale


# =============================== figure 1 ====================================
def panel_variants(ax, sens) -> None:
    """Every variant's |dlogit| against the across-state spread. The headline."""
    rows, order = sens["rows"], [
        ("gt_other", "track from another chunk"), ("gt_allinvalid", "whole block masked out"),
        ("gt_x0.5", "GT x 0.5"), ("gt_randdir", "random direction, same norm"),
        ("gt", "the true GT track"), ("gt_shuffle", "shuffled across patches"),
        ("gt_neg", "GT negated"), ("gt_x10", "GT x 10"), ("rand_10x", "pure noise x10"),
    ]
    across = float(np.std([r["logit_zero"] for r in rows]))
    rng = np.random.default_rng(0)
    for i, (k, label) in enumerate(order):
        d = np.abs([r[f"logit_{k}"] - r["logit_zero"] for r in rows])
        d = np.clip(d, 1e-5, None)
        ax.scatter(d, i + rng.uniform(-0.18, 0.18, len(d)), s=5, color=ORANGE,
                   alpha=0.3, linewidths=0, zorder=3)
        med = float(np.median(d))
        ax.plot([med], [i], marker="|", ms=14, mew=2.0, color=INK, zorder=5)
        ax.text(med, i - 0.30, f"{med:.4f}", fontsize=6.5, color=INK,
                ha="center", va="bottom", zorder=6)
    ax.axvline(across, color=INK, lw=1.2, ls="--", zorder=4)
    ax.text(across * 0.82, -0.62, f"across-state\nlogit sigma = {across:.2f}",
            fontsize=7, color=INK, ha="right", va="top", fontweight="bold")
    ax.set_yticks(range(len(order)), [l for _, l in order])
    ax.set_xscale("log")
    ax.set_xlim(4e-6, 30)
    ax.set_ylim(-1.25, len(order) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel("|logit(variant) - logit(zero track)|, 122 chunks\n"
                  "(tick = median; values below 1e-5 clipped to the left edge)")
    ax.set_title("(a) Swapping the track barely moves the score", loc="left")
    ax.grid(axis="x", zorder=0)
    ax.legend(handles=[Line2D([], [], marker="|", ls="", ms=10, mew=2, color=INK,
                              label="median"),
                       Line2D([], [], marker="o", ls="", ms=3, color=ORANGE, alpha=0.5,
                              label="one chunk")], loc="upper right",
              bbox_to_anchor=(1.02, -0.19), ncol=2)


def panel_attention(ax, path) -> None:
    """Where CLS spends its self-attention, stacked per layer."""
    nl = path[0]["n_layers"]
    f = lambda k: float(np.mean([x[k] for x in path]))  # noqa: E731
    segs = []
    for i in range(nl):
        tv = f(f"cls_attn_L{i}_track_valid")
        ti = f(f"cls_attn_L{i}_track_all") - tv
        dn = f(f"cls_attn_L{i}_dino")
        rest = f(f"cls_attn_L{i}_state") + f(f"cls_attn_L{i}_cls")
        segs.append((dn, tv, ti, rest))
    segs = np.array(segs)
    x = np.arange(nl)
    names = ["head DINO (391 tokens)", "tokens that carry a track (~24)",
             "constant invalid tokens (~367)", "state + CLS (3, < 0.3%)"]
    cols = [BLUE, ORANGE, CONST, MUTED]
    bottom = np.zeros(nl)
    for j, (nm, c) in enumerate(zip(names, cols)):
        ax.bar(x, segs[:, j], 0.72, bottom=bottom, color=c, label=nm,
               edgecolor=SURF, linewidth=1.2, zorder=3)
        bottom += segs[:, j]
    for i in range(nl):   # direct labels on the two segments that matter
        ax.text(i, segs[i, 0] / 2, f"{segs[i,0]*100:.0f}%", ha="center", va="center",
                fontsize=6.5, color="#ffffff", zorder=5)
        y = segs[i, 0] + segs[i, 1] / 2
        ax.annotate(f"{segs[i,1]*100:.1f}%", (i, y), xytext=(13, 0),
                    textcoords="offset points", fontsize=6.5, color=ORANGE,
                    ha="left", va="center", zorder=5,
                    arrowprops=dict(arrowstyle="-", lw=0.7, color=ORANGE))
    ax.set_xticks(x, [f"L{i}" for i in range(nl)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("CLS self-attention mass (sums to 1 per layer)")
    ax.set_title("(b) L0 glances at the track, L1+ pours it into the constant block", loc="left")
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)


def panel_pertoken(ax, path) -> None:
    """Same attention, per token - removes the "there are more of them" effect."""
    nl = path[0]["n_layers"]
    nv = np.array([x["n_valid"] for x in path], float)
    x = np.arange(nl)
    series = []
    for i in range(nl):
        tv = np.array([a[f"cls_attn_L{i}_track_valid"] for a in path])
        ta = np.array([a[f"cls_attn_L{i}_track_all"] for a in path])
        dn = np.array([a[f"cls_attn_L{i}_dino"] for a in path])
        series.append([(tv / (nv / N_SEQ)).mean(), ((ta - tv) / ((N_PATCH - nv) / N_SEQ)).mean(),
                       (dn / (N_PATCH / N_SEQ)).mean()])
    series = np.array(series)
    for j, (nm, c, m) in enumerate([("tokens that carry a track", ORANGE, "o"),
                                    ("constant invalid tokens", MUTED, "s"),
                                    ("head DINO", BLUE, "^")]):
        ax.plot(x, series[:, j], marker=m, color=c, label=nm, ms=5, zorder=4)
    ax.axhline(1.0, color=INK, lw=1.0, ls="--", zorder=3)
    ax.text(3.0, 1.06, "uniform share = 1.0", fontsize=6.5, color=INK, ha="center")
    ax.annotate(f"L0: {series[0,0]:.2f}x\n(over-attended)", (0, series[0, 0]), xytext=(16, -2),
                textcoords="offset points", fontsize=6.5, color=ORANGE, va="top",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=ORANGE))
    ax.set_xticks(x, [f"L{i}" for i in range(nl)])
    ax.set_ylim(0, 2.0)
    ax.set_ylabel("per-token attention / uniform share")
    ax.set_title("(c) Not \"too few tokens to notice\": L0 already over-attends", loc="left")
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper right")


def panel_gradient(ax, path) -> None:
    """The four gradient quantities: same norm, opposite verdicts."""
    f = lambda k: float(np.mean([x[k] for x in path]))  # noqa: E731
    labels = ["element count\nnumel", "gradient norm\n||dlogit/dx||",
              "per-element gradient\n||g||/sqrt(numel)",
              "whole-block leverage\n||g||*rms(x)*sqrt(numel)"]
    keys = ["numel_", "grad_", "grad_per_elem_", "leverage_"]
    y = np.arange(len(labels))
    h = 0.34
    for off, nm, c, who in ((+h / 2, "track", ORANGE, "track"), (-h / 2, "head DINO", BLUE, "dino")):
        vals = [f(k + who) for k in keys]
        ax.barh(y + off, vals, h, color=c, label=nm, edgecolor=SURF, linewidth=1.0, zorder=3)
        for yy, v in zip(y + off, vals):
            txt = (f"{v:,.0f}" if v >= 100 else f"{v:.2f}" if v >= 1
                   else f"{v:.4f}" if v >= 0.01 else f"{v:.6f}".rstrip("0"))
            ax.text(v * 1.35, yy, txt, fontsize=6.5, color=INK, va="center", zorder=5)
    for yy, k in zip(y, keys):
        r = f(k + "track") / f(k + "dino")
        ax.text(2e-6, yy + 0.42, f"track / DINO = {r:.3g}x", fontsize=6.5,
                color=INK2, va="center", style="italic")
    ax.set_yticks(y, labels)
    ax.set_xscale("log")
    ax.set_xlim(1e-6, 1e8)
    ax.invert_yaxis()
    ax.set_xlabel("normalized units (log scale)")
    ax.set_title("(d) Per number the track wins 17x, per block DINO wins 17x", loc="left")
    ax.grid(axis="x", zorder=0)
    ax.legend(loc="lower right")


def panel_firstorder(ax, sens, path) -> None:
    """Gradient (local) cannot predict the GT->zero swap (finite, non-linear)."""
    k = {(x["episode"], x["chunk"]): x for x in path}
    a = np.array([[r["logit_gt"] - r["logit_zero"],
                   k[(r["episode"], r["chunk"])]["dlogit_firstorder_zero_swap"]]
                  for r in sens["rows"] if (r["episode"], r["chunk"]) in k])
    corr = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
    lim = 12.0
    ax.plot([-lim, lim], [-lim, lim], color=AXIS, lw=1.0, ls="--", zorder=2)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=2)
    ax.axvline(0, color=AXIS, lw=0.8, zorder=2)
    ax.scatter(a[:, 0], a[:, 1], s=14, color=ORANGE, alpha=0.6, linewidths=0, zorder=4)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("measured  logit(GT) - logit(zero)")
    ax.set_ylabel("first-order prediction  g . (0 - x)")
    ax.text(0.03, 0.96, f"corr = {corr:+.3f}\ndashed = where linearity would put it",
            transform=ax.transAxes, fontsize=7, color=INK, va="top")
    ax.set_title("(e) The gradient cannot predict this swap: it is no small perturbation", loc="left")
    ax.grid(zorder=0)


def panel_budget(ax, path) -> None:
    """The 785-token budget: how little of it can carry a track."""
    nv = float(np.mean([x["n_valid"] for x in path]))
    parts = [("head DINO", N_PATCH, BLUE), ("carries a track", nv, ORANGE),
             ("constant invalid", N_PATCH - nv, CONST), ("state+CLS", 3, MUTED)]
    left = 0.0
    for nm, n, c in parts:
        ax.barh([0], [n], 0.5, left=left, color=c, edgecolor=SURF, linewidth=1.2, zorder=3)
        if n > 50:
            ax.text(left + n / 2, 0, f"{nm}\n{n:.0f}", ha="center", va="center",
                    fontsize=7, color="#ffffff" if c in (BLUE,) else INK, zorder=5)
        left += n
    ax.annotate(f"only {nv:.1f} tokens carry a track\n"
                f"= {nv/N_SEQ*100:.1f}% of the sequence, {nv/N_PATCH*100:.1f}% of the patches",
                (N_PATCH + nv / 2, 0.26), xytext=(0, 34), textcoords="offset points",
                fontsize=7.5, color=ORANGE, ha="center", va="bottom", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color=ORANGE))
    ax.set_xlim(0, N_SEQ)
    ax.set_ylim(-0.45, 0.95)
    ax.set_yticks([])
    ax.set_xlabel(f"self-attention sequence position ({N_SEQ} tokens in total)")
    ax.set_title("(f) The track's whole pathway: 1171 numbers vs DINO's 400k", loc="left")
    ax.spines["left"].set_visible(False)


def figure1(sens, path, out: Path) -> None:
    fig = plt.figure(figsize=(16.5, 9.2), dpi=160)
    gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.26,
                          left=0.065, right=0.985, top=0.9, bottom=0.085)
    panel_variants(fig.add_subplot(gs[0, 0]), sens)
    panel_attention(fig.add_subplot(gs[0, 1]), path)
    panel_pertoken(fig.add_subplot(gs[0, 2]), path)
    panel_gradient(fig.add_subplot(gs[1, 0]), path)
    panel_firstorder(fig.add_subplot(gs[1, 1]), sens, path)
    panel_budget(fig.add_subplot(gs[1, 2]), path)
    fig.suptitle("Does the track-SVM read its track?  "
                 "(scan_object, 122 decision chunks, measured 2026-10-29)",
                 fontsize=12, fontweight="bold", color=INK, x=0.065, ha="left", y=0.965)
    fig.text(0.065, 0.932,
             "The observation (3-view RGB-D + state) is held fixed; only batch[\"track\"] / "
             "batch[\"valid\"] is replaced. The dominant cause is training-target degeneracy "
             "(one track per state); `valid` covering only 6.2% of the patches is a second, "
             "compounding bottleneck.",
             fontsize=8, color=INK2, ha="left")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# =============================== figure 2 ====================================
def panel_coverage(ax, rec, px, title: str) -> None:
    rgb = np.asarray(rec["multiview"]["head"]["rgb"])
    v = np.asarray(rec["valid_mask"], bool)
    ax.imshow(rgb[..., ::-1] if rgb.shape[-1] == 3 else rgb, interpolation="nearest")
    ax.scatter(px[~v, 0], px[~v, 1], s=3.5, color=CONST, alpha=0.75, linewidths=0, zorder=3)
    ax.scatter(px[v, 0], px[v, 1], s=13, color=ORANGE, linewidths=0.4,
               edgecolors="#ffffff", zorder=4)
    ax.set_xlim(0, IMG_HW[1]); ax.set_ylim(IMG_HW[0], 0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(AXIS)
    ax.set_title(f"{title}   valid {int(v.sum())}/391 = {v.sum()/391*100:.1f}%",
                 loc="left", fontsize=8)


def panel_coverage_hist(ax, chain: Path) -> None:
    nv = []
    for p in sorted(chain.glob("episode*_chain.pkl")):
        for rec in pickle.load(open(p, "rb")):
            if rec.get("multiview") is not None:
                nv.append(int(np.asarray(rec["valid_mask"], bool).sum()))
    nv = np.array(nv)
    ax.hist(nv, bins=np.arange(0, 60, 3), color=ORANGE, edgecolor=SURF, linewidth=1.0, zorder=3)
    ax.axvline(nv.mean(), color=INK, lw=1.2, ls="--", zorder=4)
    ax.text(nv.mean() - 1.2, ax.get_ylim()[1] * 0.92, f"mean {nv.mean():.1f}",
            fontsize=7, color=INK, ha="right")
    ax.set_xlabel("patches with a GT track in one chunk (out of 391)")
    ax.set_ylabel("number of chunks")
    ax.set_title(f"(d) valid coverage over {len(nv)} chunks: {nv.min()}-{nv.max()}, "
                 f"never above 10%", loc="left")
    ax.grid(axis="y", zorder=0)


def panel_gap_weight(ax, chain: Path) -> None:
    """GAP never masks the track loss - it only soft-weights by movement."""
    tau, k = 0.01, 500.0
    W = {"invalid": 0.0, "static": 0.0, "moving": 0.0}
    disp_v, disp_i = [], []
    for p in sorted(chain.glob("episode*_chain.pkl")):
        for rec in pickle.load(open(p, "rb")):
            if rec.get("multiview") is None:
                continue
            gt = np.asarray(rec["action_track_gt"], np.float64)
            v = np.asarray(rec["valid_mask"], bool)
            d = np.linalg.norm(gt[:, -1, :], axis=-1)
            w = 1.0 / (1.0 + np.exp(-k * (d - tau)))
            W["invalid"] += w[~v].sum()
            W["static"] += w[v & (d <= tau)].sum()
            W["moving"] += w[v & (d > tau)].sum()
            disp_v += list(d[v]); disp_i += list(d[~v])
    tot = sum(W.values())
    xs = np.linspace(0, 0.05, 400)
    ax.plot(xs * 100, 1 / (1 + np.exp(-k * (xs - tau))), color=INK, lw=1.4, zorder=5,
            label="GAP movement weight\nsigmoid(500 * (|disp| - 1cm))")
    ax.axvline(tau * 100, color=AXIS, lw=1.0, ls="--", zorder=3)
    ax2 = ax.twiny()                      # a density strip, not a second value axis
    ax2.set_xlim(0, 5); ax2.set_xticks([]); ax2.spines["top"].set_visible(False)
    h_v, edges = np.histogram(np.array(disp_v) * 100, bins=40, range=(0, 5), density=True)
    h_i, _ = np.histogram(np.array(disp_i) * 100, bins=40, range=(0, 5), density=True)
    c = (edges[:-1] + edges[1:]) / 2
    ax2.fill_between(c, 0, h_i / max(h_i.max(), 1e-9) * 0.42, color=CONST, zorder=2,
                     step="mid", label="displacement of invalid patches (always 0)")
    ax2.fill_between(c, 0, h_v / max(h_v.max(), 1e-9) * 0.42, color=ORANGE, alpha=0.55,
                     zorder=3, step="mid", label="displacement of valid patches")
    ax.scatter([0], [1 / (1 + np.exp(5))], s=55, color=CONST, edgecolors=INK,
               linewidths=1.0, zorder=6)
    ax.text(0.02, 0.985,
            f"invalid patches land on the grey dot at |disp| = 0:\n"
            f"weight = {1/(1+np.exp(5)):.5f}, the same as a valid-but-static point,\n"
            f"yet they take {W['invalid']/tot*100:.1f}% of the track loss",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5, color=INK, fontweight="bold")
    ax.set_xlim(-0.15, 5); ax.set_ylim(-0.03, 1.34)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("GT last-step displacement |disp| (cm)")
    ax.set_ylabel("per-patch loss weight")
    ax.set_title("(e) GAP's loss never uses the mask, only a soft \"did it move\" weight",
                 loc="left")
    ax.grid(zorder=0)
    hs, ls = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(hs + h2, ls + l2, loc="center right")


def panel_gap_split(ax, chain: Path) -> None:
    tau, k = 0.01, 500.0
    W = np.zeros(3)
    for p in sorted(chain.glob("episode*_chain.pkl")):
        for rec in pickle.load(open(p, "rb")):
            if rec.get("multiview") is None:
                continue
            gt = np.asarray(rec["action_track_gt"], np.float64)
            v = np.asarray(rec["valid_mask"], bool)
            d = np.linalg.norm(gt[:, -1, :], axis=-1)
            w = 1.0 / (1.0 + np.exp(-k * (d - tau)))
            W += [w[~v].sum(), w[v & (d <= tau)].sum(), w[v & (d > tau)].sum()]
    W = W / W.sum() * 100
    names = ["invalid: arm proper + background", "valid, static", "valid, moving"]
    drop = [-0.37, -0.60, -0.37]
    left = 0.0
    for (n, w, c, dy) in zip(names, W, SEQ, drop):
        ax.barh([0], [w], 0.5, left=left, color=c, edgecolor=SURF, linewidth=1.2, zorder=3)
        ax.text(left + w / 2, 0, f"{w:.1f}%", ha="center", va="center", fontsize=9,
                fontweight="bold", color="#ffffff" if c == SEQ[2] else INK, zorder=5)
        ax.text(left + w / 2, dy, n, ha="center", va="top", fontsize=7, color=INK2, zorder=5)
        if dy < -0.4:
            ax.plot([left + w / 2] * 2, [-0.30, dy + 0.02], color=AXIS, lw=0.7, zorder=4)
        left += w
    ax.set_xlim(0, 100); ax.set_ylim(-0.95, 0.4)
    ax.set_yticks([]); ax.set_xlabel("share of GAP's track-loss weight mass (%)")
    ax.set_title("(f) 22% of the track supervision teaches \"arm and background do not move\"",
                 loc="left")
    ax.spines["left"].set_visible(False)


def figure2(chain: Path, out: Path) -> None:
    eps = sorted(chain.glob("episode*_chain.pkl"))
    picks = []
    for p in eps[:1] + eps[len(eps) // 2:len(eps) // 2 + 1] + eps[-1:]:
        chunks = [c for c in pickle.load(open(p, "rb")) if c.get("multiview") is not None]
        ep = p.stem.replace("_chain", "")
        if chunks:
            i = min(2, len(chunks) - 1)
            picks.append((chunks[i], f"({'abc'[len(picks)]}) {ep} chunk{i}"))
    px = patch_pixel_centres()

    fig = plt.figure(figsize=(16.5, 8.4), dpi=160)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.16,
                          left=0.05, right=0.985, top=0.885, bottom=0.085,
                          height_ratios=[1.05, 1.0])
    for j, (rec, title) in enumerate(picks):
        panel_coverage(fig.add_subplot(gs[0, j]), rec, px, title)
    panel_coverage_hist(fig.add_subplot(gs[1, 0]), chain)
    panel_gap_weight(fig.add_subplot(gs[1, 1]), chain)
    panel_gap_split(fig.add_subplot(gs[1, 2]), chain)
    fig.suptitle("What `valid` actually covers, and how GAP treats the same mask",
                 fontsize=12, fontweight="bold", color=INK, x=0.05, ha="left", y=0.962)
    fig.text(0.05, 0.925,
             "valid_point_mask = depth_valid & is_object (SAPIEN segmentation), where is_object "
             "covers the task objects AND the gripper finger links. Only the arm proper and the "
             "static background are excluded - the arm deliberately, since its pose is already "
             "known exactly from robot_state.",
             fontsize=8, color=INK2, ha="left")
    fig.legend(handles=[Patch(facecolor=ORANGE,
                              label="valid: has a GT track (task objects + gripper links)"),
                        Patch(facecolor=CONST,
                              label="invalid: no GT track (arm proper / background / invalid depth)")],
               loc="upper right", bbox_to_anchor=(0.985, 0.955), ncol=2)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sens", required=True, type=Path)
    p.add_argument("--chain", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    style()
    sens = json.loads((a.sens / "track_sensitivity.json").read_text())
    path = json.loads((a.sens / "track_pathway.json").read_text())
    figure1(sens, path, a.out / "fig1_track_findings.png")
    figure2(a.chain, a.out / "fig2_valid_coverage.png")


if __name__ == "__main__":
    main()
