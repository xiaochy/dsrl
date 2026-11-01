"""Video of the track-sensitivity measurement: what the model is shown on the
left, what the swap does to its logit on the right.

One frame per decision chunk (the anchor frame is the only one whose multiview
observation was captured). Left: the head camera with the GT track drawn on the
patches that have one -- each arrow runs from the patch centre to where that 3D
point ends up 16 steps later, projected with the head K/E. Right top: the logit
change each track variant causes AT THIS FIXED OBSERVATION, on the same axis as
the across-state spread (grey band) so the two are directly comparable. Right
bottom: the logit over the episode for a few variants, with a cursor.

Needs tests/measure_track_sensitivity.py to have been run first.

    python tests/render_track_sensitivity.py \
        --chain_dir .../chain_results/scan_object_track_sensitivity \
        --sens .../outputs/track_sens/track_sensitivity.json \
        --out .../results/videos/track_sensitivity.mp4
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402

PATCH_GRID = (17, 23)
IMG_HW = (240, 320)
DINO_HW = (PATCH_GRID[0] * 14, PATCH_GRID[1] * 14)
CURVES = ("zero", "gt", "gt_neg", "rand_10x")


def patch_pixel_centres() -> np.ndarray:
    """[391, 2] native-pixel (x, y) patch centres -- same grid as svm/preprocess.py."""
    ph, pw = PATCH_GRID
    gx, gy = np.meshgrid((np.arange(pw) + 0.5) * 14, (np.arange(ph) + 0.5) * 14)
    scale = np.array([IMG_HW[1] / DINO_HW[1], IMG_HW[0] / DINO_HW[0]])
    return np.stack([gx.ravel(), gy.ravel()], -1) * scale


def project(pts_world: np.ndarray, K: np.ndarray, E: np.ndarray) -> np.ndarray:
    """[N,3] world -> [N,2] pixels (OpenCV convention, E = [R|t] world->cam)."""
    cam = pts_world @ E[:, :3].T + E[:, 3]
    z = np.clip(cam[:, 2:3], 1e-6, None)
    return (cam / z) @ K[:3, :3].T[:, :2]


def draw_track(rgb: np.ndarray, rec: dict, scale: int = 3) -> np.ndarray:
    """Head RGB with the GT track of the valid patches drawn as arrows."""
    img = cv2.resize(rgb, (rgb.shape[1] * scale, rgb.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    mv = rec["multiview"]["head"]
    K, E = np.asarray(mv["K"], np.float64), np.asarray(mv["E"], np.float64)
    centres = np.asarray(rec["patch_centers"], np.float64)
    gt = np.asarray(rec["action_track_gt"], np.float64)
    valid = np.asarray(rec["valid_mask"], bool)
    px = patch_pixel_centres()

    idx = np.where(valid)[0]
    if len(idx) == 0:
        return img
    end = project(centres[idx] + gt[idx, -1], K, E)
    mag = np.linalg.norm(gt[idx, -1], axis=-1)
    hi = max(float(mag.max()), 1e-6)
    for n, i in enumerate(idx):
        a = (int(round(px[i, 0] * scale)), int(round(px[i, 1] * scale)))
        b = (int(round(end[n, 0] * scale)), int(round(end[n, 1] * scale)))
        # cold -> hot with displacement magnitude
        c = cv2.applyColorMap(np.uint8([[mag[n] / hi * 255]]), cv2.COLORMAP_JET)[0, 0].tolist()
        cv2.circle(img, a, 2, c, -1, cv2.LINE_AA)
        if max(abs(b[0] - a[0]), abs(b[1] - a[1])) >= 2:
            cv2.arrowedLine(img, a, b, c, 1, cv2.LINE_AA, tipLength=0.3)
    return img


def to_h264(path: Path) -> None:
    """Re-encode in place to H.264.

    cv2.VideoWriter only offers MPEG-4 Part 2 here (fourcc "mp4v"); browsers,
    QuickTime and most modern players refuse that stream, so the file looks
    corrupt even though it is fine. ffmpeg -> H.264 yuv420p +faststart plays
    everywhere and is about half the size.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("WARNING: ffmpeg not found; leaving the mp4v stream (may not open)")
        return
    tmp = path.with_name(path.stem + "_h264.mp4")
    r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(path),
                        "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)])
    if r.returncode == 0:
        tmp.replace(path)
    else:
        tmp.unlink(missing_ok=True)
        print("WARNING: ffmpeg failed; leaving the mp4v stream (may not open)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chain_dir", required=True, type=Path)
    p.add_argument("--sens", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--episodes", type=int, nargs="*", default=None, help="default: all in --sens")
    p.add_argument("--hold", type=float, default=1.2, help="seconds per chunk")
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sens = json.loads(args.sens.read_text())
    variants, rows = sens["variants"], sens["rows"]
    across = float(np.std([r["logit_zero"] for r in rows]))

    eps = args.episodes if args.episodes else sorted({r["episode"] for r in rows})
    shown = [k for k in variants if k != "zero"]

    writer, size = None, None
    for ep in eps:
        ep_rows = [r for r in rows if r["episode"] == ep]
        if not ep_rows:
            continue
        with open(args.chain_dir / f"episode{ep}_chain.pkl", "rb") as f:
            chunks = [c for c in pickle.load(f) if c.get("multiview") is not None]
        series = {k: np.array([r[f"logit_{k}"] for r in ep_rows]) for k in CURVES if f"logit_{k}" in ep_rows[0]}

        for r in ep_rows:
            rec = chunks[r["chunk"]]
            left = draw_track(np.asarray(rec["multiview"]["head"]["rgb"]), rec)

            fig = plt.figure(figsize=(7.2, left.shape[0] / 100.0), dpi=100)
            ax1 = fig.add_axes([0.26, 0.56, 0.71, 0.38])
            ax2 = fig.add_axes([0.10, 0.09, 0.87, 0.33])

            d = np.array([r[f"logit_{k}"] - r["logit_zero"] for k in shown])
            y = np.arange(len(shown))
            ax1.axvspan(-across, across, color="0.85", zorder=0,
                        label=f"across-state spread $\\pm${across:.2f}")
            ax1.barh(y, d, color=["#c0392b" if abs(x) > 0.1 else "#2980b9" for x in d], zorder=3)
            ax1.axvline(0, color="k", lw=0.8, zorder=4)
            ax1.set_yticks(y, shown, fontsize=7)
            ax1.invert_yaxis()
            ax1.set_xlabel("logit(variant) $-$ logit(zero track)   (symlog, linear within $\\pm$0.01)",
                           fontsize=8)
            ax1.tick_params(labelsize=7)
            # the effect spans ~0.001 to ~7, and the across-state band is ~5:
            # a linear axis shows either the band or the bars, never both
            ax1.set_xscale("symlog", linthresh=0.01, linscale=0.4)
            lim = max(float(np.abs(d).max()) * 2.0, across * 1.5, 0.1)
            ax1.set_xlim(-lim, lim)
            ax1.grid(axis="x", alpha=0.3, zorder=1)
            ax1.legend(fontsize=6, loc="lower right")
            ax1.set_title(f"ep{ep} chunk{r['chunk']} (frame {r['start_frame']})   "
                          f"valid patches {r['n_valid']}/391   GT rms {r['gt_rms'] * 100:.2f} cm   "
                          f"logit(zero)={r['logit_zero']:+.3f}", fontsize=8)

            x = np.arange(len(ep_rows))
            for k, s in series.items():
                ax2.plot(x, s, marker="o", ms=2.5, lw=1.2, label=k)
            ax2.axvline(r["chunk"], color="k", ls="--", lw=0.8)
            ax2.set_xlabel("decision chunk", fontsize=8)
            ax2.set_ylabel("logit", fontsize=8)
            ax2.tick_params(labelsize=7)
            ax2.legend(fontsize=6, ncol=len(series))
            ax2.grid(alpha=0.3)

            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            right = np.asarray(canvas.buffer_rgba())[..., :3][..., ::-1].copy()
            plt.close(fig)

            h = max(left.shape[0], right.shape[0])
            pad = lambda im: np.pad(im, ((0, h - im.shape[0]), (0, 0), (0, 0)), constant_values=255)  # noqa: E731
            frame = np.hstack([pad(left), pad(right)])
            if writer is None:
                size = (frame.shape[1], frame.shape[0])
                writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size)
            for _ in range(max(1, int(round(args.hold * args.fps)))):
                writer.write(frame)
        print(f"ep{ep}: {len(ep_rows)} chunks rendered")

    if writer is not None:
        writer.release()
        to_h264(args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
