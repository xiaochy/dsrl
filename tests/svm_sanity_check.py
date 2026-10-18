"""
SVM reward model sanity check: does Mercury007/mv-track-svm-scan-object
still separate success/failure once fed the DYNAMICS MODEL's PREDICTED
track (what the real DSRL reward loop will use) instead of the simulator
ground-truth track it was trained on? The README's own stated limitation
is exactly this distribution-shift risk -- this is the empirical check.

Consumes a chain_results/<task>_svm_sanity/episodeN_chain.pkl directory
produced by GAP-tte-.../svm_reward_sanity_pilot.sbatch (DYNAMICS_CHAIN_DIR +
CHAIN_COLLECT_MULTIVIEW=1 -- real policy actions, real multiview captures,
real success/failure labels from _rollouts.json).

Usage:
    python svm_sanity_check.py \
        --chain_dir .../chain_results/scan_object_svm_sanity \
        --rollouts_json .../results_svm_sanity_pilot/scan_object/GAP/demo_clean_depth/demo_clean/seed_0/200/_rollouts.json \
        --dynamics_ckpt .../dynamics-model-rollout-worldmodel/data/outputs/2026.09.28/19.53.05_.../checkpoints/200.ckpt
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from reward import build_svm_obs, load_dynamics_model, load_svm_reward_model, predict_track_for_action  # noqa: E402


def load_success_map(rollouts_json):
    with open(rollouts_json) as f:
        data = json.load(f)
    return {int(e["episode"]): bool(e["success"]) for e in data.get("episodes", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain_dir", required=True)
    ap.add_argument("--rollouts_json", required=True)
    ap.add_argument("--dynamics_ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Loading dynamics model: {args.dynamics_ckpt}")
    dynamics_model = load_dynamics_model(args.dynamics_ckpt, device)
    print("Loading SVM reward model (real downloaded checkpoint)...")
    svm = load_svm_reward_model(device=device)

    success_map = load_success_map(args.rollouts_json)
    paths = sorted(glob.glob(os.path.join(args.chain_dir, "episode*_chain.pkl")),
                   key=lambda p: int(re.search(r"episode(\d+)_chain\.pkl$", p).group(1)))
    print(f"{len(success_map)} episodes labeled, {len(paths)} chain pkl files")

    rows = []  # (episode, chunk_idx, outcome, logit, reward, pred_track_policy_logit, pred_track_policy_reward)
    skipped_no_multiview = 0
    for p in paths:
        ep = int(re.search(r"episode(\d+)_chain\.pkl$", p).group(1))
        if ep not in success_map:
            continue
        outcome = "success" if success_map[ep] else "failure"
        with open(p, "rb") as f:
            chunks = pickle.load(f)
        for ci, c in enumerate(chunks):
            mv = c.get("multiview")
            if mv is None:
                skipped_no_multiview += 1
                continue
            track_dyn = predict_track_for_action(
                dynamics_model, dino_feat=c["dino_features"], agent_pos=c["agent_pos"],
                patch_centers=c["patch_centers"], action=c["action"], device=device,
            )  # [391, H, 3]
            obs_dyn = build_svm_obs(mv, c["agent_pos"], track_dyn, c["valid_mask"])
            logit_dyn = float(svm.logits(obs_dyn)[0])
            reward_dyn = float(svm.rewards(obs_dyn)[0])

            # Also score with the POLICY's own track head output (already saved,
            # no extra model call) -- same obs otherwise, isolates "does using
            # the dynamics model's track vs. the policy's own track change
            # anything" from "does predicted track (either source) work at all".
            Hc = min(track_dyn.shape[1], c["pred_track_policy"].shape[1])
            obs_pol = build_svm_obs(mv, c["agent_pos"], c["pred_track_policy"][:, :Hc], c["valid_mask"])
            logit_pol = float(svm.logits(obs_pol)[0])
            reward_pol = float(svm.rewards(obs_pol)[0])

            rows.append((ep, ci, outcome, logit_dyn, reward_dyn, logit_pol, reward_pol))

    if skipped_no_multiview:
        print(f"WARNING: {skipped_no_multiview} chunks had no multiview data "
             "(CHAIN_COLLECT_MULTIVIEW not set for that run, or wrist camera "
             "depth missing) -- skipped")
    if not rows:
        print("No scoreable chunks found -- aborting")
        return

    import collections
    by_outcome = collections.defaultdict(list)
    for ep, ci, outcome, ld, rd, lp, rp in rows:
        by_outcome[outcome].append((ld, rd, lp, rp))

    def summarize(label, idx):
        print(f"\n--- {label} ---")
        for outcome in ("success", "failure"):
            vals = [r[idx] for r in by_outcome[outcome]]
            if not vals:
                continue
            arr = np.array(vals)
            print(f"  {outcome:8s} n={len(arr):4d}  mean={arr.mean():+.4f}  "
                 f"std={arr.std():.4f}  min={arr.min():+.4f}  max={arr.max():+.4f}")
        if by_outcome["success"] and by_outcome["failure"]:
            s = np.array([r[idx] for r in by_outcome["success"]])
            f = np.array([r[idx] for r in by_outcome["failure"]])
            # simple per-step AUC: P(success logit > failure logit) over all pairs
            auc = float((s[:, None] > f[None, :]).mean() + 0.5 * (s[:, None] == f[None, :]).mean())
            print(f"  per-step AUC (success-step logit > failure-step logit): {auc:.4f}")
            print(f"  (reference, from the model's own README, SAME task but GT track: 0.923)")

    summarize("Using DYNAMICS MODEL's predicted track (the real DSRL reward signal)", 1)
    summarize("Using POLICY's own track head (for comparison, not what DSRL will use)", 3)

    # Does the dynamics-model-based reward actually AGREE with the
    # policy-track-based reward per chunk, or do they just happen to land
    # on similar AUCs from different signals? Near-identical AUC alone
    # doesn't prove the two track sources give the SVM model the same
    # information -- direct per-chunk correlation does.
    reward_dyn_all = np.array([r[4] for r in rows])
    reward_pol_all = np.array([r[6] for r in rows])
    logit_dyn_all = np.array([r[3] for r in rows])
    logit_pol_all = np.array([r[5] for r in rows])
    pearson_r = float(np.corrcoef(reward_dyn_all, reward_pol_all)[0, 1])
    mean_abs_diff = float(np.abs(reward_dyn_all - reward_pol_all).mean())
    agree_sign = float((np.sign(logit_dyn_all) == np.sign(logit_pol_all)).mean())
    print(f"\n--- Per-chunk agreement: dynamics-track reward vs policy-track reward ---")
    print(f"  Pearson r = {pearson_r:.4f}  (across all {len(rows)} chunks)")
    print(f"  mean |reward_dyn - reward_pol| = {mean_abs_diff:.4f}")
    print(f"  fraction with same logit sign (same side of 0) = {agree_sign:.4f}")
    print(f"  (high correlation here means the two track sources give the SVM model near-")
    print(f"   identical information for THIS pool, despite being measurably different tracks")
    print(f"   by FDE/ADE in DYNAMICS_CHAIN_REPORT.md -- i.e. the reward model may be robust")
    print(f"   to track-source precision and mostly reading off coarser motion patterns. Low")
    print(f"   correlation would mean the matching AUCs are a coincidence of this small pool,")
    print(f"   not evidence the two track sources are interchangeable.)")

    n_eps = len(set(ep for ep, *_ in rows))
    n_succ_eps = len(set(ep for ep, ci, outcome, *_ in rows if outcome == "success"))
    print(f"\n{len(rows)} chunks scored across {n_eps} episodes "
         f"({n_succ_eps} success, {n_eps - n_succ_eps} failure)")


if __name__ == "__main__":
    main()
