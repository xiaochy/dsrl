"""
Sanity check for MVActionSVMReward (Mercury007/mv-action-svm-scan-object):
does it separate success/failure steps on real scan_object rollouts, using
the SAME chain_results/*.pkl data svm_sanity_check.py (the track model's
sanity check) already consumes? No dynamics-model prediction needed here --
the action model scores (state, action) directly, using the REAL executed
action already stored in each chunk record.

This only checks success-vs-failure discrimination on the REAL executed
action (the README's own recommended RL-reward use case). It does NOT
check per-candidate discrimination for best-of-N (the README's flagged
"barely reads its action input" limitation) -- that needs a separate
script sampling fresh candidates from the same observation, not yet
written; treat a good AUC here as support for the RL-reward use only.

Usage:
    python action_svm_sanity_check.py \
        --chain_dir .../chain_results/scan_object_svm_sanity_large \
        --rollouts_json .../results_svm_sanity_pilot_large/scan_object/GAP/demo_clean_depth/demo_clean/seed_0/200/_rollouts.json
"""
import argparse
import collections
import glob
import json
import os
import pickle
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from reward import build_svm_action_obs, load_action_svm_reward_model  # noqa: E402


def load_success_map(rollouts_json):
    with open(rollouts_json) as f:
        data = json.load(f)
    return {int(e["episode"]): bool(e["success"]) for e in data.get("episodes", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain_dir", required=True)
    ap.add_argument("--rollouts_json", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print("Loading MVActionSVMReward (Mercury007/mv-action-svm-scan-object)...")
    svm = load_action_svm_reward_model(device=device)

    success_map = load_success_map(args.rollouts_json)
    paths = sorted(glob.glob(os.path.join(args.chain_dir, "episode*_chain.pkl")),
                   key=lambda p: int(re.search(r"episode(\d+)_chain\.pkl$", p).group(1)))
    print(f"{len(success_map)} episodes labeled, {len(paths)} chain pkl files")

    rows = []  # (episode, chunk_idx, outcome, logit, reward)
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
            obs = build_svm_action_obs(mv, c["agent_pos"], c["action"])
            logit = float(svm.logits(obs)[0])
            from svm import svm_reward
            reward = float(svm_reward(np.asarray([logit]))[0])
            rows.append((ep, ci, outcome, logit, reward))

    if skipped_no_multiview:
        print(f"WARNING: {skipped_no_multiview} chunks had no multiview data -- skipped")
    if not rows:
        print("No scoreable chunks found -- aborting")
        return

    by_outcome = collections.defaultdict(list)
    for ep, ci, outcome, logit, reward in rows:
        by_outcome[outcome].append((logit, reward))

    print("\n--- MVActionSVMReward on real executed actions ---")
    for outcome in ("success", "failure"):
        vals = by_outcome[outcome]
        if not vals:
            continue
        rewards = np.array([r for _, r in vals])
        print(f"  {outcome:8s} n={len(rewards):4d}  mean={rewards.mean():+.4f}  "
             f"std={rewards.std():.4f}  min={rewards.min():+.4f}  max={rewards.max():+.4f}")
    if by_outcome["success"] and by_outcome["failure"]:
        s = np.array([l for l, _ in by_outcome["success"]])
        f = np.array([l for l, _ in by_outcome["failure"]])
        auc = float((s[:, None] > f[None, :]).mean() + 0.5 * (s[:, None] == f[None, :]).mean())
        print(f"  per-step AUC (success-step logit > failure-step logit): {auc:.4f}")
        print(f"  (reference, from the model's own README, held-out eval: 0.917)")

    n_eps = len(set(ep for ep, *_ in rows))
    n_succ_eps = len(set(ep for ep, ci, outcome, *_ in rows if outcome == "success"))
    print(f"\n{len(rows)} chunks scored across {n_eps} episodes "
         f"({n_succ_eps} success, {n_eps - n_succ_eps} failure)")


if __name__ == "__main__":
    main()
