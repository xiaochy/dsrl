"""
Scores one episode's chain pkl with BOTH reward models (track-SVM via the
dynamics model's predicted track, and the new action-SVM directly on the
real executed action) and dumps a JSON of per-chunk {start_frame, n_frames,
track_reward, track_logit, action_reward, action_logit} for visualization
(reward curve synced against the episode's own rollout video).

Usage:
    python score_episode_both_rewards.py --chain_pkl <path> --out <path.json>
"""
import argparse
import json
import pickle
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from reward import (  # noqa: E402
    build_svm_obs, build_svm_action_obs, load_dynamics_model,
    load_svm_reward_model, load_action_svm_reward_model, predict_track_for_action,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain_pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dynamics_ckpt", default="/project_data/held/vxiao/Bimanual_Manipulation_Project/dynamics-model-rollout-worldmodel/data/outputs/2026.09.28/19.53.05_GAP_scan_object_rollout-worldmodel/checkpoints/200.ckpt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    print("Loading dynamics model...")
    dynamics_model = load_dynamics_model(args.dynamics_ckpt, device)
    print("Loading track-SVM reward model...")
    track_svm = load_svm_reward_model(device=device)
    print("Loading action-SVM reward model (reusing track-SVM's DINOv2)...")
    action_svm = load_action_svm_reward_model(device=device, dino=track_svm.dino)

    with open(args.chain_pkl, "rb") as f:
        chunks = pickle.load(f)

    rows = []
    for ci, c in enumerate(chunks):
        mv = c.get("multiview")
        if mv is None:
            continue
        track_pred = predict_track_for_action(
            dynamics_model, dino_feat=c["dino_features"], agent_pos=c["agent_pos"],
            patch_centers=c["patch_centers"], action=c["action"], device=device,
        )
        track_obs = build_svm_obs(mv, c["agent_pos"], track_pred, c["valid_mask"])
        track_logit = float(track_svm.logits(track_obs)[0])
        track_reward = float(track_svm.rewards(track_obs)[0])

        action_obs = build_svm_action_obs(mv, c["agent_pos"], c["action"])
        action_logit = float(action_svm.logits(action_obs)[0])
        action_reward = float(action_svm.rewards(action_obs)[0])

        rows.append({
            "chunk": ci, "start_frame": c["start_frame"], "n_frames": c["n_frames"],
            "track_logit": track_logit, "track_reward": track_reward,
            "action_logit": action_logit, "action_reward": action_reward,
        })
        print(f"chunk {ci}: start_frame={c['start_frame']} track_reward={track_reward:+.4f} action_reward={action_reward:+.4f}")

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {len(rows)} chunk scores to {args.out}")


if __name__ == "__main__":
    main()
