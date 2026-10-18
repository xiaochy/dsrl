"""
Persistent subprocess: loads the (frozen) action-conditioned dynamics model
from dynamics-model-rollout-worldmodel ONCE, then services best-of-N
scoring requests over stdin/stdout (length-prefixed pickle), for the
lifetime of one eval job.

WHY A SUBPROCESS, NOT JUST A FUNCTION CALL: this process's gap_policy
import resolves to dynamics-model-rollout-worldmodel's copy (this script
lives there / imports it first, nothing else is ever on this process's
sys.path). The CALLER (GAP-tte-.../deploy_policy.py) has ALREADY imported
GAP-tte-flattenmlp-concatdit-actionrope's OWN same-named gap_policy package
to run the policy itself -- Python's module cache is keyed by name only,
so if both repos' GAPPolicy classes were needed in ONE process,
hydra.utils.instantiate's string-based `gap_policy.policy.gap.GAPPolicy`
lookup would resolve to whichever one is already in sys.modules,
regardless of which checkpoint's cfg asked for it (confirmed: this is
worse than the collision DYNAMICS_CHAIN_REPORT.md documents, which was
"only one of the two is ever actually instantiated in that process,
import order just picks the wrong one" -- here BOTH classes are genuinely
needed live, simultaneously, which no import-order fix or path trick
solves within one process). A real OS-level subprocess is the actual fix:
each process gets its own sys.modules, no collision possible.

Protocol (simple, not meant to be a general RPC framework): each message
is a pickled dict, length-prefixed with a 8-byte big-endian length header,
written/read via stdout/stdin.

Request:  {"dino_feat": np.ndarray, "agent_pos": np.ndarray,
           "patch_centers": np.ndarray, "multiview": dict,
           "actions": np.ndarray [N,H,16], "track_valid": np.ndarray}
Response: {"rewards": np.ndarray [N], "logits": np.ndarray [N],
           "best_idx": int} or {"error": str} on failure (caller should
          treat an error response as "fall back to actions[0]", never
          crash the live rollout over a scoring failure).

Usage (spawned by dynamics_scorer_client.py, not run directly):
    python dynamics_scorer_worker.py --dynamics_ckpt <path> --device cuda
"""
from __future__ import annotations

import argparse
import pickle
import struct
import sys
import traceback

import numpy as np

sys.path.insert(0, "/project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object")
from reward import build_svm_obs_batched, load_dynamics_model, load_svm_reward_model, predict_tracks_batched  # noqa: E402


def read_msg(stream) -> dict:
    header = stream.read(8)
    if len(header) < 8:
        return None  # EOF -- caller closed the pipe, shut down
    n = struct.unpack(">Q", header)[0]
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError("stdin closed mid-message")
        buf += chunk
    return pickle.loads(buf)


def write_msg(stream, obj: dict):
    buf = pickle.dumps(obj, protocol=4)
    stream.write(struct.pack(">Q", len(buf)))
    stream.write(buf)
    stream.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamics_ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.stderr.write(f"[dynamics_scorer_worker] loading dynamics model: {args.dynamics_ckpt}\n")
    dynamics_model = load_dynamics_model(args.dynamics_ckpt, args.device)
    sys.stderr.write("[dynamics_scorer_worker] loading SVM reward model...\n")
    svm = load_svm_reward_model(device=args.device)
    sys.stderr.write("[dynamics_scorer_worker] ready\n")
    sys.stderr.flush()

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        try:
            req = read_msg(stdin)
        except EOFError:
            break
        if req is None:
            break
        try:
            tracks = predict_tracks_batched(
                dynamics_model, dino_feat=req["dino_feat"], agent_pos=req["agent_pos"],
                patch_centers=req["patch_centers"], actions=req["actions"], device=args.device,
            )  # [N, 391, H, 3]
            obs = build_svm_obs_batched(req["multiview"], req["agent_pos"], tracks, req["track_valid"])
            logits = svm.logits(obs)
            rewards = svm.rewards(obs)
            best_idx = int(np.argmax(rewards))
            write_msg(stdout, {"rewards": rewards, "logits": logits, "best_idx": best_idx})
        except Exception:
            write_msg(stdout, {"error": traceback.format_exc()})


if __name__ == "__main__":
    main()
