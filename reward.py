"""
DSRL reward signal: action -> (frozen, separately-trained) dynamics model's
predicted track -> mv_svm reward model's dense reward. Neither the dynamics
model nor the SVM reward model's weights are touched here or anywhere in
the DSRL loop -- only the noise-steering SAC actor (gap_base_policy.py's
caller) is trained.

Why the DYNAMICS MODEL's track and not GAP's own joint-diffusion track
head: DYNAMICS_CHAIN_REPORT.md's whole point was establishing the
action-conditioned dynamics model is the more accurate track predictor for
a GIVEN action on most task/outcome splits (see that report for the
scan_object-specific numbers) -- and more importantly, it keeps GAP's own
weights fully out of the reward computation, so the reward is an honest
external judge of "what does this NOISE-STEERED action actually cause"
rather than partly reflecting GAP's own (frozen but still internally
consistent) self-assessment.

Verified (2026-10-13) against the REAL downloaded SVM checkpoint
(SVM-reference/checkpoints/mv-track-svm-scan-object) with synthetic
multiview/track data: shapes and dtypes line up, rewards() runs end to end
and returns finite values in [-2, 2] (lam*clip(logit,-20,20) range). NOT
yet verified with a real dynamics-model track prediction or real camera
captures (needs the scan_object dynamics-model checkpoint + a real
CHAIN_COLLECT_MULTIVIEW rollout, neither done as of this writing).
"""
from __future__ import annotations

import sys
from typing import Dict, Optional

import numpy as np
import torch

sys.path.insert(0, "/data/vxiao/bimanual_manipulation/dynamics-model-rollout-worldmodel")
sys.path.insert(0, "/data/vxiao/bimanual_manipulation/dynamics-model-rollout-worldmodel/scripts")
# SVM-reference's remote repo was updated 2026-10-13 (renamed mv_svm -> svm,
# refactored model.py/preprocess.py/reward_model.py, added MVActionSVMReward)
# -- import the live package, not the stale mirrored copy under
# checkpoints/mv-track-svm-scan-object/mv_svm/ left over from the first
# download, so both track and action models load through the same, current
# code.
sys.path.insert(0, "/data/vxiao/bimanual_manipulation/SVM-reference")

CAMERAS = ("head", "left_wrist", "right_wrist")
ACTION_SVM_CKPT_DIR = "/data/vxiao/bimanual_manipulation/SVM-reference/checkpoints/mv-action-svm-scan-object"


def load_dynamics_model(ckpt_path: str, device: str = "cuda"):
    """Thin re-export so callers don't need to know eval_track.py's path.
    ckpt must be an action_conditioned_world_model checkpoint (see
    dynamics-model-rollout-worldmodel/gap_policy/policy/gap.py)."""
    from eval_track import load_model  # noqa: E402
    policy = load_model(ckpt_path, device)
    assert getattr(policy, "action_conditioned_world_model", False), \
        f"{ckpt_path} is not an action_conditioned_world_model checkpoint"
    return policy


def load_svm_reward_model(device: str = "cuda", dino=None):
    """Loads from the local snapshot downloaded into SVM-reference/ --
    swap for MVTrackSVMReward.from_pretrained('Mercury007/mv-track-svm-scan-object')
    if that local copy ever goes stale."""
    from svm import MVTrackSVMReward  # noqa: E402
    ckpt_dir = "/data/vxiao/bimanual_manipulation/SVM-reference/checkpoints/mv-track-svm-scan-object"
    return MVTrackSVMReward(ckpt_dir, device=device, dino=dino)


def load_action_svm_reward_model(device: str = "cuda", dino=None):
    """MVActionSVMReward: scores (state, 16-step commanded EEF-pose chunk)
    directly -- no track, no dynamics-model dependency at all (see this
    repo's README "Known limitations": the action model barely reads its
    action input -- useless for ranking candidates at test time, but the
    README's own recommended use is as a dense RL reward, which is what
    DSRLScanObjectEnv._compute_reward uses it for).

    STATUS (2026-10-13): HF repo Mercury007/mv-action-svm-scan-object is
    not yet downloadable (`hf download` returns "Model ... not found",
    401 on a plain fetch too) -- blocked on that becoming public/
    accessible before this can be exercised. ACTION_SVM_CKPT_DIR points
    at where `hf download Mercury007/mv-action-svm-scan-object --local-dir
    checkpoints/mv-action-svm-scan-object` (run from SVM-reference/) will
    land once it works.
    """
    from svm import MVActionSVMReward  # noqa: E402
    return MVActionSVMReward(ACTION_SVM_CKPT_DIR, device=device, dino=dino)


def predict_track_for_action(dynamics_model, dino_feat: np.ndarray, agent_pos: np.ndarray,
                             patch_centers: np.ndarray, action: np.ndarray, device: str) -> np.ndarray:
    """One action -> [391, H, 3] predicted track (meters, world-frame
    displacement), via the frozen dynamics model. Thin wrapper around
    eval_track.predict_tracks -- see that function's docstring."""
    from eval_track import predict_tracks  # noqa: E402
    return predict_tracks(dynamics_model, dino_feat=dino_feat, agent_pos=agent_pos,
                          patch_centers=patch_centers, device=device, gt_action=action)


@torch.no_grad()
def predict_tracks_batched(dynamics_model, dino_feat: np.ndarray, agent_pos: np.ndarray,
                           patch_centers: np.ndarray, actions: np.ndarray, device: str) -> np.ndarray:
    """
    N candidate actions, ONE shared observation -> [N, 391, H, 3] predicted
    tracks in a single forward pass (same expand-along-batch trick
    deploy_policy.py's get_action_candidates uses for N candidate ACTIONS --
    here it's N candidate track PREDICTIONS for a fixed obs, one dynamics-
    model call instead of N). Used by dynamics_scorer_worker.py for best-of-N
    scoring.

    Args:
        dino_feat: [391, 1024] or [1, 391, 1024] (single obs, unbatched)
        agent_pos: [16]
        patch_centers: [391, 3]
        actions: [N, H, action_dim] RAW unnormalized candidate actions

    Returns:
        [N, 391, H, 3] meters, world-frame displacement
    """
    assert getattr(dynamics_model, "action_conditioned_world_model", False)
    N = actions.shape[0]
    dino_t = torch.from_numpy(np.asarray(dino_feat, dtype=np.float32))
    if dino_t.dim() == 2:
        dino_t = dino_t.unsqueeze(0)  # [1, 391, 1024]
    dino_t = dino_t.unsqueeze(0).to(device)  # [1, 1, 391, 1024]

    obs = {
        "agent_pos": torch.from_numpy(np.asarray(agent_pos, dtype=np.float32)).unsqueeze(0).expand(N, -1).to(device),
        "patch_centers": torch.from_numpy(np.asarray(patch_centers, dtype=np.float32)).unsqueeze(0).expand(N, -1, -1).to(device),
        "dino_features": dino_t.expand(N, -1, -1, -1).to(device),
    }
    action_t = torch.from_numpy(np.asarray(actions, dtype=np.float32)).to(device)  # [N, H, action_dim]
    result = dynamics_model.predict_track(obs, action=action_t)
    return result["track_pred"].cpu().numpy()  # [N, 391, H, 3]


def build_svm_obs(multiview: Dict[str, Dict[str, np.ndarray]], state: np.ndarray,
                  track: np.ndarray, track_valid: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Assembles the flat obs dict MVTrackSVMReward.logits/rewards expects,
    from the per-camera dict _collect_multiview_obs (deploy_policy.py)
    produces plus the dynamics model's predicted track.

    Args:
        multiview: {"head"/"left_wrist"/"right_wrist": {"rgb","depth","K","E"}},
            one UNBATCHED set (single decision step) -- see
            _collect_multiview_obs's docstring for exact shapes/units.
        state: [16] raw EEF-pose state (same convention as everywhere else
            this project uses "agent_pos"/"state").
        track: [391, H, 3] meters, world-frame displacement -- from
            predict_track_for_action. If H > 16, only the first 16 steps
            are used (mv_svm was trained with track_horizon=16).
        track_valid: [391] bool.

    Returns:
        obs: dict with leading batch dim 1 (what MVTrackSVMReward.rewards
            expects) for every key.
    """
    out: Dict[str, np.ndarray] = {}
    for cam in CAMERAS:
        if cam not in multiview:
            raise KeyError(f"multiview obs missing '{cam}' -- was CHAIN_COLLECT_MULTIVIEW=1 set, "
                           f"and did the task config have collect_wrist_camera+depth enabled?")
        c = multiview[cam]
        out[f"{cam}_rgb"] = c["rgb"][None].astype(np.uint8)
        out[f"{cam}_depth"] = c["depth"][None].astype(np.float32)
        out[f"{cam}_K"] = c["K"][None].astype(np.float32)
        out[f"{cam}_E"] = c["E"][None].astype(np.float32)
    out["state"] = np.asarray(state, dtype=np.float32)[None]
    out["track"] = np.asarray(track[:, :16], dtype=np.float32)[None]
    out["track_valid"] = np.asarray(track_valid, dtype=bool)[None]
    return out


def build_svm_obs_batched(multiview: Dict[str, Dict[str, np.ndarray]], state: np.ndarray,
                          tracks: np.ndarray, track_valid: np.ndarray) -> Dict[str, np.ndarray]:
    """
    build_svm_obs's batched counterpart: N candidate tracks against the
    SAME (shared, pre-action) multiview/state/valid-mask -- the camera obs
    and state don't depend on which candidate action is being scored, only
    `track` does (see this module's docstring on why -- the dynamics-
    model's track is the only thing that varies per candidate).

    Args:
        tracks: [N, 391, H, 3] meters -- from predict_tracks_batched.

    Returns:
        obs: dict with leading batch dim N for every key (cameras/state
            repeated N times, track genuinely different per row).
    """
    N = tracks.shape[0]
    out: Dict[str, np.ndarray] = {}
    for cam in CAMERAS:
        if cam not in multiview:
            raise KeyError(f"multiview obs missing '{cam}'")
        c = multiview[cam]
        out[f"{cam}_rgb"] = np.broadcast_to(c["rgb"].astype(np.uint8), (N, *c["rgb"].shape)).copy()
        out[f"{cam}_depth"] = np.broadcast_to(c["depth"].astype(np.float32), (N, *c["depth"].shape)).copy()
        out[f"{cam}_K"] = np.broadcast_to(c["K"].astype(np.float32), (N, *c["K"].shape)).copy()
        out[f"{cam}_E"] = np.broadcast_to(c["E"].astype(np.float32), (N, *c["E"].shape)).copy()
    out["state"] = np.broadcast_to(np.asarray(state, dtype=np.float32), (N, len(state))).copy()
    out["track"] = np.asarray(tracks[:, :, :16], dtype=np.float32)
    out["track_valid"] = np.broadcast_to(np.asarray(track_valid, dtype=bool), (N, len(track_valid))).copy()
    return out


def build_svm_action_obs(multiview: Dict[str, Dict[str, np.ndarray]], state: np.ndarray,
                         action: np.ndarray) -> Dict[str, np.ndarray]:
    """build_svm_obs's counterpart for MVActionSVMReward: no track/
    track_valid/dynamics-model step at all -- the SAME action chunk
    about to be (or that was) executed is the input directly.

    Args:
        state: [16] raw EEF-pose state.
        action: [16, 16] raw (unnormalized) commanded EEF-pose chunk,
            same convention as everywhere else in this project.
    """
    out: Dict[str, np.ndarray] = {}
    for cam in CAMERAS:
        if cam not in multiview:
            raise KeyError(f"multiview obs missing '{cam}' -- was CHAIN_COLLECT_MULTIVIEW=1 set, "
                           f"and did the task config have collect_wrist_camera+depth enabled?")
        c = multiview[cam]
        out[f"{cam}_rgb"] = c["rgb"][None].astype(np.uint8)
        out[f"{cam}_depth"] = c["depth"][None].astype(np.float32)
        out[f"{cam}_K"] = c["K"][None].astype(np.float32)
        out[f"{cam}_E"] = c["E"][None].astype(np.float32)
    out["state"] = np.asarray(state, dtype=np.float32)[None]
    out["action"] = np.asarray(action, dtype=np.float32)[None]
    return out


def build_svm_action_obs_batched(multiview: Dict[str, Dict[str, np.ndarray]], state: np.ndarray,
                                 actions: np.ndarray) -> Dict[str, np.ndarray]:
    """N candidate actions against the SAME shared multiview/state --
    unlike the track model's batched builder, this needs no dynamics-model
    call first (no predict_tracks_batched counterpart) since `action` is
    exactly the raw candidate the caller already has.

    Args:
        actions: [N, 16, 16] raw candidate action chunks.
    """
    N = actions.shape[0]
    out: Dict[str, np.ndarray] = {}
    for cam in CAMERAS:
        if cam not in multiview:
            raise KeyError(f"multiview obs missing '{cam}'")
        c = multiview[cam]
        out[f"{cam}_rgb"] = np.broadcast_to(c["rgb"].astype(np.uint8), (N, *c["rgb"].shape)).copy()
        out[f"{cam}_depth"] = np.broadcast_to(c["depth"].astype(np.float32), (N, *c["depth"].shape)).copy()
        out[f"{cam}_K"] = np.broadcast_to(c["K"].astype(np.float32), (N, *c["K"].shape)).copy()
        out[f"{cam}_E"] = np.broadcast_to(c["E"].astype(np.float32), (N, *c["E"].shape)).copy()
    out["state"] = np.broadcast_to(np.asarray(state, dtype=np.float32), (N, len(state))).copy()
    out["action"] = np.asarray(actions, dtype=np.float32)
    return out


def compute_action_reward(svm_reward_model, multiview: Dict[str, Dict[str, np.ndarray]],
                          state: np.ndarray, action: np.ndarray) -> float:
    """End-to-end action-SVM reward for one decision step: no dynamics
    model, no subprocess isolation needed (MVActionSVMReward lives in the
    independent SVM-reference package, no gap_policy collision risk --
    unlike the track path, this can run in-process alongside GAP's own
    policy or the DSRL env)."""
    obs = build_svm_action_obs(multiview, state, action)
    reward = svm_reward_model.rewards(obs)
    return float(reward[0])


def compute_reward(dynamics_model, svm_reward_model, multiview: Dict[str, Dict[str, np.ndarray]],
                   dino_feat: np.ndarray, agent_pos: np.ndarray, patch_centers: np.ndarray,
                   track_valid: np.ndarray, action: np.ndarray, device: str) -> float:
    """End-to-end: action -> dynamics-model track -> SVM reward. One
    decision step at a time (not batched across env steps -- DSRL's
    vectorized envs call this once per sub-env per step, see
    env.py's DSRLScanObjectEnv.step, which is still a stub)."""
    track = predict_track_for_action(dynamics_model, dino_feat, agent_pos, patch_centers, action, device)
    obs = build_svm_obs(multiview, agent_pos, track, track_valid)
    reward = svm_reward_model.rewards(obs)
    return float(reward[0])
