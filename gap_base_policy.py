"""
DSRL-compatible wrapper around GAPPolicy. Matches the exact interface
DSRL-reference/utils.py's DPPOBasePolicyWrapper exposes to train_dsrl.py's
DiffusionPolicyEnvWrapper:

    diffused_actions = base_policy(obs, initial_noise, return_numpy=True)

obs here is whatever encode_obs.py in deploy_policy.py would normally build
for GAP (rgb -> dino_features, state, patch_centers), already extracted --
this wrapper does NOT run the vision backbone itself (that happens once in
the env step, shared with the reward computation, not re-run per SAC
candidate).

The actual DDIM-noise-steering mechanics live in gap_policy.policy.gap's
GAPPolicy.predict_action_from_noise -- see that method's docstring (and
conditional_sample_from_noise's) for how this stays in-distribution: the
same DDIM reverse process as normal inference, with an externally supplied
initial noise tensor instead of a fresh torch.randn draw. GAP's weights are
never touched by DSRL training; only the noise distribution the SAC actor
learns to sample from changes.

Verified (2026-10-13, synthetic obs/weights): correct output shape,
deterministic given (noise, generator), and different noise genuinely
produces different actions. NOT yet verified against a real trained
scan_object checkpoint (none exists yet as of this writing).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch


class GAPBasePolicyWrapper:
    def __init__(self, gap_policy, horizon: int, action_dim: int, device: str = "cuda"):
        """
        Args:
            gap_policy: a loaded, .eval()'d GAPPolicy (see
                dynamics-model-rollout-worldmodel/scripts/eval_track.py's
                load_model, or the GAP-tte repo's own checkpoint-loading
                path -- this must be the scan_object POLICY checkpoint,
                i.e. a normal joint action+track GAPPolicy, NOT the
                action-conditioned dynamics model -- that one never
                predicts actions at all, see action_conditioned_world_model
                in dynamics-model-rollout-worldmodel's gap.py).
            horizon: GAP's action horizon (16 throughout this project).
            action_dim: GAP's raw action dim (16: 2 arms x 8).
        """
        self.policy = gap_policy.eval()
        self.horizon = horizon
        self.action_dim = action_dim
        self.device = device

    @torch.no_grad()
    def __call__(self, obs: Dict[str, torch.Tensor], initial_noise, return_numpy: bool = True):
        """
        Args:
            obs: dict with GAP's raw (unnormalized) obs_dict keys --
                'agent_pos' [B,16], 'dino_features' [B,1,391,1024], and
                'patch_centers' [B,391,3] for 4D RoPE. B = cfg.env.n_envs
                (DSRL's parallel-env count).
            initial_noise: [B, horizon*action_dim] or [B, horizon,
                action_dim], the SAC agent's sampled action (numpy or
                torch) -- this IS the DSRL noise-steering action, bounded
                by cfg.train.action_magnitude in cfg/dsrl_scan_object.yaml.

        Returns:
            diffused_actions: [B, horizon, action_dim], numpy if
                return_numpy else torch -- RAW (unnormalized) EEF-pose
                action chunk, ready to hand to TASK_ENV.take_action per
                step (same convention as deploy_policy.py's eval()).
        """
        if not torch.is_tensor(initial_noise):
            initial_noise = torch.as_tensor(initial_noise, dtype=torch.float32)
        initial_noise = initial_noise.to(self.device)
        if initial_noise.dim() == 2:
            initial_noise = initial_noise.view(-1, self.horizon, self.action_dim)

        obs_t = {
            k: (v if torch.is_tensor(v) else torch.as_tensor(v, dtype=torch.float32)).to(self.device)
            for k, v in obs.items()
        }

        action = self.policy.predict_action_from_noise(obs_t, initial_noise)
        if return_numpy:
            action = action.detach().cpu().numpy()
        return action
