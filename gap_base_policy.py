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
    def __init__(self, gap_policy, horizon: int, action_dim: int, device: str = "cuda",
                 track_seed: int = 0):
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
            track_seed: seed for the generator handed to
                conditional_sample_from_noise. REQUIRED for a controlled
                experiment, and previously missing: with joint_diffusion the
                track branch draws its own torch.randn, and passing
                generator=None makes that draw come from the GLOBAL torch RNG.
                The track influences the action, so the policy's output was a
                function of (obs, DSRL noise, whatever global RNG state the
                process happened to be in) -- not reproducible run to run, and
                not a function the DSRL agent can control.

                GAP's own eval path does NOT have this problem: it passes
                self._sample_generator (deploy_policy.py:507/:561, seeded at
                :163-164 with an explicit comment that without it "re-running
                the identical eval can silently land on a different success
                rate purely from sampling noise"). The DSRL path simply never
                plumbed it through.

                Why it matters beyond reproducibility: at the deployed
                num_inference_steps=10 the DSRL noise's authority over the
                action is ~1/370, so uncontrolled track noise can easily
                dominate the agent's own influence on the outcome. For RL that
                is unobservable, uncontrollable return variance sitting on top
                of an already tiny signal. Found 2026-10-19, when a recorded
                rollout disagreed with the outcome-sensitivity measurement on
                the same scene and nominally the same noise draw.
        """
        self.policy = gap_policy.eval()
        self.horizon = horizon
        self.action_dim = action_dim
        self.device = device
        self.track_seed = track_seed
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(track_seed)

    def reseed_track(self, seed: int):
        """Re-seed the track generator, e.g. at the start of each episode so
        repeated runs of one scene see an IDENTICAL track-noise stream and the
        only thing varying is the DSRL action."""
        self.track_seed = seed
        self._generator.manual_seed(seed)

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
                torch) -- this IS the DSRL noise-steering action. Nothing
                here clips or rescales it; the bound, if any, comes from the
                SAC actor's own parameterization. With
                cfg.train.squash_actions=True it is hard-bounded by
                cfg.train.action_magnitude (and so cannot reproduce the tails
                of the N(0,1) that GAP's DDIM was trained on); with
                squash_actions=False it is an unbounded Gaussian.

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

        action = self.policy.predict_action_from_noise(
            obs_t, initial_noise, generator=self._generator)
        if return_numpy:
            action = action.detach().cpu().numpy()
        return action
