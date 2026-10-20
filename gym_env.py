"""
Thin gymnasium.Env wrapper around env.py's DSRLScanObjectEnv, giving it the
interface stable_baselines3's SAC (DSRL-reference/stable-baselines3, a local
fork, not pip stable-baselines3) needs to train against directly:

    action_space:      Box(-action_magnitude, action_magnitude,
                        shape=(horizon*action_dim,)) -- the SAC agent's
                        output IS the DSRL noise-steering action (see
                        arXiv:2506.15799); DSRLScanObjectEnv.step() already
                        does noise -> diffused-action -> execute -> reward
                        internally, so this wrapper does NOT need DSRL-
                        reference's own DiffusionPolicyEnvWrapper/
                        ActionChunkWrapper (those exist to bolt a base
                        policy onto a vanilla robomimic/gym env from the
                        OUTSIDE; here the base-policy call already lives
                        inside DSRLScanObjectEnv.step()).

    observation_space: Box, flat vector. DSRLScanObjectEnv's own obs is a
                        dict with a 391x1024 unpooled DinoV2 patch grid
                        (needed internally by GAP's own RoPE4D cross-
                        attention) -- too large and structured to hand SAC's
                        plain MlpPolicy directly. PROVISIONAL CHOICE: mean-
                        pool dino_features over the 391 patches (-> 1024)
                        and concat with state (16) = 1040-dim. This throws
                        away spatial detail SAC's actor/critic might
                        benefit from; swapping in a smarter encoder (e.g. a
                        small CNN/attention head over the patch grid, or
                        SB3's MultiInputPolicy on the dict obs directly) is
                        a reasonable thing to revisit once this pipeline is
                        confirmed training at all -- flagged here rather
                        than guessed at further.

Returns gymnasium's 5-tuple from step() (obs, reward, terminated, truncated,
info) -- SB3's VecEnv wrapping (DummyVecEnv) expects this and folds it into
the 4-tuple VecEnv API itself.
"""
from __future__ import annotations

import numpy as np
import gymnasium
from gymnasium import spaces

from env import DSRLScanObjectEnv


def _flatten_obs(obs: dict) -> np.ndarray:
    return np.concatenate([
        np.asarray(obs["state"], dtype=np.float32),
        np.asarray(obs["dino_features"], dtype=np.float32).mean(axis=0),
    ])


class DSRLScanObjectGymEnv(gymnasium.Env):
    def __init__(self, task_name: str = "scan_object", task_config: str = "demo_clean_depth",
                gap_ckpt: str = "", dynamics_ckpt: str = "", device: str = "cuda",
                seed: int = 0, step_lim: int = 300, action_magnitude: float = 1.5,
                reward_source: str = "track", expert_check: bool = False,
                num_inference_steps: int = 0, fixed_track_noise: bool = True):
        self.env = DSRLScanObjectEnv(
            task_name=task_name, task_config=task_config, gap_ckpt=gap_ckpt,
            dynamics_ckpt=dynamics_ckpt, device=device, seed=seed, step_lim=step_lim,
            reward_source=reward_source, expert_check=expert_check,
            num_inference_steps=num_inference_steps,
            fixed_track_noise=fixed_track_noise,
        )
        # Surfaced so the training entry point can validate the noise
        # parameterization against it without reaching through two wrappers.
        self.prediction_type = self.env.prediction_type
        horizon, action_dim = self.env.base_policy.horizon, self.env.base_policy.action_dim
        self.action_space = spaces.Box(
            low=-action_magnitude * np.ones(horizon * action_dim, dtype=np.float32),
            high=action_magnitude * np.ones(horizon * action_dim, dtype=np.float32),
            dtype=np.float32,
        )
        # 16 (state) + 1024 (mean-pooled dino_features) -- see module docstring.
        obs_dim = 16 + 1024
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(obs_dim, dtype=np.float32),
            high=np.inf * np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        return _flatten_obs(obs), {}

    def step(self, action):
        obs, reward, done, info = self.env.step(np.asarray(action, dtype=np.float32))
        terminated = bool(info["success"])
        truncated = bool(done and not terminated)
        return _flatten_obs(obs), float(reward), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        self.env.close()

    def reset_seed_counter(self, seed: int):
        self.env.reset_seed_counter(seed)
