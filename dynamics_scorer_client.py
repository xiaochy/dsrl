"""
Client for dynamics_scorer_worker.py -- spawns the worker subprocess ONCE
(model loading happens there, not per call) and exposes a plain function
call interface to the caller (deploy_policy.py's eval(), or env.py).
See dynamics_scorer_worker.py's module docstring for why this is a real
subprocess rather than an in-process call.
"""
from __future__ import annotations

import subprocess
import sys
from typing import Dict, Optional

import numpy as np

from dynamics_scorer_worker import read_msg, write_msg

_WORKER_SCRIPT = "/project_data/held/vxiao/Bimanual_Manipulation_Project/dsrl-scan-object/dynamics_scorer_worker.py"


class DynamicsScorerClient:
    def __init__(self, dynamics_ckpt: str, device: str = "cuda", python: Optional[str] = None):
        self.proc = subprocess.Popen(
            [python or sys.executable, _WORKER_SCRIPT,
             "--dynamics_ckpt", dynamics_ckpt, "--device", device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,  # stderr inherited -> visible in the job log
            bufsize=0,
        )

    def score(self, dino_feat: np.ndarray, agent_pos: np.ndarray, patch_centers: np.ndarray,
             multiview: Dict[str, Dict[str, np.ndarray]], actions: np.ndarray,
             track_valid: np.ndarray) -> Dict[str, np.ndarray]:
        """
        actions: [N, H, action_dim] raw candidate actions.
        Returns {"rewards": [N], "logits": [N], "best_idx": int}, or raises
        RuntimeError (with the worker's traceback) on failure -- caller's
        responsibility to decide whether to fall back to actions[0] rather
        than let a scoring hiccup kill a live rollout.
        """
        write_msg(self.proc.stdin, {
            "dino_feat": dino_feat, "agent_pos": agent_pos, "patch_centers": patch_centers,
            "multiview": multiview, "actions": actions, "track_valid": track_valid,
        })
        resp = read_msg(self.proc.stdout)
        if resp is None:
            raise RuntimeError("dynamics_scorer_worker died (EOF on stdout) -- check its stderr in the job log")
        if "error" in resp:
            raise RuntimeError(f"dynamics_scorer_worker scoring failed:\n{resp['error']}")
        return resp

    def close(self):
        """Best-effort teardown -- a slow/stuck worker here must never turn
        into a crash this late (confirmed real 2026-10-16: action_magnitude
        sweep jobs 4033944's `wait(timeout=10)` raised TimeoutExpired AFTER
        the actual eval had already completed and printed its result,
        turning a successful run into rc=1)."""
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)
