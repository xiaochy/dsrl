"""
Locates the nondeterminism that invalidated the outcome-sensitivity numbers.

tests/measure_outcome_sensitivity.py --fixed_noise replays one scene K times
with an IDENTICAL action-noise stream and fixed_track_noise=True, so every
outcome should agree. Measured 2026-10-19: flip rate 0.10 at
num_inference_steps=10 and 0.33 at 1 -- i.e. the pipeline is NOT a
deterministic function of (scene, action noise), and the flip rates measured
with VARYING noise therefore cannot be attributed to the noise.

The track RNG is already ruled out (all four runs had fixed_track_noise=True).
This separates the two remaining candidates by comparing the ACTION CHUNKS
rather than the outcomes:

    chunks bit-identical, outcomes differ  -> SAPIEN physics / contact
                                              dynamics. The env is chaotic and
                                              flip counting cannot work at
                                              this sample size, whatever the
                                              policy does.
    chunks differ                          -> policy-side float
                                              nondeterminism (cuDNN/TF32
                                              kernel selection, DINOv2 or the
                                              denoiser), amplified downstream.
                                              Report the first diverging step
                                              and the magnitude.

Both replays run in the SAME process on the same device, so this is not
testing cross-process reproducibility -- only whether one process can
reproduce itself.

Usage (from ROBOTWIN_ROOT, RoboTwin conda env):
    python3 tests/diagnose_nondeterminism.py --gap_ckpt <path> \
        --scene 100200009 --num_inference_steps 10
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_path)

from env import DSRLScanObjectEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap_ckpt", required=True)
    ap.add_argument("--task_name", default="scan_object")
    ap.add_argument("--task_config", default="demo_clean_depth")
    ap.add_argument("--scene", type=int, required=True)
    ap.add_argument("--env_seed", type=int, default=1001)
    ap.add_argument("--n_repeat", type=int, default=2)
    ap.add_argument("--step_lim", type=int, default=500)
    ap.add_argument("--num_inference_steps", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    env = DSRLScanObjectEnv(
        task_name=args.task_name, task_config=args.task_config,
        gap_ckpt=args.gap_ckpt, dynamics_ckpt="", device=args.device,
        seed=args.env_seed, step_lim=args.step_lim, reward_source="none",
        num_inference_steps=args.num_inference_steps,
    )
    noise_dim = env.base_policy.horizon * env.base_policy.action_dim

    runs = []
    for r in range(args.n_repeat):
        env.reset(seed=args.scene)
        # IDENTICAL stream every repeat -- the whole point.
        rng = np.random.default_rng([args.scene, 0])
        chunks, obs_hashes = [], []
        done = success = False
        while not done:
            noise = rng.standard_normal(noise_dim).astype(np.float32)
            # Reproduce step()'s own encode so the chunk can be captured
            # without changing env.step's behaviour.
            import torch
            o = env._encode(env._last_observation)
            obs_hashes.append(hashlib.sha1(
                np.ascontiguousarray(o["state"]).tobytes()).hexdigest()[:12])
            obs_t = {
                "agent_pos": torch.as_tensor(o["state"], dtype=torch.float32).unsqueeze(0),
                "dino_features": torch.as_tensor(o["dino_features"],
                                                 dtype=torch.float32).unsqueeze(0),
            }
            if o.get("patch_centers") is not None:
                obs_t["patch_centers"] = torch.as_tensor(o["patch_centers"],
                                                         dtype=torch.float32).unsqueeze(0)
            chunk = env.base_policy(obs_t, noise[None])[0]
            chunks.append(np.asarray(chunk, dtype=np.float64))

            _, _, done, info = env.step(noise)
            success = success or bool(info.get("success", False))
        runs.append({"success": success, "chunks": chunks,
                     "obs_hashes": obs_hashes, "n_steps": len(chunks)})
        print(f"[diag] repeat {r}: success={success} steps={len(chunks)}", flush=True)
    env.close()

    a, b = runs[0], runs[1]
    print(f"\n=== scene {args.scene}, steps={env.num_inference_steps}, "
          f"identical noise, {args.n_repeat} repeats ===")
    print(f"  outcomes      : {[r['success'] for r in runs]}  "
          f"{'DIFFER' if len({r['success'] for r in runs}) > 1 else 'agree'}")
    print(f"  episode length: {[r['n_steps'] for r in runs]}")

    n = min(a["n_steps"], b["n_steps"])
    first_obs_div = next((i for i in range(n)
                          if a["obs_hashes"][i] != b["obs_hashes"][i]), None)
    first_act_div = next((i for i in range(n)
                          if not np.array_equal(a["chunks"][i], b["chunks"][i])), None)
    print(f"  first step whose OBSERVATION differs : {first_obs_div}")
    print(f"  first step whose ACTION CHUNK differs: {first_act_div}")
    if first_act_div is not None:
        d = np.abs(a["chunks"][first_act_div] - b["chunks"][first_act_div])
        print(f"    |delta| at that step: max {d.max():.3e}  mean {d.mean():.3e}")
    if first_obs_div == 0:
        print("  >> the very first observation already differs: reset(seed=) is "
              "not restoring an identical scene")
    elif first_act_div == 0:
        print("  >> identical first observation but different first action: "
              "POLICY-side float nondeterminism (cuDNN/TF32 kernel choice)")
    elif first_act_div is None and first_obs_div is not None:
        print("  >> every action chunk is bit-identical yet the observations "
              "diverge: SAPIEN physics nondeterminism, amplified by contact")

    out = args.out or os.path.join(base_path, "results",
                                   f"nondeterminism_{args.scene}_steps"
                                   f"{args.num_inference_steps or 'default'}.json")
    with open(out, "w") as f:
        json.dump({"scene": args.scene,
                   "num_inference_steps": env.num_inference_steps,
                   "outcomes": [r["success"] for r in runs],
                   "n_steps": [r["n_steps"] for r in runs],
                   "first_obs_divergence": first_obs_div,
                   "first_action_divergence": first_act_div}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
