"""
Records head-camera video of specific rollouts, replaying the EXACT noise
streams tests/measure_outcome_sensitivity.py used, so the recorded outcomes
reproduce the measured ones.

Point of this: at num_inference_steps=1 the noise flips the outcome on 9 of
15 scenes (daily_logs/2026-10-19.md), at the deployed 10 steps on 0 of 15.
The numbers say the noise matters at 1 step; they do not say WHAT it changes
behaviourally -- grasp pose, timing, which object is approached, or whether
the arm simply drifts. That needs looking at the rollouts.

Noise reproducibility: np.random.default_rng([scene, k]), identical to the
sensitivity script, so draw k of scene S here is the same draw k of scene S
there.

Each file is renamed after the episode to carry its outcome, so a directory
listing alone shows the flip.

Usage (from ROBOTWIN_ROOT, RoboTwin conda env):
    python3 tests/record_rollout_videos.py --gap_ckpt <path> \
        --scenes 100200009 100200025 100200034 --n_noise 3 \
        --num_inference_steps 1 --video_dir <dir>
"""
import argparse
import json
import os
import sys
import time

import numpy as np

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_path)

from env import DSRLScanObjectEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap_ckpt", required=True)
    ap.add_argument("--task_name", default="scan_object")
    ap.add_argument("--task_config", default="demo_clean_depth")
    ap.add_argument("--scenes", type=int, nargs="+", required=True,
                    help="episode seeds to replay (e.g. the flipped ones)")
    ap.add_argument("--n_noise", type=int, default=3)
    ap.add_argument("--env_seed", type=int, default=1001)
    ap.add_argument("--step_lim", type=int, default=500)
    ap.add_argument("--num_inference_steps", type=int, default=0)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--video_substep_freq", type=int, default=0,
                    help="write an extra frame every N physics substeps. 0 "
                         "keeps RoboTwin's own granularity, which is one frame "
                         "per take_action taken BEFORE that action runs -- ~16 "
                         "frames per chunk, each a widely separated waypoint, "
                         "so the arm appears to teleport (measured 15.2-16.1 "
                         "frames/chunk on the first recordings). Use ~5 with "
                         "--video_fps 30 for smooth motion.")
    ap.add_argument("--video_fps", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    env = DSRLScanObjectEnv(
        task_name=args.task_name, task_config=args.task_config,
        gap_ckpt=args.gap_ckpt, dynamics_ckpt="", device=args.device,
        seed=args.env_seed, step_lim=args.step_lim, reward_source="none",
        num_inference_steps=args.num_inference_steps,
        video_dir=args.video_dir,
        video_substep_freq=args.video_substep_freq,
        video_fps=args.video_fps,
    )
    steps_tag = env.num_inference_steps
    noise_dim = env.base_policy.horizon * env.base_policy.action_dim

    records, t0 = [], time.time()
    for scene in args.scenes:
        for k in range(args.n_noise):
            label = f"steps{steps_tag}_seed{scene}_draw{k}"
            env.reset(seed=scene, video_label=label)
            rng = np.random.default_rng([scene, k])  # same stream as the measurement
            done = success = False
            steps = 0
            while not done:
                _, _, done, info = env.step(
                    rng.standard_normal(noise_dim).astype(np.float32))
                steps += 1
                success = success or bool(info.get("success", False))
            env._close_video()  # flush ffmpeg before touching the file

            src = os.path.join(args.video_dir, f"{label}.mp4")
            dst = os.path.join(args.video_dir,
                               f"{label}_{'SUCCESS' if success else 'FAIL'}.mp4")
            if os.path.exists(src):
                os.replace(src, dst)
            records.append({"seed": scene, "draw": k, "success": success,
                            "chunk_steps": steps, "video": os.path.basename(dst)})
            print(f"[record] seed={scene} draw={k} success={success} "
                  f"steps={steps} -> {os.path.basename(dst)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    env.close()

    print(f"\n=== recorded {len(records)} rollouts at "
          f"num_inference_steps={steps_tag} into {args.video_dir} ===")
    for scene in args.scenes:
        outs = [r["success"] for r in records if r["seed"] == scene]
        tag = "FLIPPED" if 0 < sum(outs) < len(outs) else "unanimous"
        print(f"  {scene}: {[int(x) for x in outs]}  {tag}")

    with open(os.path.join(args.video_dir,
                           f"_index_steps{steps_tag}.json"), "w") as f:
        json.dump({"num_inference_steps": steps_tag, "rollouts": records}, f, indent=2)


if __name__ == "__main__":
    main()
