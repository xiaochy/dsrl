"""
Does the DDIM initial noise change the OUTCOME, not just the action?

This is the quantity DSRL actually needs and the one nothing here has
measured yet. `tests/compare_policy_paths.py` and
`tests/sweep_inference_steps_authority.py` measure how far the noise moves
the action chunk (authority 0.0035 at the deployed 10 steps, 0.046 at 1).
But an RL agent over the noise can only learn if different noise leads to
different REWARD, and reward here is terminal task success. If every noise
draw on a given scene produces the same outcome, the return is a constant
function of the action and there is no gradient to climb, whatever the
authority ratio says.

Method: replay each fixed scene K times -- identical seed, identical
SAPIEN initial state -- with a different noise stream each time, and count
how often the K outcomes disagree. No reward model is involved, which
matters because the action-SVM's own documented limitation is that it
"barely reads its action input" (candidate-to-candidate logit spread ~0.004
vs ~3 across states, daily_logs/2026-10-13.md); that is a property of the
reward model and must not be used to argue anything about the policy.

Reading the result:
    flip rate ~0        the noise cannot change the outcome. DSRL cannot
                        work on this checkpoint at this step count, and
                        neither can best-of-N, no matter how good the
                        reward model is.
    flip rate >> 0      there is real headroom, and the per-scene success
                        spread upper-bounds what perfect noise selection
                        could buy: an oracle picks the best of K, so
                        mean(any success in K) - mean(single-draw success)
                        is the gain available to a perfect steerer.

Usage (from ROBOTWIN_ROOT, RoboTwin conda env):
    python3 tests/measure_outcome_sensitivity.py --gap_ckpt <path> \
        --n_scenes 15 --n_noise 3 --num_inference_steps 1
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
    ap.add_argument("--env_seed", type=int, default=1001)
    ap.add_argument("--n_scenes", type=int, default=15)
    ap.add_argument("--n_noise", type=int, default=3,
                    help="noise draws per scene (K)")
    ap.add_argument("--step_lim", type=int, default=500)
    ap.add_argument("--num_inference_steps", type=int, default=0)
    ap.add_argument("--expert_check", action="store_true")
    ap.add_argument("--fixed_noise", action="store_true",
                    help="DETERMINISM CONTROL: give every draw the SAME action "
                         "noise instead of a different one. Any disagreement "
                         "is then pipeline nondeterminism, not noise "
                         "sensitivity, and the flip rates measured without "
                         "this flag cannot be attributed to the noise. This "
                         "control was missing from the first run of this "
                         "script (2026-10-19) and a recorded rollout "
                         "disagreeing with its result is what exposed the "
                         "gap -- the joint-diffusion track branch was drawing "
                         "from the global torch RNG, see "
                         "gap_base_policy.GAPBasePolicyWrapper.track_seed.")
    ap.add_argument("--free_track_noise", action="store_true",
                    help="reproduce the old uncontrolled behaviour (track "
                         "noise from the global torch RNG) for comparison")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    env = DSRLScanObjectEnv(
        task_name=args.task_name, task_config=args.task_config,
        gap_ckpt=args.gap_ckpt, dynamics_ckpt="", device=args.device,
        seed=args.env_seed, step_lim=args.step_lim, reward_source="none",
        expert_check=args.expert_check,
        num_inference_steps=args.num_inference_steps,
        fixed_track_noise=not args.free_track_noise,
    )
    noise_dim = env.base_policy.horizon * env.base_policy.action_dim

    # Resolve the scene seeds FIRST, under whatever filter is in force, so
    # every noise replay lands on the same scene list.
    scene_seeds = []
    for _ in range(args.n_scenes):
        env.reset()
        scene_seeds.append(env._now_seed - 1)
    print(f"[scenes] {len(scene_seeds)} seeds: {scene_seeds}", flush=True)

    rows, t0 = [], time.time()
    for si, scene in enumerate(scene_seeds):
        outcomes = []
        for k in range(args.n_noise):
            env.reset(seed=scene)
            # Distinct, reproducible noise stream per (scene, k).
            # --fixed_noise pins the draw index to 0 so every repeat gets an
            # identical action-noise stream (see the flag's help).
            rng = np.random.default_rng([scene, 0 if args.fixed_noise else k])
            done = success = False
            steps = 0
            while not done:
                _, _, done, info = env.step(
                    rng.standard_normal(noise_dim).astype(np.float32))
                steps += 1
                success = success or bool(info.get("success", False))
            outcomes.append(bool(success))
            print(f"[sens] scene {si + 1}/{len(scene_seeds)} seed={scene} "
                  f"draw {k + 1}/{args.n_noise} success={success} steps={steps} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        rows.append({"seed": scene, "outcomes": outcomes,
                     "n_success": sum(outcomes), "flipped": 0 < sum(outcomes) < len(outcomes)})
    env.close()

    K = args.n_noise
    n = len(rows)
    flipped = sum(r["flipped"] for r in rows)
    single = sum(r["n_success"] for r in rows) / (n * K)   # a single random draw
    oracle = sum(r["n_success"] > 0 for r in rows) / n     # best of K
    worst = sum(r["n_success"] == K for r in rows) / n     # worst of K

    mode = ("DETERMINISM CONTROL (identical noise every draw)"
            if args.fixed_noise else "varying noise")
    print(f"\n=== outcome sensitivity: steps={env.num_inference_steps}, "
          f"{n} scenes x {K} draws, expert_check={args.expert_check}, "
          f"track_noise={'free/global-RNG' if args.free_track_noise else 'fixed per scene'} "
          f"-- {mode} ===")
    if args.fixed_noise:
        print("  >> with identical noise, ANY disagreement is pipeline "
              "nondeterminism; a nonzero rate here invalidates attributing "
              "the varying-noise flip rate to the noise")
    print(f"  scenes whose {K} outcomes disagree : {flipped}/{n} = {flipped / n:.4f}")
    print(f"  single random draw                 : {single:.4f}")
    print(f"  oracle best-of-{K} (upper bound)    : {oracle:.4f}")
    print(f"  worst-of-{K} (lower bound)          : {worst:.4f}")
    print(f"  headroom a perfect steerer could buy: "
          f"{oracle - single:+.4f}")
    print("  >> flip rate ~0 means the outcome does not depend on the noise, "
          "so there is nothing for DSRL or best-of-N to select over")

    out = args.out or os.path.join(
        base_path, "results",
        f"outcome_sensitivity_steps{args.num_inference_steps or 'default'}"
        f"_expert{int(args.expert_check)}_{n}x{K}"
        f"{'_FIXEDNOISE' if args.fixed_noise else ''}"
        f"{'_freetrack' if args.free_track_noise else ''}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"num_inference_steps": env.num_inference_steps,
                   "fixed_noise_control": args.fixed_noise,
                   "fixed_track_noise": not args.free_track_noise,
                   "env_seed": args.env_seed, "n_scenes": n, "n_noise": K,
                   "expert_check": args.expert_check,
                   "flip_rate": flipped / n, "single_draw": single,
                   "oracle_best_of_k": oracle, "worst_of_k": worst,
                   "headroom": oracle - single, "scenes": rows}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
