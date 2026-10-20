"""
Success rate of the UNSTEERED frozen GAP policy, measured through the DSRL
env, on a named seed batch, with PER-SEED outcomes recorded.

WHY THIS EXISTS: the project's two headline numbers were never comparable.

  * 75.33% is P(success | the scripted expert solves the seed). RoboTwin's
    protocol hardcodes expert_check=True (GAP scripts/eval_policy.py:254)
    and silently skips every seed the expert fails, so those seeds never
    reach the policy. Measured from three completed 50-episode
    scan_object/demo_clean_depth evals' own stored seed lists, the expert
    solves only ~28% of raw seeds.
  * 0.400 (the step-0 DSRL eval that four different mechanism states all
    landed on) is UNCONDITIONED -- env.py used every seed as-is.

Mixing the two manufactured a "35pp gap" that three separate hypotheses
were then spent chasing (tanh squashing, RGB channel order, tensor shape --
all falsified, see daily_logs/2026-10-19.md).

This script closes the comparison from the other side: it records which
seed each episode ran on, so the SAME rollouts yield both rates --
unconditioned, and conditioned on the expert mask (from
tests/measure_expert_pass_rate.py, or from env.py's own expert_check
cache). reward_source="none" keeps it cheap: no dynamics model, no SVM, no
scorer subprocess, and nothing in step() touching _collect_multiview_obs.

The noise fed in is an exact np.random.randn draw -- DSRLScanObjectEnv.step
passes its argument straight to the base policy with no scaling, so this
is the diffusion policy sampling its own DDIM init noise, i.e. the real
"no steering" reference DSRL has to beat.

Usage (from ROBOTWIN_ROOT, RoboTwin conda env):
    python3 tests/eval_unsteered_baseline.py --env_seed 1001 --n 30 \
        --gap_ckpt <path>
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
    # env.py derives its episode seeds as 100000*(1+env_seed), the same
    # formula eval_policy.py uses for st_seed -- so env_seed=1001 is the
    # batch every DSRL eval ran on (cfg.seed=1 -> eval_seed=cfg.seed+1000).
    ap.add_argument("--env_seed", type=int, default=1001)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--step_lim", type=int, default=500)
    ap.add_argument("--expert_check", action="store_true",
                    help="skip expert-unsolvable seeds, matching RoboTwin's "
                         "own protocol (gives a number directly comparable "
                         "to the 75.33% baseline)")
    ap.add_argument("--reward_source", default="none",
                    choices=["none", "track", "action"],
                    help="\"none\" (default) is the cheap path and also removes "
                         "every _collect_multiview_obs / _verify_action_track_setup "
                         "call from step(). Pass \"track\" to re-run the SAME "
                         "rollouts with the reward path present and nothing else "
                         "changed -- that isolates whether the reward path "
                         "perturbs the live sim, which the historical 0.400 "
                         "numbers (all reward_source=track) could not separate "
                         "from their other difference, an SAC actor instead of "
                         "np.random for the noise.")
    ap.add_argument("--dynamics_ckpt", default="",
                    help="required by --reward_source track")
    ap.add_argument("--num_inference_steps", type=int, default=0,
                    help="override the policy's DDIM step count (0 = leave the "
                         "deployed value, 10). Fewer steps give the initial "
                         "noise x_T more authority under "
                         "prediction_type='sample' (see "
                         "tests/sweep_inference_steps_authority.py) but also "
                         "change the policy, so success rate has to be "
                         "re-measured here before anything is built on it.")
    ap.add_argument("--noise_seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rng = np.random.default_rng(args.noise_seed)
    env = DSRLScanObjectEnv(
        task_name=args.task_name, task_config=args.task_config,
        gap_ckpt=args.gap_ckpt, dynamics_ckpt=args.dynamics_ckpt,
        device=args.device, seed=args.env_seed, step_lim=args.step_lim,
        reward_source=args.reward_source, expert_check=args.expert_check,
        num_inference_steps=args.num_inference_steps,
    )
    noise_dim = env.base_policy.horizon * env.base_policy.action_dim

    records = []
    t0 = time.time()
    for ep in range(args.n):
        env.reset()
        seed_used = env._now_seed - 1  # reset() already advanced it
        done, success, steps = False, False, 0
        while not done:
            noise = rng.standard_normal(noise_dim).astype(np.float32)
            _, _, done, info = env.step(noise)
            steps += 1
            success = success or bool(info.get("success", False))
        records.append({"episode": ep, "seed": seed_used,
                        "success": success, "chunk_steps": steps})
        n_s = sum(r["success"] for r in records)
        print(f"[unsteered] ep={ep} seed={seed_used} success={success} steps={steps} "
              f"running={n_s}/{len(records)}={n_s/len(records):.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    env.close()
    n_succ = sum(r["success"] for r in records)
    print(f"\n=== unsteered GAP, env_seed={args.env_seed}, n={args.n}, "
          f"expert_check={args.expert_check}: "
          f"{n_succ}/{len(records)} = {n_succ / len(records):.4f} ===")
    if args.expert_check:
        print(f"=== comparable to the 75.33% baseline (both conditioned on "
              f"expert-solvable); {len(env.skipped_seeds)} seeds skipped ===")

    out = args.out or os.path.join(
        base_path, "results",
        f"unsteered_{args.env_seed}_{args.n}_expert{int(args.expert_check)}"
        f"_steps{args.num_inference_steps or 'default'}"
        f"_rew{args.reward_source}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"env_seed": args.env_seed, "n": args.n,
                   "num_inference_steps": env.num_inference_steps,
                   "reward_source": args.reward_source,
                   "expert_check": args.expert_check,
                   "task_config": args.task_config,
                   "n_success": n_succ, "success_rate": n_succ / len(records),
                   "skipped_seeds": env.skipped_seeds,
                   "episodes": records}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
