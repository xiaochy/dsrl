"""
Final deterministic evaluation of a trained DSRL SAC checkpoint: runs the
actor's MEAN action (no exploration sampling), same convention as
DeterministicEvalCallback used during training, but with enough episodes
(50/seed x 3 seeds = 150) to actually trust the number -- the n=3-episodes-
per-point running estimate logged during training was only ever meant to
catch gross wiring problems or a completely flat/declining curve, not to
be read as a final success rate (too noisy: only 0/33/67/100% possible).

Seeds default to a range (5000+) far outside anything touched during
training (training rollout: cfg.seed=1 -> now_seed from 200001; training's
own eval_env: cfg.seed+1000=1001 -> now_seed from ~100,100,001, incremented
by a few hundred episodes at most across all training runs so far) or any
other scan_object eval run in this project -- no collision risk.

Usage:
    python dsrl_final_eval.py --ckpt <path/to/checkpoint/final.zip> \
        --gap_ckpt <path> --dynamics_ckpt <path> --reward_source track \
        --seed 5000 --n_episodes 50
"""
import argparse
import os
import sys

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, "..", "DSRL-reference", "stable-baselines3"))

from stable_baselines3 import SAC  # noqa: E402

from gym_env import DSRLScanObjectGymEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gap_ckpt", required=True)
    ap.add_argument("--dynamics_ckpt", default="")
    ap.add_argument("--reward_source", default="track", choices=["track", "action"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--step_lim", type=int, default=500)
    ap.add_argument("--action_magnitude", type=float, default=1.5)
    ap.add_argument("--expert_check", action="store_true",
                    help="replicate RoboTwin's scripted-expert seed filter, so "
                         "the result is P(success | expert solves) -- the same "
                         "quantity GAP's published numbers report. Without it "
                         "this is an unconditioned rate over the raw seed "
                         "stream and must NOT be compared to them: the expert "
                         "solves only ~28%% of raw scan_object seeds (measured "
                         "2026-10-19, see env.py's module docstring).")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"Loading SAC checkpoint from {args.ckpt}")
    model = SAC.load(args.ckpt, device=args.device)

    env = DSRLScanObjectGymEnv(
        task_name="scan_object", task_config="demo_clean_depth",
        gap_ckpt=args.gap_ckpt, dynamics_ckpt=args.dynamics_ckpt, device=args.device,
        seed=args.seed, step_lim=args.step_lim, action_magnitude=args.action_magnitude,
        reward_source=args.reward_source, expert_check=args.expert_check,
    )

    successes = []
    for ep in range(args.n_episodes):
        obs, _ = env.reset()
        done = False
        ep_success = False
        ep_rew = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_rew += float(reward)
            ep_success = ep_success or bool(info.get("success", False))
        successes.append(ep_success)
        print(f"[dsrl_final_eval] seed={args.seed} episode={ep} success={ep_success} ep_reward={ep_rew:.4f}")

    env.close()
    n_succ = sum(successes)
    print(f"\n=== seed={args.seed}: {n_succ}/{len(successes)} = {n_succ / len(successes):.4f} "
          f"(expert_check={args.expert_check}) ===")


if __name__ == "__main__":
    main()
