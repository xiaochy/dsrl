"""
Measures RoboTwin's expert-check pass rate on a given seed batch.

WHY: RoboTwin/script/eval_policy.py:200 hardcodes `expert_check = True`,
and its while-loop (:218-251) only counts a seed toward `test_num` if the
SCRIPTED EXPERT solves it -- seeds where play_once() raises UnStableError,
raises anything else, or ends with `not (plan_success and check_success())`
are skipped with `now_seed += 1; continue` and never shown to the policy.
So the 75.33% plain-policy baseline is a success rate CONDITIONED on
expert-solvable scenes.

env.py's DSRLScanObjectEnv.reset() calls setup_demo() directly and skips
that filter entirely ("Every seed is used as-is", env.py:21), so every
DSRL eval number -- including the 0.400 that four different mechanism
states all landed on -- is an UNCONDITIONED rate over the raw seed stream.

If the expert pass rate on the DSRL eval batch is p, the directly
comparable baseline figure is ~0.7533 * p, not 0.7533. This script
measures p, and dumps the per-seed mask so a policy eval restricted to the
passing subset can be scored against the baseline on equal terms.

No GAP checkpoint, no DinoV2, no diffusion -- expert planning only.

Usage (from ROBOTWIN_ROOT, RoboTwin conda env):
    python3 tests/measure_expert_pass_rate.py --start_seed 100200000 --n 30
"""
import argparse
import json
import os
import sys
import time

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_path)

from env import ROBOTWIN_ROOT, _build_task_args, _class_decorator  # noqa: E402

sys.path.insert(0, ROBOTWIN_ROOT)
from envs.utils.create_actor import UnStableError  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_name", default="scan_object")
    ap.add_argument("--task_config", default="demo_clean_depth")
    ap.add_argument("--start_seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    task_args = _build_task_args(args.task_name, args.task_config)
    # eval_policy.py:221 forces render_freq=0 for the expert-check pass and
    # only restores it for the policy rollout; mirror that.
    task_args["render_freq"] = 0
    TASK_ENV = _class_decorator(args.task_name)

    records = []
    t0 = time.time()
    for i in range(args.n):
        seed = args.start_seed + i
        status, passed = "ok", False
        try:
            TASK_ENV.setup_demo(now_ep_num=i, seed=seed, is_test=True, **task_args)
            TASK_ENV.play_once()
            plan_ok = bool(TASK_ENV.plan_success)
            check_ok = bool(TASK_ENV.check_success())
            passed = plan_ok and check_ok
            status = "ok" if passed else f"expert_failed(plan={plan_ok},check={check_ok})"
            TASK_ENV.close_env()
        except UnStableError as e:
            status = f"UnStableError({e})"
            TASK_ENV.close_env()
        except Exception as e:  # noqa: BLE001 -- mirrors eval_policy.py's bare except
            status = f"{type(e).__name__}({e})"
            TASK_ENV.close_env()
        records.append({"seed": seed, "expert_pass": passed, "status": status})
        print(f"[expert_check] seed={seed} pass={passed} status={status} "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)

    n_pass = sum(r["expert_pass"] for r in records)
    p = n_pass / len(records)
    print(f"\n=== expert pass rate: {n_pass}/{len(records)} = {p:.4f} ===")
    print(f"=== baseline 0.7533 conditioned on these seeds would read as "
          f"0.7533 * {p:.4f} = {0.7533 * p:.4f} unconditioned ===")
    print(f"=== DSRL step-0 eval measured 0.4000 unconditioned; conditioned on "
          f"expert-solvable that is at most {0.4 / p if p else float('nan'):.4f} ===")

    out = args.out or os.path.join(base_path, "results",
                                   f"expert_pass_{args.start_seed}_{args.n}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"task_name": args.task_name, "task_config": args.task_config,
                   "start_seed": args.start_seed, "n": args.n,
                   "n_pass": n_pass, "pass_rate": p, "records": records}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
