"""Level-1 diagnostic: reconstructs the SAC entropy tuner's state from a run's
stdout log and says whether the tuner is at equilibrium or collapsing.

SB3 logs ent_coef (alpha) and ent_coef_loss but not log_prob. Since
(sac.py:240)

    ent_coef_loss = -(log_ent_coef * (log_prob + target_entropy))

log_prob can be recovered as

    log_prob = -ent_coef_loss / log(alpha) - target_entropy

The tuner's fixed point is (log_prob + target_entropy) == 0, i.e. the policy's
entropy H = -log_prob equals target_entropy. With target_entropy=0 in this
project's 256-dim noise space the term sits at about -174 from the first update
on, so alpha is driven down monotonically and never reaches equilibrium -- that
is the collapse diagnosed in daily_logs/2026-10-13.md and re-confirmed in
2026-10-15.md.

Numerical caveat: when alpha ~ 1 both the numerator and log(alpha) go to zero
together. The ratio stays well conditioned as long as the logged ent_coef_loss
keeps significant digits, but once it rounds to 0 the reconstruction is 0/0 and
is reported as indeterminate -- which itself means "sitting at equilibrium".

Usage:
    python3 tests/analyze_ent_coef.py outputs/targetent_diag2.out --target 174.5
    python3 tests/analyze_ent_coef.py outputs/dsrl_train_track_fixed_3k-4031196.out
"""
from __future__ import annotations

import argparse
import math
import re


def _series(path: str):
    with open(path, errors="ignore") as f:
        txt = f.read()
    alpha = [float(x) for x in re.findall(r"ent_coef\s*\|\s*([-\d.e+]+)", txt)]
    loss = [float(x) for x in re.findall(r"ent_coef_loss\s*\|\s*([-\d.e+]+)", txt)]
    steps = [int(x) for x in re.findall(r"total_timesteps\s*\|\s*([\d]+)", txt)]
    n = min(len(alpha), len(loss))
    steps = steps[:n] if len(steps) >= n else list(range(n))
    return steps, alpha[:n], loss[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    ap.add_argument("--target", type=float, default=0.0,
                    help="target_entropy the run used (0.0 for all runs before "
                         "the 2026-10-18 fix)")
    args = ap.parse_args()

    steps, alpha, loss = _series(args.logfile)
    if not alpha:
        print("no ent_coef rows found -- training may not have started yet")
        return

    print(f"\n{args.logfile}   target_entropy={args.target}\n")
    print(f"{'step':>8}{'alpha':>11}{'loss':>12}{'log_prob':>11}{'balance':>10}")
    print("-" * 52)

    balances = []
    for s, a, l in zip(steps, alpha, loss):
        la = math.log(a) if a > 0 else float("nan")
        if not math.isfinite(la) or abs(la) < 1e-9 or l == 0.0:
            print(f"{s:>8}{a:>11.4g}{l:>12.4g}{'n/a':>11}{'~0':>10}")
            continue
        lp = -l / la - args.target
        bal = lp + args.target
        balances.append(bal)
        print(f"{s:>8}{a:>11.4g}{l:>12.4g}{lp:>11.1f}{bal:>+10.2f}")

    a0, aN = alpha[0], alpha[-1]
    print(f"\nalpha: {a0:.4g} -> {aN:.4g}  ({len(alpha)} points)")
    if balances:
        print(f"balance (log_prob + target): mean {sum(balances)/len(balances):+.2f}, "
              f"last {balances[-1]:+.2f}")

    # Verdict. The thresholds are read off the known-good and known-bad runs
    # rather than chosen a priori: the failed runs fell below 0.06 while the
    # balance term sat near -174.
    print()
    if aN < 0.06 and a0 / max(aN, 1e-9) > 5:
        print("VERDICT: collapsing -- alpha has fallen by >5x into the range the "
              "2026-10-13/15 failures reached. The entropy term has lost its "
              "authority over the actor.")
    elif aN > 10:
        print("VERDICT: alpha running away upward -- the entropy term is fighting "
              "the critic with escalating force; the policy risks becoming "
              "noise-dominated and ignoring reward.")
    elif balances and abs(sum(balances) / len(balances)) < 5:
        print("VERDICT: at equilibrium -- the balance term is near zero, so the "
              "tuner is holding entropy at the target instead of crushing alpha.")
    else:
        print("VERDICT: alpha is stable but the balance term is not near zero; "
              "check whether target_entropy matches this policy's actual entropy.")


if __name__ == "__main__":
    main()
