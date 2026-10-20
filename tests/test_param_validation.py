"""
Exercises validate_noise_parameterization() for every prediction_type /
noise-setting combination -- including the epsilon paths, which cannot be
reached with the checkpoint on disk (it is prediction_type='sample').

No GPU, no checkpoint, no SAPIEN: the function takes cfg and the
prediction_type string, so every branch is reachable directly. That is the
reason it lives at module level in train_dsrl_scan_object.py rather than
nested inside main().

Must run from ROBOTWIN_ROOT like everything else here: importing
train_dsrl_scan_object pulls in RoboTwin's `envs` package, which loads assets
via paths relative to cwd.

Usage:
    cd $ROBOTWIN_ROOT && python3 <this repo>/tests/test_param_validation.py
"""
import os
import sys
import math

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "..", "DSRL-reference", "stable-baselines3"))

from omegaconf import OmegaConf  # noqa: E402

from train_dsrl_scan_object import validate_noise_parameterization  # noqa: E402

GAUSS_ENT = 256 * 0.5 * math.log(2 * math.pi * math.e)  # 363.25


def cfg_of(squash, gauss_init, magnitude, target_ent):
    return OmegaConf.create({
        "act_steps": 16, "action_dim": 16,
        "train": {"squash_actions": squash, "standard_gauss_init": gauss_init,
                  "action_magnitude": magnitude, "target_ent": target_ent},
    })


CASES = [
    # (name, prediction_type, cfg, expect_raise)
    ("epsilon + squashed  -> must RAISE (discards epsilon's whole point)",
     "epsilon", cfg_of(True, False, 1.5, 0.0), True),
    ("epsilon + unsquashed, no gauss_init -> must RAISE",
     "epsilon", cfg_of(False, False, 1.0, GAUSS_ENT), True),
    ("epsilon + the documented preset -> must PASS",
     "epsilon", cfg_of(False, True, 1.0, GAUSS_ENT), False),
    ("epsilon + preset but wrong target_ent -> PASS with a warning",
     "epsilon", cfg_of(False, True, 1.0, 0.0), False),
    ("sample (this project's checkpoint) -> PASS with the authority note",
     "sample", cfg_of(True, False, 1.5, 0.0), False),
    ("v_prediction -> PASS with an unrecognized warning",
     "v_prediction", cfg_of(True, False, 1.5, 0.0), False),
]


def main():
    failures = []
    for name, ptype, cfg, expect_raise in CASES:
        print(f"\n--- {name}")
        try:
            validate_noise_parameterization(cfg, ptype)
            raised = False
        except ValueError as e:
            raised = True
            print(f"    ValueError: {str(e)[:110]}...")
        verdict = "OK" if raised == expect_raise else "FAILED"
        if raised != expect_raise:
            failures.append(name)
        print(f"    raised={raised} expected={expect_raise}  -> {verdict}")

    # the override must downgrade the hard error to a warning
    print("\n--- DSRL_ALLOW_PARAM_MISMATCH=1 must downgrade the error")
    os.environ["DSRL_ALLOW_PARAM_MISMATCH"] = "1"
    try:
        validate_noise_parameterization("epsilon" and cfg_of(True, False, 1.5, 0.0),
                                        "epsilon")
        print("    raised=False expected=False  -> OK")
    except ValueError:
        failures.append("override did not downgrade")
        print("    raised=True expected=False  -> FAILED")
    finally:
        del os.environ["DSRL_ALLOW_PARAM_MISMATCH"]

    print(f"\n=== {len(CASES) + 1 - len(failures)}/{len(CASES) + 1} passed ===")
    for f in failures:
        print(f"  FAILED: {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
