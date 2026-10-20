# Patches against upstream repos

## `sb3-sac-policies-squash_actions.patch`

Applies to `DSRL-reference/stable-baselines3` (the DSRL paper repo's own SB3
fork, `github.com/ajwagen/stable-baselines3-dsrl`). That is not our repo, so
the change cannot be pushed upstream; it is kept here instead.

    cd DSRL-reference/stable-baselines3
    git apply ../../dsrl-scan-object/patches/sb3-sac-policies-squash_actions.patch

Two things in one file (`stable_baselines3/sac/policies.py`, +40/-9):

1. **`squash_actions` option** (default `True`, so nothing existing changes).
   `False` swaps `SquashedDiagGaussianDistribution` for the unbounded
   `DiagGaussianDistribution`. Needed because DSRL's action IS the noise fed
   to a diffusion model's reverse process: tanh makes `|a| > action_magnitude`
   unreachable, so a squashed actor cannot emit the 13.3% of N(0,1) mass
   beyond +-1.5 (measured: matching std 0.98 vs 1.00, but kurtosis 1.61 vs
   3.01 and `P(|a|>1.5)` exactly 0 vs 0.133). Requires
   `action_magnitude=1.0` -- see the comment at the `squash_output=True` call.

2. **A pre-existing latent bug fix**: `Actor._get_constructor_parameters`
   omitted `standard_gauss_init`, which changes forward-pass behaviour. Without
   it `SAC.load()` rebuilds a *different* actor and loads the weights into it,
   so every eval and every resume was silently wrong. Both flags are now in
   the constructor params; round-trip verified.
