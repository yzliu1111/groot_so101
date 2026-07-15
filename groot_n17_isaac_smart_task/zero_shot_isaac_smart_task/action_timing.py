"""Pure helpers for aligning policy actions with Isaac environment steps."""

from __future__ import annotations

import math


def resolve_env_steps_per_policy_action(
    policy_action_hz: float,
    env_step_dt_s: float,
    *,
    ratio_tolerance: float = 1e-4,
) -> tuple[int, float]:
    """Return zero-order-hold steps and the resulting effective policy rate.

    The runner currently supports an integer number of Isaac environment steps
    per policy action.  Failing on a non-integer ratio is deliberate: silently
    rounding it would make an action chunk run at a different speed from the
    dataset/model time base.
    """

    policy_hz = float(policy_action_hz)
    env_dt = float(env_step_dt_s)
    tolerance = float(ratio_tolerance)
    if not math.isfinite(policy_hz) or policy_hz <= 0.0:
        raise ValueError(f"policy_action_hz must be finite and positive, got {policy_action_hz}")
    if not math.isfinite(env_dt) or env_dt <= 0.0:
        raise ValueError(f"env_step_dt_s must be finite and positive, got {env_step_dt_s}")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f"ratio_tolerance must be finite and non-negative, got {ratio_tolerance}")

    exact_steps = (1.0 / policy_hz) / env_dt
    env_steps = int(round(exact_steps))
    if env_steps < 1:
        raise ValueError(
            "policy action rate is faster than the Isaac environment step rate: "
            f"policy_action_hz={policy_hz:g} env_step_hz={1.0 / env_dt:g}"
        )
    if not math.isclose(exact_steps, float(env_steps), rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(
            "policy period is not an integer number of Isaac environment steps; "
            "choose a compatible --policy-action-hz or add an explicit resampler: "
            f"policy_action_hz={policy_hz:g} env_step_dt_s={env_dt:g} "
            f"exact_env_steps={exact_steps:.8g}"
        )

    effective_policy_hz = 1.0 / (env_steps * env_dt)
    return env_steps, effective_policy_hz
