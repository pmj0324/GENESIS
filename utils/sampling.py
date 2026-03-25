from __future__ import annotations

from numbers import Integral


def _coerce_int(value, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got bool")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise TypeError(f"{name} must be an integer, got {type(value).__name__}")


def validate_sampling_steps(
    steps,
    *,
    name: str = "steps",
    max_steps: int | None = None,
) -> int:
    """
    Validate step count used by samplers.

    Rules:
      - steps must be an integer
      - steps > 0
      - steps != 1  (single-step sampling is intentionally disabled)
      - if max_steps is provided, steps <= max_steps
    """
    steps = _coerce_int(steps, name=name)
    if steps <= 0:
        raise ValueError(f"{name} must be > 0, got {steps}")
    if steps == 1:
        raise ValueError(f"{name}=1 is not supported; use >= 2")
    if max_steps is not None:
        max_steps = _coerce_int(max_steps, name="max_steps")
        if steps > max_steps:
            raise ValueError(f"{name}={steps} exceeds max_steps={max_steps}")
    return steps
