"""
Flow Matching inference ODE solver utility.

Implements three solver modes with a shared interface:
  - euler  : fixed-step 1st order
  - heun   : fixed-step 2nd order (NFE = 2 * n_steps)
  - dopri5 : adaptive-step Dormand-Prince via torchdiffeq.odeint

Design notes:
  - Default integrates in t-direction 0 -> 1.
  - Also supports reversed integration (e.g. 1 -> 0) via t_start/t_end.
  - Keeps batch dimension and spatial dimensions unchanged.
  - Tracks and logs actual NFE (number of velocity-field evaluations).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Literal

import torch


SolverName = Literal["euler", "heun", "dopri5"]
VelocityFn = Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]


# Suggested presets requested by user:
# - validation during training: euler 50 steps
# - paper evaluation: heun 50 steps (=100 NFE)
# - GT reference: dopri5, rtol/atol=1e-5
VAL_SOLVER_DEFAULT = {"solver": "euler", "n_steps": 50}
PAPER_SOLVER_DEFAULT = {"solver": "heun", "n_steps": 50}
GT_SOLVER_DEFAULT = {"solver": "dopri5", "rtol": 1e-5, "atol": 1e-5}


@dataclass
class ODEInferenceStats:
    solver: str
    nfe: int
    n_steps: Optional[int] = None
    rtol: Optional[float] = None
    atol: Optional[float] = None


def select_solver(solver: str) -> SolverName:
    """Normalize and validate solver name."""
    name = str(solver).strip().lower()
    if name not in {"euler", "heun", "dopri5"}:
        raise ValueError(
            f"Unknown ODE solver: {solver!r}. Options: euler / heun / dopri5"
        )
    return name  # type: ignore[return-value]


class FlowMatchingODESolver:
    """Inference ODE solver wrapper for velocity fields v_theta(x, t, cond)."""

    def __init__(
        self,
        velocity_fn: VelocityFn,
        *,
        log_nfe: bool = True,
    ):
        self.velocity_fn = velocity_fn
        self.log_nfe = bool(log_nfe)
        self.last_stats: Optional[ODEInferenceStats] = None

    def _vf(
        self,
        x: torch.Tensor,
        t_scalar: float,
        cond: Optional[torch.Tensor],
        nfe_counter: dict,
    ) -> torch.Tensor:
        """Evaluate v_theta and increment NFE counter."""
        B = x.shape[0]
        t_b = torch.full((B,), float(t_scalar), device=x.device, dtype=torch.float32)
        nfe_counter["nfe"] += 1
        v = self.velocity_fn(x, t_b, cond)
        if v.shape != x.shape:
            raise ValueError(
                f"velocity_fn shape mismatch: expected {tuple(x.shape)}, got {tuple(v.shape)}"
            )
        return v

    @torch.no_grad()
    def sample(
        self,
        x0: torch.Tensor,
        cond: Optional[torch.Tensor],
        solver: str = "heun",
        n_steps: int = 50,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ) -> torch.Tensor:
        """
        Solve dx/dt = v_theta(x, t, cond) from t=t_start to t=t_end.

        Args:
            x0: Initial state at t=0, shape [B, C, ...].
            cond: Conditioning tensor (typically [B, cond_dim]) or None.
            solver: 'euler' | 'heun' | 'dopri5'.
            n_steps: Fixed step count for euler/heun.
            rtol/atol: Adaptive tolerance for dopri5.
            t_start/t_end: Integration interval endpoints.

        Returns:
            x1: Final state at t=1, same shape as x0.
        """
        mode = select_solver(solver)
        x = x0
        if cond is not None and cond.device != x.device:
            cond = cond.to(x.device)
        t0 = float(t_start)
        t1 = float(t_end)
        if t0 == t1:
            raise ValueError(f"t_start and t_end must differ, got {t0} == {t1}")

        nfe_counter = {"nfe": 0}

        if mode in {"euler", "heun"}:
            if int(n_steps) <= 0:
                raise ValueError(f"n_steps must be > 0 for {mode}, got {n_steps}")
            steps = int(n_steps)
            ts = torch.linspace(t0, t1, steps + 1, device=x.device, dtype=torch.float32)

            if mode == "euler":
                for i in range(steps):
                    t_cur = ts[i].item()
                    dt = (ts[i + 1] - ts[i]).item()
                    v = self._vf(x, t_cur, cond, nfe_counter)
                    x = x + dt * v
            else:
                for i in range(steps):
                    t_cur = ts[i].item()
                    t_next = ts[i + 1].item()
                    dt = t_next - t_cur
                    k1 = self._vf(x, t_cur, cond, nfe_counter)
                    x_pred = x + dt * k1
                    k2 = self._vf(x_pred, t_next, cond, nfe_counter)
                    x = x + 0.5 * dt * (k1 + k2)

            self.last_stats = ODEInferenceStats(
                solver=mode, nfe=nfe_counter["nfe"], n_steps=steps
            )
            if self.log_nfe:
                print(
                    f"[flow_ode] solver={mode}  t=[{t0:.3f}->{t1:.3f}]  "
                    f"n_steps={steps}  nfe={nfe_counter['nfe']}",
                    flush=True,
                )
            return x

        # dopri5 (adaptive) via torchdiffeq
        try:
            from torchdiffeq import odeint
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "dopri5 requires torchdiffeq. Install with: pip install torchdiffeq"
            ) from exc

        if float(rtol) <= 0 or float(atol) <= 0:
            raise ValueError(f"rtol/atol must be > 0, got rtol={rtol}, atol={atol}")

        def ode_rhs(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            t_val = float(t.detach().cpu().item())
            return self._vf(y, t_val, cond, nfe_counter)

        t_eval = torch.tensor([t0, t1], device=x.device, dtype=torch.float32)
        sol = odeint(ode_rhs, x, t_eval, method="dopri5", rtol=float(rtol), atol=float(atol))
        x1 = sol[-1]

        self.last_stats = ODEInferenceStats(
            solver=mode,
            nfe=nfe_counter["nfe"],
            rtol=float(rtol),
            atol=float(atol),
        )
        if self.log_nfe:
            print(
                f"[flow_ode] solver=dopri5  t=[{t0:.3f}->{t1:.3f}]  "
                f"rtol={float(rtol):.1e}  atol={float(atol):.1e}  "
                f"nfe={nfe_counter['nfe']}",
                flush=True,
            )
        return x1
