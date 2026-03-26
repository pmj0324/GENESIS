from __future__ import annotations


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _pick(*values, default=None):
    for value in values:
        if value is not None:
            return value
    return default


def _method(value, *, default: str) -> str:
    if value is None:
        return default
    return str(value).strip().lower()


def resolve_sampler_config(cfg: dict, framework: str) -> dict:
    """
    Resolve generative sampler settings with consistent precedence.

    Priority:
      1) generative.sampler.*
      2) framework-local legacy location (if any)
      3) built-in default

    Notes:
      - EDM keeps `eta` as a compatibility alias.
      - If EDM `S_churn` is missing and `eta > 0`, we map:
            S_churn = eta * steps
        so legacy eta configs keep stochastic behavior.
    """
    gcfg = _as_dict(cfg.get("generative"))
    scfg = _as_dict(gcfg.get("sampler"))
    dcfg = _as_dict(gcfg.get("diffusion"))
    ecfg = _as_dict(gcfg.get("edm"))

    out = {"cfg_scale": float(_pick(scfg.get("cfg_scale"), default=1.0))}

    if framework == "flow_matching":
        out["method"] = _method(_pick(scfg.get("method"), scfg.get("sampler_a")), default="heun")
        out["steps"] = int(_pick(scfg.get("steps"), default=25))
        out["rtol"] = float(_pick(scfg.get("rtol"), default=1e-5))
        out["atol"] = float(_pick(scfg.get("atol"), default=1e-5))
        return out

    if framework == "diffusion":
        out["method"] = _method(_pick(scfg.get("method"), scfg.get("sampler_a")), default="ddim")
        out["steps"] = int(_pick(scfg.get("steps"), default=50))
        out["eta"] = float(_pick(scfg.get("eta"), dcfg.get("eta"), default=0.0))
        return out

    if framework == "edm":
        steps = int(_pick(scfg.get("steps"), ecfg.get("steps"), default=40))
        eta = float(_pick(scfg.get("eta"), ecfg.get("eta"), default=0.0))
        s_churn_raw = _pick(
            scfg.get("S_churn"),
            scfg.get("s_churn"),
            scfg.get("churn"),
            ecfg.get("S_churn"),
            ecfg.get("s_churn"),
            ecfg.get("churn"),
            default=None,
        )
        s_churn = float(s_churn_raw) if s_churn_raw is not None else (eta * steps if eta > 0 else 0.0)

        out.update(
            {
                "method": _method(_pick(scfg.get("method"), scfg.get("sampler_a")), default="heun"),
                "steps": steps,
                "eta": eta,
                "S_churn": s_churn,
                "sigma_min": float(_pick(scfg.get("sigma_min"), ecfg.get("sigma_min"), default=0.002)),
                "sigma_max": float(_pick(scfg.get("sigma_max"), ecfg.get("sigma_max"), default=80.0)),
                "rho": float(_pick(scfg.get("rho"), ecfg.get("rho"), default=7.0)),
            }
        )
        return out

    raise ValueError(f"Unknown framework: {framework!r}")
