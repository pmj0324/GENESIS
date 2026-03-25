from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analysis.correlation import compute_correlation_errors
from analysis.cross_spectrum import compute_cross_power_spectrum_2d, compute_spectrum_errors
from analysis.power_spectrum import compute_power_spectrum_2d
from dataloader.normalization import Normalizer
from diffusion.ddpm import GaussianDiffusion
from diffusion.schedules import build_schedule
from diffusion.samplers_edm import make_sigma_schedule
from flow_matching.samplers import EulerSampler
from utils.sampler_config import resolve_sampler_config


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None):
        if cond is None:
            return self.conv(x)
        return self.conv(x) + 0.0 * cond[:, 0].view(-1, 1, 1, 1)


def _expect_error(fn, *args, **kwargs) -> str:
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        return type(exc).__name__
    raise AssertionError("expected an exception, but call succeeded")


def run_checks() -> dict:
    np.random.seed(0)
    torch.manual_seed(0)

    results: dict = {}

    # 1) Normalizer round-trip
    norm_cfg = {
        "Mcdm": {"method": "softclip", "center": 10.876, "scale": 0.590, "clip_c": 4.5},
        "Mgas": {"method": "affine", "center": 10.344, "scale": 0.627},
        "T": {"method": "minmax", "center": 4.2234, "scale": 0.8163, "min_z": -3.0, "max_z": 3.0},
    }
    normalizer = Normalizer(norm_cfg)
    x = (10 ** np.random.uniform(2.0, 12.0, size=(4, 3, 32, 32))).astype(np.float32)
    z = normalizer.normalize_numpy(x)
    xr = normalizer.denormalize_numpy(z)
    rel = np.abs(xr - x) / (np.abs(x) + 1e-12)
    results["normalizer_roundtrip"] = {
        "mean_rel_error": float(rel.mean()),
        "max_rel_error": float(rel.max()),
        "passed": bool(rel.mean() < 1e-4 and rel.max() < 5e-3),
    }

    # 2) power_spectrum(auto) parity
    field = np.random.randn(64, 64)
    k1, p1 = compute_power_spectrum_2d(field)
    k2, p2 = compute_cross_power_spectrum_2d(field, field)
    rel_l2 = float(np.linalg.norm(p1 - p2) / (np.linalg.norm(p1) + 1e-12))
    results["power_spectrum_parity"] = {
        "k_equal": bool(np.allclose(k1, k2)),
        "relative_l2_error": rel_l2,
        "passed": bool(np.allclose(k1, k2) and rel_l2 < 1e-10),
    }

    # 3) metrics finite sanity
    xt = 10 ** np.random.uniform(2.0, 8.0, size=(6, 3, 32, 32)).astype(np.float32)
    xg = 10 ** np.random.uniform(2.0, 8.0, size=(6, 3, 32, 32)).astype(np.float32)
    spec = compute_spectrum_errors(np.log10(xt), np.log10(xg), n_bins=12)
    corr = compute_correlation_errors(np.log10(xt), np.log10(xg), n_bins=12)
    results["metric_finite_checks"] = {
        "spec_mean_error_finite": bool(np.isfinite(spec["Mcdm"]["mean_error"])),
        "corr_max_delta_finite": bool(np.isfinite(corr["Mcdm-Mgas"]["max_delta_r"])),
        "passed": bool(
            np.isfinite(spec["Mcdm"]["mean_error"])
            and np.isfinite(corr["Mcdm-Mgas"]["max_delta_r"])
        ),
    }

    # 4) step validation checks
    model = DummyModel()
    schedule = build_schedule("cosine", T=32)
    diffusion = GaussianDiffusion(schedule)
    cond = torch.randn(2, 6)

    step_checks = {
        "ddim_steps_0": _expect_error(
            diffusion.ddim_sample, model, (2, 3, 16, 16), cond, 0, 0.0, 1.0, False
        ),
        "ddim_steps_1": _expect_error(
            diffusion.ddim_sample, model, (2, 3, 16, 16), cond, 1, 0.0, 1.0, False
        ),
        "ddim_steps_gt_T": _expect_error(
            diffusion.ddim_sample, model, (2, 3, 16, 16), cond, 33, 0.0, 1.0, False
        ),
        "flow_steps_1": _expect_error(
            EulerSampler().sample, model, (2, 3, 16, 16), cond, 1, 1.0, False
        ),
        "edm_steps_1": _expect_error(make_sigma_schedule, 1),
    }
    step_checks["passed"] = all(err in {"ValueError", "TypeError"} for err in step_checks.values())
    results["step_validation"] = step_checks

    # 5) EDM eta compatibility mapping
    sampler_cfg = resolve_sampler_config(
        {
            "generative": {
                "framework": "edm",
                "sampler": {"steps": 40, "eta": 0.2},
                "edm": {},
            }
        },
        "edm",
    )
    results["edm_eta_alias"] = {
        "eta": sampler_cfg["eta"],
        "S_churn": sampler_cfg["S_churn"],
        "expected_s_churn": 8.0,
        "passed": bool(abs(sampler_cfg["S_churn"] - 8.0) < 1e-8),
    }

    results["overall_passed"] = all(
        bool(section.get("passed", False)) for section in results.values() if isinstance(section, dict)
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run math/runtime checks for GENESIS audit")
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    report = run_checks()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"saved: {args.out_json}")
    print(f"overall_passed={report['overall_passed']}")


if __name__ == "__main__":
    main()
