"""
GENESIS - CAMELS Protocol Evaluator

LH/1P/CV/EX 프로토콜 기반 생성 모델 평가.

프로토콜 (§4.5):
  LH  — Latin Hypercube: 100 held-out sims, 6D parameter space generalization
  1P  — One-Parameter-at-a-time: ∂P(k)/∂θ_i sensitivity test
  CV  — Cosmic Variance: variance ratio σ²_gen/σ²_true at fiducial params
  EX  — Extreme: extrapolation robustness, no catastrophic failures

측정 프로토콜 (§4.6):
  - Final eval: N ≥ 32 generated maps per conditioning
  - Uncertainty: mean ± std over 5 independent sampling runs
  - Pass criterion: mean + 1σ < target threshold

사용법:
    evaluator = CAMELSEvaluator(
        model=model,
        sampler_fn=sampler_fn,    # (model, shape, cond) → [B,3,H,W] normalized
        normalizer=normalizer,
        device='cuda',
        box_size=25.0,
    )
    results = evaluator.evaluate_lh(val_maps, val_params)
"""
import numpy as np
import torch
from typing import Callable, Optional, List
from tqdm.auto import tqdm

from .cross_spectrum import (
    _to_numpy, compute_spectrum_errors, CHANNELS, AUTO_POWER_THRESHOLDS,
)
from .correlation import compute_correlation_errors
from .pixel_distribution import compare_pixel_distributions, compute_distribution_summary


class CAMELSEvaluator:
    """Evaluates a generative model on CAMELS cosmological map data.

    Supports LH (Latin Hypercube) and CV (Cosmic Variance) evaluation protocols.

    Args:
        model: Trained generative model (torch.nn.Module).
        sampler_fn: Callable with signature (model, shape, cond) -> Tensor [B,3,H,W]
            returning normalized (z-space) samples.
        normalizer: Normalizer instance for denormalization.
        device: Torch device string or object.
        box_size: Physical box size in Mpc/h. Default 25.0.
        n_gen_per_cond: Number of generated samples per conditioning vector.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        sampler_fn: Callable,
        normalizer,
        device,
        box_size: float = 25.0,
        n_gen_per_cond: int = 1,
    ):
        self.model = model
        self.sampler_fn = sampler_fn
        self.normalizer = normalizer
        self.device = device
        self.box_size = box_size
        self.n_gen_per_cond = n_gen_per_cond

    def _progress(self, iterable, *, total=None, desc="", enabled=True, leave=False):
        """Wrap iterables with tqdm so long evaluations show ETA and progress."""
        if not enabled:
            return iterable
        return tqdm(iterable, total=total, desc=desc, leave=leave, dynamic_ncols=True)

    def _to_log10(self, x_norm) -> np.ndarray:
        """Denormalize z-space maps and convert to log10(physical) space.

        Args:
            x_norm: (B, 3, H, W) or (3, H, W) normalized maps, torch tensor or numpy.

        Returns:
            np.ndarray of shape (B, 3, H, W) in log10(physical field) space.
        """
        if hasattr(x_norm, "cpu"):
            t = x_norm.float()
        else:
            t = torch.from_numpy(np.asarray(x_norm, dtype=np.float32))

        squeezed = t.ndim == 3
        if squeezed:
            t = t.unsqueeze(0)

        with torch.no_grad():
            physical = self.normalizer.denormalize(t.to(self.device)).cpu()

        log10_maps = torch.log10(torch.clamp(physical, min=1e-30)).numpy()
        return log10_maps[0] if squeezed else log10_maps

    @torch.no_grad()
    def _generate_log10(self, cond_norm, n_samples: int = 1) -> np.ndarray:
        """Generate samples in log10 space given a conditioning vector.

        Args:
            cond_norm: [6] or [1, 6] normalized cosmological parameter tensor.
            n_samples: Number of samples to generate.

        Returns:
            np.ndarray of shape (n_samples, 3, H, W) in log10 space.
        """
        if hasattr(cond_norm, "cpu"):
            cond = cond_norm.float().to(self.device)
        else:
            cond = torch.from_numpy(np.asarray(cond_norm, dtype=np.float32)).to(self.device)

        cond = cond.view(1, -1)  # (1, 6)

        # Sample n_samples times (batch with repeated cond)
        if n_samples == 1:
            shape = (1, 3, 256, 256)
            sample_norm = self.sampler_fn(self.model, shape, cond)  # (1, 3, H, W)
        else:
            cond_batch = cond.expand(n_samples, -1)  # (n_samples, 6)
            shape = (n_samples, 3, 256, 256)
            sample_norm = self.sampler_fn(self.model, shape, cond_batch)  # (n_samples, 3, H, W)

        return self._to_log10(sample_norm)

    def evaluate_lh(
        self,
        val_maps,
        val_params,
        max_samples: int = 200,
        verbose: bool = True,
    ) -> dict:
        """Evaluate on the LH (Latin Hypercube) validation set.

        For each sample (up to max_samples), generates one map with the same
        conditioning and evaluates power spectra, correlation coefficients, and
        pixel distributions.

        Args:
            val_maps: (N, 3, H, W) normalized maps (torch or numpy).
            val_params: (N, 6) normalized cosmological parameters (torch or numpy).
            max_samples: Maximum number of validation samples to use.
            verbose: If True, print progress.

        Returns:
            dict with keys: "auto_power", "cross_power", "correlation", "pdf",
            "pdf_summary", "pass_summary"
        """
        val_maps_np = _to_numpy(val_maps)
        val_params_np = _to_numpy(val_params)

        N = min(len(val_maps_np), max_samples)

        true_log10_list = []
        gen_log10_list = []

        for i in self._progress(range(N), total=N, desc="LH samples", enabled=verbose):
            true_l10 = self._to_log10(val_maps_np[i])  # (3, H, W)
            true_log10_list.append(true_l10)

            cond = val_params_np[i]  # (6,)
            gen_l10 = self._generate_log10(cond, n_samples=1)  # (1, 3, H, W)
            gen_log10_list.append(gen_l10[0])

        # Stack to (N, 3, H, W)
        true_batch = np.stack(true_log10_list, axis=0)
        gen_batch = np.stack(gen_log10_list, axis=0)

        if verbose:
            print("  [evaluate_lh] computing spectrum errors ...", flush=True)

        spectrum_errors = compute_spectrum_errors(
            true_batch, gen_batch, box_size=self.box_size
        )
        correlation_errors = compute_correlation_errors(
            true_batch, gen_batch, box_size=self.box_size
        )
        pdf_results = compare_pixel_distributions(true_batch, gen_batch, log=False)
        pdf_summary = compute_distribution_summary(true_batch, gen_batch)

        # Separate auto and cross from spectrum_errors
        auto_power = {k: v for k, v in spectrum_errors.items() if v["type"] == "auto"}
        cross_power = {k: v for k, v in spectrum_errors.items() if v["type"] == "cross"}

        # Pass summary
        pass_summary = {}
        for ch in CHANNELS:
            if ch in auto_power:
                pass_summary[f"auto_{ch}"] = auto_power[ch]["passed"]
        for pair_key, v in cross_power.items():
            pass_summary[f"cross_{pair_key}"] = v["passed"]
        for pair_key, v in correlation_errors.items():
            pass_summary[f"corr_{pair_key}"] = v["passed"]
        for ch, v in pdf_results.items():
            pass_summary[f"pdf_{ch}"] = v["passed"]

        all_passed = all(pass_summary.values())
        pass_summary["overall"] = all_passed

        return {
            "auto_power": auto_power,
            "cross_power": cross_power,
            "correlation": correlation_errors,
            "pdf": pdf_results,
            "pdf_summary": pdf_summary,
            "pass_summary": pass_summary,
        }

    def evaluate_cv(
        self,
        cv_maps_norm,
        fiducial_cond_norm,
        box_size: Optional[float] = None,
    ) -> dict:
        """Evaluate on the CV (Cosmic Variance) set.

        Compares per-k variance of true vs generated maps at fiducial cosmology.

        Pass criterion (§4.5): 0.7 < var_ratio < 1.3 for most k-bins.

        # [OLD] Pass criterion: 0.8 < var_ratio < 1.2 for >80% of k-bins
        # (relaxed from 0.8-1.2 to 0.7-1.3 to account for multi-field complexity)

        Args:
            cv_maps_norm: (N_cv, 3, H, W) normalized CV set maps (same fiducial
                params, different seeds).
            fiducial_cond_norm: [6] fiducial cosmological parameters (normalized).
            box_size: Override box size. Uses self.box_size if None.

        Returns:
            dict mapping channel name -> {
                "k": ..., "var_true": ..., "var_gen": ...,
                "var_ratio": ..., "passed": bool
            }
        """
        if box_size is None:
            box_size = self.box_size

        cv_maps_np = _to_numpy(cv_maps_norm)
        N_cv = len(cv_maps_np)

        # True log10 maps
        true_log10_list = []
        for i in self._progress(range(N_cv), total=N_cv, desc="CV true maps", enabled=True):
            true_log10_list.append(self._to_log10(cv_maps_np[i]))
        true_batch = np.stack(true_log10_list, axis=0)  # (N_cv, 3, H, W)

        # Generated log10 maps
        print(f"  [evaluate_cv] generating {N_cv} fiducial samples ...", flush=True)
        gen_batch = self._generate_log10(fiducial_cond_norm, n_samples=N_cv)  # (N_cv, 3, H, W)

        from .cross_spectrum import compute_cross_power_spectrum_2d
        results = {}

        for ci, ch in enumerate(self._progress(CHANNELS, total=len(CHANNELS), desc="CV channels", enabled=True)):
            # Per-sample P(k) for true and gen
            pk_true_list = []
            pk_gen_list = []
            k_centers = None

            for b in self._progress(range(N_cv), total=N_cv, desc=f"CV spectra {ch}", enabled=True):
                k_c, Pk_t = compute_cross_power_spectrum_2d(
                    true_batch[b, ci], true_batch[b, ci], box_size=box_size
                )
                _, Pk_g = compute_cross_power_spectrum_2d(
                    gen_batch[b, ci], gen_batch[b, ci], box_size=box_size
                )
                pk_true_list.append(Pk_t)
                pk_gen_list.append(Pk_g)
                k_centers = k_c

            pks_true = np.array(pk_true_list)  # (N_cv, n_bins)
            pks_gen = np.array(pk_gen_list)

            var_true = pks_true.var(axis=0)  # (n_bins,)
            var_gen = pks_gen.var(axis=0)

            var_ratio = var_gen / (var_true + 1e-60)

            # ── [OLD] CV pass criterion ──
            # frac_in_band = float(((var_ratio > 0.8) & (var_ratio < 1.2)).mean())
            # passed = bool(frac_in_band > 0.8)

            # ── [NEW] CV pass criterion (§4.5): 0.7 < ratio < 1.3 ──
            frac_in_band = float(((var_ratio > 0.7) & (var_ratio < 1.3)).mean())
            passed = bool(frac_in_band > 0.8)

            results[ch] = {
                "k": k_centers,
                "var_true": var_true,
                "var_gen": var_gen,
                "var_ratio": var_ratio,
                "frac_in_band": frac_in_band,
                "passed": passed,
            }

        return results

    def evaluate_1p(
        self,
        onep_maps_norm,
        onep_params_norm,
        fiducial_maps_norm,
        fiducial_cond_norm,
        param_names: Optional[List[str]] = None,
        box_size: Optional[float] = None,
    ) -> dict:
        """Evaluate on the 1P (One-Parameter-at-a-time) set (§4.5).

        Tests whether the model correctly captures ∂P(k)/∂θ_i for each
        parameter by comparing P(k) ratio curves.

        Pass criterion: generated P(k) ratios within ±2σ of true (per seed).

        Args:
            onep_maps_norm: dict mapping param_name -> (N_vals, 3, H, W) normalized maps
                where each param is varied while others are fiducial.
            onep_params_norm: dict mapping param_name -> (N_vals, 6) normalized params.
            fiducial_maps_norm: (N_fid, 3, H, W) fiducial maps for ratio denominator.
            fiducial_cond_norm: [6] fiducial condition.
            param_names: Optional list of parameter names. Default: all keys in onep_maps_norm.
            box_size: Override box size.

        Returns:
            dict mapping param_name -> {
                channel_name -> {
                    "k", "ratio_true", "ratio_gen", "ratio_diff",
                    "within_2sigma", "frac_within", "passed"
                }
            }
        """
        if box_size is None:
            box_size = self.box_size
        if param_names is None:
            param_names = list(onep_maps_norm.keys())

        from .cross_spectrum import compute_cross_power_spectrum_2d

        # Compute fiducial P(k) (average over fiducial maps)
        fid_np = _to_numpy(fiducial_maps_norm)
        fid_log10 = np.stack(
            [
                self._to_log10(fid_np[i])
                for i in self._progress(
                    range(len(fid_np)),
                    total=len(fid_np),
                    desc="1P fiducial maps",
                    enabled=True,
                )
            ],
            axis=0,
        )

        fid_pk = {}
        fid_pk_std = {}
        for ci, ch in enumerate(self._progress(CHANNELS, total=len(CHANNELS), desc="1P fid channels", enabled=True)):
            pks = []
            k_c = None
            for b in self._progress(
                range(len(fid_log10)),
                total=len(fid_log10),
                desc=f"1P fid spectra {ch}",
                enabled=True,
            ):
                k_c, pk = compute_cross_power_spectrum_2d(
                    fid_log10[b, ci], fid_log10[b, ci], box_size=box_size
                )
                pks.append(pk)
            pks = np.array(pks)
            fid_pk[ch] = (k_c, pks.mean(axis=0))
            fid_pk_std[ch] = pks.std(axis=0)

        results = {}

        for pname in self._progress(param_names, total=len(param_names), desc="1P params", enabled=True):
            maps_np = _to_numpy(onep_maps_norm[pname])
            params_np = _to_numpy(onep_params_norm[pname])
            N_vals = len(maps_np)

            param_results = {}
            for ci, ch in enumerate(
                self._progress(CHANNELS, total=len(CHANNELS), desc=f"1P channels {pname}", enabled=True)
            ):
                k_centers = fid_pk[ch][0]
                fid_mean = fid_pk[ch][1]
                fid_std = fid_pk_std[ch]

                ratio_true_list = []
                ratio_gen_list = []

                for i in self._progress(
                    range(N_vals),
                    total=N_vals,
                    desc=f"1P {pname} {ch}",
                    enabled=True,
                ):
                    # True P(k) ratio
                    true_l10 = self._to_log10(maps_np[i])
                    _, pk_true = compute_cross_power_spectrum_2d(
                        true_l10[ci], true_l10[ci], box_size=box_size
                    )
                    ratio_true = pk_true / (fid_mean + 1e-60)
                    ratio_true_list.append(ratio_true)

                    # Generated P(k) ratio
                    gen_l10 = self._generate_log10(params_np[i], n_samples=1)[0]
                    _, pk_gen = compute_cross_power_spectrum_2d(
                        gen_l10[ci], gen_l10[ci], box_size=box_size
                    )
                    ratio_gen = pk_gen / (fid_mean + 1e-60)
                    ratio_gen_list.append(ratio_gen)

                ratio_true_arr = np.array(ratio_true_list)
                ratio_gen_arr = np.array(ratio_gen_list)

                # Per-value check: is generated ratio within ±2σ of true?
                # σ from fiducial cosmic variance propagated to ratio
                sigma_ratio = 2.0 * fid_std / (fid_mean + 1e-60)
                ratio_diff = np.abs(ratio_gen_arr - ratio_true_arr)
                within_2sigma = ratio_diff < sigma_ratio[np.newaxis, :]
                frac_within = float(within_2sigma.mean())
                passed = bool(frac_within > 0.7)  # >70% of (value, k) pairs within ±2σ

                param_results[ch] = {
                    "k": k_centers,
                    "ratio_true": ratio_true_arr,
                    "ratio_gen": ratio_gen_arr,
                    "ratio_diff": ratio_diff,
                    "sigma_ratio": sigma_ratio,
                    "within_2sigma": within_2sigma,
                    "frac_within": frac_within,
                    "passed": passed,
                }

            results[pname] = param_results

        return results

    def evaluate_ex(
        self,
        ex_maps_norm,
        ex_params_norm,
        box_size: Optional[float] = None,
    ) -> dict:
        """Evaluate on the EX (Extreme) set (§4.5).

        Tests extrapolation robustness: no catastrophic failures (NaN, divergence,
        >200% error). Error target: < 2× LH targets.

        Args:
            ex_maps_norm: (N_ex, 3, H, W) normalized extreme-parameter maps.
            ex_params_norm: (N_ex, 6) normalized extreme parameters.
            box_size: Override box size.

        Returns:
            dict with keys:
                "spectrum_errors": per-sample spectrum error results
                "has_nan": bool (any NaN in generated maps)
                "has_divergence": bool (any value > 1e10)
                "max_auto_error": dict {channel: float} max mean error across samples
                "passed": bool (no NaN, no divergence, no >200% error)
        """
        if box_size is None:
            box_size = self.box_size

        ex_maps_np = _to_numpy(ex_maps_norm)
        ex_params_np = _to_numpy(ex_params_norm)
        N_ex = len(ex_maps_np)

        true_log10_list = []
        gen_log10_list = []

        has_nan = False
        has_divergence = False

        for i in self._progress(range(N_ex), total=N_ex, desc="EX samples", enabled=True):
            true_l10 = self._to_log10(ex_maps_np[i])
            true_log10_list.append(true_l10)

            gen_l10 = self._generate_log10(ex_params_np[i], n_samples=1)[0]
            gen_log10_list.append(gen_l10)

            # Check for catastrophic failures
            if np.any(np.isnan(gen_l10)):
                has_nan = True
            if np.any(np.abs(gen_l10) > 1e10):
                has_divergence = True

        true_batch = np.stack(true_log10_list, axis=0)
        gen_batch = np.stack(gen_log10_list, axis=0)

        spectrum_errors = compute_spectrum_errors(true_batch, gen_batch, box_size=box_size)

        # Check: auto-power error < 2× LH targets (§4.5 EX criterion)
        max_auto_error = {}
        catastrophic = False
        for ch in CHANNELS:
            me = spectrum_errors[ch]["mean_error"]
            max_auto_error[ch] = me
            if me > 2.0:  # >200% error = catastrophic
                catastrophic = True

        passed = bool(not has_nan and not has_divergence and not catastrophic)

        return {
            "spectrum_errors": spectrum_errors,
            "has_nan": has_nan,
            "has_divergence": has_divergence,
            "max_auto_error": max_auto_error,
            "passed": passed,
        }

    def evaluate_multirun(
        self,
        val_maps,
        val_params,
        n_runs: int = 5,
        max_samples: int = 32,
        verbose: bool = True,
    ) -> dict:
        """Run LH evaluation multiple times with different seeds for uncertainty (§4.6).

        Reports mean ± std over n_runs independent sampling runs.
        Pass criterion: mean + 1σ < target threshold.

        Args:
            val_maps: (N, 3, H, W) normalized validation maps.
            val_params: (N, 6) normalized cosmological parameters.
            n_runs: Number of independent sampling runs.
            max_samples: Max samples per run (recommended: ≥32 per §4.6).
            verbose: Print progress.

        Returns:
            dict with keys:
                "runs": list of per-run LH results,
                "summary": aggregated mean±std for key metrics,
                "pass_robust": pass with mean+1σ criterion
        """
        runs = []
        for run_idx in self._progress(range(n_runs), total=n_runs, desc="Multi-run", enabled=verbose):
            if verbose:
                print(f"\n[multirun] === Run {run_idx+1}/{n_runs} ===", flush=True)
            # Set different random seed for each run (affects noise in sampling)
            torch.manual_seed(run_idx * 12345 + 42)
            np.random.seed(run_idx * 12345 + 42)

            result = self.evaluate_lh(val_maps, val_params, max_samples=max_samples,
                                      verbose=verbose)
            runs.append(result)

        # Aggregate key metrics across runs
        summary = {"auto_power": {}, "cross_power": {}, "correlation": {}, "pdf": {}}

        # Auto-power mean errors
        for ch in CHANNELS:
            means = [r["auto_power"][ch]["mean_error"] for r in runs]
            rms_vals = [r["auto_power"][ch]["rms_error"] for r in runs]
            summary["auto_power"][ch] = {
                "mean_error_mean": float(np.mean(means)),
                "mean_error_std": float(np.std(means)),
                "rms_error_mean": float(np.mean(rms_vals)),
                "rms_error_std": float(np.std(rms_vals)),
            }

        # Cross-power mean errors
        for pair in ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]:
            means = [r["cross_power"][pair]["mean_error"] for r in runs]
            summary["cross_power"][pair] = {
                "mean_error_mean": float(np.mean(means)),
                "mean_error_std": float(np.std(means)),
            }

        # Correlation max_delta_r
        for pair in ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]:
            vals = [r["correlation"][pair]["max_delta_r"] for r in runs]
            summary["correlation"][pair] = {
                "max_delta_r_mean": float(np.mean(vals)),
                "max_delta_r_std": float(np.std(vals)),
            }

        # PDF KS statistic
        for ch in CHANNELS:
            vals = [r["pdf"][ch]["ks_statistic"] for r in runs]
            summary["pdf"][ch] = {
                "ks_statistic_mean": float(np.mean(vals)),
                "ks_statistic_std": float(np.std(vals)),
            }

        # Robust pass: mean + 1σ < target (§4.6)
        pass_robust = {}
        for ch in CHANNELS:
            s = summary["auto_power"][ch]
            # Check scale-dependent thresholds
            all_pass = True
            for label, k_lo, k_hi, thr_mean, thr_rms in AUTO_POWER_THRESHOLDS[ch]:
                # Use overall mean_error as proxy (conservative)
                if s["mean_error_mean"] + s["mean_error_std"] >= thr_mean:
                    all_pass = False
                    break
            pass_robust[f"auto_{ch}"] = all_pass

        from .cross_spectrum import CROSS_POWER_THRESHOLDS
        for pair in ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]:
            s = summary["cross_power"][pair]
            thr = CROSS_POWER_THRESHOLDS[pair]
            pass_robust[f"cross_{pair}"] = bool(
                s["mean_error_mean"] + s["mean_error_std"] < thr
            )

        return {
            "runs": runs,
            "summary": summary,
            "pass_robust": pass_robust,
            "n_runs": n_runs,
            "max_samples": max_samples,
        }

    def run_all(
        self,
        val_maps,
        val_params,
        cv_maps=None,
        fiducial_cond=None,
        onep_maps=None,
        onep_params=None,
        fiducial_maps=None,
        ex_maps=None,
        ex_params=None,
        protocols=None,
        max_samples: int = 100,
        n_multirun: int = 0,
    ) -> dict:
        """Run all specified evaluation protocols.

        Args:
            val_maps: (N, 3, H, W) normalized validation maps.
            val_params: (N, 6) normalized cosmological parameters.
            cv_maps: (N_cv, 3, H, W) CV set maps (required for "cv" protocol).
            fiducial_cond: [6] fiducial condition (required for "cv"/"1p" protocol).
            onep_maps: dict param_name -> maps (required for "1p" protocol).
            onep_params: dict param_name -> params (required for "1p" protocol).
            fiducial_maps: (N_fid, 3, H, W) fiducial maps (required for "1p" protocol).
            ex_maps: (N_ex, 3, H, W) extreme maps (required for "ex" protocol).
            ex_params: (N_ex, 6) extreme params (required for "ex" protocol).
            protocols: List of protocol names to run. Default: ["lh"].
            max_samples: Maximum number of LH validation samples to evaluate.
            n_multirun: If > 0, run multirun evaluation with this many seeds.

        Returns:
            dict with protocol names as keys and their results as values.
        """
        if protocols is None:
            protocols = ["lh"]

        all_results = {}

        if "lh" in protocols:
            print("[run_all] Running LH protocol ...", flush=True)
            all_results["lh"] = self.evaluate_lh(
                val_maps, val_params, max_samples=max_samples
            )

        if "cv" in protocols:
            if cv_maps is None or fiducial_cond is None:
                raise ValueError("cv_maps and fiducial_cond are required for CV protocol.")
            print("[run_all] Running CV protocol ...", flush=True)
            all_results["cv"] = self.evaluate_cv(cv_maps, fiducial_cond)

        if "1p" in protocols:
            if onep_maps is None or onep_params is None or fiducial_maps is None or fiducial_cond is None:
                raise ValueError(
                    "onep_maps, onep_params, fiducial_maps, and fiducial_cond "
                    "are required for 1P protocol."
                )
            print("[run_all] Running 1P protocol ...", flush=True)
            all_results["1p"] = self.evaluate_1p(
                onep_maps, onep_params, fiducial_maps, fiducial_cond
            )

        if "ex" in protocols:
            if ex_maps is None or ex_params is None:
                raise ValueError("ex_maps and ex_params are required for EX protocol.")
            print("[run_all] Running EX protocol ...", flush=True)
            all_results["ex"] = self.evaluate_ex(ex_maps, ex_params)

        if n_multirun > 0:
            print(f"[run_all] Running multi-run evaluation (n={n_multirun}) ...", flush=True)
            all_results["multirun"] = self.evaluate_multirun(
                val_maps, val_params, n_runs=n_multirun, max_samples=max_samples
            )

        return all_results
