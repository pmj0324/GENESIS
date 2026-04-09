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
        sample_shape: Optional[tuple[int, int, int]] = None,
    ):
        self.model = model
        self.sampler_fn = sampler_fn
        self.normalizer = normalizer
        self.device = device
        self.box_size = box_size
        self.n_gen_per_cond = n_gen_per_cond
        if sample_shape is None:
            sample_shape = (3, 256, 256)
        if len(sample_shape) != 3 or any(int(v) <= 0 for v in sample_shape):
            raise ValueError(f"sample_shape must be (C,H,W) with positive integers, got {sample_shape!r}")
        self.sample_shape = tuple(int(v) for v in sample_shape)

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
    def _generate_log10(self, cond_norm, n_samples: Optional[int] = None) -> np.ndarray:
        """Generate samples in log10 space given a conditioning vector.

        Args:
            cond_norm: [6], [1, 6], or [N, 6] normalized parameter tensor/array.
            n_samples: Number of samples to generate. If None, inferred from cond rows.

        Returns:
            np.ndarray of shape (n_samples, 3, H, W) in log10 space.
        """
        if hasattr(cond_norm, "cpu"):
            cond = cond_norm.float().to(self.device)
        else:
            cond = torch.from_numpy(np.asarray(cond_norm, dtype=np.float32)).to(self.device)

        if cond.ndim == 1:
            cond = cond.view(1, -1)
        elif cond.ndim != 2:
            raise ValueError(f"cond_norm must be 1D or 2D, got shape={tuple(cond.shape)}")

        if n_samples is None:
            n_samples = int(cond.shape[0])

        if cond.shape[0] == 1 and n_samples > 1:
            cond_batch = cond.expand(n_samples, -1)
        elif cond.shape[0] == n_samples:
            cond_batch = cond
        else:
            raise ValueError(
                f"cond batch mismatch: cond_rows={cond.shape[0]} n_samples={n_samples}"
            )

        shape = (n_samples, *self.sample_shape)
        sample_norm = self.sampler_fn(self.model, shape, cond_batch)  # (n_samples, 3, H, W)
        if isinstance(sample_norm, tuple):
            sample_norm = sample_norm[0]

        return self._to_log10(sample_norm)

    @torch.no_grad()
    def _generate_log10_in_chunks(
        self,
        conds_norm,
        *,
        batch_size: int = 8,
        desc: Optional[str] = None,
        enabled: bool = True,
    ) -> np.ndarray:
        """Generate log10 maps for varying conditions using chunked batches."""
        conds_np = _to_numpy(conds_norm).astype(np.float32, copy=False)
        if conds_np.ndim == 1:
            conds_np = conds_np[None, :]
        if conds_np.ndim != 2 or conds_np.shape[1] != 6:
            raise ValueError(f"expected conds shape [N,6], got {conds_np.shape}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        outputs = []
        starts = range(0, len(conds_np), batch_size)
        if desc is not None:
            starts = self._progress(
                starts,
                total=(len(conds_np) + batch_size - 1) // batch_size,
                desc=desc,
                enabled=enabled,
            )
        for start in starts:
            end = min(start + batch_size, len(conds_np))
            batch = conds_np[start:end]
            outputs.append(self._generate_log10(batch, n_samples=len(batch)))
        return (
            np.concatenate(outputs, axis=0)
            if outputs
            else np.empty((0, *self.sample_shape), dtype=np.float32)
        )

    @torch.no_grad()
    def _to_physical(self, x_norm) -> np.ndarray:
        """Denormalize normalized maps to physical space (no log10 transform).

        Args:
            x_norm: (B,3,H,W) or (3,H,W) normalized maps (torch tensor or numpy).

        Returns:
            np.ndarray, same spatial shape, in physical units.
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
        arr = physical.numpy().astype(np.float32, copy=False)
        return arr[0] if squeezed else arr

    @torch.no_grad()
    def _generate_norm(self, cond_norm, n_samples: Optional[int] = None) -> np.ndarray:
        """Generate normalized (z-space) samples given a conditioning vector.

        Returns:
            np.ndarray of shape (n_samples, 3, H, W) in normalized space.
        """
        if hasattr(cond_norm, "cpu"):
            cond = cond_norm.float().to(self.device)
        else:
            cond = torch.from_numpy(np.asarray(cond_norm, dtype=np.float32)).to(self.device)
        if cond.ndim == 1:
            cond = cond.view(1, -1)
        elif cond.ndim != 2:
            raise ValueError(f"cond_norm must be 1D or 2D, got shape={tuple(cond.shape)}")
        if n_samples is None:
            n_samples = int(cond.shape[0])
        if cond.shape[0] == 1 and n_samples > 1:
            cond_batch = cond.expand(n_samples, -1)
        elif cond.shape[0] == n_samples:
            cond_batch = cond
        else:
            raise ValueError(
                f"cond batch mismatch: cond_rows={cond.shape[0]} n_samples={n_samples}"
            )
        shape = (n_samples, *self.sample_shape)
        out = self.sampler_fn(self.model, shape, cond_batch)
        if isinstance(out, tuple):
            out = out[0]
        return out.detach().cpu().numpy().astype(np.float32, copy=False)

    @torch.no_grad()
    def _generate_norm_in_chunks(
        self,
        conds_norm,
        *,
        batch_size: int = 8,
        desc: Optional[str] = None,
        enabled: bool = True,
    ) -> np.ndarray:
        """Generate normalized samples for varying conditions in chunks.

        Returns:
            np.ndarray of shape (N, 3, H, W) in normalized space.
        """
        conds_np = _to_numpy(conds_norm).astype(np.float32, copy=False)
        if conds_np.ndim == 1:
            conds_np = conds_np[None, :]
        if conds_np.ndim != 2 or conds_np.shape[1] != 6:
            raise ValueError(f"expected conds shape [N,6], got {conds_np.shape}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        outputs = []
        starts = range(0, len(conds_np), batch_size)
        if desc is not None:
            starts = self._progress(
                starts,
                total=(len(conds_np) + batch_size - 1) // batch_size,
                desc=desc,
                enabled=enabled,
            )
        for start in starts:
            end = min(start + batch_size, len(conds_np))
            outputs.append(self._generate_norm(conds_np[start:end], n_samples=end - start))
        return (
            np.concatenate(outputs, axis=0)
            if outputs
            else np.empty((0, *self.sample_shape), dtype=np.float32)
        )

    def _norm_batch_to_log10(self, x_norm_np: np.ndarray) -> np.ndarray:
        """Convert a batch of normalized maps to log10(physical) space.

        Args:
            x_norm_np: (N,3,H,W) float32 numpy in normalized space.

        Returns:
            (N,3,H,W) float32 numpy in log10(physical) space.
        """
        t = torch.from_numpy(x_norm_np.astype(np.float32))
        with torch.no_grad():
            physical = self.normalizer.denormalize(t.to(self.device)).cpu()
        return torch.log10(torch.clamp(physical, min=1e-30)).numpy().astype(np.float32, copy=False)

    def _norm_batch_to_physical(self, x_norm_np: np.ndarray) -> np.ndarray:
        """Convert a batch of normalized maps to physical space (no log transform).

        Args:
            x_norm_np: (N,3,H,W) float32 numpy in normalized space.

        Returns:
            (N,3,H,W) float32 numpy in physical space.
        """
        t = torch.from_numpy(x_norm_np.astype(np.float32))
        with torch.no_grad():
            physical = self.normalizer.denormalize(t.to(self.device)).cpu()
        return physical.numpy().astype(np.float32, copy=False)

    def _compute_lh_metrics(
        self,
        true_batch: np.ndarray,
        gen_batch: np.ndarray,
        *,
        pdf_log: bool = False,
    ) -> dict:
        """Compute all LH evaluation metrics for a (true, generated) batch pair.

        Args:
            true_batch: (N,3,H,W) true maps.
            gen_batch: (N,3,H,W) generated maps.
            pdf_log: If True, convert to log10 before PDF comparison (use when
                inputs are in physical space). False when inputs are already log10.

        Returns:
            dict with keys: auto_power, cross_power, correlation, pdf,
            pdf_summary, pass_summary.
        """
        spectrum_errors = compute_spectrum_errors(
            true_batch, gen_batch, box_size=self.box_size
        )
        correlation_errors = compute_correlation_errors(
            true_batch, gen_batch, box_size=self.box_size
        )
        pdf_results = compare_pixel_distributions(true_batch, gen_batch, log=pdf_log)
        pdf_summary = compute_distribution_summary(true_batch, gen_batch)

        auto_power = {k: v for k, v in spectrum_errors.items() if v["type"] == "auto"}
        cross_power = {k: v for k, v in spectrum_errors.items() if v["type"] == "cross"}

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
        pass_summary["overall"] = all(pass_summary.values())

        return {
            "auto_power": auto_power,
            "cross_power": cross_power,
            "correlation": correlation_errors,
            "pdf": pdf_results,
            "pdf_summary": pdf_summary,
            "pass_summary": pass_summary,
        }

    def evaluate_lh(
        self,
        val_maps,
        val_params,
        max_samples: int = 200,
        verbose: bool = True,
    ) -> dict:
        """Evaluate on the LH (Latin Hypercube) validation set.

        Generates each map ONCE and computes metrics in both log10(physical) and
        physical space to avoid redundant model inference.

        Args:
            val_maps: (N, 3, H, W) normalized maps (torch or numpy).
            val_params: (N, 6) normalized cosmological parameters (torch or numpy).
            max_samples: Maximum number of validation samples to use.
            verbose: If True, print progress.

        Returns:
            dict with keys:
                "log10"    — metrics computed on log10(physical) fields (primary,
                             thresholds calibrated for this space).
                "physical" — metrics computed on raw physical fields (amplitude check;
                             thresholds are log10-calibrated, treat pass/fail as indicative).
            Each sub-dict has keys: auto_power, cross_power, correlation, pdf,
            pdf_summary, pass_summary.
        """
        val_maps_np = _to_numpy(val_maps).astype(np.float32, copy=False)
        val_params_np = _to_numpy(val_params)
        N = min(len(val_maps_np), max_samples)

        # Generate normalized samples ONCE — avoids double model inference
        gen_norm = self._generate_norm_in_chunks(
            val_params_np[:N], batch_size=8, desc="LH generate", enabled=verbose
        )

        # Convert true and generated maps to both spaces in one pass
        true_norm = val_maps_np[:N]
        if verbose:
            print("  [evaluate_lh] converting to log10 and physical space ...", flush=True)
        true_log10 = self._norm_batch_to_log10(true_norm)
        gen_log10  = self._norm_batch_to_log10(gen_norm)
        true_phys  = self._norm_batch_to_physical(true_norm)
        gen_phys   = self._norm_batch_to_physical(gen_norm)

        if verbose:
            print("  [evaluate_lh] computing log10 space metrics ...", flush=True)
        log10_results = self._compute_lh_metrics(true_log10, gen_log10, pdf_log=False)

        if verbose:
            print("  [evaluate_lh] computing physical space metrics ...", flush=True)
        phys_results = self._compute_lh_metrics(true_phys, gen_phys, pdf_log=True)

        return {"log10": log10_results, "physical": phys_results}

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
            gen_batch = self._generate_log10_in_chunks(
                params_np,
                batch_size=8,
                desc=f"1P generate {pname}",
                enabled=True,
            )

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
                    gen_l10 = gen_batch[i]
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
        or auto-power mean error > 2× the LH threshold for each channel).

        Args:
            ex_maps_norm: (N_ex, 3, H, W) normalized extreme-parameter maps.
            ex_params_norm: (N_ex, 6) or (N_cond, 6) normalized extreme parameters.
                If N_cond != N_ex, params are expanded to match maps.
            box_size: Override box size.

        Returns:
            dict with keys:
                "spectrum_errors": per-sample spectrum error results
                "has_nan": bool (any NaN in generated maps)
                "has_divergence": bool (any value > 1e10)
                "max_auto_error": dict {channel: float} max mean error across samples
                "auto_error_threshold_2x_lh": dict {channel: float} pass threshold
                "passed": bool (no NaN, no divergence, no catastrophic error)
        """
        if box_size is None:
            box_size = self.box_size

        ex_maps_np = _to_numpy(ex_maps_norm)
        ex_params_np = _to_numpy(ex_params_norm)
        N_ex = len(ex_maps_np)
        N_cond = len(ex_params_np)
        if N_cond <= 0:
            raise ValueError("EX params are empty.")

        # EX files may store one condition per extreme simulation (e.g., 4 rows),
        # while maps include multiple projections per simulation (e.g., 60 rows).
        if N_cond == N_ex:
            ex_cond_batch = ex_params_np
        elif N_ex % N_cond == 0:
            ex_cond_batch = np.repeat(ex_params_np, N_ex // N_cond, axis=0)
        else:
            reps = (N_ex + N_cond - 1) // N_cond
            ex_cond_batch = np.tile(ex_params_np, (reps, 1))[:N_ex]

        true_log10_list = [
            self._to_log10(ex_maps_np[i])
            for i in self._progress(range(N_ex), total=N_ex, desc="EX samples", enabled=True)
        ]
        true_batch = np.stack(true_log10_list, axis=0)
        gen_batch = self._generate_log10_in_chunks(
            ex_cond_batch,
            batch_size=8,
            desc="EX generate",
            enabled=True,
        )

        has_nan = bool(np.isnan(gen_batch).any())
        has_divergence = bool((np.abs(gen_batch) > 1e10).any())

        spectrum_errors = compute_spectrum_errors(true_batch, gen_batch, box_size=box_size)

        # Check: auto-power error < 2× LH targets (§4.5 EX criterion)
        max_auto_error = {}
        auto_error_threshold_2x_lh = {}
        catastrophic = False
        for ch in CHANNELS:
            me = spectrum_errors[ch]["mean_error"]
            max_auto_error[ch] = me
            lh_mean_threshold = max(
                (thr_mean for _, _, _, thr_mean, _ in AUTO_POWER_THRESHOLDS.get(ch, [])),
                default=1.0,
            )
            ex_threshold = 2.0 * lh_mean_threshold
            auto_error_threshold_2x_lh[ch] = ex_threshold
            if me > ex_threshold:
                catastrophic = True

        passed = bool(not has_nan and not has_divergence and not catastrophic)

        return {
            "spectrum_errors": spectrum_errors,
            "has_nan": has_nan,
            "has_divergence": has_divergence,
            "max_auto_error": max_auto_error,
            "auto_error_threshold_2x_lh": auto_error_threshold_2x_lh,
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

        # Aggregate key metrics across runs — use log10 space results
        summary = {"auto_power": {}, "cross_power": {}, "correlation": {}, "pdf": {}}

        # Auto-power mean errors
        for ch in CHANNELS:
            means = [r["log10"]["auto_power"][ch]["mean_error"] for r in runs]
            rms_vals = [r["log10"]["auto_power"][ch]["rms_error"] for r in runs]
            summary["auto_power"][ch] = {
                "mean_error_mean": float(np.mean(means)),
                "mean_error_std": float(np.std(means)),
                "rms_error_mean": float(np.mean(rms_vals)),
                "rms_error_std": float(np.std(rms_vals)),
            }

        # Cross-power mean errors
        for pair in ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]:
            means = [r["log10"]["cross_power"][pair]["mean_error"] for r in runs]
            summary["cross_power"][pair] = {
                "mean_error_mean": float(np.mean(means)),
                "mean_error_std": float(np.std(means)),
            }

        # Correlation max_delta_r
        for pair in ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]:
            vals = [r["log10"]["correlation"][pair]["max_delta_r"] for r in runs]
            summary["correlation"][pair] = {
                "max_delta_r_mean": float(np.mean(vals)),
                "max_delta_r_std": float(np.std(vals)),
            }

        # PDF KS statistic
        for ch in CHANNELS:
            vals = [r["log10"]["pdf"][ch]["ks_statistic"] for r in runs]
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

        val_maps_np = _to_numpy(val_maps)
        if val_maps_np.ndim == 4:
            self.sample_shape = tuple(int(v) for v in val_maps_np.shape[1:4])

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
