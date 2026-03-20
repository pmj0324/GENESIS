"""
GENESIS - CAMELS Protocol Evaluator

LH/1P/CV/EX 프로토콜 기반 생성 모델 평가.

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
from typing import Callable, Optional

from .cross_spectrum import _to_numpy, compute_spectrum_errors, CHANNELS
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

        for i in range(N):
            if verbose and (i % 10 == 0 or i == N - 1):
                print(f"  [evaluate_lh] {i+1}/{N} ...", flush=True)

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
        for i in range(N_cv):
            true_log10_list.append(self._to_log10(cv_maps_np[i]))
        true_batch = np.stack(true_log10_list, axis=0)  # (N_cv, 3, H, W)

        # Generated log10 maps
        gen_batch = self._generate_log10(fiducial_cond_norm, n_samples=N_cv)  # (N_cv, 3, H, W)

        from .cross_spectrum import compute_cross_power_spectrum_2d
        results = {}

        for ci, ch in enumerate(CHANNELS):
            # Per-sample P(k) for true and gen
            pk_true_list = []
            pk_gen_list = []
            k_centers = None

            for b in range(N_cv):
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

            # passed: 0.8 < var_ratio < 1.2 for >80% of k bins
            frac_in_band = float(((var_ratio > 0.8) & (var_ratio < 1.2)).mean())
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

    def run_all(
        self,
        val_maps,
        val_params,
        cv_maps=None,
        fiducial_cond=None,
        protocols=None,
    ) -> dict:
        """Run all specified evaluation protocols.

        Args:
            val_maps: (N, 3, H, W) normalized validation maps.
            val_params: (N, 6) normalized cosmological parameters.
            cv_maps: (N_cv, 3, H, W) CV set maps (required for "cv" protocol).
            fiducial_cond: [6] fiducial condition (required for "cv" protocol).
            protocols: List of protocol names to run. Default: ["lh"].

        Returns:
            dict with protocol names as keys and their results as values.
        """
        if protocols is None:
            protocols = ["lh"]

        all_results = {}

        if "lh" in protocols:
            print("[run_all] Running LH protocol ...", flush=True)
            all_results["lh"] = self.evaluate_lh(val_maps, val_params)

        if "cv" in protocols:
            if cv_maps is None or fiducial_cond is None:
                raise ValueError("cv_maps and fiducial_cond are required for CV protocol.")
            print("[run_all] Running CV protocol ...", flush=True)
            all_results["cv"] = self.evaluate_cv(cv_maps, fiducial_cond)

        return all_results
