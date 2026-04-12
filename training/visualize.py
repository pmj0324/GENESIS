"""
GENESIS - EpochVisualizer

에폭 끝마다 호출:
  1. 샘플 맵         → plots/ep{N:04d}_samples.png       3 cols × (Real + viz samplers)
  2. Power spectrum  → plots/ep{N:04d}_power_spectrum.png
                       (각 채널별 상단 P(k) + 하단 |ΔP|/P_true 상대오차 패널,
                        여러 viz sampler curve 지원)
  3. Loss curve      → plots/loss.png                    (선형 + log-y loss; 매 에폭 덮어씀)
  4. Learning rate   → plots/lr.png                      (lr 히스토리 있을 때만; 매 에폭 덮어씀)

샘플/파워/메트릭은 기본적으로 best 갱신 시에만 생성되며,
viz.every_epoch=true 이면 매 epoch 생성된다.
요약 파일:
  - plots/best_samples.png
  - plots/best_power_spectrum.png
  - plots/latest_samples.png
  - plots/latest_power_spectrum.png
  best_*는 val loss best 갱신 시에만 갱신되고,
  latest_*는 가장 최근에 렌더링한 epoch 결과를 가리킨다.

conditioning:
  ref_cond = val_dataset[0][1]  →  val set 첫 번째 샘플의 z-score 정규화된 cosmological params
  순서: [Ω_m, σ_8, A_SN1, A_SN2, A_AGN1, A_AGN2] (IllustrisTNG LH suite)
  시각화 시 denormalize해서 실제 물리값으로 표시.

maps:
  denormalize 하면 physical linear field 값으로 복원
  샘플 맵은 viz.samples.field_space 설정(log / linear)에 따라 표시.

power spectrum:
  ref_map[0]: val set 첫 번째 실제 맵 (ref_cond와 동일 시뮬레이션)
  val avg   : val set 처음 N장 평균 P(k)
  viz samplers: ref_cond 조건으로 생성된 샘플들의 P(k)
  linear: denormalized physical field 그대로 P(k) 계산
  log   : denormalized field를 log10(max(x, 1e-30))로 변환 후 P(k) 계산
  매 에폭 linear/log를 2행으로 함께 표시
  estimator:
    - genesis (기본)
    - diffusion_hmc (호환 모드; linear=normalize=True, log=normalize=False)
"""

import json
import shutil
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from analysis.power_spectrum import compute_power_spectrum_2d
from analysis.cross_spectrum import compute_spectrum_errors
from analysis.correlation import compute_correlation_errors
from analysis.pixel_distribution import compare_pixel_distributions
from dataloader.normalization import (
    Normalizer,
    ParamNormalizer,
    denormalize_params,
    PARAM_NAMES,
)

CHANNEL_NAMES = ["Mcdm", "Mgas", "T"]
CHANNEL_LABELS = ["log₁₀(Mcdm)", "log₁₀(Mgas)", "log₁₀(T)"]
CMAPS         = ["viridis", "plasma", "inferno"]
SAMPLER_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
SAMPLER_LINESTYLES = ["-", "--", "-.", ":"]

SamplerFn = Callable[[torch.nn.Module, Tuple, torch.Tensor], torch.Tensor]
SamplerSpec = Tuple[str, SamplerFn]


class EpochVisualizer:
    """
    Args:
        sampler_a  : 기본 viz sampler (호환용)
        sampler_b  : 기본 metric sampler (호환용)
        samplers   : 샘플/파워 figure에 표시할 sampler 목록
        metric_sampler: per-epoch metric 계산에 사용할 sampler
        plot_dir   : 그림 저장 폴더
        ref_maps   : [N, 3, 256, 256]  실제 val 맵 (정규화된 상태)
        ref_cond   : [6]  고정 conditioning 벡터 (z-score 정규화)
        norm_cfg   : metadata.yaml의 normalization 섹션 (dict)
        device     : 샘플링 디바이스
    """

    def __init__(
        self,
        sampler_a:  SamplerSpec,
        sampler_b:  SamplerSpec,
        plot_dir,
        ref_maps,
        ref_cond,
        norm_cfg:   Dict,
        device,
        param_norm_cfg: Optional[Dict] = None,
        eval_conds: Optional[torch.Tensor] = None,   # [N, 6] — 에폭 메트릭용 N개 조건
        eval_n:     int = 15,                         # 메트릭 평가에 사용할 샘플 수
        viz_cfg:    Optional[Dict] = None,
        samplers: Optional[list[SamplerSpec]] = None,
        metric_sampler: Optional[SamplerSpec] = None,
    ):
        self.samplers = list(samplers) if samplers is not None else [sampler_a, sampler_b]
        if len(self.samplers) == 0:
            raise ValueError("At least one visualization sampler is required.")

        self.metric_sampler = metric_sampler or sampler_b
        self.metric_name, self.metric_fn = self.metric_sampler
        self.name_a, self.fn_a = self.samplers[0]
        self.name_b, self.fn_b = self.metric_sampler
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.device   = device
        self.viz_cfg  = viz_cfg or {}
        self.viz_every_epoch = bool(self.viz_cfg.get("every_epoch", False))

        # normalizer (maps denormalize 용)
        self.normalizer = Normalizer(norm_cfg) if norm_cfg else None
        self.param_normalizer = ParamNormalizer(param_norm_cfg) if param_norm_cfg else None

        samples_cfg = self.viz_cfg.get("samples", {})
        power_cfg = self.viz_cfg.get("power", {})
        metrics_cfg = self.viz_cfg.get("metrics", {})
        self.samples_field_space = self._normalize_field_space(
            samples_cfg.get("field_space", "log")
        )
        requested_power_spaces = power_cfg.get("field_spaces")
        if requested_power_spaces is None:
            requested_power_spaces = [
                power_cfg.get("field_space", "log"),
                *(power_cfg.get("extra_field_spaces", ["linear"]) or []),
            ]
        if isinstance(requested_power_spaces, str):
            requested_power_spaces = [requested_power_spaces]

        power_spaces: list[str] = []
        for space in requested_power_spaces:
            norm_space = self._normalize_field_space(space)
            if norm_space not in power_spaces:
                power_spaces.append(norm_space)
        # From now on, always render both spaces in the same figure.
        for required in ("linear", "log"):
            if required not in power_spaces:
                power_spaces.append(required)
        self.power_plot_spaces = power_spaces
        self.power_spectrum_estimator = self._normalize_power_spectrum_estimator(
            power_cfg.get("estimator", "genesis")
        )
        self.metrics_field_space = self._normalize_field_space(
            metrics_cfg.get("field_space", "log")
        )

        # ref_maps → numpy [N, 3, 256, 256] (정규화 상태)
        if hasattr(ref_maps, "cpu"):
            ref_maps = ref_maps.cpu().numpy()
        ref_maps_np = np.asarray(ref_maps, dtype=np.float32)
        if ref_maps_np.ndim == 3:
            ref_maps_np = ref_maps_np[None, ...]
        if ref_maps_np.ndim != 4:
            raise ValueError(f"ref_maps must be [N,3,H,W] or [3,H,W], got shape={ref_maps_np.shape}")

        requested_n = max(1, int(eval_n))
        available_n = len(ref_maps_np)
        use_n = min(requested_n, available_n)
        if use_n < requested_n:
            print(
                f"  [viz] eval_n fallback: requested={requested_n}, "
                f"available={available_n} -> using N={use_n}",
                flush=True,
            )

        self.ref_maps_norm = ref_maps_np[:use_n]
        # denormalize → physical linear field 값
        self.ref_maps_linear = self._denorm_maps(self.ref_maps_norm)

        # ref_cond: z-score 정규화된 값 보관
        if hasattr(ref_cond, "cpu"):
            ref_cond = ref_cond.cpu()
        self.ref_cond_norm = ref_cond.numpy().flatten()          # [6] 정규화값
        self.ref_cond_raw  = self._denorm_params(ref_cond).numpy().flatten()  # [6] 실제값
        self.ref_cond      = ref_cond.view(1, -1).float().to(device)

        # 에폭 메트릭용: N개의 (맵, 조건) 쌍
        n = min(use_n, len(self.ref_maps_norm))
        self.eval_maps_norm = self.ref_maps_norm[:n]          # [N, 3, H, W] normalized
        if eval_conds is not None:
            ec = eval_conds[:n]
            self.eval_conds = (ec.cpu() if hasattr(ec, "cpu") else torch.from_numpy(ec)).float().to(device)
        else:
            # fallback: ref_cond 반복
            self.eval_conds = self.ref_cond.repeat(n, 1)

    # ── Denormalize helpers ────────────────────────────────────────────────────

    @staticmethod
    def _normalize_field_space(field_space: str) -> str:
        value = str(field_space).strip().lower()
        aliases = {
            "log10": "log",
            "log_field": "log",
            "log-field": "log",
            "linear_field": "linear",
            "linear-field": "linear",
        }
        value = aliases.get(value, value)
        if value not in {"log", "linear"}:
            raise ValueError(f"Unknown field_space: {field_space!r}. Options: log / linear")
        return value

    def _denorm_maps(self, maps_norm: np.ndarray) -> np.ndarray:
        """[N, 3, H, W] or [3, H, W] 정규화 → physical linear field"""
        if self.normalizer is None:
            return maps_norm
        t = torch.from_numpy(maps_norm)
        squeezed = t.ndim == 3
        if squeezed:
            t = t.unsqueeze(0)
        out = self.normalizer.denormalize(t).numpy()
        return out[0] if squeezed else out

    def _denorm_params(self, params_norm: torch.Tensor) -> torch.Tensor:
        """[6] or [B, 6] normalized params → raw physical params."""
        if self.param_normalizer is not None:
            arr = params_norm.detach().cpu().numpy() if hasattr(params_norm, "detach") else np.asarray(params_norm)
            out = self.param_normalizer.denormalize_numpy(arr)
            return torch.as_tensor(out, dtype=torch.float32)
        return denormalize_params(params_norm)

    def _to_field_space(self, maps_linear: np.ndarray, field_space: str) -> np.ndarray:
        maps_linear = np.asarray(maps_linear, dtype=np.float32)
        if field_space == "log":
            return np.log10(np.clip(maps_linear, 1e-30, None))
        return maps_linear

    def _field_space_from_norm(self, maps_norm: np.ndarray, field_space: str) -> np.ndarray:
        return self._to_field_space(self._denorm_maps(maps_norm), field_space)

    def _field_space_label(self, field_space: str) -> str:
        return "log10 field" if field_space == "log" else "linear field"

    @staticmethod
    def _normalize_power_spectrum_estimator(value: str) -> str:
        """Normalize estimator alias for power spectrum plotting.

        Supported:
          - genesis
          - diffusion_hmc (diffusion-hmc compatibility)
        """
        mode = str(value).strip().lower()
        aliases = {
            "default": "genesis",
            "genesis_default": "genesis",
            "diffusion-hmc": "diffusion_hmc",
            "dhmc": "diffusion_hmc",
            "compat": "diffusion_hmc",
            "compatibility": "diffusion_hmc",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"genesis", "diffusion_hmc"}:
            raise ValueError(
                f"Unknown power estimator: {value!r}. "
                "Options: genesis / diffusion_hmc"
            )
        return mode

    def _power_estimator_label(self) -> str:
        return "GENESIS" if self.power_spectrum_estimator == "genesis" else "diffusion-hmc compatible"

    def _compute_power_spectrum(self, field: np.ndarray, field_space: str) -> tuple[np.ndarray, np.ndarray]:
        """Compute P(k) with selected estimator.

        diffusion-hmc 호환 모드에서는 원본 비교 코드 관례를 따른다:
          - linear field  -> normalize=True
          - log10 field   -> normalize=False
        """
        if self.power_spectrum_estimator == "genesis":
            return compute_power_spectrum_2d(field)

        use_norm = field_space == "linear"
        return compute_power_spectrum_2d(
            field,
            mode="diffusion_hmc",
            diffusion_hmc_normalize=use_norm,
            diffusion_hmc_smoothed=0.25,
        )

    def _plot_positive_loglog(self, ax, k: np.ndarray, pk: np.ndarray, *args, **kwargs) -> None:
        mask = np.isfinite(k) & np.isfinite(pk) & (k > 0) & (pk > 0)
        if mask.sum() == 0:
            return
        ax.loglog(k[mask], pk[mask], *args, **kwargs)

    @staticmethod
    def _compute_relative_error_percent(
        k_ref: np.ndarray,
        pk_ref: np.ndarray,
        k_cmp: np.ndarray,
        pk_cmp: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return |P_cmp - P_ref| / |P_ref| * 100 on a shared positive-k grid."""
        k_ref = np.asarray(k_ref, dtype=np.float64)
        pk_ref = np.asarray(pk_ref, dtype=np.float64)
        k_cmp = np.asarray(k_cmp, dtype=np.float64)
        pk_cmp = np.asarray(pk_cmp, dtype=np.float64)

        ref_mask = np.isfinite(k_ref) & np.isfinite(pk_ref) & (k_ref > 0)
        cmp_mask = np.isfinite(k_cmp) & np.isfinite(pk_cmp) & (k_cmp > 0)
        if ref_mask.sum() == 0 or cmp_mask.sum() == 0:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

        k_ref_valid = k_ref[ref_mask]
        pk_ref_valid = pk_ref[ref_mask]
        k_cmp_valid = k_cmp[cmp_mask]
        pk_cmp_valid = pk_cmp[cmp_mask]

        if (
            k_ref_valid.shape == k_cmp_valid.shape
            and np.allclose(k_ref_valid, k_cmp_valid, rtol=1e-6, atol=0.0)
        ):
            pk_cmp_eval = pk_cmp_valid
            k_eval = k_ref_valid
            pk_ref_eval = pk_ref_valid
        else:
            k_lo = max(float(k_ref_valid.min()), float(k_cmp_valid.min()))
            k_hi = min(float(k_ref_valid.max()), float(k_cmp_valid.max()))
            overlap = (k_ref_valid >= k_lo) & (k_ref_valid <= k_hi)
            if overlap.sum() == 0:
                return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
            k_eval = k_ref_valid[overlap]
            pk_ref_eval = pk_ref_valid[overlap]
            pk_cmp_eval = np.interp(k_eval, k_cmp_valid, pk_cmp_valid)

        denom = np.clip(np.abs(pk_ref_eval), 1e-30, None)
        rel_err = np.abs(pk_cmp_eval - pk_ref_eval) / denom * 100.0
        mask = np.isfinite(k_eval) & np.isfinite(rel_err) & (k_eval > 0)
        return k_eval[mask], rel_err[mask]

    def _cond_str(self) -> str:
        """실제 cosmological parameter 값 한 줄 문자열"""
        return "  ".join(
            f"{n}={v:.4f}" for n, v in zip(PARAM_NAMES, self.ref_cond_raw)
        )

    @staticmethod
    def _tensor_to_numpy(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    # ── Main call ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def __call__(self, epoch: int, model, history: dict, is_best: bool = False):
        model.eval()
        shape = (1, 3, 256, 256)

        try:
            # loss/lr plot은 매 epoch 업데이트
            self._plot_loss(history)

            if not self.viz_every_epoch and not is_best:
                print(f"  [viz] skip heavy plots at epoch {epoch+1:04d} (not best)", flush=True)
                return

            sampled_maps: list[tuple[str, np.ndarray]] = []
            for sampler_name, sampler_fn in self.samplers:
                print(f"  [viz] {sampler_name} sampling ...", flush=True)
                try:
                    raw_sample = sampler_fn(model, shape, self.ref_cond)
                    raw_sample = raw_sample[0] if isinstance(raw_sample, tuple) else raw_sample
                    raw_sample = self._tensor_to_numpy(raw_sample)
                    if raw_sample.ndim == 4 and raw_sample.shape[0] == 1:
                        raw_sample = raw_sample[0]
                    sampled_maps.append((sampler_name, self._denorm_maps(raw_sample)))
                except Exception as sampler_exc:
                    print(
                        f"  [viz] WARNING: {sampler_name} sampling failed at "
                        f"epoch {epoch+1:04d} — {sampler_exc}",
                        flush=True,
                    )

            if sampled_maps:
                self._plot_samples(epoch, sampled_maps, is_best=is_best)
                self._plot_power_spectrum(
                    epoch,
                    sampled_maps,
                    self.power_plot_spaces,
                    is_best=is_best,
                )
            else:
                print(
                    f"  [viz] WARNING: all visualization samplers failed at "
                    f"epoch {epoch+1:04d}; sample/power plots skipped",
                    flush=True,
                )

            self._print_metrics(epoch, model, history)

            torch.cuda.empty_cache()
            mode = "epoch-update" if self.viz_every_epoch else "best-update"
            if not sampled_maps:
                mode = f"{mode}, metrics-only"
            print(f"  [viz] saved({mode}) → {self.plot_dir}", flush=True)
        except Exception as e:
            print(f"  [viz] WARNING: plot failed at epoch {epoch+1} — {e}", flush=True)
            plt.close("all")
        finally:
            model.train()

    # ── 1. 샘플 맵 (3×3) ──────────────────────────────────────────────────────

    def _update_aliases(self, src_path: Path, *, best_name: str, latest_name: str, is_best: bool) -> None:
        if is_best:
            shutil.copy(src_path, self.plot_dir / best_name)
        shutil.copy(src_path, self.plot_dir / latest_name)

    def _plot_samples(
        self,
        epoch: int,
        sampled_maps: list[tuple[str, np.ndarray]],
        *,
        is_best: bool,
    ):
        """
        n rows × 3 cols
          rows : Real val[0] | viz samplers...
          cols : Mcdm | Mgas | T
          각 subplot 독립 colorbar
        """
        field_space = self.samples_field_space
        real = self.ref_maps_linear[0]   # [3, 256, 256] physical linear field
        rows = [("Real (val[0])", real), *sampled_maps]
        n_rows = len(rows)

        fig, axes = plt.subplots(n_rows, 3, figsize=(14, max(4.8, 3.5 * n_rows)), squeeze=False)
        fig.suptitle(
            f"Epoch {epoch+1:04d}  –  Field  [{self._field_space_label(field_space)}]\n"
            f"{self._cond_str()}",
            fontsize=10,
        )

        ref_field = self._to_field_space(real, field_space)
        ch_ranges = []
        for ci in range(3):
            d = ref_field[ci][np.isfinite(ref_field[ci])]
            vmin = float(np.percentile(d, 1))  if len(d) else -1.0
            vmax = float(np.percentile(d, 99)) if len(d) else  1.0
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                vmin, vmax = -1.0, 1.0
            ch_ranges.append((vmin, vmax))

        for ri, (row_label, img) in enumerate(rows):
            for ci, (ch_label, cmap) in enumerate(zip(CHANNEL_LABELS, CMAPS)):
                ax  = axes[ri, ci]
                vmin, vmax = ch_ranges[ci]
                data = self._to_field_space(img[ci], field_space)
                data = np.clip(data, vmin - 3, vmax + 3)   # 극단값 표시는 허용하되 무한대 방지
                data = np.where(np.isfinite(data), data, vmin)
                im  = ax.imshow(data, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
                ax.set_title(f"{row_label}  –  {ch_label}", fontsize=9)
                ax.axis("off")
                fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

        path = self.plot_dir / f"ep{epoch+1:04d}_samples.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        self._update_aliases(
            path,
            best_name="best_samples.png",
            latest_name="latest_samples.png",
            is_best=is_best,
        )
        plt.close(fig)

    # ── 2. Power spectrum ──────────────────────────────────────────────────────

    def _plot_power_spectrum(
        self,
        epoch: int,
        sampled_maps: list[tuple[str, np.ndarray]],
        field_spaces: list[str],
        *,
        is_best: bool,
    ):
        """
        linear/log field space 기준 P(k) 비교 + 하단 relative error 패널
          True mean    : ref_cond와 동일 시뮬레이션의 실제 맵 N장 평균 P(k)
          True 16-84%  : same-condition realization band
          viz samplers : ref_cond 조건으로 생성된 샘플
        """
        n_rows = len(field_spaces)
        n_ref = int(len(self.ref_maps_linear))
        height_ratios = []
        n_plot_rows = n_rows * 2 + max(0, n_rows - 1)
        for row_idx, _ in enumerate(field_spaces):
            height_ratios.extend([3.5, 1.2])
            if row_idx < n_rows - 1:
                # Add a dedicated spacer row so the upper error-panel xlabel
                # never collides with the next field-space title.
                height_ratios.append(0.5)
        fig, axes = plt.subplots(
            n_plot_rows,
            3,
            figsize=(15.5, 5.6 * n_rows),
            squeeze=False,
            sharex="col",
            gridspec_kw={
                "height_ratios": height_ratios,
                "hspace": 0.12,
                "wspace": 0.20,
            },
        )
        fig.suptitle(
            f"Epoch {epoch+1:04d}  –  Power Spectrum + Relative Error  "
            f"[linear + log10 | {self._power_estimator_label()}]\n"
            f"cond: {self._cond_str()}",
            fontsize=9,
            y=0.992,
        )

        spacer_rows = range(2, n_plot_rows, 3)
        for spacer_row in spacer_rows:
            for c in range(3):
                axes[spacer_row, c].axis("off")

        for row, field_space in enumerate(field_spaces):
            base_row = row * 3
            for c, ch_name in enumerate(CHANNEL_NAMES):
                ax = axes[base_row, c]
                err_ax = axes[base_row + 1, c]
                ref_all = [self._to_field_space(m[c], field_space) for m in self.ref_maps_linear]

                true_curves = np.asarray(
                    [self._compute_power_spectrum(m, field_space)[1] for m in ref_all],
                    dtype=np.float64,
                )
                k_true, _ = self._compute_power_spectrum(ref_all[0], field_space)
                pk_true_mean = true_curves.mean(axis=0)
                pk_true_lo = np.percentile(true_curves, 16, axis=0)
                pk_true_hi = np.percentile(true_curves, 84, axis=0)

                pos_band = (
                    (k_true > 0)
                    & np.isfinite(pk_true_lo)
                    & np.isfinite(pk_true_hi)
                    & (pk_true_hi > 0)
                )
                if np.any(pos_band):
                    ax.fill_between(
                        k_true[pos_band],
                        np.clip(pk_true_lo[pos_band], 1e-30, None),
                        np.clip(pk_true_hi[pos_band], 1e-30, None),
                        color="black",
                        alpha=0.16,
                        linewidth=0,
                        label=f"True 16-84% (N={n_ref})",
                    )
                self._plot_positive_loglog(ax, k_true, pk_true_mean, "k-", lw=2.0, label=f"True mean (N={n_ref})")

                err_max = 0.0
                for si, (sampler_name, sample_map) in enumerate(sampled_maps):
                    sample_field = self._to_field_space(sample_map[c], field_space)
                    k_s, pk_s = self._compute_power_spectrum(sample_field, field_space)
                    color = SAMPLER_COLORS[si % len(SAMPLER_COLORS)]
                    linestyle = SAMPLER_LINESTYLES[(si // len(SAMPLER_COLORS)) % len(SAMPLER_LINESTYLES)]
                    self._plot_positive_loglog(
                        ax,
                        k_s,
                        pk_s,
                        color=color,
                        linestyle=linestyle,
                        lw=1.5,
                        label=f"{sampler_name} (generated)",
                    )

                    k_err, rel = self._compute_relative_error_percent(k_true, pk_true_mean, k_s, pk_s)
                    if len(rel) > 0:
                        err_ax.semilogx(k_err, rel, color=color, linestyle=linestyle, lw=1.3)
                        err_max = max(err_max, float(np.nanmax(rel)))

                ax.set_title(f"{ch_name} [{self._field_space_label(field_space)}]", fontsize=10)
                ax.set_ylabel("P(k)")
                ax.legend(
                    fontsize=6 if len(sampled_maps) > 3 else 7,
                    loc="upper right",
                    framealpha=0.9,
                    ncol=2 if len(sampled_maps) > 4 else 1,
                )
                ax.grid(True, alpha=0.3, which="both")
                ax.tick_params(axis="x", labelbottom=False)

                err_ax.axhline(0.0, color="black", lw=0.8, alpha=0.6)
                err_ax.grid(True, alpha=0.3, which="both")
                if row == n_rows - 1:
                    err_ax.set_xlabel("k  [h/Mpc]")
                else:
                    err_ax.set_xlabel("")
                    err_ax.tick_params(axis="x", labelbottom=False)
                if c == 0:
                    err_ax.set_ylabel(r"$|\Delta P| / P_{\rm true}$ [%]")

                if np.isfinite(err_max) and err_max > 0:
                    err_ax.set_ylim(0.0, err_max * 1.15)
                else:
                    err_ax.set_ylim(0.0, 1.0)

        fig.subplots_adjust(left=0.06, right=0.985, bottom=0.07, top=0.89)
        path = self.plot_dir / f"ep{epoch+1:04d}_power_spectrum.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        self._update_aliases(
            path,
            best_name="best_power_spectrum.png",
            latest_name="latest_power_spectrum.png",
            is_best=is_best,
        )
        plt.close(fig)

    # ── 3. Per-epoch Metrics ───────────────────────────────────────────────────

    @torch.no_grad()
    def _print_metrics(self, epoch: int, model, history: dict):
        """
        eval_maps (N개) 각각에 대해 metric sampler로 생성 → log10 공간에서 메트릭 계산 후 출력.
        기본은 viz.sampler_b 또는 별도 metric_sampler.
        """
        try:
            N = self.eval_maps_norm.shape[0]

            # ── 배치 샘플링 (metric sampler 사용) ─────────────────────────────
            print(f"  [metrics] {self.metric_name} sampling (N={N}) ...", flush=True)
            shape   = (N, 3, self.eval_maps_norm.shape[-2], self.eval_maps_norm.shape[-1])
            raw_gen = self.metric_fn(model, shape, self.eval_conds)   # [N, 3, H, W] normalized
            raw_gen = raw_gen[0] if isinstance(raw_gen, tuple) else raw_gen

            # ── 평가 공간 변환 ────────────────────────────────────────────────
            field_space = self.metrics_field_space
            true_eval = self._field_space_from_norm(self.eval_maps_norm, field_space)   # [N, 3, H, W]
            gen_eval = self._field_space_from_norm(
                self._tensor_to_numpy(raw_gen),
                field_space,
            )

            # ── Auto/Cross Power Spectrum ──────────────────────────────────────
            spec_err  = compute_spectrum_errors(true_eval, gen_eval)
            corr_err  = compute_correlation_errors(true_eval, gen_eval)
            pdf_res   = compare_pixel_distributions(true_eval, gen_eval, log=False, ks_subsample=20000)

            # ── 출력 ──────────────────────────────────────────────────────────
            sep = "─" * 78
            print(f"\n  {sep}")
            print(f"  Epoch {epoch+1:04d} Metrics  (N={N}, {self.metric_name})  [{self._field_space_label(field_space)}]")
            print(f"  {sep}")

            # Auto-Power (scale-dependent thresholds, §4.1)
            print("  Auto-Power  [mean | max | rms]  (field+scale-dependent thresholds):")
            for ch in ["Mcdm", "Mgas", "T"]:
                e = spec_err[ch]
                ok = "✓" if e["passed"] else "✗"
                print(f"    {ch:<6}  {e['mean_error']*100:5.1f}%  {e['max_error']*100:5.1f}%  {e['rms_error']*100:5.1f}%  {ok}")
                # Show per-range breakdown
                scale_errors = e.get("scale_errors", {})
                for label, se in scale_errors.items():
                    se_ok = "✓" if se["passed"] else "✗"
                    print(f"      {label:>5}: mean={se['mean_error']*100:5.1f}%(<{se['threshold_mean']*100:.0f}%)  rms={se['rms_error']*100:5.1f}%(<{se['threshold_rms']*100:.0f}%)  {se_ok}")

            # Cross-Power (pair-dependent thresholds, §4.2)
            # ── [OLD] print("  Cross-Power [mean rel.err]:")
            # ── [OLD] uniform threshold: 10%
            print("  Cross-Power [mean rel.err]  (pair-dependent thresholds):")
            for pair in ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]:
                e = spec_err[pair]
                ok = "✓" if e["passed"] else "✗"
                thr = e.get("threshold", 0.15) * 100
                print(f"    {pair:<12}  {e['mean_error']*100:5.1f}% (<{thr:.0f}%)  {ok}")

            # Correlation (pair-dependent, Data-Driven Table 8)
            # [OLD] print("  Correlation [max Δr]  (k<5: <0.1, k≥5: <0.2):")
            print("  Correlation [max Δr]  (Mcdm-Mgas: <0.1 | T-pairs: <0.3):")
            for pair in ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]:
                c = corr_err[pair]
                ok = "✓" if c["passed"] else "✗"
                print(f"    {pair:<12}  Δr={c['max_delta_r']:.3f}  {ok}")
                scale_errors = c.get("scale_errors", {})
                for label, se in scale_errors.items():
                    se_ok = "✓" if se["passed"] else "✗"
                    print(f"      {label:>5}: Δr={se['max_delta_r']:.3f} (<{se['threshold']:.1f})  {se_ok}")

            # PDF (§4.4: KS D < 0.05 primary criterion)
            # ── [OLD] print("  PDF KS-test [statistic | p-value]:")
            # ── [OLD] criterion: p-value > 0.05
            print("  PDF  [KS D<0.05 | μ_err<5% | σ_err<10%]:")
            for ch in ["Mcdm", "Mgas", "T"]:
                p = pdf_res[ch]
                ok = "✓" if p["passed"] else "✗"
                mean_rel = p.get("mean_rel_error", float("nan")) * 100
                std_rel = p.get("std_rel_error", float("nan")) * 100
                print(f"    {ch:<6}  D={p['ks_statistic']:.3f}  μ={mean_rel:.1f}%  σ={std_rel:.1f}%  {ok}")

            print(f"  {sep}\n")

            # ── JSON 저장 ────────────────────────────────────────────────────
            self._save_metrics_json(epoch, N, spec_err, corr_err, pdf_res, history)

        except Exception as e:
            print(f"  [metrics] WARNING: metric computation failed — {e}", flush=True)

    def _save_metrics_json(
        self,
        epoch: int,
        n_samples: int,
        spec_err: dict,
        corr_err: dict,
        pdf_res: dict,
        history: dict,
    ) -> None:
        """메트릭 JSON 저장.

        저장 위치:
          - {plot_dir}/../metrics_history.json
          - {plot_dir}/../metrics_best.json

        기본값은 best(val_loss 최소) 기준 1개만 유지하고,
        viz.every_epoch=true 이면 epoch별 누적 기록을 유지한다.
        """
        # ── Auto-Power ────────────────────────────────────────────────────────
        auto = {}
        for ch in ["Mcdm", "Mgas", "T"]:
            e = spec_err[ch]
            auto[ch] = {
                "mean_error": round(e["mean_error"], 5),
                "max_error":  round(e["max_error"],  5),
                "rms_error":  round(e["rms_error"],  5),
                "passed":     e["passed"],
                "scale_errors": {
                    label: {
                        "mean_error": round(se["mean_error"], 5),
                        "rms_error":  round(se["rms_error"],  5),
                        "threshold_mean": se["threshold_mean"],
                        "threshold_rms":  se["threshold_rms"],
                        "passed": se["passed"],
                        "n_bins": se["n_bins"],
                    }
                    for label, se in e.get("scale_errors", {}).items()
                },
            }

        # ── Cross-Power ───────────────────────────────────────────────────────
        cross = {}
        for pair in ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]:
            e = spec_err[pair]
            cross[pair] = {
                "mean_error": round(e["mean_error"], 5),
                "max_error":  round(e["max_error"],  5),
                "rms_error":  round(e["rms_error"],  5),
                "threshold":  e.get("threshold", None),
                "passed":     e["passed"],
            }

        # ── Correlation ───────────────────────────────────────────────────────
        corr = {}
        for pair in ["Mcdm-Mgas", "Mcdm-T", "Mgas-T"]:
            c = corr_err[pair]
            corr[pair] = {
                "max_delta_r": round(c["max_delta_r"], 5),
                "passed":      c["passed"],
                "scale_errors": {
                    label: {
                        "max_delta_r": round(se["max_delta_r"], 5),
                        "threshold":   se["threshold"],
                        "passed":      se["passed"],
                    }
                    for label, se in c.get("scale_errors", {}).items()
                },
            }

        # ── PDF ───────────────────────────────────────────────────────────────
        pdf = {}
        for ch in ["Mcdm", "Mgas", "T"]:
            p = pdf_res[ch]
            pdf[ch] = {
                "ks_statistic":  round(float(p["ks_statistic"]), 5),
                "mean_rel_error": round(float(p.get("mean_rel_error", float("nan"))), 5),
                "std_rel_error":  round(float(p.get("std_rel_error",  float("nan"))), 5),
                "passed": p["passed"],
            }

        # ── overall pass ──────────────────────────────────────────────────────
        passed_overall = (
            all(auto[ch]["passed"] for ch in auto)
            and all(cross[p]["passed"] for p in cross)
            and all(corr[p]["passed"] for p in corr)
            and all(pdf[ch]["passed"] for ch in pdf)
        )

        # 현재 epoch/최적 epoch 정보(학습 loss history 기준)
        val_hist = history.get("val_loss", []) if isinstance(history, dict) else []
        current_val_loss = float("nan")
        best_epoch_by_val = epoch + 1
        best_val_loss = float("nan")
        if isinstance(val_hist, list) and len(val_hist) > 0:
            arr = np.asarray(val_hist, dtype=np.float64)
            if epoch < len(arr):
                current_val_loss = float(arr[epoch])
            finite = np.isfinite(arr)
            if finite.any():
                best_idx = int(np.nanargmin(arr))
                best_epoch_by_val = best_idx + 1
                best_val_loss = float(arr[best_idx])

        record = {
            "epoch":          epoch + 1,
            "val_loss":       current_val_loss,
            "best_epoch_by_val_loss": best_epoch_by_val,
            "best_val_loss":  best_val_loss,
            "n_samples":      n_samples,
            "power_spectrum_estimator": self.power_spectrum_estimator,
            "auto_power":     auto,
            "cross_power":    cross,
            "correlation":    corr,
            "pdf":            pdf,
            "passed_overall": passed_overall,
        }

        # ── 기존 기록 로드 (list/dict 모두 허용) ───────────────────────────────
        metrics_path = self.plot_dir.parent / "metrics_history.json"
        records: list[dict] = []
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        records = [x for x in loaded if isinstance(x, dict)]
                    elif isinstance(loaded, dict):
                        records = [loaded]
            except Exception:
                records = []

        # epoch 중복 방지: 같은 epoch 있으면 덮어씀
        records = [r for r in records if int(r.get("epoch", -1)) != int(record["epoch"])]
        records.append(record)
        records.sort(key=lambda r: int(r.get("epoch", 0)))

        # ── best(val_loss 최소) epoch 레코드 계산 ─────────────────────────────
        best_record = None
        for r in records:
            if int(r.get("epoch", -1)) == int(best_epoch_by_val):
                best_record = r
                break
        if best_record is None and len(records) > 0:
            # 이론상 거의 안 오지만, epoch 누락 시 근접 epoch fallback
            best_record = min(
                records,
                key=lambda r: abs(int(r.get("epoch", 0)) - int(best_epoch_by_val)),
            )
        if best_record is None:
            best_record = record

        history_payload = records if self.viz_every_epoch else [best_record]
        with open(metrics_path, "w") as f:
            json.dump(history_payload, f, indent=2, allow_nan=True)

        best_path = self.plot_dir.parent / "metrics_best.json"
        with open(best_path, "w") as f:
            json.dump(best_record, f, indent=2, allow_nan=True)

        save_mode = "per-epoch" if self.viz_every_epoch else "best-only"
        print(f"  [metrics] saved({save_mode}) → {metrics_path}", flush=True)

    # ── 4. Loss curve ──────────────────────────────────────────────────────────

    def _plot_loss(self, history: dict):
        train = history["train_loss"]
        val = history["val_loss"]
        n = len(train)
        if n == 0:
            return
        epochs = np.arange(1, n + 1, dtype=float)
        lr_hist = history.get("lr", [])
        lr_ok = len(lr_hist) == n

        fig, (ax_lin, ax_log) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        ax_lin.plot(epochs, train, label="train", color="steelblue", linewidth=1.5)
        ax_lin.plot(epochs, val, label="val", color="tomato", linewidth=1.5)
        ax_lin.set_ylabel("Loss")
        ax_lin.set_title("Training loss (linear)")
        ax_lin.grid(True, alpha=0.3)
        ax_lin.legend(loc="upper right")

        eps = 1e-12
        ax_log.semilogy(epochs, np.maximum(train, eps), label="train", color="steelblue", linewidth=1.5)
        ax_log.semilogy(epochs, np.maximum(val, eps), label="val", color="tomato", linewidth=1.5)
        ax_log.set_xlabel("Epoch")
        ax_log.set_ylabel("Loss (log scale)")
        ax_log.set_title("Training loss (log y)")
        ax_log.legend(loc="upper right")
        ax_log.grid(True, alpha=0.3, which="both")

        fig.tight_layout()
        fig.savefig(self.plot_dir / "loss.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

        if lr_ok:
            fig_lr, ax_lr = plt.subplots(figsize=(8, 4))
            ax_lr.plot(epochs, lr_hist, color="tab:green", linewidth=1.5)
            ax_lr.set_xlabel("Epoch")
            ax_lr.set_ylabel("Learning rate")
            ax_lr.set_title("Learning rate")
            ax_lr.set_yscale("log")
            ax_lr.grid(True, alpha=0.3, which="both")
            fig_lr.tight_layout()
            fig_lr.savefig(self.plot_dir / "lr.png", dpi=100, bbox_inches="tight")
            plt.close(fig_lr)
