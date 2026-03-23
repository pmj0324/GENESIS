"""
GENESIS - EpochVisualizer

에폭 끝마다 호출:
  1. 샘플 맵         → plots/ep{N:04d}_samples.png       3×3 grid (Real / sampler_a / sampler_b)
  2. Power spectrum  → plots/ep{N:04d}_power_spectrum.png
  3. Loss curve      → plots/loss.png                    (선형 + log-y loss; 매 에폭 덮어씀)
  4. Learning rate   → plots/lr.png                      (lr 히스토리 있을 때만; 매 에폭 덮어씀)

conditioning:
  ref_cond = val_dataset[0][1]  →  val set 첫 번째 샘플의 z-score 정규화된 cosmological params
  순서: [Ω_m, σ_8, A_SN1, A_SN2, A_AGN1, A_AGN2] (IllustrisTNG LH suite)
  시각화 시 denormalize해서 실제 물리값으로 표시.

maps:
  affine 정규화 → denormalize 하면 log10(field) 값으로 복원
  샘플 맵은 log10 스케일로 표시.

power spectrum:
  ref_map[0]: val set 첫 번째 실제 맵 (ref_cond와 동일 시뮬레이션)
  val avg   : val set 처음 8장 평균 P(k)
  sampler_a/b: ref_cond 조건으로 생성된 샘플의 P(k)
  → 모두 denormalize된 log10 맵 기준 P(k)
"""

import json
import shutil
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from analysis.power_spectrum import compute_power_spectrum_2d
from analysis.cross_spectrum import compute_spectrum_errors
from analysis.correlation import compute_correlation_errors
from analysis.pixel_distribution import compare_pixel_distributions
from dataloader.normalization import Normalizer, denormalize_params, PARAM_NAMES

CHANNEL_NAMES = ["Mcdm", "Mgas", "T"]
CHANNEL_LABELS = ["log₁₀(Mcdm)", "log₁₀(Mgas)", "log₁₀(T)"]
CMAPS         = ["viridis", "plasma", "inferno"]

SamplerFn = Callable[[torch.nn.Module, Tuple, torch.Tensor], torch.Tensor]


class EpochVisualizer:
    """
    Args:
        sampler_a  : (이름, 샘플 함수)  e.g. ("DDPM", ddpm_fn)
        sampler_b  : (이름, 샘플 함수)  e.g. ("DDIM", ddim_fn)
        plot_dir   : 그림 저장 폴더
        ref_maps   : [N, 3, 256, 256]  실제 val 맵 (정규화된 상태)
        ref_cond   : [6]  고정 conditioning 벡터 (z-score 정규화)
        norm_cfg   : metadata.yaml의 normalization 섹션 (dict)
        device     : 샘플링 디바이스
    """

    def __init__(
        self,
        sampler_a:  Tuple[str, SamplerFn],
        sampler_b:  Tuple[str, SamplerFn],
        plot_dir,
        ref_maps,
        ref_cond,
        norm_cfg:   Dict,
        device,
        eval_conds: Optional[torch.Tensor] = None,   # [N, 6] — 에폭 메트릭용 N개 조건
        eval_n:     int = 8,                          # 메트릭 평가에 사용할 샘플 수
    ):
        self.name_a, self.fn_a = sampler_a
        self.name_b, self.fn_b = sampler_b
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.device   = device

        # normalizer (maps denormalize 용)
        self.normalizer = Normalizer(norm_cfg) if norm_cfg else None

        # ref_maps → numpy [N, 3, 256, 256] (정규화 상태)
        if hasattr(ref_maps, "cpu"):
            ref_maps = ref_maps.cpu().numpy()
        self.ref_maps_norm = np.asarray(ref_maps[:8], dtype=np.float32)
        # denormalize → log10(field) 값
        self.ref_maps = self._denorm_maps(self.ref_maps_norm)

        # ref_cond: z-score 정규화된 값 보관
        if hasattr(ref_cond, "cpu"):
            ref_cond = ref_cond.cpu()
        self.ref_cond_norm = ref_cond.numpy().flatten()          # [6] 정규화값
        self.ref_cond_raw  = denormalize_params(ref_cond).numpy().flatten()  # [6] 실제값
        self.ref_cond      = ref_cond.view(1, -1).float().to(device)

        # 에폭 메트릭용: N개의 (맵, 조건) 쌍
        n = min(eval_n, len(self.ref_maps_norm))
        self.eval_maps_norm = self.ref_maps_norm[:n]          # [N, 3, H, W] normalized
        if eval_conds is not None:
            ec = eval_conds[:n]
            self.eval_conds = (ec.cpu() if hasattr(ec, "cpu") else torch.from_numpy(ec)).float().to(device)
        else:
            # fallback: ref_cond 반복
            self.eval_conds = self.ref_cond.repeat(n, 1)

    # ── Denormalize helpers ────────────────────────────────────────────────────

    def _denorm_maps(self, maps_norm: np.ndarray) -> np.ndarray:
        """[N, 3, H, W] or [3, H, W] 정규화 → log10(field)"""
        if self.normalizer is None:
            return maps_norm
        t = torch.from_numpy(maps_norm)
        squeezed = t.ndim == 3
        if squeezed:
            t = t.unsqueeze(0)
        out = self.normalizer.denormalize(t).numpy()
        return out[0] if squeezed else out

    def _cond_str(self) -> str:
        """실제 cosmological parameter 값 한 줄 문자열"""
        return "  ".join(
            f"{n}={v:.4f}" for n, v in zip(PARAM_NAMES, self.ref_cond_raw)
        )

    # ── Main call ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def __call__(self, epoch: int, model, history: dict):
        model.eval()
        shape = (1, 3, 256, 256)

        try:
            print(f"  [viz] {self.name_a} sampling ...", flush=True)
            raw_a    = self.fn_a(model, shape, self.ref_cond)[0].cpu().numpy()
            sample_a = self._denorm_maps(raw_a)

            print(f"  [viz] {self.name_b} sampling ...", flush=True)
            raw_b    = self.fn_b(model, shape, self.ref_cond)[0].cpu().numpy()
            sample_b = self._denorm_maps(raw_b)

            self._plot_samples(epoch, sample_a, sample_b)
            self._plot_power_spectrum(epoch, sample_a, sample_b)
            self._plot_loss(history)
            self._print_metrics(epoch, model)

            torch.cuda.empty_cache()
            print(f"  [viz] saved → {self.plot_dir}", flush=True)
        except Exception as e:
            print(f"  [viz] WARNING: plot failed at epoch {epoch+1} — {e}", flush=True)
            plt.close("all")
        finally:
            model.train()

    # ── 1. 샘플 맵 (3×3) ──────────────────────────────────────────────────────

    def _plot_samples(self, epoch: int, sample_a: np.ndarray, sample_b: np.ndarray):
        """
        3 rows × 3 cols  (denormalized 값)
          rows : Real val[0] | sampler_a | sampler_b
          cols : Mcdm | Mgas | T
          각 subplot 독립 colorbar
        """
        real = self.ref_maps[0]   # [3, 256, 256] denormalized
        rows = [("Real (val[0])", real), (self.name_a, sample_a), (self.name_b, sample_b)]

        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        fig.suptitle(
            f"Epoch {epoch+1:04d}  –  Field  [denormalized]\n"
            f"{self._cond_str()}",
            fontsize=10,
        )

        # vmin/vmax: real 데이터 log10 값 기준으로 고정
        # (10^x 대신 log10 공간 그대로 표시 → overflow 없음)
        ref_log = np.log10(np.clip(self.ref_maps[0], 1e-30, None))  # [3, H, W]
        ch_ranges = []
        for ci in range(3):
            d = ref_log[ci][np.isfinite(ref_log[ci])]
            vmin = float(np.percentile(d, 1))  if len(d) else -1.0
            vmax = float(np.percentile(d, 99)) if len(d) else  1.0
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                vmin, vmax = -1.0, 1.0
            ch_ranges.append((vmin, vmax))

        for ri, (row_label, img) in enumerate(rows):
            for ci, (ch_label, cmap) in enumerate(zip(CHANNEL_LABELS, CMAPS)):
                ax  = axes[ri, ci]
                vmin, vmax = ch_ranges[ci]
                # log10 공간으로 변환, 비정상값 클립
                data = np.log10(np.clip(img[ci], 1e-30, None))
                data = np.clip(data, vmin - 3, vmax + 3)   # 극단값 표시는 허용하되 무한대 방지
                data = np.where(np.isfinite(data), data, vmin)
                im  = ax.imshow(data, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
                ax.set_title(f"{row_label}  –  {ch_label}", fontsize=9)
                ax.axis("off")
                fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

        path = self.plot_dir / f"ep{epoch+1:04d}_samples.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        shutil.copy(path, self.plot_dir / "latest_samples.png")
        plt.close(fig)

    # ── 2. Power spectrum ──────────────────────────────────────────────────────

    def _plot_power_spectrum(self, epoch: int, sample_a: np.ndarray, sample_b: np.ndarray):
        """
        denormalized log10 맵 기준 P(k) 비교
          Real val[0]  : ref_cond와 동일 시뮬레이션의 실제 맵
          Real val avg : val set 처음 8장 평균 P(k)
          sampler_a/b  : ref_cond 조건으로 생성된 샘플
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(
            f"Epoch {epoch+1:04d}  –  Power Spectrum  [log₁₀ field, denormalized]\n"
            f"cond: {self._cond_str()}",
            fontsize=9,
        )

        for c, (ch_name, ax) in enumerate(zip(CHANNEL_NAMES, axes)):
            k_r0, Pk_r0 = compute_power_spectrum_2d(self.ref_maps[0][c])
            Pk_avg = np.mean(
                [compute_power_spectrum_2d(m[c])[1] for m in self.ref_maps], axis=0
            )
            k_a, Pk_a = compute_power_spectrum_2d(sample_a[c])
            k_b, Pk_b = compute_power_spectrum_2d(sample_b[c])

            ax.loglog(k_r0, Pk_r0, "k-",  lw=2.0, label="Real val[0] (same cond)")
            ax.loglog(k_r0, Pk_avg, "k--", lw=1.2, label="Real val avg (N=8)", alpha=0.6)
            ax.loglog(k_a,  Pk_a,   "r-",  lw=1.5, label=f"{self.name_a} (generated)")
            ax.loglog(k_b,  Pk_b,   "b-",  lw=1.5, label=f"{self.name_b} (generated)")

            ax.set_title(ch_name, fontsize=11)
            ax.set_xlabel("k  [pixel freq]")
            ax.set_ylabel("P(k)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, which="both")

        fig.tight_layout()
        path = self.plot_dir / f"ep{epoch+1:04d}_power_spectrum.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        shutil.copy(path, self.plot_dir / "latest_power_spectrum.png")
        plt.close(fig)

    # ── 3. Per-epoch Metrics ───────────────────────────────────────────────────

    @torch.no_grad()
    def _print_metrics(self, epoch: int, model):
        """
        eval_maps (N개) 각각에 대해 sampler_b로 생성 → log10 공간에서 메트릭 계산 후 출력.
        sampler_b (보통 DDIM)로 N개를 한 번에 배치 샘플링.
        """
        try:
            N = self.eval_maps_norm.shape[0]

            # ── 배치 샘플링 (sampler_b 사용) ──────────────────────────────────
            print(f"  [metrics] {self.name_b} sampling (N={N}) ...", flush=True)
            shape   = (N, 3, self.eval_maps_norm.shape[-2], self.eval_maps_norm.shape[-1])
            raw_gen = self.fn_b(model, shape, self.eval_conds)   # [N, 3, H, W] normalized
            raw_gen = raw_gen[0] if isinstance(raw_gen, tuple) else raw_gen

            # ── log10 공간 변환 ───────────────────────────────────────────────
            true_log = self._to_log10(self.eval_maps_norm)   # [N, 3, H, W]
            gen_log  = self._to_log10(raw_gen.cpu().numpy() if hasattr(raw_gen, "cpu") else raw_gen)

            # ── Auto/Cross Power Spectrum ──────────────────────────────────────
            spec_err  = compute_spectrum_errors(true_log, gen_log)
            corr_err  = compute_correlation_errors(true_log, gen_log)
            pdf_res   = compare_pixel_distributions(true_log, gen_log, log=False, ks_subsample=20000)

            # ── 출력 ──────────────────────────────────────────────────────────
            sep = "─" * 78
            print(f"\n  {sep}")
            print(f"  Epoch {epoch+1:04d} Metrics  (N={N}, {self.name_b})  [log₁₀ space]")
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
            self._save_metrics_json(epoch, N, spec_err, corr_err, pdf_res)

        except Exception as e:
            print(f"  [metrics] WARNING: metric computation failed — {e}", flush=True)

    def _save_metrics_json(
        self,
        epoch: int,
        n_samples: int,
        spec_err: dict,
        corr_err: dict,
        pdf_res: dict,
    ) -> None:
        """에폭 metrics를 JSON으로 저장 (누적 history 파일에 append).

        저장 위치: {plot_dir}/../metrics_history.json
        각 에폭 항목 구조:
            epoch, n_samples, auto_power, cross_power, correlation, pdf, passed_overall
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

        record = {
            "epoch":          epoch + 1,
            "n_samples":      n_samples,
            "auto_power":     auto,
            "cross_power":    cross,
            "correlation":    corr,
            "pdf":            pdf,
            "passed_overall": passed_overall,
        }

        # ── 누적 history 파일에 append ────────────────────────────────────────
        metrics_path = self.plot_dir.parent / "metrics_history.json"
        history: list = []
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    history = json.load(f)
            except Exception:
                history = []
        # epoch 중복 방지: 같은 epoch 있으면 덮어씀
        history = [r for r in history if r.get("epoch") != record["epoch"]]
        history.append(record)
        history.sort(key=lambda r: r["epoch"])
        with open(metrics_path, "w") as f:
            json.dump(history, f, indent=2, allow_nan=True)
        print(f"  [metrics] saved → {metrics_path}", flush=True)

    def _to_log10(self, x) -> np.ndarray:
        """normalized z-space → log10(physical field)"""
        if hasattr(x, "cpu"):
            x = x.cpu().numpy()
        x = np.asarray(x, dtype=np.float32)
        t = torch.from_numpy(x)
        if t.ndim == 3:
            t = t.unsqueeze(0)
        phys = self.normalizer.denormalize(t).numpy()          # 10^(z*scale+center)
        return np.log10(np.clip(phys, 1e-30, None))            # log10 space

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
