"""
GENESIS - EpochVisualizer

에폭 끝마다 호출:
  1. 샘플 맵         → plots/ep{N:04d}_samples.png       3×3 grid (Real / sampler_a / sampler_b)
  2. Power spectrum  → plots/ep{N:04d}_power_spectrum.png
  3. Loss curve      → plots/loss.png                    (매 에폭 덮어씀)

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

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from typing import Callable, Dict, Tuple

from analysis.power_spectrum import compute_power_spectrum_2d
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
        sampler_a: Tuple[str, SamplerFn],
        sampler_b: Tuple[str, SamplerFn],
        plot_dir,
        ref_maps,
        ref_cond,
        norm_cfg: Dict,
        device,
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

        fig.savefig(
            self.plot_dir / f"ep{epoch+1:04d}_samples.png",
            dpi=100, bbox_inches="tight",
        )
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
        fig.savefig(
            self.plot_dir / f"ep{epoch+1:04d}_power_spectrum.png",
            dpi=100, bbox_inches="tight",
        )
        plt.close(fig)

    # ── 3. Loss curve ──────────────────────────────────────────────────────────

    def _plot_loss(self, history: dict):
        epochs = range(1, len(history["train_loss"]) + 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, history["train_loss"], label="train", color="steelblue", linewidth=1.5)
        ax.plot(epochs, history["val_loss"],   label="val",   color="tomato",    linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plot_dir / "loss.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
