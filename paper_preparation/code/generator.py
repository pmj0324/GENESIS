"""Checkpoint + sampler wrapper that produces n_gen samples per condition.

We delegate the heavy lifting (model construction, sampler build, denorm) to
`sample.py:load_model_and_normalizer` / `generate_samples`, which is the same
entry point used by the existing `samples/dopri5/` generation. This guarantees
the per-condition outputs here match what the rest of the project produces,
modulo seed and batching.

The wrapper adds:
    * a stable seed-per-condition convention (cond_id offset from a base),
    * automatic chunking when n_gen > a max GPU batch,
    * returning *normalized*-space outputs (so we can store directly to NPY
      with no extra conversion).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

# Make repo root importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sample import load_model_and_normalizer  # noqa: E402

__all__ = ["GenesisGenerator"]


class GenesisGenerator:
    """High-level n_gen-sample-per-condition generator.

    Example
    -------
    >>> g = GenesisGenerator(
    ...     config="runs/.../config_resume.yaml",
    ...     checkpoint="runs/.../best.pt",
    ...     solver="dopri5", steps=50, cfg_scale=1.0,
    ... )
    >>> samples_norm = g.generate(theta_norm, n_gen=32, seed_base=42, cond_id=17)
    >>> samples_norm.shape
    (32, 3, 256, 256)
    """

    def __init__(
        self,
        config: str,
        checkpoint: str,
        solver: Optional[str] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        device: str = "cuda",
        model_source: str = "auto",
        max_batch: int = 32,
        run_gpu_test: bool = True,
    ):
        self.device = device
        self.max_batch = int(max_batch)
        self.config_path = str(config)
        self.checkpoint_path = str(checkpoint)
        self.solver = solver
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.rtol = rtol
        self.atol = atol
        self.run_gpu_test = bool(run_gpu_test)

        self.model, self.normalizer, self.sampler_fn, self.cfg = (
            load_model_and_normalizer(
                config=config,
                checkpoint_path=checkpoint,
                device=device,
                model_source=model_source,
                cfg_scale=cfg_scale,
                solver=solver,
                steps=steps,
                rtol=rtol,
                atol=atol,
            )
        )

        sampler_cfg = self.cfg.get("generative", {}).get("sampler", {})
        self.sampler_method = sampler_cfg.get("method", "?")
        self.sampler_steps = sampler_cfg.get("steps", "?")
        self.sampler_cfg_scale = sampler_cfg.get("cfg_scale", None)
        self.sampler_rtol = sampler_cfg.get("rtol", None)
        self.sampler_atol = sampler_cfg.get("atol", None)
        self.use_amp = (
            "cuda" in str(self.device)
            and torch.cuda.is_available()
            and hasattr(torch, "autocast")
        )

        # ── GPU 테스트 ──────────────────────────────────────────────────────
        if self.run_gpu_test:
            self._test_gpu_usage()

    @torch.no_grad()
    def generate(
        self,
        theta_norm: np.ndarray,
        n_gen: int,
        seed_base: int = 42,
        cond_id: int = 0,
        sample_shape: tuple = (3, 256, 256),
    ) -> np.ndarray:
        """Generate `n_gen` samples for one condition, returning a normalized
        numpy array of shape (n_gen, *sample_shape)."""
        theta_norm = np.asarray(theta_norm, dtype=np.float32).reshape(-1)
        if theta_norm.size != 6:
            raise ValueError(f"theta_norm must have 6 elements, got {theta_norm.size}")

        seed = int(seed_base) + int(cond_id)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if "cuda" in str(self.device) and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        cond_full = (
            torch.from_numpy(theta_norm)
            .to(self.device)
            .unsqueeze(0)
            .expand(n_gen, -1)
            .contiguous()
        )

        n_chunks = (n_gen + self.max_batch - 1) // self.max_batch
        out_chunks = []
        chunk_ranges = [
            (start, min(start + self.max_batch, n_gen))
            for start in range(0, n_gen, self.max_batch)
        ]
        for start, end in tqdm(chunk_ranges, desc="  chunks", unit="chunk", leave=False, dynamic_ncols=True):
            cond_chunk = cond_full[start:end]
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = self.sampler_fn(
                        self.model, (end - start, *sample_shape), cond_chunk
                    )
            else:
                out = self.sampler_fn(
                    self.model, (end - start, *sample_shape), cond_chunk
                )
            if isinstance(out, tuple):
                out = out[0]
            out_chunks.append(out.detach().float().cpu().numpy())
        return np.concatenate(out_chunks, axis=0).astype(np.float32, copy=False)

    # ── GPU 테스트 메서드 ────────────────────────────────────────────────────

    def _test_gpu_usage(self):
        """GPU가 실제로 사용되고 있는지 테스트."""
        print("\n[GPU Test] ────────────────────────────────────────")
        print(f"[GPU Test] Device: {self.device}")
        print(f"[GPU Test] CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"[GPU Test] CUDA device count: {torch.cuda.device_count()}")
            print(f"[GPU Test] Current CUDA device: {torch.cuda.current_device()}")
            print(f"[GPU Test] CUDA device name: {torch.cuda.get_device_name(0)}")

            # 메모리 테스트: 메모리 할당 전후 비교
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated() / 1024**2  # MB

            # Swin은 patch/window/downsample 제약이 있어서 지나치게 작은 입력으로는
            # forward 자체가 불가능하다. 실제 생성 해상도와 동일한 256을 사용해
            # GPU 경로를 검증하되, 1회 forward만 수행해 오버헤드는 제한한다.
            test_side = 256
            test_cond = torch.zeros(1, 6, device=self.device, dtype=torch.float32)
            test_x = torch.randn(1, 3, test_side, test_side, device=self.device, dtype=torch.float32)
            test_t = torch.full((1,), 0.5, device=self.device, dtype=torch.float32)
            with torch.inference_mode():
                result = self.model(test_x, test_t, test_cond).abs().mean()
            torch.cuda.synchronize()

            mem_after = torch.cuda.memory_allocated() / 1024**2  # MB
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB

            print(f"[GPU Test] Memory before test: {mem_before:.1f} MB")
            print(f"[GPU Test] Memory after test: {mem_after:.1f} MB")
            print(f"[GPU Test] Peak memory used: {peak_mem:.1f} MB")
            print(f"[GPU Test] Memory delta: {mem_after - mem_before:.1f} MB")
            print(f"[GPU Test] Test result: {result.item():.2e} (✓ 연산 성공)")

            # 모델이 GPU에 있는지 확인
            model_device = next(self.model.parameters()).device
            print(f"[GPU Test] Model device: {model_device}")
            if str(model_device) == str(self.device):
                print("[GPU Test] ✓ 모델이 올바른 device에 로드됨")
            else:
                print(f"[GPU Test] ✗ 경고: 모델이 {model_device}에 있음 (지정: {self.device})")
        else:
            print("[GPU Test] ✗ CUDA 사용 불가 - CPU로 실행 중")

        print("[GPU Test] ────────────────────────────────────────\n")

    # ── metadata for manifest writing ────────────────────────────────────────

    def manifest_dict(self) -> dict:
        return {
            "checkpoint": self.checkpoint_path,
            "config": self.config_path,
            "device": self.device,
            "sampler_method": self.sampler_method,
            "sampler_steps": self.sampler_steps,
            "sampler_cfg_scale": self.sampler_cfg_scale,
            "sampler_rtol": self.sampler_rtol,
            "sampler_atol": self.sampler_atol,
            "max_batch": self.max_batch,
            "use_amp": self.use_amp,
            "run_gpu_test": self.run_gpu_test,
        }
