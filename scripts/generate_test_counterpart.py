"""
scripts/generate_test_counterpart.py

test 데이터셋과 동일한 구조의 생성 데이터셋 제작.

  - test_params.npy : (1500, 6)  → 100 unique 조건 × 15 반복
  - 각 unique 조건마다 15개 생성 → (1500, 3, 256, 256) normalized space

출력:
  samples/test_counterpart/
    generated_maps.npy    (1500, 3, 256, 256)  float32  normalized
    generated_params.npy  (1500, 6)             float32  normalized (z-score)
    metadata.json

사용:
  cd /home/work/cosmology/GENESIS
  python scripts/generate_test_counterpart.py
  python scripts/generate_test_counterpart.py --solver heun --steps 25
  python scripts/generate_test_counterpart.py --solver euler --steps 50 --seed 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sample import load_model_and_normalizer

# ── 기본 설정 ─────────────────────────────────────────────────────────────────

CONFIG = (
    REPO_ROOT
    / "runs/flow/unet/unet_flow_0330_ft_last_cosine_restarts_t0_3/config_resume.yaml"
)
CKPT = (
    REPO_ROOT
    / "runs/flow/unet/unet_flow_0330_ft_last_cosine_restarts_t0_3/best.pt"
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="test 데이터셋과 동일 구조의 generated 데이터셋 생성"
    )
    p.add_argument("--config",     default=str(CONFIG))
    p.add_argument("--checkpoint", default=str(CKPT))
    p.add_argument("--solver",     default="euler",
                   choices=["euler", "heun", "rk4", "dopri5"])
    p.add_argument("--steps",      type=int, default=50)
    p.add_argument("--cfg-scale",  type=float, default=1.0)
    p.add_argument("--seed",       type=int, default=42,
                   help="베이스 시드 (조건별로 seed+cond_idx 사용)")
    p.add_argument("--maps-per-sim", type=int, default=15,
                   help="unique 조건 당 생성할 샘플 수 (기본: 15)")
    p.add_argument("--output-dir", default="samples/test_counterpart")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--data-dir",   default=None,
                   help="데이터 디렉토리 override (기본: config에서 읽음)")
    return p.parse_args()


# ── 생성 (normalized space) ────────────────────────────────────────────────────

@torch.no_grad()
def generate_normalized(
    model,
    sampler_fn,
    params_norm_batch: np.ndarray,   # (n_samples, 6)  normalized
    seed: int,
    device: str,
    sample_shape: tuple = (3, 256, 256),
) -> np.ndarray:
    """sampler_fn 직접 호출 → normalized space tensor 반환 (denormalize 없음).

    Returns
    -------
    np.ndarray  shape (n_samples, 3, H, W)  float32  normalized
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if "cuda" in str(device) and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n = len(params_norm_batch)
    cond = torch.from_numpy(params_norm_batch.astype(np.float32)).to(device)  # (n, 6)

    out = sampler_fn(model, (n, *sample_shape), cond)
    if isinstance(out, tuple):
        out = out[0]

    return out.detach().float().cpu().numpy().astype(np.float32, copy=False)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" GENESIS — Test Counterpart Generator")
    print(f" solver={args.solver}  steps={args.steps}  cfg={args.cfg_scale}")
    print(f" maps_per_sim={args.maps_per_sim}  seed_base={args.seed}")
    print(f" output → {out_dir}")
    print("=" * 60)

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    model, normalizer, sampler_fn, cfg = load_model_and_normalizer(
        config=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
        cfg_scale=args.cfg_scale,
        solver=args.solver,
        steps=args.steps,
    )

    # ── 데이터셋 경로 ─────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir) if args.data_dir else Path(cfg["data"]["data_dir"])
    test_maps_path   = data_dir / "test_maps.npy"
    test_params_path = data_dir / "test_params.npy"

    if not test_params_path.exists():
        raise FileNotFoundError(f"test_params.npy 없음: {test_params_path}")

    all_params = np.load(test_params_path)          # (1500, 6)  normalized
    n_total    = len(all_params)

    # 데이터에서 maps_per_sim 자동 감지 (연속된 동일 파라미터 블록 길이)
    ref = all_params[0]
    auto_mps = 1
    while auto_mps < n_total and np.allclose(all_params[auto_mps], ref, atol=1e-5):
        auto_mps += 1
    data_mps = auto_mps  # 데이터 내 실제 maps_per_sim

    # 생성할 샘플 수 = CLI 지정값 (기본: 데이터와 동일)
    mps = args.maps_per_sim if args.maps_per_sim != 15 else data_mps
    n_unique = n_total // data_mps

    print(f"\n[data] test_params: {all_params.shape}")
    print(f"[data] maps_per_sim 자동 감지: {data_mps}")
    print(f"[data] unique 조건: {n_unique}  ×  {mps} 생성 = {n_unique * mps}")

    # unique 조건 추출 (데이터 구조 기준 n*data_mps 번째 행)
    unique_params = all_params[np.arange(n_unique) * data_mps]   # (100, 6)

    # 생성 결과 버퍼
    sample_shape = (3, 256, 256)
    gen_maps   = np.empty((n_unique * mps, *sample_shape), dtype=np.float32)
    gen_params = np.empty((n_unique * mps, all_params.shape[1]), dtype=np.float32)

    t0 = time.time()
    for i, p_norm in enumerate(unique_params):
        seed_i = args.seed + i
        # (mps, 6) 조건 배치
        cond_batch = np.tile(p_norm[None], (mps, 1))           # (15, 6)

        samples = generate_normalized(
            model=model,
            sampler_fn=sampler_fn,
            params_norm_batch=cond_batch,
            seed=seed_i,
            device=device,
            sample_shape=sample_shape,
        )
        # 버퍼에 저장
        start = i * mps
        gen_maps[start : start + mps]   = samples
        gen_params[start : start + mps] = cond_batch

        elapsed = time.time() - t0
        eta     = elapsed / (i + 1) * (n_unique - i - 1)
        print(
            f"  [{i+1:3d}/{n_unique}]  "
            f"seed={seed_i}  "
            f"elapsed={elapsed:.0f}s  eta={eta:.0f}s  "
            f"range=[{samples.min():.3f}, {samples.max():.3f}]"
        )

    # ── 저장 ─────────────────────────────────────────────────────────────────
    maps_out   = out_dir / "generated_maps.npy"
    params_out = out_dir / "generated_params.npy"

    np.save(maps_out,   gen_maps)
    np.save(params_out, gen_params)
    print(f"\n  saved: generated_maps.npy    {gen_maps.shape}  (normalized)")
    print(f"  saved: generated_params.npy  {gen_params.shape}  (z-score normalized)")

    # metadata
    meta = {
        "checkpoint":      str(Path(args.checkpoint).resolve()),
        "config":          str(Path(args.config).resolve()),
        "solver":          args.solver,
        "steps":           args.steps,
        "cfg_scale":       args.cfg_scale,
        "seed_base":       args.seed,
        "n_unique_conds":  int(n_unique),
        "maps_per_sim":    int(mps),
        "total_samples":   int(n_unique * mps),
        "output_shape":    list(gen_maps.shape),
        "space":           "normalized",
        "source_data_dir": str(data_dir),
        "device":          device,
        "elapsed_sec":     round(time.time() - t0, 1),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  saved: metadata.json")

    print(f"\n[done]  총 {n_unique * mps}개 생성  ({time.time()-t0:.1f}s)")
    print(f"        → {out_dir}")


if __name__ == "__main__":
    main()
