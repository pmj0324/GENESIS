"""
GENESIS - Dataset Builder

두 가지 역할:
  1. build()       : 채널별 .npy 3개 → Maps_3ch_*.npy 합치기 (1회)
  2. build_splits(): Maps_3ch_*.npy + params → 정규화 + train/val/test 분리 저장 (1회)
  3. materialize_augmentation():
                  : 정규화된 train split을 D4 대칭으로 물리적으로 증설해 새 데이터셋 저장

사용법:
  # Step 1: 채널 합치기 (이미 완료된 경우 스킵)
  python -m dataloader.build_dataset stack \\
      --maps-dir /path/to/CAMELS/IllustrisTNG

  # Step 2: 정규화 + split 저장
  python -m dataloader.build_dataset splits \\
      --maps-path  /path/to/Maps_3ch_IllustrisTNG_LH_z=0.00.npy \\
      --params-path /path/to/params_LH_IllustrisTNG.txt \\
      --out-dir    GENESIS-data/ \\
      --norm-config configs/base.yaml

  # Step 3: train split 물리적 증설 (D4, 최대 8배가 가장 자연스러움)
  python -m dataloader.build_dataset augment \\
      --data-dir GENESIS-data/affine_default \\
      --out-dir  GENESIS-data/affine_default_x8 \\
      --copies   8

출력 (GENESIS-data/):
  train_maps.npy    [12000, 3, 256, 256]  float32  정규화됨
  train_params.npy  [12000, 6]            float32  zscore
  val_maps.npy      [1500,  3, 256, 256]
  val_params.npy    [1500,  6]
  test_maps.npy     [1500,  3, 256, 256]
  test_params.npy   [1500,  6]
  metadata.yaml     ← 정규화 설정, split 정보 기록
"""

import numpy as np
import yaml
import shutil
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloader.normalization import Normalizer, normalize_params_numpy

SUITE        = "IllustrisTNG"
REDSHIFT     = "z=0.00"
FIELDS       = ["Mcdm", "Mgas", "T"]
MAPS_PER_SIM = 15
OUT_NAME     = f"Maps_3ch_{SUITE}_LH_{REDSHIFT}.npy"


# ── Step 1: 채널 합치기 ────────────────────────────────────────────────────────

def build(maps_dir: Path, out_dir: Path | None = None):
    maps_dir = Path(maps_dir)
    out_path = Path(out_dir) / OUT_NAME if out_dir else maps_dir / OUT_NAME

    if out_path.exists():
        print(f"[build] 이미 존재함: {out_path}")
        print("  덮어쓰려면 기존 파일을 삭제 후 재실행.")
        return

    print(f"[build] 소스: {maps_dir}")
    channels = []
    for field in tqdm(FIELDS, desc="loading channels"):
        path = maps_dir / f"Maps_{field}_{SUITE}_LH_{REDSHIFT}.npy"
        tqdm.write(f"  {path.name}")
        m = np.load(path).astype(np.float32)
        tqdm.write(f"    shape={m.shape}  dtype={m.dtype}")
        channels.append(m)

    print("stacking ...", flush=True)
    stacked = np.stack(channels, axis=1)   # [N, 3, 256, 256]
    print(f"  shape={stacked.shape}  size={stacked.nbytes / 1e9:.2f}GB")

    print(f"saving → {out_path} ...")
    fp = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=stacked.dtype, shape=stacked.shape
    )
    chunk = 500
    for i in tqdm(range(0, len(stacked), chunk), desc="saving"):
        fp[i:i+chunk] = stacked[i:i+chunk]
    del fp
    print("[build] done.")


# ── Step 2: 정규화 + split 저장 ────────────────────────────────────────────────

def build_splits(
    maps_path:   Path,
    params_path: Path,
    out_dir:     Path,
    norm_config: dict,
    train_ratio: float = 0.8,
    val_ratio:   float = 0.1,
    seed:        int   = 42,
):
    maps_path   = Path(maps_path)
    params_path = Path(params_path)
    out_dir     = Path(out_dir)

    metadata_path = out_dir / "metadata.yaml"
    if metadata_path.exists():
        print(f"[build_splits] 이미 존재함: {out_dir}")
        print("  덮어쓰려면 기존 폴더를 삭제 후 재실행.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 로드 ──────────────────────────────────────────────────────────────────
    print(f"[build_splits] maps   : {maps_path}")
    print(f"[build_splits] params : {params_path}")

    print("loading maps ...", flush=True)
    maps_raw = np.load(maps_path).astype(np.float32)     # [15000, 3, 256, 256]
    params_raw = np.loadtxt(params_path, dtype=np.float32)  # [1000, 6]
    print(f"  maps={maps_raw.shape}  params={params_raw.shape}")

    # ── 정규화 ────────────────────────────────────────────────────────────────
    print("normalizing maps ...", flush=True)
    normalizer = Normalizer(norm_config)
    chunk = 500
    for i in tqdm(range(0, len(maps_raw), chunk), desc="normalizing"):
        maps_raw[i:i+chunk] = normalizer.normalize_numpy(maps_raw[i:i+chunk])

    params_exp = np.repeat(params_raw, MAPS_PER_SIM, axis=0)   # [15000, 6]
    params_norm = normalize_params_numpy(params_exp)             # zscore

    # ── split ─────────────────────────────────────────────────────────────────
    n_sims  = len(params_raw)
    rng     = np.random.default_rng(seed)
    sim_idx = rng.permutation(n_sims)
    n_train = int(n_sims * train_ratio)
    n_val   = int(n_sims * val_ratio)

    splits = {
        "train": sim_idx[:n_train],
        "val":   sim_idx[n_train : n_train + n_val],
        "test":  sim_idx[n_train + n_val :],
    }

    # ── 저장 ──────────────────────────────────────────────────────────────────
    sizes = {}
    for split_name, sims in splits.items():
        map_idx = np.concatenate([
            np.arange(s * MAPS_PER_SIM, (s + 1) * MAPS_PER_SIM) for s in sims
        ])
        m = maps_raw[map_idx]
        p = params_norm[map_idx]
        sizes[split_name] = len(m)

        maps_out   = out_dir / f"{split_name}_maps.npy"
        params_out = out_dir / f"{split_name}_params.npy"

        print(f"saving {split_name}: maps={m.shape}  params={p.shape}")
        fp = np.lib.format.open_memmap(maps_out, mode="w+", dtype=m.dtype, shape=m.shape)
        chunk = 500
        for i in tqdm(range(0, len(m), chunk), desc=f"  {split_name}_maps", leave=False):
            fp[i:i+chunk] = m[i:i+chunk]
        del fp
        np.save(params_out, p)

    # ── metadata ──────────────────────────────────────────────────────────────
    metadata = {
        "created":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_maps":  str(maps_path.resolve()),
        "source_params": str(params_path.resolve()),
        "normalization": norm_config,
        "split": {
            "train_ratio": train_ratio,
            "val_ratio":   val_ratio,
            "seed":        seed,
            "n_sims":      n_sims,
            "maps_per_sim": MAPS_PER_SIM,
        },
        "sizes": sizes,
    }
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)

    print(f"[build_splits] done. → {out_dir}")
    print(f"  train={sizes['train']}  val={sizes['val']}  test={sizes['test']}")
    print(f"  metadata → {metadata_path}")


# ── Step 3: train split D4 물리적 증설 ────────────────────────────────────────

def _copy_array_file(src: Path, dst: Path) -> None:
    arr = np.load(src, mmap_mode="r")
    fp = np.lib.format.open_memmap(dst, mode="w+", dtype=arr.dtype, shape=arr.shape)
    chunk = 500
    for i in tqdm(range(0, len(arr), chunk), desc=f"copy {src.stem}", leave=False):
        fp[i:i+chunk] = arr[i:i+chunk]
    del fp


def materialize_augmentation(
    data_dir: Path,
    out_dir: Path,
    copies: int = 8,
):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)

    if copies <= 0:
        raise ValueError(f"copies must be positive, got {copies}")

    metadata_in = data_dir / "metadata.yaml"
    metadata_out = out_dir / "metadata.yaml"
    if metadata_out.exists():
        print(f"[augment] 이미 존재함: {out_dir}")
        print("  덮어쓰려면 기존 폴더를 삭제 후 재실행.")
        return

    required = [
        data_dir / "train_maps.npy",
        data_dir / "train_params.npy",
        data_dir / "val_maps.npy",
        data_dir / "val_params.npy",
        data_dir / "test_maps.npy",
        data_dir / "test_params.npy",
        metadata_in,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    out_dir.mkdir(parents=True, exist_ok=True)

    train_maps = np.load(data_dir / "train_maps.npy", mmap_mode="r")
    train_params = np.load(data_dir / "train_params.npy", mmap_mode="r")
    n_train = len(train_maps)
    out_shape = (n_train * copies, *train_maps.shape[1:])

    print(f"[augment] source={data_dir}")
    print(f"[augment] output={out_dir}")
    print(f"[augment] train_maps={train_maps.shape} train_params={train_params.shape}")
    print(f"[augment] copies={copies} -> augmented_train_maps={out_shape}")

    train_maps_out = out_dir / "train_maps.npy"
    train_params_out = out_dir / "train_params.npy"

    maps_fp = np.lib.format.open_memmap(
        train_maps_out, mode="w+", dtype=train_maps.dtype, shape=out_shape
    )
    params_aug = np.empty((n_train * copies, train_params.shape[1]), dtype=np.float32)
    chunk = 128

    for copy_idx in tqdm(range(copies), desc="augment train"):
        start = copy_idx * n_train
        end = start + n_train
        transform_idx = copy_idx % 8
        for i in range(0, n_train, chunk):
            j = min(i + chunk, n_train)
            batch = np.asarray(train_maps[i:j], dtype=np.float32)
            k = transform_idx % 4
            batch = np.rot90(batch, k=k, axes=(-2, -1))
            if transform_idx >= 4:
                batch = np.flip(batch, axis=-1)
            maps_fp[start + i:start + j] = np.ascontiguousarray(batch)
        params_aug[start:end] = train_params[:]

    del maps_fp
    np.save(train_params_out, params_aug)

    _copy_array_file(data_dir / "val_maps.npy", out_dir / "val_maps.npy")
    shutil.copy2(data_dir / "val_params.npy", out_dir / "val_params.npy")
    _copy_array_file(data_dir / "test_maps.npy", out_dir / "test_maps.npy")
    shutil.copy2(data_dir / "test_params.npy", out_dir / "test_params.npy")

    with open(metadata_in) as f:
        metadata = yaml.safe_load(f)

    metadata["created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata["materialized_augmentation"] = {
        "source_data_dir": str(data_dir.resolve()),
        "method": "D4",
        "copies": int(copies),
        "train_only": True,
    }
    metadata.setdefault("sizes", {})
    metadata["sizes"]["train"] = int(n_train * copies)

    with open(metadata_out, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)

    print(f"[augment] done. -> {out_dir}")
    print(f"  train={n_train * copies}  val={metadata['sizes'].get('val')}  test={metadata['sizes'].get('test')}")
    print(f"  metadata -> {metadata_out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GENESIS dataset builder")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # stack
    p_stack = sub.add_parser("stack", help="채널별 npy → Maps_3ch_*.npy")
    p_stack.add_argument("--maps-dir", type=Path, required=True,
                         help="CAMELS IllustrisTNG 디렉토리")
    p_stack.add_argument("--out-dir",  type=Path, default=None)

    # splits
    p_splits = sub.add_parser("splits", help="정규화 + train/val/test 분리 저장")
    p_splits.add_argument("--maps-path",   type=Path, required=True)
    p_splits.add_argument("--params-path", type=Path, required=True)
    p_splits.add_argument("--out-dir",     type=Path, required=True)
    p_splits.add_argument("--norm-config", type=Path, required=True,
                          help="normalization 섹션이 포함된 YAML (예: configs/base.yaml)")
    p_splits.add_argument("--train-ratio", type=float, default=0.8)
    p_splits.add_argument("--val-ratio",   type=float, default=0.1)
    p_splits.add_argument("--seed",        type=int,   default=42)

    # augment
    p_aug = sub.add_parser("augment", help="정규화된 train split을 D4 대칭으로 물리적 증설")
    p_aug.add_argument("--data-dir", type=Path, required=True,
                       help="기존 정규화 데이터 디렉토리")
    p_aug.add_argument("--out-dir",  type=Path, required=True,
                       help="증설된 새 데이터 디렉토리")
    p_aug.add_argument("--copies",   type=int, default=8,
                       help="train split 반복/대칭 증설 배수 (기본: 8)")

    args = parser.parse_args()

    if args.cmd == "stack":
        build(maps_dir=args.maps_dir, out_dir=args.out_dir)

    elif args.cmd == "splits":
        with open(args.norm_config) as f:
            cfg = yaml.safe_load(f)
        build_splits(
            maps_path=args.maps_path,
            params_path=args.params_path,
            out_dir=args.out_dir,
            norm_config=cfg["normalization"],
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

    elif args.cmd == "augment":
        materialize_augmentation(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            copies=args.copies,
        )
