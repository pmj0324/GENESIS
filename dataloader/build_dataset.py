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
      --maps-dir /path/to/CAMELS/IllustrisTNG [--suite LH|CV|EX|1P]

  # Step 2: 정규화 + split 저장
  python -m dataloader.build_dataset splits \\
      --maps-path  /path/to/Maps_3ch_IllustrisTNG_LH_z=0.00.npy \\
      --params-path /path/to/params_LH_IllustrisTNG.txt \\
      --out-dir    GENESIS-data/ \\
      --norm-config configs/base.yaml \\
      --split-strategy stratified_1d \\
      --stratify-param Omega_m \\
      --stratify-bins 10

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
  split_train.npy   [800]               int64  (canonical)
  split_val.npy     [100]               int64  (canonical)
  split_test.npy    [100]               int64  (canonical)
  train_sim_ids.npy [800]               int64  (legacy alias)
  val_sim_ids.npy   [100]               int64  (legacy alias)
  test_sim_ids.npy  [100]               int64  (legacy alias)
  metadata.yaml     ← 정규화 설정, split 정보 기록
"""

import numpy as np
import yaml
import shutil
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from dataloader.normalization import (
    Normalizer,
    fit_param_normalization,
    normalize_params_numpy,
    split_normalization_config,
)
from dataloader.recipe import build_normalization_recipe, save_normalization_recipe

SUITE        = "IllustrisTNG"
REDSHIFT     = "z=0.00"
FIELDS       = ["Mcdm", "Mgas", "T"]
PARAM_NAMES  = ["Omega_m", "sigma_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"]
MAPS_PER_SIM = 15


# ── Step 1: 채널 합치기 ────────────────────────────────────────────────────────

def build(maps_dir: Path, out_dir: Path | None = None, suite: str = "LH"):
    maps_dir = Path(maps_dir)
    out_name = f"Maps_3ch_{SUITE}_{suite}_{REDSHIFT}.npy"
    out_path = Path(out_dir) / out_name if out_dir else maps_dir / out_name

    if out_path.exists():
        print(f"[build] 이미 존재함: {out_path}")
        print("  덮어쓰려면 기존 파일을 삭제 후 재실행.")
        return

    print(f"[build] 소스: {maps_dir}  suite={suite}")
    channels = []
    for field in tqdm(FIELDS, desc="loading channels"):
        path = maps_dir / f"Maps_{field}_{SUITE}_{suite}_{REDSHIFT}.npy"
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

def _resolve_param_index(param: str | int) -> int:
    if isinstance(param, int):
        idx = int(param)
    else:
        text = str(param).strip()
        if text.lstrip("-").isdigit():
            idx = int(text)
        else:
            aliases = {name.lower(): i for i, name in enumerate(PARAM_NAMES)}
            aliases.update(
                {
                    "omegam": 0,
                    "omega_m": 0,
                    "sigma8": 1,
                    "sigma_8": 1,
                }
            )
            key = text.lower()
            if key not in aliases:
                raise ValueError(
                    f"Unknown stratify param: {param!r}. "
                    f"Use one of {PARAM_NAMES} or index 0..{len(PARAM_NAMES)-1}."
                )
            idx = aliases[key]

    if idx < 0 or idx >= len(PARAM_NAMES):
        raise ValueError(
            f"stratify param index out of range: {idx}. "
            f"Valid range is 0..{len(PARAM_NAMES)-1}."
        )
    return idx


def _make_stratify_labels(params: np.ndarray, param_idx: int, n_bins: int):
    values = params[:, param_idx].astype(np.float64)
    if n_bins <= 1:
        labels = np.zeros(len(values), dtype=np.int64)
        edges = np.array([values.min(), values.max()], dtype=np.float64)
        return labels, edges

    vmin, vmax = float(np.min(values)), float(np.max(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Non-finite values found while building stratify labels.")

    if vmax <= vmin:
        labels = np.zeros(len(values), dtype=np.int64)
        edges = np.array([vmin, vmax], dtype=np.float64)
        return labels, edges

    # Equal-width bins over the selected parameter range (pd.cut 스타일).
    edges = np.linspace(vmin, vmax, int(n_bins) + 1, dtype=np.float64)
    labels = np.digitize(values, edges[1:-1], right=False).astype(np.int64)
    return labels, edges


def _stratified_pick(
    ids: np.ndarray,
    labels: np.ndarray,
    n_pick: int,
    rng: np.random.Generator,
):
    n_total = len(ids)
    if n_pick <= 0:
        return np.empty(0, dtype=np.int64), ids.copy()
    if n_pick >= n_total:
        return ids.copy(), np.empty(0, dtype=np.int64)

    unique, counts = np.unique(labels, return_counts=True)
    target = counts.astype(np.float64) * (float(n_pick) / float(n_total))
    picks = np.floor(target).astype(np.int64)

    rem = int(n_pick - picks.sum())
    if rem > 0:
        frac = target - picks
        order = np.argsort(-frac)
        for i in order:
            if rem == 0:
                break
            if picks[i] < counts[i]:
                picks[i] += 1
                rem -= 1

    picked_chunks = []
    other_chunks = []
    for cls, n_cls_pick in zip(unique, picks):
        cls_ids = ids[labels == cls].copy()
        rng.shuffle(cls_ids)
        n_cls_pick = int(n_cls_pick)
        picked_chunks.append(cls_ids[:n_cls_pick])
        other_chunks.append(cls_ids[n_cls_pick:])

    picked = np.concatenate(picked_chunks).astype(np.int64, copy=False)
    other = np.concatenate(other_chunks).astype(np.int64, copy=False)
    rng.shuffle(picked)
    rng.shuffle(other)
    return picked, other


def _split_sim_ids(
    n_sims: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    strategy: str,
    strat_labels: np.ndarray | None = None,
):
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"Invalid ratios: train_ratio={train_ratio}, val_ratio={val_ratio}. "
            "Require 0 < train_ratio, 0 <= val_ratio, train_ratio+val_ratio < 1."
        )

    n_train = int(n_sims * train_ratio)
    n_val = int(n_sims * val_ratio)
    n_test = n_sims - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Invalid split sizes: train={n_train}, val={n_val}, test={n_test}. "
            "Please adjust ratios."
        )

    sim_ids = np.arange(n_sims, dtype=np.int64)
    rng = np.random.default_rng(seed)

    if strategy == "random":
        perm = rng.permutation(sim_ids)
        return {
            "train": perm[:n_train],
            "val": perm[n_train : n_train + n_val],
            "test": perm[n_train + n_val :],
        }

    if strategy == "stratified_1d":
        if strat_labels is None or len(strat_labels) != n_sims:
            raise ValueError("stratified_1d split requires strat_labels for all simulations.")
        labels = np.asarray(strat_labels, dtype=np.int64)
        train_ids, valtest_ids = _stratified_pick(sim_ids, labels, n_train, rng)
        val_ids, test_ids = _stratified_pick(
            valtest_ids, labels[valtest_ids], n_val, rng
        )
        return {"train": train_ids, "val": val_ids, "test": test_ids}

    raise ValueError(
        f"Unknown split strategy: {strategy!r}. "
        "Options: random / stratified_1d"
    )


def build_splits(
    maps_path:   Path,
    params_path: Path,
    out_dir:     Path,
    map_norm_config: dict,
    param_norm_cfg: dict | None = None,
    param_norm_mode: str | None = None,
    train_ratio: float = 0.8,
    val_ratio:   float = 0.1,
    seed:        int   = 42,
    split_strategy: str = "stratified_1d",
    stratify_param: str | int = "Omega_m",
    stratify_bins: int = 10,
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
    normalizer = Normalizer(map_norm_config)
    chunk = 500
    for i in tqdm(range(0, len(maps_raw), chunk), desc="normalizing"):
        maps_raw[i:i+chunk] = normalizer.normalize_numpy(maps_raw[i:i+chunk])

    # ── parameter normalization ───────────────────────────────────────────────
    param_cfg = dict(param_norm_cfg or {})
    if param_norm_mode is not None:
        param_cfg["method"] = param_norm_mode

    # Priority:
    # 1) CLI override (--param-norm-mode): always fit from params_raw
    # 2) YAML has fitted stats: reuse them as-is for exact reproducibility
    # 3) Otherwise: fit from params_raw using YAML/default method
    if param_norm_mode is not None:
        param_mode = str(param_cfg.get("method", "legacy_zscore")).strip().lower()
        print(f"[build_splits] param normalization: fit mode={param_mode} (CLI override)")
        param_norm_recipe = fit_param_normalization(params_raw, mode=param_mode)
    elif param_cfg.get("stats"):
        param_norm_recipe = param_cfg
        recipe_mode = str(param_norm_recipe.get("method", "custom")).strip().lower()
        print(f"[build_splits] param normalization: using provided YAML stats (method={recipe_mode})")
    else:
        param_mode = str(param_cfg.get("method", "legacy_zscore")).strip().lower()
        print(f"[build_splits] param normalization: fit mode={param_mode}")
        param_norm_recipe = fit_param_normalization(params_raw, mode=param_mode)

    params_norm_sim = normalize_params_numpy(params_raw, param_norm_recipe)
    params_exp = np.repeat(params_norm_sim, MAPS_PER_SIM, axis=0)   # [15000, 6]

    # ── split ─────────────────────────────────────────────────────────────────
    n_sims = len(params_raw)
    split_strategy = str(split_strategy).strip().lower()
    stratify_cfg = None
    strat_labels = None
    if split_strategy == "stratified_1d":
        stratify_idx = _resolve_param_index(stratify_param)
        strat_labels, strat_edges = _make_stratify_labels(
            params_raw, param_idx=stratify_idx, n_bins=int(stratify_bins)
        )
        uniq, cnt = np.unique(strat_labels, return_counts=True)
        stratify_cfg = {
            "param_index": int(stratify_idx),
            "param_name": PARAM_NAMES[stratify_idx],
            "n_bins": int(stratify_bins),
            "bin_edges": strat_edges.tolist(),
            "label_counts": {int(u): int(c) for u, c in zip(uniq, cnt)},
        }
        print(
            "[build_splits] split strategy: stratified_1d "
            f"(param={PARAM_NAMES[stratify_idx]}, bins={int(stratify_bins)})"
        )
    elif split_strategy == "random":
        print("[build_splits] split strategy: random")
    else:
        raise ValueError(
            f"Unknown split strategy: {split_strategy!r}. "
            "Options: random / stratified_1d"
        )

    splits = _split_sim_ids(
        n_sims=n_sims,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        strategy=split_strategy,
        strat_labels=strat_labels,
    )
    print(
        f"[build_splits] simulations: train={len(splits['train'])} "
        f"val={len(splits['val'])} test={len(splits['test'])}"
    )

    # ── 저장 ──────────────────────────────────────────────────────────────────
    sizes = {}
    for split_name, sims in splits.items():
        sim_ids = np.asarray(sims, dtype=np.int64)
        # Canonical split-id filenames.
        np.save(out_dir / f"split_{split_name}.npy", sim_ids)
        # Backward compatibility alias.
        np.save(out_dir / f"{split_name}_sim_ids.npy", sim_ids)
        map_idx = np.concatenate([
            np.arange(s * MAPS_PER_SIM, (s + 1) * MAPS_PER_SIM) for s in sims
        ])
        m = maps_raw[map_idx]
        p = params_exp[map_idx]
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
        "normalization": map_norm_config,
        "param_normalization": param_norm_recipe,
        "split": {
            "train_ratio": train_ratio,
            "val_ratio":   val_ratio,
            "seed":        seed,
            "n_sims":      n_sims,
            "maps_per_sim": MAPS_PER_SIM,
            "strategy": split_strategy,
            "stratify": stratify_cfg,
            "id_files": {
                "train": "split_train.npy",
                "val": "split_val.npy",
                "test": "split_test.npy",
            },
            "legacy_id_files": {
                "train": "train_sim_ids.npy",
                "val": "val_sim_ids.npy",
                "test": "test_sim_ids.npy",
            },
        },
        "sizes": sizes,
    }
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)

    print(f"[build_splits] done. → {out_dir}")
    print(f"  train={sizes['train']}  val={sizes['val']}  test={sizes['test']}")
    print(f"  metadata → {metadata_path}")


# ── Step 3: train split D4 물리적 증설 ────────────────────────────────────────

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

    shutil.copy2(data_dir / "val_maps.npy",   out_dir / "val_maps.npy")
    shutil.copy2(data_dir / "val_params.npy", out_dir / "val_params.npy")
    shutil.copy2(data_dir / "test_maps.npy",  out_dir / "test_maps.npy")
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
    p_stack.add_argument("--suite",    type=str,  default="LH",
                         choices=["LH", "CV", "EX", "1P"],
                         help="CAMELS suite (기본: LH)")

    # splits
    p_splits = sub.add_parser("splits", help="정규화 + train/val/test 분리 저장")
    p_splits.add_argument("--maps-path",   type=Path, required=True)
    p_splits.add_argument("--params-path", type=Path, required=True)
    p_splits.add_argument("--out-dir",     type=Path, required=True)
    p_splits.add_argument("--norm-config", type=Path, required=True,
                          help="normalization 섹션이 포함된 YAML (예: configs/base.yaml)")
    p_splits.add_argument(
        "--param-norm-mode",
        type=str,
        choices=["legacy_zscore", "astro_mixed"],
        default=None,
        help="Optional override for the params normalization mode from YAML.",
    )
    p_splits.add_argument("--train-ratio", type=float, default=0.8)
    p_splits.add_argument("--val-ratio",   type=float, default=0.1)
    p_splits.add_argument("--seed",        type=int,   default=42)
    p_splits.add_argument(
        "--split-strategy",
        type=str,
        choices=["stratified_1d", "random"],
        default="stratified_1d",
        help=(
            "Simulation-level split strategy. "
            "Default is stratified_1d (recommended)."
        ),
    )
    p_splits.add_argument(
        "--stratify-param",
        type=str,
        default="Omega_m",
        help=(
            "Parameter for stratified_1d split. "
            "Use name (Omega_m, sigma_8, ...) or index (0..5)."
        ),
    )
    p_splits.add_argument(
        "--stratify-bins",
        type=int,
        default=10,
        help="Number of equal-width bins for stratified_1d split.",
    )

    # augment
    p_aug = sub.add_parser("augment", help="정규화된 train split을 D4 대칭으로 물리적 증설")
    p_aug.add_argument("--data-dir", type=Path, required=True,
                       help="기존 정규화 데이터 디렉토리")
    p_aug.add_argument("--out-dir",  type=Path, required=True,
                       help="증설된 새 데이터 디렉토리")
    p_aug.add_argument("--copies",   type=int, default=8,
                       help="train split 반복/대칭 증설 배수 (기본: 8)")

    # recipe
    p_recipe = sub.add_parser("recipe", help="원본 맵에서 정규화 레시피 YAML 생성")
    p_recipe.add_argument("--maps-path",   type=Path, required=True,
                          help="원본 맵 .npy 파일 (raw, positive-valued)")
    p_recipe.add_argument("--params-path", type=Path, default=None,
                          help="params txt/npy (파라미터 정규화 stats를 YAML에 포함할 때)")
    p_recipe.add_argument("--lower-percentile", type=float, default=0.0,
                          help="log-space min 경계 퍼센타일 (0=true min, 1=p1)")
    p_recipe.add_argument("--upper-percentile", type=float, default=100.0,
                          help="log-space max 경계 퍼센타일 (100=true max, 99=p99)")
    p_recipe.add_argument("--center-stat", choices=["mean", "median"], default="mean",
                          help="min-max 스케일 후 빼줄 통계값")
    p_recipe.add_argument("--range-mode",  choices=["centered", "symmetric"], default="centered",
                          help="centered: mean/median 빼기, symmetric: [-1, 1] 매핑")
    p_recipe.add_argument("--param-mode",  choices=["legacy_zscore", "astro_mixed"], default=None,
                          help="파라미터 정규화 방식 (미지정 시 YAML에 포함 안 함)")
    p_recipe.add_argument("--out", type=Path, default=None,
                          help="출력 YAML 경로 (기본: configs/normalization/<자동생성>.yaml)")

    args = parser.parse_args()

    if args.cmd == "stack":
        build(maps_dir=args.maps_dir, out_dir=args.out_dir, suite=args.suite)

    elif args.cmd == "splits":
        with open(args.norm_config) as f:
            cfg = yaml.safe_load(f)
        map_norm_config, param_norm_config = split_normalization_config(cfg.get("normalization", {}))
        build_splits(
            maps_path=args.maps_path,
            params_path=args.params_path,
            out_dir=args.out_dir,
            map_norm_config=map_norm_config,
            param_norm_cfg=param_norm_config,
            param_norm_mode=args.param_norm_mode,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            split_strategy=args.split_strategy,
            stratify_param=args.stratify_param,
            stratify_bins=args.stratify_bins,
        )

    elif args.cmd == "augment":
        materialize_augmentation(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            copies=args.copies,
        )

    elif args.cmd == "recipe":
        maps = np.load(args.maps_path, mmap_mode="r")
        payload = build_normalization_recipe(
            maps,
            lower_percentile=args.lower_percentile,
            upper_percentile=args.upper_percentile,
            center_stat=args.center_stat,
            range_mode=args.range_mode,
            param_mode=args.param_mode,
        )
        if args.param_mode is not None and args.params_path is not None:
            if args.params_path.suffix.lower() == ".npy":
                params = np.load(args.params_path).astype(np.float32, copy=False)
            else:
                params = np.loadtxt(args.params_path, dtype=np.float32)
            payload.setdefault("normalization", {})["params"] = fit_param_normalization(
                params, mode=args.param_mode
            )
        if args.out is None:
            suffix = f"{args.lower_percentile:g}_{args.upper_percentile:g}_{args.range_mode}"
            if args.range_mode == "centered":
                suffix = f"{suffix}_{args.center_stat}"
            if args.param_mode:
                suffix = f"{suffix}_{args.param_mode}"
            out = Path("configs/normalization") / f"{args.maps_path.stem}_{suffix}.yaml"
        else:
            out = args.out
        save_normalization_recipe(payload, out)
        print(f"saved: {out}")
