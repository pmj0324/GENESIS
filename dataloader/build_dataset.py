"""
GENESIS - Dataset Builder

사용법:
  # Step 1: 채널 합치기 + params 페어링 → zarr 저장
  python -m dataloader.build_dataset stack \\
      --maps-dir /path/to/CAMELS/IllustrisTNG --suite LH

  # Step 2: 정규화 + train/val/test split → dataset.zarr 저장
  python -m dataloader.build_dataset splits \\
      --raw-zarr IllustrisTNG_LH.zarr \\
      --out dataset.zarr \\
      --norm-config configs/normalization/xxx.yaml

  # Step 3: D4 augmentation 물리적 증설
  python -m dataloader.build_dataset augment \\
      --data-path dataset.zarr \\
      --out dataset_x8.zarr \\
      --copies 8

  # Step 4: raw zarr에서 정규화 레시피 YAML 생성
  python -m dataloader.build_dataset recipe \\
      --raw-zarr IllustrisTNG_LH.zarr \\
      --out configs/normalization/my_recipe.yaml

stack 출력: {out_dir}/IllustrisTNG_{suite}.zarr
  maps     [N, 3, 256, 256]  float32  raw  chunks=(8,3,256,256)  lz4 압축
  params   [N, 6]            float32  map-level 확장 (1P는 단위 변환 완료)
  sim_ids  [N]               int32    maps[i]가 몇 번 sim인지
  .attrs   suite, fields, param_names, n_sims, maps_per_sim, ...

수트별 특이사항:
  LH  : 1000 sims × 15 maps = 15000  (훈련용)
  CV  : 27   sims × 15 maps = 405    (평가용)
  EX  : 4    sims × 15 maps = 60     (평가용)
  1P  : 30   sims × 15 maps = 450    (평가용, 30sim만 사용 + 단위 변환)
"""

import shutil
import numpy as np
import yaml
import zarr
from numcodecs import Blosc
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

# 1P suite: astrophysical params cols 2-5가 물리값으로 저장됨 → LH multiplier로 변환
_1P_N_SIMS     = 30   # 우주론 파라미터가 변동하는 sim만 사용 (rows 0-29)
_1P_FIDUCIALS  = [3.6, 1.0, 7.4, 20.0]  # cols 2,3,4,5 피듀셜 물리값


# ── Step 1: 채널 합치기 + params 페어링 → zarr ────────────────────────────────

def build(maps_dir: Path, out_dir: Path | None = None, suite: str = "LH"):
    """3채널 합치기 + params 페어링 → zarr 저장.

    출력: {out_dir}/IllustrisTNG_{suite}.zarr
      maps     [N, 3, 256, 256]  float32  raw
      params   [N, 6]            float32  map-level 확장 (1P는 단위 변환 적용)
      sim_ids  [N]               int32    maps[i]가 몇 번 sim인지
    """
    maps_dir = Path(maps_dir)
    out_dir  = Path(out_dir) if out_dir else maps_dir
    out_path = out_dir / f"{SUITE}_{suite}.zarr"

    if out_path.exists():
        print(f"[build] 이미 존재함: {out_path}")
        print("  덮어쓰려면 기존 파일을 삭제 후 재실행.")
        return

    # ── params 로드 ───────────────────────────────────────────────────────────
    params_path = maps_dir / f"params_{suite}_{SUITE}.txt"
    params_raw  = np.loadtxt(params_path, dtype=np.float32)

    n_sims_total = len(params_raw)
    n_sims_use   = _1P_N_SIMS if suite == "1P" else n_sims_total
    params_raw   = params_raw[:n_sims_use]

    # 1P: 첫 6열만 사용 + astrophysical params → LH multiplier 단위로 변환
    if suite == "1P":
        params_raw = params_raw[:, :6].copy()
        for col, fid in enumerate(_1P_FIDUCIALS, start=2):
            params_raw[:, col] /= fid

    n_sims  = len(params_raw)
    n_maps  = n_sims * MAPS_PER_SIM

    # ── maps 로드 + 스택 ──────────────────────────────────────────────────────
    print(f"[build] suite={suite}  n_sims={n_sims}  n_maps={n_maps}")
    channels = []
    for field in tqdm(FIELDS, desc="loading channels"):
        path = maps_dir / f"Maps_{field}_{SUITE}_{suite}_{REDSHIFT}.npy"
        tqdm.write(f"  {path.name}")
        m = np.load(path, mmap_mode="r")[:n_maps]
        tqdm.write(f"    shape={m.shape}")
        channels.append(np.asarray(m, dtype=np.float32))

    print("stacking ...", flush=True)
    maps_stacked = np.stack(channels, axis=1)   # [N, 3, 256, 256]
    print(f"  shape={maps_stacked.shape}  raw_size={maps_stacked.nbytes/1e9:.2f}GB")
    del channels

    # ── params / sim_ids → map-level 확장 ────────────────────────────────────
    params_expanded = np.repeat(params_raw, MAPS_PER_SIM, axis=0)          # [N, 6]
    sim_ids         = np.repeat(np.arange(n_sims, dtype=np.int32), MAPS_PER_SIM)  # [N]

    # ── zarr 저장 ─────────────────────────────────────────────────────────────
    print(f"saving → {out_path} ...")
    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)

    store = zarr.open_group(str(out_path), mode="w")
    store.create_dataset(
        "maps", data=maps_stacked,
        chunks=(8, 3, 256, 256), compressor=compressor, dtype="float32",
    )
    store.create_dataset("params",  data=params_expanded, dtype="float32")
    store.create_dataset("sim_ids", data=sim_ids,         dtype="int32")
    store.attrs.update({
        "suite":         suite,
        "sim":           SUITE,
        "redshift":      REDSHIFT,
        "fields":        FIELDS,
        "param_names":   PARAM_NAMES,
        "n_sims":        int(n_sims),
        "maps_per_sim":  MAPS_PER_SIM,
        "n_maps":        int(n_maps),
        "1p_converted":  suite == "1P",
    })

    print(f"[build] done → {out_path}")
    size_gb = sum(
        store[k].nbytes_stored for k in ("maps", "params", "sim_ids")
    ) / 1e9
    print(f"  maps={maps_stacked.shape}  compressed={size_gb:.2f}GB")


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
    raw_zarr_path:   Path,
    out_path:        Path,
    map_norm_config: dict,
    param_norm_cfg:  dict | None = None,
    param_norm_mode: str | None = None,
    train_ratio:     float = 0.8,
    val_ratio:       float = 0.1,
    seed:            int   = 42,
    split_strategy:  str   = "stratified_1d",
    stratify_param:  str | int = "Omega_m",
    stratify_bins:   int   = 10,
):
    """raw zarr → 정규화 + train/val/test split → dataset.zarr 저장.

    입력: build()가 만든 {SUITE}_{suite}.zarr
    출력: dataset.zarr
      train/  maps [N_train, 3, 256, 256]  params [N_train, 6]  sim_ids [N_train]
      val/    maps [N_val,   3, 256, 256]  params [N_val,   6]  sim_ids [N_val]
      test/   maps [N_test,  3, 256, 256]  params [N_test,  6]  sim_ids [N_test]
      .attrs  normalization, param_normalization, split_info, ...

    메모리 최적화:
    - params/sim_ids만 한번에 로드 (~360KB) → split 결정
    - maps는 sim 단위 스트리밍 (15장씩) → peak ~30MB (기존 ~11.7GB)
    """
    raw_zarr_path = Path(raw_zarr_path)
    out_path      = Path(out_path)

    if out_path.exists():
        print(f"[build_splits] 이미 존재함: {out_path}")
        print("  덮어쓰려면 기존 파일을 삭제 후 재실행.")
        return

    # ── params / sim_ids 로드 (가벼움, maps는 아직 안 읽음) ──────────────────
    print(f"[build_splits] source: {raw_zarr_path}")
    raw = zarr.open_group(str(raw_zarr_path), mode="r")
    raw_attrs   = dict(raw.attrs)

    params_map  = np.array(raw["params"],  dtype=np.float32)  # [N, 6]
    sim_ids_map = np.array(raw["sim_ids"], dtype=np.int32)     # [N]

    maps_per_sim = int(raw_attrs["maps_per_sim"])
    n_sims       = int(raw_attrs["n_sims"])
    n_maps       = len(params_map)

    params_raw = params_map[::maps_per_sim]   # [n_sims, 6]  sim-level
    print(f"  maps={raw['maps'].shape}  n_sims={n_sims}")

    # ── 파라미터 정규화 fit ───────────────────────────────────────────────────
    param_cfg = dict(param_norm_cfg or {})
    if param_norm_mode is not None:
        param_cfg["method"] = param_norm_mode

    if param_norm_mode is not None:
        param_mode = str(param_cfg.get("method", "legacy_zscore")).strip().lower()
        print(f"[build_splits] param norm: fit mode={param_mode} (CLI override)")
        param_norm_recipe = fit_param_normalization(params_raw, mode=param_mode)
    elif param_cfg.get("stats"):
        param_norm_recipe = param_cfg
        print(f"[build_splits] param norm: using provided YAML stats")
    else:
        param_mode = str(param_cfg.get("method", "legacy_zscore")).strip().lower()
        print(f"[build_splits] param norm: fit mode={param_mode}")
        param_norm_recipe = fit_param_normalization(params_raw, mode=param_mode)

    params_norm = normalize_params_numpy(params_map, param_norm_recipe)  # [N, 6]

    # ── split 결정 ────────────────────────────────────────────────────────────
    split_strategy = str(split_strategy).strip().lower()
    stratify_cfg   = None
    strat_labels   = None

    if split_strategy == "stratified_1d":
        stratify_idx = _resolve_param_index(stratify_param)
        strat_labels, strat_edges = _make_stratify_labels(
            params_raw, param_idx=stratify_idx, n_bins=int(stratify_bins)
        )
        uniq, cnt = np.unique(strat_labels, return_counts=True)
        stratify_cfg = {
            "param_index": int(stratify_idx),
            "param_name":  PARAM_NAMES[stratify_idx],
            "n_bins":      int(stratify_bins),
            "bin_edges":   strat_edges.tolist(),
            "label_counts": {int(u): int(c) for u, c in zip(uniq, cnt)},
        }
        print(f"[build_splits] strategy: stratified_1d (param={PARAM_NAMES[stratify_idx]}, bins={stratify_bins})")
    elif split_strategy == "random":
        print("[build_splits] strategy: random")
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy!r}. Options: random / stratified_1d")

    splits = _split_sim_ids(
        n_sims=n_sims, train_ratio=train_ratio, val_ratio=val_ratio,
        seed=seed, strategy=split_strategy, strat_labels=strat_labels,
    )
    print(
        f"[build_splits] sims: train={len(splits['train'])}  "
        f"val={len(splits['val'])}  test={len(splits['test'])}"
    )

    # ── 출력 zarr 사전 할당 (maps는 아직 비어있음) ────────────────────────────
    print(f"allocating → {out_path} ...")
    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
    out   = zarr.open_group(str(out_path), mode="w")
    sizes = {}

    # sim 순서(split 내 위치)를 기억: sim_id → (split_name, write_offset)
    sim_write_map: dict[int, tuple[str, int]] = {}

    for split_name, sims in splits.items():
        sims_arr = np.asarray(sims, dtype=np.int64)
        n        = len(sims_arr) * maps_per_sim
        sizes[split_name] = n

        grp = out.require_group(split_name)
        grp.create_dataset("maps",
                           shape=(n, 3, 256, 256), dtype="float32",
                           chunks=(8, 3, 256, 256), compressor=compressor)

        # params / sim_ids: 벡터화 인덱싱으로 한번에 채움
        map_idx = (sims_arr[:, None] * maps_per_sim + np.arange(maps_per_sim)).ravel()
        grp.create_dataset("params",  data=params_norm[map_idx],  dtype="float32")
        grp.create_dataset("sim_ids", data=sim_ids_map[map_idx],  dtype="int32")

        for pos, sim_id in enumerate(sims_arr):
            sim_write_map[int(sim_id)] = (split_name, pos * maps_per_sim)

        print(f"  {split_name}: {n} maps  ({len(sims_arr)} sims)")

    # ── maps: sim 단위 스트리밍 정규화 + 저장 ────────────────────────────────
    # raw zarr를 sim 순서(0→n_sims-1)로 순차 읽음 → 순차 I/O 최적
    # peak 메모리: maps_per_sim × 3 × 256 × 256 × 4 bytes = ~30MB
    print("normalizing maps (streaming) ...", flush=True)
    normalizer = Normalizer(map_norm_config)

    for sim_id in tqdm(range(n_sims), desc="normalizing"):
        split_name, write_offset = sim_write_map[sim_id]
        src_start = sim_id * maps_per_sim
        src_end   = src_start + maps_per_sim

        maps_chunk = np.array(raw["maps"][src_start:src_end], dtype=np.float32)
        maps_norm  = normalizer.normalize_numpy(maps_chunk)
        out[split_name]["maps"][write_offset : write_offset + maps_per_sim] = maps_norm

    # ── attrs (metadata) ──────────────────────────────────────────────────────
    out.attrs.update({
        "created":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source":              str(raw_zarr_path.resolve()),
        "normalization":       map_norm_config,
        "param_normalization": param_norm_recipe,
        "split": {
            "train_ratio":  train_ratio,
            "val_ratio":    val_ratio,
            "seed":         seed,
            "n_sims":       n_sims,
            "maps_per_sim": maps_per_sim,
            "strategy":     split_strategy,
            "stratify":     stratify_cfg,
            "sim_ids": {
                k: splits[k].tolist() for k in ("train", "val", "test")
            },
        },
        "sizes": sizes,
        **{k: raw_attrs[k] for k in ("suite", "sim", "redshift", "fields", "param_names")},
    })

    print(f"[build_splits] done → {out_path}")
    print(f"  train={sizes['train']}  val={sizes['val']}  test={sizes['test']}")


# ── Step 3: train split D4 물리적 증설 ────────────────────────────────────────

def materialize_augmentation(
    data_path: Path,
    out_path:  Path,
    copies:    int = 8,
):
    """dataset.zarr의 train을 D4 대칭으로 물리적 증설 → 새 zarr 저장.

    train만 증설 (copies배). val/test는 그대로 복사.
    copies=8: D4 8종 변환 (rot×4 × flip×2) 각 1회씩.
    """
    data_path = Path(data_path)
    out_path  = Path(out_path)

    if copies <= 0:
        raise ValueError(f"copies must be positive, got {copies}")
    if out_path.exists():
        print(f"[augment] 이미 존재함: {out_path}")
        print("  덮어쓰려면 기존 파일을 삭제 후 재실행.")
        return

    src = zarr.open_group(str(data_path), mode="r")
    out = zarr.open_group(str(out_path),  mode="w")

    train_maps   = src["train/maps"]
    train_params = src["train/params"]
    train_sids   = src["train/sim_ids"]
    n_train      = len(train_maps)

    print(f"[augment] source={data_path}")
    print(f"[augment] output={out_path}")
    print(f"[augment] train={train_maps.shape}  copies={copies} → {n_train*copies} maps")

    # ── train 증설 ────────────────────────────────────────────────────────────
    compressor  = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
    out_shape   = (n_train * copies, *train_maps.shape[1:])
    maps_out    = out.require_group("train").create_dataset(
        "maps", shape=out_shape, dtype="float32",
        chunks=(8, 3, 256, 256), compressor=compressor,
    )
    params_out  = out["train"].create_dataset(
        "params", shape=(n_train * copies, train_params.shape[1]), dtype="float32",
    )
    sids_out    = out["train"].create_dataset(
        "sim_ids", shape=(n_train * copies,), dtype="int32",
    )

    chunk = 128
    for copy_idx in tqdm(range(copies), desc="augment train"):
        start         = copy_idx * n_train
        transform_idx = copy_idx % 8
        for i in range(0, n_train, chunk):
            j     = min(i + chunk, n_train)
            batch = np.array(train_maps[i:j], dtype=np.float32)
            k     = transform_idx % 4
            if k > 0:
                batch = np.rot90(batch, k=k, axes=(-2, -1))
            if transform_idx >= 4:
                batch = np.flip(batch, axis=-1)
            maps_out[start + i : start + j] = np.ascontiguousarray(batch)
        params_out[start : start + n_train] = train_params[:]
        sids_out[start   : start + n_train] = train_sids[:]

    # ── val / test 그대로 복사 ────────────────────────────────────────────────
    for split in ("val", "test"):
        grp = out.require_group(split)
        for key in ("maps", "params", "sim_ids"):
            arr = np.array(src[f"{split}/{key}"])
            kw  = dict(chunks=(8, 3, 256, 256), compressor=compressor) \
                  if key == "maps" else {}
            grp.create_dataset(key, data=arr, dtype=arr.dtype, **kw)

    # ── attrs ─────────────────────────────────────────────────────────────────
    attrs = dict(src.attrs)
    attrs["created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    attrs["augmentation"] = {"source": str(data_path.resolve()), "method": "D4", "copies": copies}
    attrs.setdefault("sizes", {})
    attrs["sizes"]["train"] = n_train * copies
    out.attrs.update(attrs)

    print(f"[augment] done → {out_path}")
    print(f"  train={n_train*copies}  val={attrs['sizes'].get('val')}  test={attrs['sizes'].get('test')}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GENESIS dataset builder")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # stack
    p_stack = sub.add_parser("stack", help="채널별 npy + params → {SUITE}_{suite}.zarr")
    p_stack.add_argument("--maps-dir", type=Path, required=True,
                         help="CAMELS IllustrisTNG 디렉토리")
    p_stack.add_argument("--out-dir",  type=Path, default=None)
    p_stack.add_argument("--suite",    type=str,  default="LH",
                         choices=["LH", "CV", "EX", "1P"],
                         help="CAMELS suite (기본: LH)")

    # splits
    p_splits = sub.add_parser("splits", help="raw zarr → 정규화 + train/val/test → dataset.zarr")
    p_splits.add_argument("--raw-zarr",   type=Path, required=True,
                          help="stack으로 만든 raw zarr (예: IllustrisTNG_LH.zarr)")
    p_splits.add_argument("--out",        type=Path, required=True,
                          help="출력 dataset.zarr 경로")
    p_splits.add_argument("--norm-config", type=Path, required=True,
                          help="normalization YAML (예: configs/normalization/log_p1_p99.yaml)")
    p_splits.add_argument("--param-norm-mode", type=str, default=None,
                          choices=["legacy_zscore", "astro_mixed"],
                          help="파라미터 정규화 방식 override")
    p_splits.add_argument("--train-ratio", type=float, default=0.8)
    p_splits.add_argument("--val-ratio",   type=float, default=0.1)
    p_splits.add_argument("--seed",        type=int,   default=42)
    p_splits.add_argument("--split-strategy", type=str, default="stratified_1d",
                          choices=["stratified_1d", "random"])
    p_splits.add_argument("--stratify-param", type=str, default="Omega_m")
    p_splits.add_argument("--stratify-bins",  type=int, default=10)

    # augment
    p_aug = sub.add_parser("augment", help="dataset.zarr train을 D4 대칭으로 물리적 증설")
    p_aug.add_argument("--data-path", type=Path, required=True,
                       help="원본 dataset.zarr 경로")
    p_aug.add_argument("--out",       type=Path, required=True,
                       help="증설된 새 dataset.zarr 경로")
    p_aug.add_argument("--copies",    type=int, default=8,
                       help="증설 배수 (기본: 8, D4 대칭)")

    # recipe
    p_recipe = sub.add_parser("recipe", help="raw zarr에서 정규화 레시피 YAML 생성")
    p_recipe.add_argument("--raw-zarr",  type=Path, required=True,
                          help="stack 출력 zarr (예: IllustrisTNG_LH.zarr)")
    p_recipe.add_argument("--lower-percentile", type=float, default=0.0)
    p_recipe.add_argument("--upper-percentile", type=float, default=100.0)
    p_recipe.add_argument("--center-stat", choices=["mean", "median"], default="mean")
    p_recipe.add_argument("--range-mode",  choices=["centered", "symmetric"], default="centered")
    p_recipe.add_argument("--param-mode",  choices=["legacy_zscore", "astro_mixed"], default=None,
                          help="파라미터 정규화 방식 (미지정 시 YAML에 포함 안 함)")
    p_recipe.add_argument("--out", type=Path, default=None,
                          help="출력 YAML 경로 (기본: configs/normalization/<suite>_<suffix>.yaml)")

    args = parser.parse_args()

    if args.cmd == "stack":
        build(maps_dir=args.maps_dir, out_dir=args.out_dir, suite=args.suite)
        # 출력: {maps_dir 또는 out_dir}/IllustrisTNG_{suite}.zarr

    elif args.cmd == "splits":
        with open(args.norm_config) as f:
            cfg = yaml.safe_load(f)
        map_norm_config, param_norm_config = split_normalization_config(cfg.get("normalization", {}))
        build_splits(
            raw_zarr_path=args.raw_zarr,
            out_path=args.out,
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
        shutil.copy(args.norm_config, args.out / "normalization.yaml")
        print(f"[splits] recipe saved → {args.out}/normalization.yaml")

    elif args.cmd == "augment":
        materialize_augmentation(
            data_path=args.data_path,
            out_path=args.out,
            copies=args.copies,
        )

    elif args.cmd == "recipe":
        raw = zarr.open_group(str(args.raw_zarr), mode="r")
        raw_attrs = dict(raw.attrs)
        suite = raw_attrs.get("suite", args.raw_zarr.stem)
        maps_per_sim = int(raw_attrs.get("maps_per_sim", MAPS_PER_SIM))

        # maps: zarr [N, 3, 256, 256] — loaded channel-by-channel inside recipe
        maps = raw["maps"]
        print(f"[recipe] suite={suite}  maps={maps.shape}")

        payload = build_normalization_recipe(
            maps,
            lower_percentile=args.lower_percentile,
            upper_percentile=args.upper_percentile,
            center_stat=args.center_stat,
            range_mode=args.range_mode,
            param_mode=args.param_mode,
        )

        if args.param_mode is not None:
            # sim-level params: subsample 1 row per sim
            params = np.array(raw["params"][::maps_per_sim], dtype=np.float32)
            print(f"[recipe] fitting param norm: mode={args.param_mode}  n_sims={len(params)}")
            payload.setdefault("normalization", {})["params"] = fit_param_normalization(
                params, mode=args.param_mode
            )

        if args.out is None:
            suffix = f"{args.lower_percentile:g}_{args.upper_percentile:g}_{args.range_mode}"
            if args.range_mode == "centered":
                suffix = f"{suffix}_{args.center_stat}"
            if args.param_mode:
                suffix = f"{suffix}_{args.param_mode}"
            out = Path("configs/normalization") / f"{suite}_{suffix}.yaml"
        else:
            out = args.out
        save_normalization_recipe(payload, out)
        print(f"saved: {out}")
