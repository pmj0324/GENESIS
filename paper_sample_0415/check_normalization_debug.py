#!/usr/bin/env python3
"""Debug normalization round-trips for LH split arrays.

This script checks one sample from <data_dir>/<split>_{maps,params}.npy against
the raw CAMELS source files recorded in metadata.yaml.

It supports two ways to choose a sample:
  1. --k <row-index> in the split arrays
  2. --condition-index <cond> --map-index <local-map>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.normalization import PARAM_NAMES, Normalizer, ParamNormalizer  # noqa: E402


DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs/experiments/flow/unet/"
    / "unet_flow_0414_minmaxsym_perscale_only_ft_plateau_lr2e5_f065_p4_es20.yaml"
)


def load_cfg(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_config_path(config_arg: str | None, yaml_arg: str | None) -> Path:
    chosen = yaml_arg if yaml_arg is not None else config_arg
    if chosen is None:
        return DEFAULT_CONFIG
    return Path(chosen)


def resolve_data_dir(config_path: Path, data_dir_arg: str | None) -> Path:
    if data_dir_arg is not None:
        return Path(data_dir_arg)
    cfg = load_cfg(config_path)
    data_dir = cfg.get("data", {}).get("data_dir")
    if not data_dir:
        raise ValueError("--data-dir is required because config.data.data_dir is missing.")
    return Path(data_dir)


def load_metadata(data_dir: Path) -> dict:
    meta_path = data_dir / "metadata.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.yaml not found: {meta_path}")
    with open(meta_path) as f:
        return yaml.safe_load(f)


def resolve_source_path(data_dir: Path, value: str | None, key: str) -> Path:
    if not value:
        raise KeyError(f"metadata.{key} is missing")
    path = Path(value)
    if not path.is_absolute():
        path = data_dir / path
    if not path.exists():
        raise FileNotFoundError(f"{key} not found: {path}")
    return path


def load_split_ids(data_dir: Path, split: str, meta: dict) -> np.ndarray | None:
    split_cfg = meta.get("split", {})
    id_files = split_cfg.get("id_files", {})
    legacy_files = split_cfg.get("legacy_id_files", {})

    candidates = []
    if split in id_files:
        candidates.append(data_dir / id_files[split])
    candidates.append(data_dir / f"split_{split}.npy")
    if split in legacy_files:
        candidates.append(data_dir / legacy_files[split])
    candidates.append(data_dir / f"{split}_sim_ids.npy")

    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return np.load(path)
    return None


def summarize_map_errors(x_map: np.ndarray, z_map: np.ndarray, map_norm: Normalizer) -> dict:
    z_map_from_raw = map_norm.normalize_numpy(x_map)
    x_map_rec = map_norm.denormalize_numpy(z_map)
    abs_err = np.abs(x_map - x_map_rec)
    rel_err = abs_err / np.clip(np.abs(x_map), 1e-30, None)

    summary = {
        "train_z_vs_normalize_raw_allclose": bool(np.allclose(z_map, z_map_from_raw, atol=1e-5)),
        "train_z_vs_normalize_raw_max_diff": float(np.max(np.abs(z_map - z_map_from_raw))),
        "raw_vs_denorm_train_z_max_diff": float(abs_err.max()),
        "raw_vs_denorm_train_z_mean_diff": float(abs_err.mean()),
        "roundtrip_max_diff": float(np.max(np.abs(z_map - map_norm.normalize_numpy(x_map_rec)))),
        "per_channel": {},
    }

    for idx, name in enumerate(["Mcdm", "Mgas", "T"]):
        summary["per_channel"][name] = {
            "raw_min": float(x_map[idx].min()),
            "raw_max": float(x_map[idx].max()),
            "rec_min": float(x_map_rec[idx].min()),
            "rec_max": float(x_map_rec[idx].max()),
            "max_abs_diff": float(np.max(np.abs(x_map[idx] - x_map_rec[idx]))),
            "mean_abs_diff": float(np.mean(np.abs(x_map[idx] - x_map_rec[idx]))),
            "max_rel_diff": float(rel_err[idx].max()),
        }
    return summary


def summarize_param_errors(
    x_param: np.ndarray,
    z_param: np.ndarray,
    theta_block: np.ndarray,
    param_norm: ParamNormalizer,
) -> dict:
    z_param_from_raw = param_norm.normalize_numpy(x_param)
    x_param_rec = param_norm.denormalize_numpy(z_param)
    abs_err = np.abs(x_param - x_param_rec)
    rel_err = abs_err / np.clip(np.abs(x_param), 1e-30, None)
    block_spread = np.max(np.abs(theta_block - z_param[None, :]), axis=0)

    summary = {
        "train_z_vs_normalize_raw_allclose": bool(np.allclose(z_param, z_param_from_raw, atol=1e-6)),
        "train_z_vs_normalize_raw_max_diff": float(np.max(np.abs(z_param - z_param_from_raw))),
        "raw_vs_denorm_train_z_max_diff": float(abs_err.max()),
        "raw_vs_denorm_train_z_mean_diff": float(abs_err.mean()),
        "roundtrip_max_diff": float(np.max(np.abs(z_param - param_norm.normalize_numpy(x_param_rec)))),
        "split_block_consistent_atol_1e-6": bool(
            np.allclose(theta_block, z_param[None, :], atol=1e-6, rtol=0.0)
        ),
        "split_block_max_norm_spread": {
            name: float(block_spread[i]) for i, name in enumerate(PARAM_NAMES)
        },
        "raw_values": [float(x) for x in x_param],
        "norm_values": [float(x) for x in z_param],
        "rec_values": [float(x) for x in x_param_rec],
        "per_param_abs_diff": {
            name: float(abs_err[i]) for i, name in enumerate(PARAM_NAMES)
        },
        "per_param_rel_diff": {
            name: float(rel_err[i]) for i, name in enumerate(PARAM_NAMES)
        },
    }
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--yaml", default=None, help="alias of --config")
    p.add_argument("--data-dir", default=None, help="dataset dir; if omitted, use config.data.data_dir")
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--k", type=int, default=0, help="row index inside <split>_{maps,params}.npy")
    p.add_argument("--condition-index", type=int, default=None, help="optional condition index override")
    p.add_argument("--map-index", type=int, default=None, help="optional map index within condition")
    p.add_argument("--output-json", default=None, help="optional path to save the summary json")
    p.add_argument("--print-json", action="store_true", help="print the full summary json at the end")
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    config_path = resolve_config_path(args.config, args.yaml)
    data_dir = resolve_data_dir(config_path, args.data_dir)
    meta = load_metadata(data_dir)

    maps_per_sim = int(meta.get("split", {}).get("maps_per_sim", 15))
    source_maps_path = resolve_source_path(data_dir, meta.get("source_maps"), "source_maps")
    source_params_path = resolve_source_path(data_dir, meta.get("source_params"), "source_params")

    map_cfg = meta.get("normalization", {})
    param_cfg = meta.get("param_normalization", meta.get("params_normalization", {}))
    map_norm = Normalizer(map_cfg)
    param_norm = ParamNormalizer.from_metadata(meta)

    split_maps = np.load(data_dir / f"{args.split}_maps.npy", mmap_mode="r")
    split_params = np.load(data_dir / f"{args.split}_params.npy", mmap_mode="r")
    split_ids = load_split_ids(data_dir, args.split, meta)
    raw_maps = np.load(source_maps_path, mmap_mode="r")
    raw_params = np.loadtxt(source_params_path, dtype=np.float32)

    if len(split_maps) != len(split_params):
        raise ValueError(
            f"{args.split}_maps rows ({len(split_maps)}) != {args.split}_params rows ({len(split_params)})"
        )
    if len(split_params) % maps_per_sim != 0:
        raise ValueError(
            f"{args.split}_params rows ({len(split_params)}) is not divisible by maps_per_sim ({maps_per_sim})"
        )

    n_conditions = len(split_params) // maps_per_sim
    if split_ids is not None and len(split_ids) != n_conditions:
        raise ValueError(
            f"{args.split} split ids length ({len(split_ids)}) != condition count ({n_conditions})"
        )

    if args.condition_index is not None or args.map_index is not None:
        if args.condition_index is None or args.map_index is None:
            raise ValueError("use --condition-index and --map-index together")
        sim_pos = int(args.condition_index)
        local_idx = int(args.map_index)
        if not (0 <= sim_pos < n_conditions):
            raise ValueError(f"--condition-index out of range: {sim_pos} (n_conditions={n_conditions})")
        if not (0 <= local_idx < maps_per_sim):
            raise ValueError(f"--map-index out of range: {local_idx} (maps_per_sim={maps_per_sim})")
        k = sim_pos * maps_per_sim + local_idx
    else:
        k = int(args.k)
        if not (0 <= k < len(split_maps)):
            raise ValueError(f"--k out of range: {k} (n_rows={len(split_maps)})")
        sim_pos = k // maps_per_sim
        local_idx = k % maps_per_sim

    sim_id = int(split_ids[sim_pos]) if split_ids is not None else sim_pos
    raw_map_idx = sim_id * maps_per_sim + local_idx

    x_map = np.asarray(raw_maps[raw_map_idx], dtype=np.float32)
    z_map = np.asarray(split_maps[k], dtype=np.float32)
    x_param = np.asarray(raw_params[sim_id], dtype=np.float32)
    z_param = np.asarray(split_params[k], dtype=np.float32)
    theta_block = np.asarray(
        split_params[sim_pos * maps_per_sim : (sim_pos + 1) * maps_per_sim],
        dtype=np.float32,
    )

    map_summary = summarize_map_errors(x_map, z_map, map_norm)
    param_summary = summarize_param_errors(x_param, z_param, theta_block, param_norm)

    summary = {
        "data_dir": str(data_dir.resolve()),
        "split": args.split,
        "k": int(k),
        "maps_per_sim": int(maps_per_sim),
        "condition_index": int(sim_pos),
        "map_index_within_condition": int(local_idx),
        "sim_id": int(sim_id),
        "raw_map_idx": int(raw_map_idx),
        "source_maps": str(source_maps_path.resolve()),
        "source_params": str(source_params_path.resolve()),
        "map_cfg": map_cfg,
        "param_cfg": param_cfg,
        "shapes": {
            "split_maps": list(split_maps.shape),
            "split_params": list(split_params.shape),
            "split_ids": None if split_ids is None else list(split_ids.shape),
            "raw_maps": list(raw_maps.shape),
            "raw_params": list(raw_params.shape),
        },
        "map_check": map_summary,
        "param_check": param_summary,
    }

    print("=== METADATA ===")
    print("data_dir      :", data_dir)
    print("split         :", args.split)
    print("maps_per_sim  :", maps_per_sim)
    print("source_maps   :", source_maps_path, "| exists:", source_maps_path.exists())
    print("source_params :", source_params_path, "| exists:", source_params_path.exists())
    print("normalization keys:", list(map_cfg.keys()))

    print("\n=== NORMALIZER ===")
    print("map_cfg   :", map_cfg)
    print("param_cfg :", param_cfg)

    print("\n=== SHAPES ===")
    print("split_maps  :", split_maps.shape, "dtype:", split_maps.dtype)
    print("split_params:", split_params.shape, "dtype:", split_params.dtype)
    if split_ids is not None:
        print("split_ids   :", split_ids.shape, "dtype:", split_ids.dtype)
    else:
        print("split_ids   : <none>")
    print("raw_maps    :", raw_maps.shape, "dtype:", raw_maps.dtype)
    print("raw_params  :", raw_params.shape, "dtype:", raw_params.dtype)

    print("\n=== INDEX ===")
    print(
        f"k={k} -> condition_index={sim_pos}, map_index={local_idx}, "
        f"sim_id={sim_id}, raw_map_idx={raw_map_idx}"
    )

    print("\n=== SAMPLE SHAPES ===")
    print("x_map   :", x_map.shape, "min:", float(x_map.min()), "max:", float(x_map.max()))
    print("z_map   :", z_map.shape, "min:", float(z_map.min()), "max:", float(z_map.max()))
    print("x_param :", x_param.shape, "values:", x_param.tolist())
    print("z_param :", z_param.shape, "values:", z_param.tolist())

    print("\n=== MAP CHECK ===")
    print(
        "split z vs normalize(raw)  allclose:",
        map_summary["train_z_vs_normalize_raw_allclose"],
    )
    print(
        "split z vs normalize(raw)  max diff:",
        map_summary["train_z_vs_normalize_raw_max_diff"],
    )
    print(
        "raw vs denorm(split z)     max diff:",
        map_summary["raw_vs_denorm_train_z_max_diff"],
    )
    print(
        "raw vs denorm(split z)    mean diff:",
        map_summary["raw_vs_denorm_train_z_mean_diff"],
    )
    for name, stats in map_summary["per_channel"].items():
        print(
            f"  {name:<5} raw range [{stats['raw_min']:.4g}, {stats['raw_max']:.4g}]"
            f" | rec range [{stats['rec_min']:.4g}, {stats['rec_max']:.4g}]"
            f" | max diff {stats['max_abs_diff']:.4g}"
        )
    print("roundtrip max diff:", map_summary["roundtrip_max_diff"])

    print("\n=== PARAM CHECK ===")
    print(
        "split z vs normalize(raw)  allclose:",
        param_summary["train_z_vs_normalize_raw_allclose"],
    )
    print(
        "split z vs normalize(raw)  max diff:",
        param_summary["train_z_vs_normalize_raw_max_diff"],
    )
    print(
        "raw vs denorm(split z)     max diff:",
        param_summary["raw_vs_denorm_train_z_max_diff"],
    )
    print(
        "raw vs denorm(split z)    mean diff:",
        param_summary["raw_vs_denorm_train_z_mean_diff"],
    )
    print(
        "split block consistent     :",
        param_summary["split_block_consistent_atol_1e-6"],
    )
    for idx, name in enumerate(PARAM_NAMES):
        raw_v = param_summary["raw_values"][idx]
        rec_v = param_summary["rec_values"][idx]
        diff_v = param_summary["per_param_abs_diff"][name]
        print(f"  {name:<8} raw={raw_v:.6f}  rec={rec_v:.6f}  diff={diff_v:.2e}")
    print("roundtrip max diff:", param_summary["roundtrip_max_diff"])

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"\nsummary json saved to: {out_path}")

    if args.print_json:
        print("\n=== SUMMARY JSON ===")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(parse_args())
