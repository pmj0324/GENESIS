"""
Convert training metrics JSON files to best-only format.

What it does:
  - scans <root> recursively for `metrics_history.json`
  - selects one best record (val_loss minimum epoch)
  - overwrites `metrics_history.json` as a one-element list [best_record]
  - writes `metrics_best.json` as a single dict best_record

Best epoch source priority:
  1) sibling best.pt -> best_epoch
  2) sibling best.pt -> epoch + 1 (legacy checkpoint without best_epoch)
  3) metrics entries containing val_loss -> argmin(val_loss)
  4) fallback: latest epoch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older torch versions do not support `weights_only`.
        return torch.load(path, map_location="cpu")
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _pick_best_epoch(records: list[dict], ckpt: dict | None) -> tuple[int, str, float | None]:
    if ckpt is not None:
        best_epoch = _safe_int(ckpt.get("best_epoch"))
        best_val = _safe_float(ckpt.get("best_val", ckpt.get("val_loss")))
        if best_epoch is not None and best_epoch > 0:
            return best_epoch, "checkpoint.best_epoch", best_val

        epoch_raw = _safe_int(ckpt.get("epoch"))
        if epoch_raw is not None:
            return epoch_raw + 1, "checkpoint.epoch_plus_one", best_val

    # Try val_loss inside JSON records (newer format)
    with_val = []
    for r in records:
        v = _safe_float(r.get("val_loss"))
        e = _safe_int(r.get("epoch"))
        if v is not None and e is not None:
            with_val.append((v, e))
    if len(with_val) > 0:
        best_val, best_epoch = min(with_val, key=lambda x: x[0])
        return best_epoch, "metrics_history.val_loss", best_val

    # Fallback: latest epoch
    latest = max((_safe_int(r.get("epoch")) or 0) for r in records)
    if latest <= 0:
        latest = 1
    return latest, "latest_epoch_fallback", None


def _select_best_record(records: list[dict], best_epoch: int) -> dict:
    # Exact match
    for r in records:
        if (_safe_int(r.get("epoch")) or -1) == best_epoch:
            return dict(r)

    # Closest epoch fallback
    return dict(
        min(
            records,
            key=lambda r: abs((_safe_int(r.get("epoch")) or 0) - best_epoch),
        )
    )


def migrate_file(metrics_path: Path, *, dry_run: bool = False) -> tuple[bool, str]:
    try:
        loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"failed to read JSON: {exc}"

    if isinstance(loaded, list):
        records = [x for x in loaded if isinstance(x, dict)]
    elif isinstance(loaded, dict):
        records = [loaded]
    else:
        return False, "unsupported JSON type"

    if len(records) == 0:
        return False, "empty records"

    ckpt = _load_checkpoint(metrics_path.parent / "best.pt")
    best_epoch, selector, best_val = _pick_best_epoch(records, ckpt)
    best = _select_best_record(records, best_epoch)

    # Fill explicit best metadata for downstream scripts.
    best["best_epoch_by_val_loss"] = int(best_epoch)
    if best_val is not None:
        best["best_val_loss"] = float(best_val)
        best.setdefault("val_loss", float(best_val))
    best["best_selector"] = selector

    if dry_run:
        return True, f"would keep epoch={best.get('epoch')} ({selector})"

    metrics_path.write_text(
        json.dumps([best], indent=2, allow_nan=True),
        encoding="utf-8",
    )
    (metrics_path.parent / "metrics_best.json").write_text(
        json.dumps(best, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    return True, f"kept epoch={best.get('epoch')} ({selector})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate metrics_history.json to best-only format")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("runs"),
        help="Root directory to scan (default: runs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without writing files",
    )
    args = parser.parse_args()

    paths = sorted(args.root.rglob("metrics_history.json"))
    if len(paths) == 0:
        print(f"[migrate] no metrics_history.json found under: {args.root}")
        return

    ok = 0
    fail = 0
    for p in paths:
        changed, msg = migrate_file(p, dry_run=args.dry_run)
        rel = p.as_posix()
        if changed:
            ok += 1
            print(f"[ok]  {rel}  -> {msg}")
        else:
            fail += 1
            print(f"[skip] {rel}  ({msg})")

    mode = "DRY-RUN" if args.dry_run else "WRITE"
    print(f"[migrate] mode={mode} total={len(paths)} ok={ok} skipped={fail}")


if __name__ == "__main__":
    main()
