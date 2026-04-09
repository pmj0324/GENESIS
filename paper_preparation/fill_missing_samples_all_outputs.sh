#!/usr/bin/env bash
set -euo pipefail

# Fill missing per-condition samples for every run_tag under paper_preparation/output.
#
# It reads each run's manifest.json and re-runs 01_generate_samples.py in resume mode
# (without --force), so only missing/incomplete conditions are generated.
#
# Environment overrides:
#   PROJECT_ROOT   (default: /home/work/cosmology/GENESIS)
#   OUTPUT_ROOT    (default: $PROJECT_ROOT/paper_preparation/output)
#   DEVICE         (default: cuda)
#   MAX_BATCH      (default: 64)
#   PROTOCOL       (default: all)        # lh / cv / 1p / ex / all
#   MAX_CONDS      (default: empty)      # optional smoke cap
#   ONLY_TAG       (default: empty)      # run only one run_tag
#   DRY_RUN        (default: 0)          # 1 = print commands only
#
# Example:
#   bash paper_preparation/fill_missing_samples_all_outputs.sh
#   ONLY_TAG="unet__..._ngen32" bash paper_preparation/fill_missing_samples_all_outputs.sh
#   DRY_RUN=1 bash paper_preparation/fill_missing_samples_all_outputs.sh

PROJECT_ROOT="${PROJECT_ROOT:-/home/work/cosmology/GENESIS}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/paper_preparation/output}"
DEVICE="${DEVICE:-cuda}"
MAX_BATCH="${MAX_BATCH:-64}"
PROTOCOL="${PROTOCOL:-all}"
MAX_CONDS="${MAX_CONDS:-}"
ONLY_TAG="${ONLY_TAG:-}"
DRY_RUN="${DRY_RUN:-0}"

GEN_SCRIPT="$PROJECT_ROOT/paper_preparation/scripts/01_generate_samples.py"

if [[ ! -f "$GEN_SCRIPT" ]]; then
  echo "[error] generation script not found: $GEN_SCRIPT" >&2
  exit 1
fi

if [[ ! -d "$OUTPUT_ROOT" ]]; then
  echo "[error] output root not found: $OUTPUT_ROOT" >&2
  exit 1
fi

resolve_path() {
  # Resolve ckpt/config paths stored in manifest to existing filesystem paths.
  # Supports absolute, repo-relative, and 'GENESIS/...' style relative paths.
  local raw="$1"
  local project_root="$2"
  python - "$raw" "$project_root" <<'PY'
from pathlib import Path
import sys

raw = sys.argv[1].strip()
root = Path(sys.argv[2]).resolve()

cands = []
p = Path(raw)
if p.is_absolute():
    cands.append(p)
else:
    cands.append(root / raw)
    cands.append(root.parent / raw)
    if raw.startswith("GENESIS/"):
        cands.append(root / raw[len("GENESIS/"):])
        cands.append(root.parent / raw)

for c in cands:
    if c.exists():
        print(str(c.resolve()))
        raise SystemExit(0)

print("")
PY
}

echo "[fill] PROJECT_ROOT=$PROJECT_ROOT"
echo "[fill] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[fill] PROTOCOL=$PROTOCOL DEVICE=$DEVICE MAX_BATCH=$MAX_BATCH DRY_RUN=$DRY_RUN"

processed=0
skipped=0
failed=0

while IFS= read -r run_dir; do
  run_tag="$(basename "$run_dir")"
  if [[ -n "$ONLY_TAG" && "$run_tag" != "$ONLY_TAG" ]]; then
    continue
  fi

  manifest="$run_dir/manifest.json"
  if [[ ! -f "$manifest" ]]; then
    echo "[skip] $run_tag  (manifest.json not found)"
    skipped=$((skipped + 1))
    continue
  fi

  # Extract values from manifest.
  readarray -t mf < <(python - "$manifest" <<'PY'
import json, sys
from pathlib import Path

m = json.loads(Path(sys.argv[1]).read_text())
g = m.get("generator", {})
print(g.get("checkpoint", ""))
print(g.get("config", ""))
print(m.get("n_gen", 32))
print(m.get("seed_base", 42))
print(g.get("max_batch", 64))
PY
)

  ckpt_raw="${mf[0]:-}"
  cfg_raw="${mf[1]:-}"
  n_gen="${mf[2]:-32}"
  seed_base="${mf[3]:-42}"
  manifest_max_batch="${mf[4]:-64}"

  if [[ -z "$ckpt_raw" || -z "$cfg_raw" ]]; then
    echo "[skip] $run_tag  (checkpoint/config missing in manifest)"
    skipped=$((skipped + 1))
    continue
  fi

  ckpt="$(resolve_path "$ckpt_raw" "$PROJECT_ROOT")"
  cfg="$(resolve_path "$cfg_raw" "$PROJECT_ROOT")"

  if [[ -z "$ckpt" || -z "$cfg" ]]; then
    echo "[skip] $run_tag  (cannot resolve ckpt/config path)"
    echo "       ckpt_raw=$ckpt_raw"
    echo "       cfg_raw=$cfg_raw"
    skipped=$((skipped + 1))
    continue
  fi

  # Prefer user MAX_BATCH override; otherwise keep manifest max_batch.
  use_max_batch="$MAX_BATCH"
  if [[ -z "${MAX_BATCH:-}" ]]; then
    use_max_batch="$manifest_max_batch"
  fi

  cmd=(
    python "$GEN_SCRIPT"
    --ckpt "$ckpt"
    --config "$cfg"
    --protocol "$PROTOCOL"
    --n-gen "$n_gen"
    --seed-base "$seed_base"
    --device "$DEVICE"
    --max-batch "$use_max_batch"
    --tag "$run_tag"
  )
  if [[ -n "$MAX_CONDS" ]]; then
    cmd+=(--max-conds "$MAX_CONDS")
  fi

  echo
  echo "[run] $run_tag"
  echo "      ckpt=$ckpt"
  echo "      cfg=$cfg"
  echo "      n_gen=$n_gen seed_base=$seed_base protocol=$PROTOCOL"
  echo "      cmd: ${cmd[*]}"

  if [[ "$DRY_RUN" == "1" ]]; then
    processed=$((processed + 1))
    continue
  fi

  if "${cmd[@]}"; then
    processed=$((processed + 1))
  else
    echo "[fail] $run_tag"
    failed=$((failed + 1))
  fi
done < <(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)

echo
echo "[done] processed=$processed skipped=$skipped failed=$failed"
if [[ "$failed" -gt 0 ]]; then
  exit 1
fi

