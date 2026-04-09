# Quick Plot Commands

`04_make_plots.py`를 빠르게 돌릴 때 쓰는 명령어 모음입니다.

## Run Tag

```bash
RUN_TAG="unet__0330_ft_last_cosine_restarts_t0_3__dopri5_step50_cfg1.0_ngen32"
```

## 1) Paper 결과만 가장 빠르게

Per-condition diagnostic, visualize 출력 둘 다 생략합니다.
이 모드에서는 `--n-workers`가 적용되지 않습니다 (`--no-visualize` 이기 때문).

```bash
python /home/work/cosmology/GENESIS/paper_preparation/scripts/04_make_plots.py \
  --run-tag "$RUN_TAG" \
  --protocols lh cv 1p ex \
  --no-per-cond \
  --no-visualize
```

## 2) Diagnostic은 유지, Per-condition만 생략

```bash
python /home/work/cosmology/GENESIS/paper_preparation/scripts/04_make_plots.py \
  --run-tag "$RUN_TAG" \
  --protocols lh cv 1p ex \
  --no-per-cond \
  --n-workers 8
```

## 3) Visualize 그림만 생성 (병렬)

```bash
python /home/work/cosmology/GENESIS/paper_preparation/scripts/04_make_plots.py \
  --run-tag "$RUN_TAG" \
  --protocols lh cv 1p ex \
  --visualize-only \
  --n-workers 8
```

## 4) 기본 전체 실행 (가장 오래 걸림)

```bash
python /home/work/cosmology/GENESIS/paper_preparation/scripts/04_make_plots.py \
  --run-tag "$RUN_TAG" \
  --protocols lh cv 1p ex \
  --n-workers 8
```

## 참고

- `04_make_plots.py` 실행 전 `eval/` 폴더가 필요합니다.
- `--n-workers`는 visualize 단계에서만 사용됩니다.
  - `--no-visualize`면 worker 수를 바꿔도 영향이 없습니다.
- 아직 평가를 안 했다면 먼저 아래 실행:

```bash
python /home/work/cosmology/GENESIS/paper_preparation/scripts/03_evaluate.py \
  --run-tag "$RUN_TAG" \
  --protocols lh cv 1p ex \
  --force
```
