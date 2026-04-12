# GENESIS-data 역할 정리 (Code + Recipe + Dataset)

이 문서는 `GENESIS-data/` 아래 파일/디렉토리의 역할과 데이터 생성 흐름을 정리합니다.

## 1) 디렉토리/파일 역할

- `GENESIS-data/make_normalization_recipe.py`
  - 정규화 Recipe YAML 생성 스크립트.
  - 맵 정규화 통계(`normalization.maps`)를 계산해서 저장.
  - `--params-path`를 주면 파라미터 정규화 stats(`normalization.params.stats`)도 같이 저장.

- `GENESIS-data/recipes/*.yaml`
  - 재사용 가능한 정규화 레시피 저장소.
  - `dataloader.build_dataset splits --norm-config`로 직접 사용.

- `GENESIS-data/<dataset_name>/`
  - 실제 학습/평가용 정규화 데이터셋 디렉토리.
  - `train/val/test_maps.npy`, `train/val/test_params.npy`, `split_*.npy`, `metadata.yaml` 포함.

- `GENESIS-data/<dataset_name>/metadata.yaml`
  - 실제 생성에 사용된 최종 정규화 설정 기록.
  - 맵 정규화는 `normalization`에, 파라미터 정규화는 `param_normalization`에 저장.

## 2) 표준 생성 순서

## 2-1) (선택) 3채널 맵 스택 생성

`Maps_3ch_IllustrisTNG_LH_z=0.00.npy`가 이미 있으면 생략.

```bash
python -m dataloader.build_dataset stack \
  --maps-dir /home/work/cosmology/CAMELS/IllustrisTNG
```

## 2-2) Recipe YAML 생성

아래 예시는 `log10 -> minmax_sym([-1,1])` + `astro_mixed`입니다.

```bash
python GENESIS-data/make_normalization_recipe.py \
  --maps-path /home/work/cosmology/CAMELS/IllustrisTNG/Maps_3ch_IllustrisTNG_LH_z=0.00.npy \
  --params-path /home/work/cosmology/CAMELS/IllustrisTNG/params_LH_IllustrisTNG.txt \
  --lower-percentile 0 \
  --upper-percentile 100 \
  --range-mode symmetric \
  --param-mode astro_mixed \
  --out GENESIS-data/recipes/log_minmax_sym_channelwise_astro_mixed.yaml
```

## 2-3) Recipe로 dataset 생성

```bash
python -m dataloader.build_dataset splits \
  --maps-path /home/work/cosmology/CAMELS/IllustrisTNG/Maps_3ch_IllustrisTNG_LH_z=0.00.npy \
  --params-path /home/work/cosmology/CAMELS/IllustrisTNG/params_LH_IllustrisTNG.txt \
  --out-dir GENESIS-data/log_minmax_sym_channelwise_astro_mixed \
  --norm-config GENESIS-data/recipes/log_minmax_sym_channelwise_astro_mixed.yaml
```

## 3) 파라미터 정규화 우선순위 (중요)

`build_dataset splits`에서 파라미터 정규화는 아래 우선순위로 결정됩니다.

1. `--param-norm-mode`를 CLI로 주면 항상 raw params에서 다시 fit
2. 아니고 YAML에 `normalization.params.stats`가 있으면 그 stats를 그대로 재사용
3. 둘 다 없으면 YAML의 `method`(또는 default)로 raw params에서 fit

즉, 완전 재현이 목적이면 Recipe 생성 시 `--params-path`를 함께 주는 것을 권장합니다.

## 4) 결과물

`--out-dir` 아래 생성:

- `train_maps.npy`, `val_maps.npy`, `test_maps.npy`
- `train_params.npy`, `val_params.npy`, `test_params.npy`
- `split_train.npy`, `split_val.npy`, `split_test.npy`
- `metadata.yaml`

## 5) 새 npy에 동일 정규화 재사용할 때

- 맵: `Normalizer(meta["normalization"])`
- 파라미터: `ParamNormalizer.from_metadata(meta)`

주의: `split_normalization_config(meta["normalization"])`만으로는 파라미터 stats가 안 나오므로,
파라미터는 반드시 `meta["param_normalization"]`을 사용해야 합니다.
