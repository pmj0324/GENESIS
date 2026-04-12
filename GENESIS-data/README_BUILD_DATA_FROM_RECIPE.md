# GENESIS-data 데이터 생성 가이드 (Recipe -> Dataset)

아래 순서대로 하면 됩니다.

## 0) (선택) 3채널 맵 파일이 없으면 먼저 생성

`Maps_3ch_IllustrisTNG_LH_z=0.00.npy`가 이미 있으면 이 단계는 건너뛰세요.

```bash
python -m dataloader.build_dataset stack \
  --maps-dir /home/work/cosmology/CAMELS/IllustrisTNG
```

## 1) 정규화 Recipe YAML 생성

아래 예시는 `log10 -> minmax_sym([-1, 1])` + `params: astro_mixed` 레시피를 만듭니다.

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

## 2) Recipe로 dataset 생성 (splits)

```bash
python -m dataloader.build_dataset splits \
  --maps-path /home/work/cosmology/CAMELS/IllustrisTNG/Maps_3ch_IllustrisTNG_LH_z=0.00.npy \
  --params-path /home/work/cosmology/CAMELS/IllustrisTNG/params_LH_IllustrisTNG.txt \
  --out-dir GENESIS-data/log_minmax_sym_channelwise_astro_mixed \
  --norm-config GENESIS-data/recipes/log_minmax_sym_channelwise_astro_mixed.yaml
```

## 3) 결과물 확인

`--out-dir` 아래에 아래 파일들이 생성됩니다.

- `train_maps.npy`, `val_maps.npy`, `test_maps.npy`
- `train_params.npy`, `val_params.npy`, `test_params.npy`
- `split_train.npy`, `split_val.npy`, `split_test.npy`
- `metadata.yaml`

## 핵심 정리

- `--params-path`를 주면 Recipe YAML 안에
  `normalization.params.stats`(파라미터 정규화 통계)까지 저장됩니다.
- `--params-path` 없이 만들면 `normalization.params.method`만 저장되고,
  실제 파라미터 통계는 `splits` 실행 시 fit되어 `metadata.yaml`에 저장됩니다.
