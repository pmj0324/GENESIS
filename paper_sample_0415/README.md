# paper_sample_0415

0414 `minmax_sym` UNet `last.pt` 기준 paper-style 샘플링 폴더입니다.

기본 스크립트:

```bash
python paper_sample_0415/generate_samples.py
```

LH split 기준 전용 스크립트:

```bash
python paper_sample_0415/generate_samples_LH_test.py
```

run 디렉터리의 YAML을 직접 쓰고 싶으면:

```bash
python paper_sample_0415/generate_samples.py \
  --yaml /home/work/cosmology/GENESIS/runs/flow/unet/0414_unet_flow_minmaxsym_perscale_only_ft_plateau_lr2e5_f065_p4_es20/config_resume.yaml
```

이 경우 `--ckpt`를 따로 안 주면 같은 폴더의 `last.pt`를 자동으로 찾습니다.

기본값:

- checkpoint:
  `/home/work/cosmology/GENESIS/runs/flow/unet/0414_unet_flow_minmaxsym_perscale_only_ft_plateau_lr2e5_f065_p4_es20/last.pt`
- config:
  `configs/experiments/flow/unet/unet_flow_0414_minmaxsym_perscale_only_ft_plateau_lr2e5_f065_p4_es20.yaml`
- protocol:
  `lh`
- LH split:
  `test`
- n_gen:
  `32`

출력 구조:

```text
paper_sample_0415/output/<run_tag>/
  manifest.json
  samples/lh/cond_000/
    gen_norm.npy
    true_norm.npy
    theta_norm.npy
    theta_phys.npy
    meta.json
```

예시:

```bash
python paper_sample_0415/generate_samples.py \
  --protocol lh \
  --n-gen 32
```

```bash
python paper_sample_0415/generate_samples.py \
  --protocol cv \
  --n-gen 32
```

```bash
python paper_sample_0415/generate_samples.py \
  --protocol lh \
  --lh-split test \
  --max-conds 5 \
  --tag smoke_lh_0415
```

LH split 전용 예시:

```bash
python paper_sample_0415/generate_samples_LH_test.py \
  --yaml /home/work/cosmology/GENESIS/runs/flow/unet/0414_unet_flow_minmaxsym_perscale_only_ft_plateau_lr2e5_f065_p4_es20/config_resume.yaml \
  --split test
```

```bash
python paper_sample_0415/generate_samples_LH_test.py \
  --yaml /home/work/cosmology/GENESIS/runs/flow/unet/0414_unet_flow_minmaxsym_perscale_only_ft_plateau_lr2e5_f065_p4_es20/config_resume.yaml \
  --data-dir /home/work/cosmology/GENESIS/GENESIS-data/log_minmax_sym_channelwise_astro_mixed \
  --split val \
  --n-gen 15
```

```bash
python paper_sample_0415/generate_samples_LH_test.py \
  --yaml /home/work/cosmology/GENESIS/runs/flow/unet/0414_unet_flow_minmaxsym_perscale_only_ft_plateau_lr2e5_f065_p4_es20/config_resume.yaml \
  --split train \
  --normalization-only
```

정규화만 확인:

```bash
python paper_sample_0415/generate_samples.py \
  --yaml /home/work/cosmology/GENESIS/runs/flow/unet/0414_unet_flow_minmaxsym_perscale_only_ft_plateau_lr2e5_f065_p4_es20/config_resume.yaml \
  --normalization-only
```

이 모드에서는 출력 폴더에 아래 파일이 생깁니다.

- `normalization_summary.json`
- `normalization_summary.txt`

메모:

- `lh`는 `<data_dir>/<split>_maps.npy`, `<data_dir>/<split>_params.npy`를 사용합니다.
- `cv`는 `<data_dir>/cv_maps.npy`, `<data_dir>/cv_params.npy`가 있어야 합니다.
- map 정규화는 `data_dir/metadata.yaml`의 `normalization` 섹션을 그대로 읽습니다.
- param 정규화는 `metadata.yaml`의 `param_normalization`이 있으면 그걸 쓰고, 없으면 코드의 legacy z-score 기본값을 사용합니다.
- 지금 버전은 샘플링 코드만 먼저 넣은 상태입니다. 평가/플롯은 아직 포함하지 않았습니다.

`generate_samples_LH_test.py` 메모:

- `--split`은 `train`, `val`, `test` 중 하나이고 기본값은 `test`입니다.
- `--data-dir`를 안 주면 YAML의 `data.data_dir`를 사용합니다.
- `split_<split>.npy` 또는 `<split>_sim_ids.npy`가 있으면 그 순서대로 컨디션을 따릅니다.
- `maps_per_sim`은 `metadata.yaml`의 `split.maps_per_sim`를 우선 사용하고, 없으면 파라미터 반복으로 추정합니다.
- `--n-gen` 기본값은 `15`입니다.
