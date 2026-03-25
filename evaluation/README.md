# Evaluation Package

`evaluation/`은 GENESIS 평가 코드의 기준 패키지입니다.

## 목적

- 평가 엔트리포인트를 한 폴더에서 관리
- 프레임워크별 샘플링 연결 지점을 명확히 분리
- 향후 공통 로직(`core`)과 데이터 준비(`data`) 모듈화 기반 마련

## 현재 구조

- `evaluation/cli/evaluate.py`
  - 프로토콜 기반 평가 진입점(LH/CV/1P/EX)
  - `build_sampler_fn`에서 프레임워크별 샘플러를 단일 인터페이스로 통합
- `evaluation/cli/sample_cv.py`
  - CV 조건 샘플 생성/리포트
  - `evaluate.py`의 `build_sampler_fn` 재사용
- `evaluation/core/`
  - 공통 평가 로직 이전 대상(현재 스캐폴딩)
- `evaluation/data/`
  - 평가용 데이터 준비 로직 이전 대상(현재 스캐폴딩)
- `evaluation/experimental/`
  - 실험적/비정규 평가 코드 분리 공간(현재 스캐폴딩)

## 엔트리포인트 실행

```bash
python -m evaluation.cli.evaluate --help
python -m evaluation.cli.sample_cv --help
```

호환용 래퍼:

- 루트 `evaluate.py` → `evaluation.cli.evaluate` 호출
- 루트 `sample_cv.py` → `evaluation.cli.sample_cv` 호출

## 샘플링 호출 흐름

1. CLI가 config/checkpoint/data를 로드
2. `build_sampler_fn(cfg, model, device)`가 sampler 설정 해석
3. framework별 샘플러를 `sampler_fn(model, shape, cond)` 형태로 래핑
4. evaluator 또는 CV sampler가 `sampler_fn`을 호출해 맵 생성

framework별 실제 샘플링 함수:

- Flow Matching: `flow_matching/samplers.py`
- Diffusion: `diffusion/ddpm.py`
- EDM: `diffusion/samplers_edm.py`

## 정리 원칙

- 새 평가 CLI는 `evaluation/cli/`에 추가
- 여러 CLI에서 공통으로 쓰는 함수는 `evaluation/core/`로 이동
- 프로토콜 데이터 생성 코드는 `evaluation/data/`로 단계적 이동
- 실험 코드(논문 부록, 임시 분석)는 `evaluation/experimental/`에 둠
