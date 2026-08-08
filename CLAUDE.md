# CLAUDE.md — FringeNet

이 파일은 매 세션 자동 로드되는 프로젝트 작업 메모리다.
배경·방법론의 전체 서사는 @README.md 참고. 여기에는 작업에 필요한 계약과 스펙만 둔다.

## 한 줄 요약

**FringeNet**: 반사율 스펙트럼(226채널) → 4층 박막 두께[nm] 다중출력 회귀.
미분가능 TMM을 물리 디코더로 쓰는 physics-informed 접근이 차별점.

## 목적과 우선순위 (중요)

- 채용 포트폴리오. 마감 약 3주. 리더보드 점수보다 다음이 성공 기준:
  1. 물리 단위 테스트를 통과한 TMM forward 모델
  2. 물리 손실(β) on/off ablation으로 기여를 정량화
  3. 재현 가능한 코드와 읽히는 문서
- 모든 수치·그림은 실제 실행 산출물이어야 한다. placeholder를 결과처럼 쓰지 않는다.
- 작게 일하기: 태스크 하나 → 테스트/실행 확인 → 커밋. 백로그 체크박스를 직접 갱신한다.

## 도메인 계약 (물리)

- 구조: 공기(n₀) / SiN(layer_1) / SiO₂(layer_2) / SiN(layer_3) / SiO₂(layer_4) / Si 기판(n_s). 수직입사.
- 층 j: delta_j = 2*pi * n_j * d_j / lambda
- Abelès 특성행렬 (Macleod 관례, 입사측 층부터 곱):

```
M_j = [[cos(delta_j),        i*sin(delta_j)/n_j],
       [i*n_j*sin(delta_j),  cos(delta_j)      ]]

(B, C)^T = (M_1 @ M_2 @ M_3 @ M_4) @ (1, n_s)^T
r = (n0*B - C) / (n0*B + C)
R = |r|^2
T = 4 * n0 * Re(n_s) / |n0*B + C|^2        # 무흡수 층 가정 시 R + T = 1
```

- 파장이 다른 채널끼리는 간섭하지 않는다 → R(λ)는 채널별 독립 계산 (W축 벡터화).

## 데이터 계약

- 경로: `data/raw/{train.csv, test.csv, sample_submission.csv}` — 사용자가 데이콘에서 직접 배치.
- **데이터 파일을 git에 커밋하거나 저장소에 재배포하지 않는다** (.gitignore: `data/`, `runs/`).
- `train.csv`: `layer_1..layer_4` = 두께[nm] 타깃, 컬럼 `"0".."225"` = 반사율(0~1).
- 헤더 0~225는 **비식별화된 파장 인덱스**다. 실제 nm로 단정하는 코드/서술 금지.
  단, 채널 순서는 연속 스펙트럼으로 의미 있음 (1D conv 유효).
- 검증할 가설 — Task 2에서 확인 후 아래 체크박스와 README §2를 갱신할 것:
  - [ ] 행 수 = 810,000
  - [ ] 각 layer 고유값 = {10, 20, ..., 300} (30개, 10 nm 격자)
  - [ ] 반사율 ⊂ [0, 1], 결측 없음, 중복 조합 없음
- 평가지표: MAE. 제출 파일은 `sample_submission.csv` 형식.

## 코딩 규약

- Python ≥ 3.11, PyTorch ≥ 2.x. 의존성 최소 유지(순수 PyTorch; lightning/hydra 등 대형 프레임워크는 사용자 승인 없이 추가 금지).
- 타입힌트 필수. docstring에 텐서 shape 명시. shape 규약:
  `d: (B, L)` float, `n_layers: (L, W)` complex, `lam: (W,)`, 반환 `R: (B, W)` real.
- 층 수(L=4)에 대해서만 파이썬 루프 허용. B, W축은 반드시 벡터화.
- dtype: 검증·캘리브레이션은 complex128, 학습은 complex64. d는 real 유지(autograd가 d로 흐르게).
- 포맷/린트 ruff, 테스트 pytest, 시드 고정 유틸(`src/utils/seed.py`), 설정은 `configs/*.yaml`.
- 커밋 메시지: `feat|fix|test|docs|exp: ...`

## 물리 단위 테스트 — tests/test_tmm.py (전부 green이어야 다음 단계 진행)

대상 함수: `tmm_reflectance(d, n_layers, n0, ns, lam) -> R` (배치·파장 벡터화, complex 지원, 미분가능)

1. 무층 극한: d=0 전부 → r=(n0−ns)/(n0+ns). n0=1, ns=1.5 → R=0.04.
2. 에너지 보존: 실수 굴절률 무작위 스택에서 R+T=1 (atol 1e-6, complex128).
3. λ/4 무반사: n0=1, ns=2.25, n1=1.5, d1=λ/(4·1.5) → 해당 λ에서 R < 1e-8.
4. Airy 대조: 단층 TMM ↔ 해석해 r=(r01+r12·e^{−2iδ})/(1+r01·r12·e^{−2iδ}) allclose.
5. 미분가능성: float64에서 dR/dd를 유한차분과 비교 (rtol 1e-4).
6. 흡수 기판: 복소 ns에서 0 ≤ R < 1, NaN/Inf 없음.

## 모델·학습 스펙 요약

- Baseline(대조군): MLP. 226채널을 순서 없는 피처로 취급 — 구조 bias의 기여를 재기 위한 기준선.
- Level 1(구조·표현 bias): 1D CNN. 입력 `(B, 1, 226)` → conv 스택 → GAP → MLP 헤드 → 4출력,
  `sigmoid*290+10` bound (범위는 데이터 검증 결과에 맞춰 조정).
  fringe 주기가 두께에 따라 변하므로 커널 크기 혼합 또는 dilated conv로 다중 스케일 수용영역 확보.
  각 요소(구조/다중스케일/bound)는 config 플래그로 on/off 가능해야 한다 (ablation용).
- Level 2:
  - Stage A `src/calibrate.py`: train 서브셋(~5만 행)의 (d_true, R_obs)로 forward 미지수 피팅.
    파라미터화 — λ 그리드: `lam = lam_min + cumsum(softplus(u))` (단조);
    SiN·SiO₂: Cauchy `n(λ)=A+B/λ²+C/λ⁴`, k=0 가정; Si: n,k 곡선 학습(k≥0 softplus).
    초기값은 refractiveindex.info 문헌값. 산출물: 재구성 RMSE + 분산 곡선 플롯.
  - **게이지 고정 필수**: delta = 2*pi*n*d/lambda 는 (n, lambda)의 공통 스케일에 불변이라
    n과 lambda는 동시 식별이 불가능하다. 기본값으로 **SiO₂ Cauchy 계수를 문헌값에 freeze**하고
    lambda 그리드와 나머지 물질만 학습한다. 이 제약 없이 전부 자유롭게 두지 말 것
    (손실은 내려가도 물리적으로 무의미한 해에 안착한다).
    캘리브레이션 결과의 n(λ)가 문헌값 대비 물리적으로 타당한 범위인지 반드시 육안 확인·기록.
  - 판정 게이트: 재구성 RMSE < 5e-3 (R 단위) → 물리 디코더 채택.
    실패 시 fallback: 지도학습 d→R forward emulator(NN)를 동결 디코더로 사용하고, 실패 사실을 README에 기록.
  - Stage B `src/train.py --physics`: `L = MAE(d_hat, d) + beta * L1(R_dec(d_hat), R_obs)`,
    beta 워밍업 스케줄. ablation: beta=0 vs beta>0.
- 평가: 전체/층별 MAE, 학습곡선, TMM 재구성 오차 히스토그램(신뢰도 지표), 두께 구간별 오차 분석.

## 평가 규약 (데이터가 시뮬레이션 격자라서 생기는 함정)

- **격자 스냅 금지 (또는 분리 보고)**: 타깃이 10 nm 격자 위에 있어 예측을 최근접 격자로
  반올림하면 MAE가 인위적으로 낮아진다. 이는 생성 방식의 누설이지 계측 성능이 아니다.
  기본 리포트는 raw 예측값 기준. 스냅 결과를 넣을 경우 반드시 별도 행으로 분리하고 각주를 단다.
- **split 3종 리포트**:
  1. random split (기본, 조합 보간 성능)
  2. held-out 두께 값 split — 특정 두께 값(예: layer_1의 {70, 150, 230})을 학습에서 통째로 제외
  3. 격자 밖 합성셋 — 캘리브레이션된 forward 모델로 비격자 두께 샘플 생성 후 평가
- **노이즈 강건성**: 입력 R에 가우시안 노이즈(σ = 1e-3, 5e-3, 1e-2) 주입 시 MAE 열화 곡선.
- 위 셋은 Task 4 이후 평가 스크립트에 고정 포함한다. 좋은 숫자만 고르지 않는다.

## 작업 백로그 (순서 준수, 완료 시 체크)

- [x] **Task 0 — 스캐폴드**: 디렉토리(README §4), `.gitignore`(data/, runs/, .venv 등),
  `requirements.txt`(torch, numpy, pandas, pyarrow, matplotlib, scikit-learn, pytest, ruff, pyyaml, tqdm),
  ruff 설정, seed 유틸. DoD: `pytest -q` 통과(수집 0 허용), `ruff check .` 클린.
- [ ] **Task 1 — TMM 모듈**: `src/physics/tmm.py` + 위 단위 테스트 6종. DoD: 전부 green.
- [ ] **Task 2 — 데이터 검증·로더**: `scripts/verify_data.py`(가설 체크 출력),
  `src/data/dataset.py`(parquet 캐시 권장). DoD: 검증 결과를 본 파일 체크박스와 README §2에 반영.
- [ ] **Task 3 — EDA**: (a) 한 층만 변화시킨 스펙트럼 오버레이(두께↑ → fringe 조밀 확인),
  (b) 네 층을 각각 흔들었을 때 스펙트럼 민감도 비교(어느 층이 잘 보이는가),
  (c) 반사율 분포·범위. DoD: `reports/figures/` 그림 3종 + 관찰 메모.
- [ ] **Task 4 — Baseline 학습**: 90/10 val split(시드 고정), MAE 리포트. DoD: 재현 커맨드 README 반영.
- [ ] **Task 5 — Level 1 ablation**: MLP vs 1D CNN, 단일 vs 다중 스케일, bound on/off 비교표.
- [ ] **Task 6 — Stage A 캘리브레이션**: 게이트 판정까지.
- [ ] **Task 7 — Stage B 물리 손실**: beta ablation + 신뢰도 지표 분석.
- [ ] **Task 8 — 문서화**: README 결과·그림·한계 논의 갱신.

## 하지 말 것

- `data/` 커밋, 대회 데이터의 원본·가공본 재배포 (로더는 로컬 파일만 참조)
- 비식별 파장축을 실제 파장으로 단정하는 코드·서술
- 실행으로 뒷받침되지 않은 수치 주장 (모든 표는 스크립트 산출)
- 격자 스냅 후처리를 주 결과로 제시하기 (누설이며, 분리 보고 대상)
- 캘리브레이션에서 게이지 고정 없이 n과 λ를 동시에 자유 학습시키기
- 백로그 순서 건너뛰기 (특히 Task 1 테스트 통과 전에 학습 코드 작성 금지)

## 자주 쓰는 명령

```bash
pytest -q
ruff check . && ruff format .
python scripts/verify_data.py
python -m src.train --config configs/baseline.yaml
```

## 세션 시작 체크리스트

1. 본 파일과 git log/status로 현재 상태 파악 → 백로그 최상단 미완료 Task 확인
2. 착수 전 계획을 2~4줄로 요약해 제시
3. 구현 → 테스트 → 커밋 → 체크박스 갱신
