# CLAUDE.md — FringeNet

매 세션 자동 로드되는 작업 메모리. **계약과 스펙만** 둔다 — 배경·설계는 @README.md,
진행 연표는 docs/week_N.md, 수치의 근거는 reports/<실험>.md.

**FringeNet**: 반사율 스펙트럼(226채널) → 4층 박막 두께[nm] 다중출력 회귀.
미분가능 TMM을 물리 디코더로 쓰는 physics-informed 접근이 차별점.

## 목적과 우선순위 (중요)

- 채용 포트폴리오. 마감 약 3주. 리더보드 점수보다 다음이 성공 기준:
  1. 물리 단위 테스트를 통과한 TMM forward 모델
  2. 물리 손실(β) on/off ablation으로 기여를 정량화
  3. 재현 가능한 코드와 읽히는 문서
- **모든 수치·그림은 실제 실행 산출물이어야 한다.** placeholder를 결과처럼 쓰지 않고,
  타 실험 수치를 인용할 때는 해당 `metrics.json`에서 복사한다 (기억으로 쓰지 않는다).
- 작게 일하기: 태스크 하나 → 테스트/실행 확인 → 커밋. 백로그 체크박스를 직접 갱신한다.

## 도메인 계약 (물리)

구조: 공기(n₀) / SiN(layer_1) / SiO₂(layer_2) / SiN(layer_3) / SiO₂(layer_4) / Si 기판(n_s),
수직입사. Abelès 특성행렬, Macleod 관례, 입사측 층부터 곱한다.

```
delta_j = 2*pi * n_j * d_j / lambda
M_j = [[cos(delta_j),        i*sin(delta_j)/n_j],
       [i*n_j*sin(delta_j),  cos(delta_j)      ]]
(B, C)^T = (M_1 @ M_2 @ M_3 @ M_4) @ (1, n_s)^T
r = (n0*B - C) / (n0*B + C)
R = |r|^2
T = 4 * n0 * Re(n_s) / |n0*B + C|^2        # 무흡수 층 가정 시 R + T = 1
```

복소 굴절률은 `n = n' - i*k` (k ≥ 0 이 흡수). 파장이 다른 채널끼리는 간섭하지 않으므로
R(λ)는 채널별 독립 계산이다 (W축 벡터화, 파이썬 루프는 층 수 L에만).

## 데이터 계약

- 경로 `data/raw/{train,test,sample_submission}.csv` — 사용자가 데이콘에서 직접 배치.
  parquet 캐시는 `data/cache/`에 자동 생성 (최초 29초 → 이후 3.4초).
- **데이터 파일을 git에 커밋하거나 재배포하지 않는다** (.gitignore `/data/**`, 구조는
  .gitkeep으로만 추적). **앵커(`/`)가 없으면 `src/data/` 패키지까지 무시된다.**
  **문헌 광학상수 원본은 대회 데이터가 아니므로** `src/physics/literature/*.yml`에 두고
  git 추적한다 (CC0) — `data/` 아래 두면 .gitignore에 걸려 재현이 깨진다.
- `train.csv`: `layer_1..layer_4` = 두께[nm] 타깃, 컬럼 `"0".."225"` = 반사율.
  헤더 0~225는 **비식별화된 파장 인덱스**다 — 실제 nm로 단정하는 코드/서술 금지.
  단 채널 순서는 연속 스펙트럼으로 의미 있다 (1D conv 유효).
- 검증 완료 (`scripts/verify_data.py`, 종료코드 0): train (810000, 230) · 각 층 고유값
  {10, 20, …, 300} nm 30개 · 결측·중복 없음 · **30⁴ = 810,000 전수 조합** ·
  test 10,000행(id 일치).
- **test의 두께는 격자 위에 있지 않다 — 연속이다 (중요).** 라벨이 없어 R에서 역추정했고
  독립 두 방법이 일치한다 (유계 노이즈 반증 / 디코더 역해 격자거리 평균 2.42 nm vs train
  0.37 nm). **파급**: 격자 스냅은 train/holdout에서만 누설이고 **test(제출)에서는 MAE를
  약 +1.2 nm 악화**시킨다. 그리고 격자 밖 일반화 평가셋이 이미 test로 존재한다 —
  holdout 수치는 격자 위 조합 보간, 리더보드는 격자 밖 외삽이므로 같은 것으로 말하면 안 된다.
- **반사율은 [0,1]을 벗어난다 — 가산 노이즈가 있다 (중요).** train 범위
  [−0.015117, 0.943648], 음의 반사율이 값의 0.35%·행의 46.9% (`R > 1`은 0건).
  - **σ = 0.008658** — 채널축 m차 차분 `Var(Δ^m) = C(2m,m)·σ²` 로 m을 올리면 상한이 단조
    감소하며 **m = 5~8에서 6자리 고정**된다. 평평해진다는 것 자체가 잔차가 채널축에서
    백색이라는 근거다. **두께축으로는 같은 추정이 불가능**하다 (주기 1.07~4.5개로 과소표집).
  - **노이즈는 유계다 — |ε| ≤ 0.0152.** 균등분포면 a = σ√3 = 0.014996이고, 1.83억 관측에서
    `R < −0.0152`가 **0건**이다 (가우시안이면 5σ = −0.043까지 나와야 한다).
    → **모델 판정에 직접 쓴다**: 잔차가 0.0152를 넘는 관측 하나하나가 통계 없이 모델
    오류의 증거다 (Level 2 게이트 (b)).
  - 주의: 고주파 잔차 `y[i]−(y[i−1]+y[i+1])/2` 는 2차 차분의 −1/2배라 **같은 추정량**이다
    (독립 확인이 아니다 — 새 정보는 첨도뿐).
- 평가지표 MAE. 제출은 `sample_submission.csv` 형식.

## EDA 확정 수치 (`scripts/eda.py` 산출 — 해석은 reports/eda_notes.md)

- **노이즈 분포는 균등에 가깝다** (곡률 적은 행에서 σ·초과첨도가 둘 다 균등분포 예측으로
  수렴). → 강건성 실험의 주입 노이즈는 **균등 ±0.015 기본**, 가우시안은 별개 질문으로 병기.
- **층별 민감도** (한 층만 +10 nm, RMS ΔR / SNR): layer_1 0.1105/12.7 · layer_2 0.0896/**10.3**
  · layer_3 0.1329/**15.3** · layer_4 0.1025/11.8. → 최소 SNR 10.3, **원리적 사각지대 없음**
  이므로 층별 MAE 격차는 관측 한계가 아니라 모델 문제로 해석한다. layer_4는 40~60 nm에서
  민감도가 0.0760까지 꺼지는 국소 최소가 있다 (최저 50 nm).
- **채널별 정보량 불균등**: mean |ΔR|이 채널 0~9 대비 216~225에서 약 **3배**. 노이즈는
  채널에 균일하므로(3차 차분 σ비 1.006) 대역 오른쪽 끝 SNR이 3배다 — 채널 자르기·
  downsample 주의. 물리 손실의 채널 가중은 ablation 후보(기본 균등).
- **표본 추출 주의**: train 행은 (layer_1..4) **사전식 정렬**이라 `x[:N]` 앞머리 자르기는
  표본이 아니다 (layer_1 = 10 nm 구석만 본다). 반드시 시드 고정 무작위 표집.
- **하지 말 것**: 무늬 개수를 세어 "두께에 비례한다"는 정량 법칙을 주장하기. 파장축
  비식별화 + 대역 내 무늬 수가 적음 + 4층 beat + 노이즈로 신뢰 수준이 안 나온다 (네 층
  모두 10 nm면 진폭이 노이즈 수준이라 부호가 반대로 나온다). **두께↑ → fringe 조밀은
  정성 확인까지만.**

## 코딩 규약

- Python ≥ 3.11, PyTorch ≥ 2.x. 의존성 최소 유지 (순수 PyTorch — lightning/hydra 등
  대형 프레임워크는 사용자 승인 없이 추가 금지).
- 타입힌트 필수. docstring에 텐서 shape 명시. 규약: `d: (B, L)` float,
  `n_layers: (L, W)` complex, `lam: (W,)`, 반환 `R: (B, W)` real.
- 층 수(L=4)에 대해서만 파이썬 루프 허용. B, W축은 반드시 벡터화.
- dtype: 검증·캘리브레이션은 complex128, 학습은 complex64. d는 real 유지(autograd).
- 포맷/린트 ruff, 테스트 pytest, 시드 유틸 `src/utils/seed.py`,
  설정은 `configs/<실험>/<변형>.yaml` (`experiment`·`run_name` 키 필수).
- 커밋 메시지 `feat|fix|refactor|test|docs|exp: ...` (GitHub 자동 머지 커밋은 예외).
  Colab VM 커밋은 UTC — 문서 날짜(KST)와 대조할 때 `git log --date=local`.
- **GPU가 필요한 실험은 Colab에서 돌린다** (로컬 WSL2는 CPU 전용 — 분석·검증·리포트 담당).
  학습 엔트리는 `src/train_gpu.py`(holdout 전용). CPU 경로(`src/train.py` — baseline을 만든
  경로)와 디커플, 수정 금지. 산출물 계약은 동일하고 체크포인트는 CPU 텐서로 저장돼 로컬
  `evaluate.py`와 호환. CPU↔GPU는 bit 재현이 아니라 MAE 수준에서 비교한다.
- **세션 유실 대비는 필수 규율** — Colab 런타임은 언제든 끊긴다는 전제로, GPU에서 도는
  모든 학습·장시간 스크립트는 다음 4가지를 갖춘다 (`train_gpu.py`에 구현돼 있으니 재사용이
  기본):
  1. best 갱신 즉시 체크포인트 저장 — 종료를 기다리지 않는다
  2. 에폭 단위 resume 상태 저장(+RNG) 및 Drive 미러 백업
  3. 재실행 시 완료 run 스킵 + 진행 run 재개 (재개 결과 = 무중단 실행, 테스트로 검증)
  4. 노트북 Run-All이 입력 대기 없이 end-to-end로 돌고(비밀은 정적 소스에서 로드),
     전 작업 완료+push 성공 시 런타임 자동 반납
- **Colab 노트북은 라운드(학습 세션)별 1개**: `notebooks/<대실험>/roundN_<내용>.ipynb`.
  완료된 라운드는 실행 로그 보존을 위해 수정·재실행하지 않는다. 새 라운드는 직전 노트북을
  복사해 헤더·CONFIGS 갱신 + 출력 비움으로 시작한다.
  **IDE 옛 버퍼 사고를 두 번 겪었다** — 노트북을 연 채로 커밋이 진행되면 그 뒤 에디터 저장이
  옛 버퍼 전체로 디스크를 덮어써 커밋된 셀 수정이 조용히 사라진다. JSON diff라 눈으로 못 잡으니
  **확인을 사람 눈이 아니라 명령에 맡긴다**:
  - 편집 **전**: `git status --short notebooks/` — 깨끗해야 시작한다. 연 채로 pull/checkout
    했으면 에디터에서 닫았다 다시 연다 (버퍼를 버린다).
  - 커밋 **전**: `python scripts/check_notebook_regression.py` — 최근 커밋이 추가한 줄을
    지우고 있으면 종료코드 1로 알린다. `pytest tests/test_notebooks.py`는 필수 셀 규약을 건다.

  필수 셀 규약 (`tests/test_notebooks.py`가 검사한다):
  1. Drive 마운트는 항상 `force_remount=True` (이전 세션의 stale 마운트 배제)
  2. push 셀 PAT는 정적 소스에서 자동 로드 (env → Colab Secrets → Drive 파일) — 무정지
  3. 런타임 자동 반납의 취소 대기 sleep은 **5초** (긴 대기 금지 — 유휴 과금)
  4. **push 후 Drive 체크포인트 무결성 검증, 통과해야만 반납**:
     `flush_and_unmount()` → 재마운트 → **미러의 model.pt를 다시 로드해 holdout 재추론 →
     기록된 val MAE 재현 확인**. Drive FUSE는 비동기 업로드라 세션이 죽으면 대용량 파일이
     구버전으로 남고(실사례: 3.4GB resume.pt가 5에폭 뒤처짐), git 미추적 대형 model.pt는
     Drive가 유일본이다.

## 물리 단위 테스트 — tests/test_tmm.py (전부 green이어야 다음 단계 진행)

대상: `tmm_reflectance(d, n_layers, n0, ns, lam) -> R` (배치·파장 벡터화, complex, 미분가능)

1. 무층 극한: d=0 전부 → R = 0.04 (n0=1, ns=1.5)
2. 에너지 보존: 실수 굴절률 무작위 스택에서 R+T=1 (atol 1e-6, complex128).
   **복소 ns + n0 ≠ 1 케이스 포함** — T의 n0 계수·Re(ns) 취급·복소 부호 관례를 동시에 건다
3. λ/4 무반사: n1 = √(n0·ns), d = λ/(4n1) → R < 1e-8
4. Airy 대조: 단층 TMM ↔ 해석해 `r = (r01 + r12·e^{−2iδ})/(1 + r01·r12·e^{−2iδ})`
5. 미분가능성: float64에서 dR/dd를 유한차분과 비교 (rtol 1e-4)
6. 흡수 기판: 복소 ns에서 0 ≤ R < 1, NaN/Inf 없음
7. **층 순서 고정**: 비대칭 2층을 재귀 프레넬 공식과 대조 — 1~6은 적층 순서를 뒤집어도
   전부 통과해 순서를 고정하지 못한다 (Task 1에서 발견·보강)

## 모델·학습 스펙

- **Baseline (대조군)**: MLP 512×3, Linear→BatchNorm→GELU, dropout 0, bare regression,
  입력 표준화 없음, batch 512, AdamW 1e-3/wd 1e-4, warmup 1000 + cosine.
  **holdout MAE 4.599 nm** (`reports/mlp_baseline.md`). 226채널을 순서 없는 피처로 취급.
- **Level 1 (구조 bias)**: 1D CNN. **확정 = flatten-dilated-bound, holdout MAE 2.346 nm**
  (baseline −49%, `reports/level1_cnn.md`). 국소 conv는 수용영역이 전 대역을 덮고(dilated
  RF 259) 파장축 위치가 보존될 때만(flatten) 유효 — 소박한 conv+GAP는 4배 나쁘고 채널
  셔플 대조군보다도 나쁘다. sigmoid bound가 격자 끝 오차를 지워 추가 −20%
  (**MLP 때와 반대 결론** — 강한 백본에서는 범위 밖 초과분이 남은 오차를 지배).
  **Level 2 백본 = flatten-dilated-bound.**
- **상한 기준선**: 리더보드 1등 단일 모델(213.2M skip-MLP) 원본 재현 → holdout MAE
  **0.3955 nm** (`reports/strong_baseline.md`). 0.66M CNN(2.346) 대비 322배 파라미터로 −83%.
  Task 7의 서사는 "작은 모델 + 물리가 이 격차를 얼마나 좁히나".
- **Level 2 Stage A = `src/calibrate.py`** (물리 제약 최소제곱, 자유 파라미터 1~7개).
  설계 원칙: **물리 법칙을 자유도 개수로 강제한다** — 채널별 자유 곡선을 두면 모델 오차를
  물성으로 흡수해 RMSE는 내려가도 물성값이 비물리적이 된다. `reports/stage_a.md`.
  - λ(c): `1/λ = ν₀(1 + r₁u + r₂u²)`, u = c/(W−1). 초기값은 두께축 주파수 식별
    (`src/physics/freq_id.py`)의 강건 적합 — 결정론적 닫힌형이라 run마다 재계산하고
    metrics.json에 기록한다. **채널별 226개를 그대로 고정하지 말 것** (0.445 nm 흔들림이
    그것만으로 R 오차 rms 0.0052를 만든다 — 남은 계통오차 0.0041보다 크다).
  - SiO₂: **정확한 Malitson 1965 Sellmeier 동결** (게이지). Cauchy 근사 쓰지 말 것.
  - SiN: **Luke 2015 Sellmeier**, B₁(필요시 C₁)만 자유. **문헌값 동결 금지** —
    게이지-불변 관측량 n_SiN/n_SiO₂가 문헌 대비 −2.15% 편차(박막 조성 차이)다.
  - Si 기판: **Schinke 2015 실측표 + 에너지축 3차 스플라인** 동결 (선택적으로 ΔE·k 스케일).
    코드 기본값도 이것이다 (`src.calibrate.DEFAULT_SI_SOURCE` — 기본값을 기각된 표로 두면
    `si_source`를 빠뜨린 config가 조용히 그쪽으로 돈다). **해석적 수식이 아니라 실측 원본표**
    이고, 표는 기하학 논증이 아니라 **측정으로** 골랐다. **c-Si에 Tauc-Lorentz 쓰지 말 것**
    (비정질용 — TL이 맞는 자리는 비정질 SiN의 UV 흡수 쪽이다).
  - **게이지 고정 필수**: δ = 2πnd/λ 가 (n, λ) 공통 스케일에 불변이라 동시 식별이 불가능
    하다. SiO₂를 문헌값에 freeze하고 나머지만 학습한다. 상관행렬 ν₀ ↔ SiN B₁ = −0.686이
    그 축이다. 캘리브레이션 결과 n(λ)가 문헌 대비 타당한지 반드시 육안 확인·기록.
  - **판정 게이트 (a)~(f)** — 재구성 잔차만 보는 게이트는 계통오차를 파라미터로 흡수하는
    모델을 항상 유리하게 만들므로 파라미터의 물리성과 예측력을 함께 본다:
    - (a) RMSE < 1.2σ = **0.010390**. 1.2 배수에 유도는 없으니 **계통오차 √(RMSE²−σ²)를
      함께 1차 지표로** 읽는다. (원 기준 5e-3은 폐기 — 노이즈가 있어 완벽한 모델도 도달 불가)
    - (b) **유계 노이즈 위반율** (가장 날카롭다). 잔차가 0.0152를 넘는 관측은 통계 없이
      모델 오류의 증거이고, 채널·두께로 분해하면 **어느 물리가 부족한지 국소화**해 준다.
      **현행 최선 9.99% — 미통과.** 채택 근거와 수용 리스크는 README §3.2에 명시돼 있다.
      사전 선언한 pass/fail 게이트를 못 넘긴 사실 자체를 지우지 않는다.
    - (c) 잔차 백색성 — **단독으로는 판별력이 없다** (채널별 자유도를 주면 구조가 사라진다).
    - (d) **두께 nm 역해 MAE** — R 단위를 프로젝트 단위로 옮긴 값이고 Stage B의 물리 손실이
      강제할 수 있는 정확도의 상한이다. d_true에서 출발하므로 **경쟁 성능 수치로 쓰지 말 것**.
    - (e) **채널 홀드아웃 예측** — 매끈한 물리 분산만 가능하고 채널별 자유 모델은 원리적으로
      불가능 → 파라미터화가 내용을 가진다는 증명. **대조군 없이 held/fit 비를 읽지 않는다**
      (균등 간격판은 한계 효과가 +0.3%뿐이라 판별력이 없다).
    - (f) **파라미터 물리성** — 문헌 범위 + 곡선 매끈 + **독립 두 방법 일치**. 형식
      신뢰구간은 iid 가정이라 낙관적이므로 방법 간 일치도와 **문헌표 간 계통**(세 Si 표의
      불일치가 유계 예산의 70%를 쓴다)을 함께 보고한다. 개별 추정의 산포를 요약 곡선의
      오차 막대로 쓰지 않는다.
    - 실패 시 fallback: 지도학습 d→R forward emulator(NN)를 동결 디코더로 쓰고 기록한다.
  - **확정 (2026-08-13)**: 자유 파라미터 7개 + Si 표 Schinke → RMSE **0.009573 (1.106σ)**
    ✓(a) / 계통오차 0.004085, **역해 MAE 0.340 nm**, 채널 홀드아웃 한계효과 +19.7%,
    λ 절대 스케일 검정 통과(1.00689), 물성 전부 문헌 정합(독립 두 방법 0.32~0.74%).
    **게이트 (b) 미통과 (9.99%)** — 잔차는 대역 단파장 끝(Luke 유효범위 밖 외삽)에 몰리고
    그 구간엔 문헌표 불일치가 없다(모델 부족).
  - **피팅 표본을 늘려도 좋아지지 않는다 — 측정으로 확인.** `fit_rows` 2k/8k/50k(25배)에서
    RMSE·계통오차가 **6자리 동일**하고 역해 MAE도 0.3402~0.3404다 (`configs/stage_a/fitrows-*.yaml`,
    게이트 리포트의 «표본 수 민감도»). 한계는 통계오차가 아니라 **모델 형태**이므로 train
    전체(729,000행)로 늘리는 것은 순손실이다 — 계통오차는 표본으로 줄지 않는다.
    같은 이유로 **디코더를 개선하려면 물성 모델을 손봐야 한다**(게이트 (b) 후속).
  - **Stage B 디코더 = `runs/stage_a/joint-lam3-sin2-si2-schinke/model.pt`**
    (`src.losses.DEFAULT_DECODER`).
- **Stage B = `src/losses.py` + `src/train_gpu.py`의 `train.physics` 블록**:
  `L = MAE(d_hat, d) + beta(step) * L1(R_dec(d_hat), R_obs)`, beta 선형 워밍업.
  **CPU 경로 `src/train.py`에는 없다** (동결 경로라 GPU 쪽에만 배선한다).
  사전등록한 예측은 docs/week_1.md TODO.
  - **판정 완료 — 이 손실 항은 기각됐다.** β ∈ {0,30,100,300}을 세 축에서 돌렸고
    (`beta*` 무작위 split · `heldout-thickness-beta*` held-out 두께 값 split ·
    `ft-heldout-beta*` 수렴 지점 warm start) **전부 β에 단조로 해롭다**. 적합 수준을 맞추면
    이득은 0이다 (`reports/stage_b_curves*.md`). 코드·config는 재현을 위해 남기지만
    **새 β run을 추가하지 않는다** — 물리의 값어치는 추론 시 역산과 신뢰도 지표에서 찾는다.
  - **동결은 두 겹**: `theta.requires_grad_(False)` + 디코더가 **파라미터를 아예 보유하지
    않는다**(상수 분광량만 버퍼). `nn.Parameter`라 `model.parameters()`를 통째로 옵티마이저에
    넘기면 계약이 조용히 깨지므로, 학습 모델의 서브모듈로 두지 않는다 (체크포인트
    state_dict에도 섞이지 않아 `evaluate.py` 로드 계약이 유지된다).
  - **beta=0 대조군은 물리 항을 gradient에 넣지 않고 진단으로만 기록한다** — 학습 경로가
    물리 항 도입 전과 같아야 차이를 물리 항에 귀속할 수 있다 (테스트가 비트 동일성으로 고정).
  - 학습 dtype은 **complex64**: theta가 동결이라 (λ, n, n_s)는 상수이므로 1회 계산해
    캐스팅·캐시한다 (float64 대비 오차 max 6.0e-6 = σ의 0.07%, GPU float64는 1/32 처리율).
  - 진단: 매 에폭 `train_phys`(가중 전 재구성 L1)·`val_phys`를 로그에 남기고 best 에폭
    값을 metrics.json `val_phys_l1`에 기록한다. 참 두께에서의 하한은 E|ε| = 0.0075.
  - **`train.init_from`** = 체크포인트 warm start (수렴 후 물리 fine-tune용,
    `configs/stage_b/ft-heldout-beta*.yaml`). **분할이 같아야 한다** — 다른 split의 체크포인트로
    warm start하면 출처 모델이 이미 본 행이 holdout에 들어가 누수다. `train_gpu`가 출처
    `metrics.json`의 `data` 블록을 대조해 막고, 미러에서 model.pt만 받아 metrics.json이 없으면
    대조 불가를 로그에 남긴다. config에 있으므로 resume 지문이 cold start와 갈라진다.
- **역산 refinement = `src/physics/invert.py` + `scripts/refine_inversion.py`** (Task 7의
  사전등록 2). 물리를 손실이 아니라 **추론 후 보정**으로 쓴다: 예측 `d_hat`을 출발점으로
  동결 디코더를 관측 R에 맞춰 배치 LM으로 재적합한다. 라벨·격자를 안 쓰므로 test에도 쓸 수 있다.
  - **`lm_invert`의 `d_init`이 실험의 정의다.** 게이트 (d)는 `d_true`(디코더 내재 편향 = 상한),
    refinement는 `d_hat`. 같은 함수를 쓰되 **출발점만** 다르고 그 차이가 측정의 전부다.
  - **확정 (2026-08-14)**: holdout MAE **2.3455 → 0.6110 nm (−74%)**, 98%의 행이 개선.
    격자 중앙 출발은 **77.68 nm로 실패** → CNN의 기여는 정밀도가 아니라 **올바른 fringe 분지**다.
    상한(참값 출발) 0.3336. 정본은 `reports/inversion_refine.md`.
  - **감쇠 스케일**: `damping="batch"`(게이트 (d)의 기존 동작)는 한 행의 갱신이 배치 구성에
    의존해 **청크를 바꾸면 답이 달라진다**. refinement는 `"row"`를 쓴다 (청크 불변, 테스트로 고정).
    두 값을 잘못 조합하면 조용히 다른 수치가 나오는 대신 에러가 난다.
  - **추론 비용은 손해다 — 숨기지 않는다.** 정본은 `reports/inversion_bench.md`
    (`scripts/bench_invert.py`, GPU는 `notebooks/inversion/round1_gpu-bench.ipynb`).
    행당 TMM forward가 270회(야코비안 중앙차분 8 + 시험 스텝 1) × 30회라 `cnn+LM`이 213M
    skip-MLP forward보다 **한 자릿수 느리다**(L4에서 17.26배). 서사는 정확도·모델 크기
    (2.5 MB vs 813.6 MB)에 대한 주장이지 지연에 대한 주장이 아니다.
    - **단일 배수를 인용하지 말 것** — 하드웨어에 크게 의존한다. skip-MLP는 GEMM 한 방이라
      BLAS·GPU 최적화를 온전히 받고 LM은 비정형 + 대역폭 바운드라 덜 받아서, 같은 코드가
      기계마다 다른 배수를 낸다. 배수는 리포트에서 읽고 기계를 함께 적는다.
    - **GPU 이득의 절반 이상을 float64가 먹는다** — L4에서 LM complex128은 10.3배 빨라지는데
      complex64는 24.5배다(소비자 GPU FP64 1/32). 배포 경로는 complex64가 맞고 **판정 수치는
      계속 complex128**이다. CPU↔GPU는 bit가 아니라 MAE 수준에서 일치한다(같은 표본 1.8% 차).
    - **반복수를 줄이는 것은 공짜가 아니다** — 10회는 30회보다 MAE가 0.02~0.03 nm 나쁘다
      (CPU·GPU 세 측정에서 일관). 3배 싸지는 대신 치르는 값이므로 함께 적는다.
    - 미착수 레버: 해석적 야코비안(디코더가 미분가능한데 역해는 중앙차분을 쓴다) ·
      수렴 행 조기 종료(고정 30회 대신).
- 평가: 전체/층별 MAE, 학습곡선, TMM 재구성 오차 히스토그램(신뢰도 지표), 두께 구간별 오차.

## 평가 규약 (데이터가 시뮬레이션 격자라서 생기는 함정)

- **격자 스냅 금지 (또는 분리 보고)**: train/holdout 타깃이 10 nm 격자 위에 있어 예측을
  최근접 격자로 반올림하면 MAE가 인위적으로 낮아진다(실측 2.346 → 1.287). 생성 방식의
  누설이지 계측 성능이 아니다 — 기본 리포트는 raw 예측값 기준, 스냅은 별도 행 + 각주.
  **단 test(제출)에는 절대 쓰지 말 것** — 격자 밖이라 MAE가 약 +1.2 nm 악화된다.
- **split 3종 리포트**: ① random split (조합 보간) ② **held-out 두께 값 split** —
  `data.holdout_thickness: [70, 150, 230]`, 값을 **전 층에서** 뺀다. 학습이 27⁴ = 531,441행
  이므로 **random split run과 MAE를 직접 비교하지 말 것** (같은 split 안의 대조만 유효).
  ③ 격자 밖 — **이미 test가 그 평가셋이지만 라벨이 없어 리더보드 제출로만 수치가 나온다**.
  캘리브레이션 forward로 합성하면 **같은 물리로 만든 데이터라 β>0에 유리한 순환**이므로
  주 판정에 쓰지 않는다 (최종 선택 모델만 제출로 확인).
- **노이즈 강건성**: 입력 R에 노이즈 주입 시 MAE 열화 곡선 (`scripts/evaluate_axes.py`).
  기본은 데이터와 같은 종류인 **균등 ±0.015**, 가우시안은 별개 질문으로 병기. 데이터에 이미
  σ = 0.008658이 있으므로 주입은 **추가분**임을 명시한다. 주입 실현은 수준마다 고정해
  모든 run이 같은 입력을 본다 (차이를 난수가 아니라 모델에 귀속시킨다).
- **신뢰도 지표**(README §3.4)도 같은 스크립트가 낸다.
- 위 축들은 Task 7의 측정 도구다. 좋은 숫자만 고르지 않는다.
- **실험 관리 — 대실험(experiment) / 변형(run) 2단**: 설정 `configs/<실험>/<변형>.yaml`,
  산출물 `runs/<실험>/<변형>/` = model.pt + train.log + metrics.json 세 가지만
  (metrics.json이 설정 스냅샷을 겸한다). 변형 이름은 번호가 아니라 **무엇이 다른지 드러나는
  서술형**으로 (예: `dropout0.0`). 대실험이 끝나면 `reports/<실험>.md`로 취합한다 —
  README에는 성능 수치를 두지 않는다.
- **git에는 텍스트 산출물만 추적한다** (metrics.json · train.log). 체크포인트는 Drive 미러
  `MyDrive/FringeNet/runs_mirror/<실험>/<run>/`에 3종(model.pt 포함)으로 보관한다 —
  `train_gpu.py`의 `_mirror_copy`가 학습 중에 이미 그렇게 쓰므로 별도 작업이 필요 없다.
  목록·sha256·복구 방법은 **`runs/CHECKPOINTS.md`**, 복구는
  `git show <원본 커밋>:runs/<실험>/<run>/model.pt`(과거분) 또는 Drive 사본.
  **예외 — `runs/stage_a/*/model.pt`는 git 추적**(합계 44 KB): `diagnose_calibration.py`가
  직접 로드해 `reports/stage_a_gate.md`를 재생성하므로 빠지면 재현 커맨드가 깨진다.
  텍스트 2종은 git이 정본이다 — 미러 사본으로 덮어쓰지 말 것.
- **주차별 실험 노트 `docs/week_N.md`** (첫 커밋 2026-08-08 기준 7일 단위): 날짜별 진행·
  발견·결정의 연표이고 **TODO 관리도 여기서** 한다. 백로그 체크박스를 갱신할 때 함께 갱신.

## 기록 규약 — 남길 것과 뺄 것 (포트폴리오 저장소다)

채용 포트폴리오라 **읽히는 것**이 목표다. 기록은 많을수록 좋은 게 아니라 **판단에 필요한
만큼**이다. 판정 기준 하나: *이 문장이 없으면 읽는 사람이 다른 결정을 내리는가?* 아니면 뺀다.

- **문서는 현재 사실만 쓴다.** "전에는 X였는데 정정했다", "옛 판본은 Y였다" 같은 **시간축
  서술을 리포트에 남기지 않는다.** 정정에서 배운 것이 있으면 **현재시제 규칙으로 바꿔** 쓴다
  (예: "0.368은 틀렸다" → "개별 추정의 산포를 요약 곡선의 오차 막대로 쓰지 않는다").
  시간축이 필요한 기록은 `docs/week_N.md` 연표에 **한 줄**.
- **같은 수치를 두 번 싣지 않는다.** 스크립트 산출표(`*_gate.md`, `eda_metrics.md`)가 정본이고
  리포트는 **참조 + 판단 근거**만 쓴다. 리포트에 표를 복제하면 디코더·설정이 바뀔 때 조용히
  스테일이 된다 — 실제로 그렇게 됐다.
- **손계산 수치는 스크립트로 옮긴다.** 리포트에 손으로 적은 수치는 재현이 안 되고 갱신도
  안 된다. 새 수치를 주장하려면 산출 스크립트를 `scripts/`에 남긴다.
- **코드 주석은 계약과 함정만.** 남길 것 = shape/dtype 계약 · 자명하지 않은 물리·수식 근거 ·
  실제 버그를 막는 경고. 뺄 것 = 리포트와 중복되는 서술 · 감사 이력 곁가지("N차 리뷰에서
  발견") · 코드를 읽으면 아는 내용.
- **커밋 메시지는 길어도 된다** — git이 시간축의 정본이다. 반대로 그 서사를 리포트·주석에
  옮겨 적지 않는다.
- 정리는 사후가 아니라 작성 시점에 한다. 이미 부푼 문서를 줄일 때는 **중복부터** 걷고,
  그래도 길면 근거의 깊이가 아니라 서술을 줄인다.

## 작업 백로그 (순서 준수 — 일일 진행·열린 항목은 docs/week_1.md)

- [x] **Task 0~3** — 스캐폴드 · TMM 모듈+테스트 7종 · 데이터 검증·로더 · EDA
- [x] **Task 4 — Baseline 학습**: 90/10 val split(시드 고정), MAE 리포트 → 4.599 nm
- [x] **Task 5 — Level 1 ablation**: MLP vs 1D CNN, 단일 vs 다중 스케일, bound on/off
      → flatten-dilated-bound 2.346 nm
- [x] **Task 6 — Stage A 캘리브레이션**: 게이트 판정까지 → TMM 채택 (게이트 (b) 미통과)
- [ ] **Task 7 — Stage B**: β ablation은 **종결**(아래), 남은 것은 세 가지다
  - [x] **β ablation 3라운드 — 물리 손실 항 기각.** 무작위 split · held-out 두께 값 split ·
        수렴 지점 warm start 전부에서 β에 단조로 해롭고 적합 수준을 맞추면 이득이 0이다.
        **이미 끝났으니 다시 돌리지 않는다** (`runs/stage_b/*` 12런이 정본).
  - [x] **역산 refinement — 사전등록 2 부합.** holdout MAE **2.3455 → 0.6110 nm (−74%)**
        (`reports/inversion_refine.md`). 물리 단독(격자 중앙 출발)은 77.68 nm로 실패하므로
        **CNN의 기여는 정밀도가 아니라 올바른 fringe 분지**다. 상한(참값 출발)이 0.3336이다.
        추론 비용은 `reports/inversion_bench.md` — 지연에서는 손해다(아래 스펙).
  - [ ] **평가 축 실측**: 노이즈 강건성 · 신뢰도 지표 (`scripts/evaluate_axes.py`).
        체크포인트가 git에 없어 Drive 미러에서 받아와야 한다 (`runs/CHECKPOINTS.md`).
  - [ ] **`reports/stage_b.md` 취합** — `stage_b_curves*.md`는 스크립트 산출물이고 서사가 아니다.
- [ ] **Task 8 — 문서화**: README 결과·그림·한계 논의 갱신

## 하지 말 것

- `data/` 커밋, 대회 데이터의 원본·가공본 재배포 (로더는 로컬 파일만 참조)
- 비식별 파장축을 실제 파장으로 단정하는 코드·서술
- 실행으로 뒷받침되지 않은 수치 주장 (모든 표는 스크립트 산출)
- 격자 스냅 후처리를 주 결과로 제시하기 (누설이며 분리 보고 대상)
- **캘리브레이션에서 물성(n, k)을 채널별 자유 곡선으로 두기** — λ의 매끈한 함수라는 제약이
  사라져 모델 오차를 흡수한다. RMSE는 내려가지만 나온 곡선은 물성이 아니다
  (k_Si가 λ에 대해 톱니로 나오는 식)
- **문헌 광학상수 표를 손으로 옮겨 적기** — 원본을 `src/physics/literature/`에 두고
  파싱한다. 눈대중 19점 표는 E1 봉우리를 4.3% 깎아 역해 MAE를 0.663 → 1.104 nm로 악화시킨다
  (대조군으로만 보존: `dispersion.CoarseTableNK`)
- **결정질 Si에 Tauc-Lorentz 쓰기** (비정질용 — E1·E2 임계점을 표현 못 한다)
- 캘리브레이션에서 게이지 고정 없이 n과 λ를 동시에 자유 학습시키기
- 백로그 순서 건너뛰기 (특히 Task 1 테스트 통과 전에 학습 코드 작성 금지)
- **run이 도는 중에 `src/`를 편집하기** (파라미터 목록과 초기값이 어긋나는 순간에 걸린다)
- **분할을 손으로 재구성하기** — 학습·평가·진단·노트북 전부 `prepare_from_config(cfg)` 한 곳만
  거친다. 인자를 직접 넘기면 새 split 옵션(`holdout_thickness` 등)이 조용히 빠져 **다른 split을
  재구성한다** (노트북의 Drive 무결성 검증이 정상 체크포인트를 손상으로 오판하는 형태로 걸렸다)
- **적합 수준을 맞추지 않고 일반화 격차(val − train)를 비교하기** — 덜 적합된 모델은 격차가
  자명하게 작으므로 정규화 효과의 증거가 되지 못한다. 축을 에폭이 아니라 `train_l1`으로 두고
  같은 적합에서 val을 비교한다 (`scripts/analyze_stage_b_curves.py`)
- **리포트·주석에 자기정정 이력 쌓기** — 정정은 규칙으로 바꿔 쓰고 시간축은 git·주차
  노트에 맡긴다 (위 기록 규약)

## 자주 쓰는 명령

```bash
pytest -q
ruff check . && ruff format .
python scripts/verify_data.py
python scripts/measure_noise.py                                                   # 노이즈 σ·유계 상한
python scripts/check_notebook_regression.py                                       # 노트북 되돌림 탐지
python -m src.train --config configs/mlp_baseline/dropout0.0.yaml
python -m src.calibrate --config configs/stage_a/joint-lam3-sin2-si2-schinke.yaml  # Stage A (디코더)
python scripts/diagnose_calibration.py                                            # Stage A 게이트·그림
python -m src.evaluate --run runs/mlp_baseline/dropout0.0                          # holdout 재평가

# Stage B 물리 손실 — 본 학습은 Colab GPU, 로컬은 스모크만 (약 35초).
# run-name에 -smoke를 붙이는 것이 필수다 — 서브셋 run이 완료 기록을 남기면 본 run이 스킵된다
python -m src.train_gpu --config configs/stage_b/beta100.yaml --device cpu \
  --subset 20000 --epochs 2 --run-name beta100-smoke

# 적합 수준을 맞춘 β 대조 — 학습·체크포인트 없이 커밋된 train.log만으로 돈다.
# 인자를 안 주면 라운드 1(무작위 split). 산출물 이름에 split tag가 붙어 라운드끼리
# 덮어쓰지 않는다 (tag = 대조군 run 이름에서 beta<수> 꼬리를 뗀 것).
python scripts/analyze_stage_b_curves.py                                   # 라운드 1
python scripts/analyze_stage_b_curves.py $(for b in 0 30 100 300; do \
  printf -- '--run runs/stage_b/ft-heldout-beta%s ' $b; done)              # 라운드 3

# 평가 축 (노이즈 강건성 · 신뢰도 지표) — 학습 없이 체크포인트 위에서 돈다.
# 체크포인트는 git에 없으므로 Drive 미러에서 복사한 뒤 실행한다 (runs/CHECKPOINTS.md).
# 단 `stage_b/beta0`은 `level1_cnn/flatten-dilated-bound`와 비트 동일하고 그 model.pt는
# git 히스토리에 있다 (git show <커밋>:... — runs/CHECKPOINTS.md) — 백본 축은 Drive 없이 돈다.
python scripts/evaluate_axes.py --run runs/stage_b/beta0 --run runs/stage_b/beta100

# 역산 refinement (추론 후 물리 보정) — 위와 같은 체크포인트 + Stage A 디코더. 전체 holdout
# 81,000행이 CPU 8스레드로 약 55분(팔 3개)이라 확인은 --rows로 줄여서 한다 (무작위 표본).
python scripts/refine_inversion.py --rows 5000

# 역해 LM 추론 비용 (장치·dtype·반복수). GPU 측정은 notebooks/inversion/round1_gpu-bench.ipynb.
python scripts/bench_invert.py --rows 4096
```

Stage A 경로는 최대 상주 메모리 **약 5 GB**를 쓴다 (조건부 평균이 train 전체를 올린다).

## 세션 시작 체크리스트

1. 본 파일과 git log/status로 현재 상태 파악 → 백로그 최상단 미완료 Task 확인
2. 착수 전 계획을 2~4줄로 요약해 제시
3. 구현 → 테스트 → 커밋 → 체크박스·docs 주차 노트 갱신
