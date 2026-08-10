# Week 1 실험 노트 — 2026-08-08 ~ 2026-08-14

**주 목표** (로드맵 Week 1): 스캐폴드 · TMM 모듈+테스트 · 데이터 검증 · EDA · baseline 학습
**결과**: 08-10에 전부 완료. baseline **holdout MAE 4.599 nm** 확정.

---

## 2026-08-08 (토) — Task 0~3: 스캐폴드, TMM, 데이터 검증, EDA

### Task 0 — 스캐폴드 (`73bd960`, `3550f42`)

- 디렉토리 구조, requirements, ruff/pytest 설정, 시드 유틸.
- 함정 하나 수정: `.gitignore`의 `data/` 패턴에 루트 앵커(`/`)가 없으면
  `src/data/` **파이썬 패키지까지 무시**되어 소스가 커밋에서 빠진다.

### Task 1 — 미분가능 TMM 모듈 (`a067e32`)

- `src/physics/tmm.py` + 물리 단위 테스트 **7종 green**:
  무층 극한 · 에너지 보존(R+T=1) · λ/4 무반사 · Airy 단층 대조 ·
  미분가능성(유한차분 대조) · 흡수 기판 · **비대칭 2층 vs 재귀 프레넬**(보강).
- 보강 이유: 명세 6종만으로는 층 **적층 순서를 고정하지 못했다** — 순서를 뒤집어도
  전부 통과했다. 비대칭 2층 스택을 재귀 프레넬 공식과 대조하는 7번째 테스트로 잡았다.

### Task 2 — 데이터 검증 (`057e11f`, `scripts/verify_data.py`)

- 계약 확정: train (810,000, 230), 각 층 {10, 20, …, 300} nm 30값,
  **30⁴ = 810,000 전수 조합** (중복·결측 없음), test 10,000행(id 일치).
- **발견 — 가설 "반사율 ⊂ [0, 1]"은 거짓.** 검증 항목 중 유일한 FAIL이었고,
  이 프로젝트에서 가장 파급이 큰 발견이 됐다.

  | | train | test |
  |---|---|---|
  | 범위 | [**−0.015117**, 0.943648] | [**−0.014998**, 0.940716] |
  | `R < 0` 인 값 | 636,520개 (0.35%) | 7,929개 (0.35%) |
  | `R < 0` 을 포함한 행 | **46.9%** | **47.7%** |
  | `R > 1` | 0건 | 0건 |

  음의 반사율은 물리적으로 불가능 → 참 스펙트럼 위에 **가산 노이즈 σ ≈ 0.0087**.
  서로 독립인 추정 두 가지가 일치한다 — 2차 차분(상한 0.009122, 곡률 적은 행 0.008838)과
  음수 하한 기반 `a/√3 = 0.008728`. 음수 1퍼센타일 −0.0135가 하한 −0.0151에 붙어 있는 것도
  가우시안 꼬리보다 유계(균등) 노이즈에 부합. 최대 R = 0.944라 위쪽 절단은 없다.
  유도 전체는 [reports/eda_notes.md](../reports/eda_notes.md) §4.

### Task 3 — EDA (`de41f28`, `scripts/eda.py` → `reports/eda_notes.md`)

- **두께↑ → fringe 조밀** (정성 확인, fig1 히트맵). SiN(1·3)이 SiO₂(2·4)보다 촘촘 —
  광경로차 2·n·d, 문헌 굴절률 비(≈2.0 vs 1.46)와 방향 일치.
- **층별 민감도**: +10 nm의 SNR 10.3(layer_2) ~ 15.3(layer_3) — **원리적 사각지대 없음.**
  층별 MAE 격차는 모델 문제로 해석한다. layer_4는 40~60 nm에 민감도 국소 최소(0.0760).
- **채널별 정보량**: 대역 오른쪽 끝이 왼쪽의 약 **3배** (노이즈는 채널에 균일 —
  3차 차분 σ비 1.006). 채널 자르기 주의, 물리 손실 채널 가중은 ablation 후보.
- **노이즈 분포**: 곡률 적은 행에서 σ → 0.0088, 초과 첨도 → −0.52로
  균등분포 이론값(잔차 통계 −0.6) 방향으로 **둘 다 수렴** → 균등분포에 가깝다.

---

## 2026-08-09 (일) — 자체 정정, 전수 감사, Stage A 게이트 확정

- **표본 편향 수정** (`cfe0baa`): train 행이 (layer_1..4) **사전식 정렬**이라 `x[:N]`
  앞머리 자르기는 표본이 아니다(layer_1 = 10 nm 구석만 봐서 σ 과소). 시드 고정 무작위
  표집으로 교체. "독립 추정 3개" 주장도 2개로 정정 — 고주파 잔차는 2차 차분과
  대수적으로 **같은 추정량**이라 독립 확인이 아니다(새 정보는 첨도뿐).
- **FFT 기반 분석 전부 제거** (`9131c29`): 비식별 파장축에서는 채널 간격이 파장에서
  균일한지 알 수 없어 주기 분석을 신뢰할 수 없다.
- **전수 감사** (main @ `9131c29` 기준): 물리·데이터·통계 주장 25건 독립 재검증 전부 일치,
  치명 오류 0. 발견 F1~F9(근거 교체·중복 정리 등)는 `0e13907`로 정정.
- **결정 — Stage A 판정 게이트를 (a)+(c)로 확정** (`34b7072`):
  - 원 기준 "재구성 RMSE < 5×10⁻³" **폐기** — 관측 R에 σ ≈ 0.0087 노이즈가 있어
    *완벽한* forward 모델도 도달 불가. 그대로 두면 맞는 TMM을 기각하게 된다.
  - **(a)** RMSE < 1.2σ ≈ 0.0105 와 **(c)** 잔차 백색성 진단을 병행, 둘 다 통과해야 채택.
  - **(b)** "스무딩 후 비교"는 기각 — 노이즈와 함께 무늬 곡률도 뭉개져 판별력이 떨어진다.
- Task 4 착수: baseline MLP 모델 + 팩토리 + 단위 테스트 (`c88e10c`).

---

## 2026-08-10 (월) — Task 4 완료: baseline 확정 + 감사 정정 + 문서 개편

- 학습·평가 파이프라인 (`884c47f`): holdout 10%(seed 42)를 프로젝트 공통 검증셋으로
  고정, k-fold 앙상블은 holdout 제외 90% 안에서만 접는 프로토콜.
- MLP 재설계 (`49b8d0c`, `fb8009f`): 블록형 Linear→Norm→Act→Dropout + residual 옵션,
  hidden 512×3, train.log 실시간 기록.
- output_bound off 실험 (`99fe78e`) → bare regression 채택.
- **baseline 확정 (`2e7ef92`): dropout 0.0, holdout MAE 4.599 nm.**
  dropout 0.1은 6.645 nm로 순손실(+44%) — 전수 격자라 과적합 압력이 약하다.
  취합 리포트: [reports/mlp_baseline.md](../reports/mlp_baseline.md).
- 실험 관리 2단 구조(experiment/variant) 도입, runs/ 전체 git 추적 (`c3489d0`). PR #1 머지.
- **감사 후속 정정** (`d27fbe5`): Task 4 산출물 재감사 — 치명 오류 0, 커밋된 체크포인트가
  현재 코드로 리포트 수치를 정확히 재현함을 확인. 정정한 것: 리포트 수렴 서술 2곳
  (로그와 불일치), 이관 기록·스테일 ckpt_path, **holdout이 best-epoch 선택에도 쓰인다는
  사실 명시**(min-선택 편향 ~0.01 nm 규모; k-fold 모드는 OOF 선택이라 미해당),
  파라미터 수 0.65M 실측, 1등 솔루션 출처 링크, `*.ipynb` gitignore.
- **문서 구조 개편** (이 커밋): README 슬림화(소개·구조·불변 사실만) +
  `docs/week_N.md` 주차별 실험 노트 도입. 진행 서사·발견 기록·TODO는 이 노트에서 관리.

## 2026-08-10 (월) — Task 5 착수: MLP vs 1D CNN (변인 통제 설계)

- **CNN1D 구현** (`8f8838b`): `src/models/cnn.py` — ConvBlock(다중 커널 병렬 분기,
  ablation용 플래그) 스택 → GAP → Linear 헤드. 블록 구성(Conv→BN→GELU)·출력 규약
  (bare head + bias 중앙 초기화)을 baseline MLP와 동일하게 두고 **연결 패턴만** 바꿨다.
  모델 테스트 13종 추가, 전체 42종 green.
- **변인 통제 설계** — "성능 차 = 구조 bias 기여"를 말하기 위한 장치:
  1. **파라미터 매칭**: channels (32,64,128,200,280)에서 CNN 646,340 vs MLP 646,660
     (**−0.05%**). 테스트가 ±10%를 강제한다 — 용량 차이 반론 차단.
  2. **채널 셔플 대조군**: 같은 CNN에 고정 무작위 순열(seed 7)로 채널 순서만 파괴.
     MLP는 입력 순열에 불변(첫 Linear 열 순서만 바뀜)이므로 셔플 대조군은 CNN에만
     의미가 있고, CNN vs shuffled-CNN 낙폭이 **스펙트럼 순서 정보의 기여를 직접 측정**한다.
  3. 학습 프로토콜(seed 42, holdout, AdamW 1e-3/wd 1e-4, warmup+cosine, 30ep, batch 512)
     은 baseline과 완전 동일. 조작 변인은 아키텍처 하나.
- 아키텍처 결정: 첫 블록 stride 1(226 전체 해상도 유지 — fringe 고주파 보존), 이후
  stride 2 4회. stride-2 stem은 에폭당 4분으로 빨라지지만 해상도 손실 위험이 있어 기각.
  CPU 벤치마크: CNN 에폭 ~7.1분 (MLP 26초의 ~16배, 가중치 공유로 파라미터당 연산이
  많은 conv의 본질적 비용).
- ~~full run 2종 순차 실행 시작 — 총 ~7시간 예상~~ → **CPU 학습 취소, GPU(Colab)로 전환**.
  로컬이 CPU 전용(torch 2.13.0+cpu)이라 CNN부터는 비용이 안 맞는다 (에폭 ~7분).
- **GPU 학습 경로 구축** (`8e72b43`): `src/train_gpu.py` — baseline을 만든 CPU 경로
  (`src/train.py`)는 수정하지 않는 디커플 원칙. 산출물 계약 동일, 데이터 GPU 상주,
  체크포인트는 CPU 텐서로 저장(로컬 evaluate.py 호환). holdout 전용.
  CPU 스모크(subset 2만, 2ep)에서 train.py와 **MAE 완전 일치(71.5161)** 확인 —
  같은 시드·같은 CPU 연산이면 두 경로가 같은 결과를 낸다.
  Colab 드라이버 `notebooks/colab_train.ipynb` (VM/로컬 커널 자동 분기, SMOKE 플래그,
  PAT push 셀). 워크플로: Colab에서 학습·push → 로컬 pull·분석·리포트.
- 열린 확인 사항: lr 1e-3은 MLP 기준으로 고른 값이라 CNN에 불리할 수 있음 —
  결과가 이상하면 subset lr 스윕(3e-4/1e-3/3e-3)으로 공정성 확인 예정.
  GPU에서는 스윕 비용이 싸졌으므로 본 학습 전에 돌려봐도 좋다.
- **첫 결과 (T4, 에폭 ~67초, run당 ~34분; 커밋 `0fd750a`) — 예상과 반대, 둘 다 중요**:

  | run | holdout MAE | 층별 (L1/L2/L3/L4) |
  |---|---|---|
  | MLP baseline (Task 4) | **4.599 nm** | 3.56 / 5.39 / 4.78 / 4.66 |
  | CNN single-scale (GAP) | 18.161 nm | 12.29 / 23.53 / 17.71 / 19.12 |
  | CNN **shuffled** (대조군) | 12.234 nm | 9.93 / 14.54 / 12.52 / 11.94 |

  체크포인트 3종 모두 로컬 CPU에서 기록 수치 정확 재현 확인 (scratchpad 분석 스크립트).
  1. **CNN(GAP)이 MLP보다 4배 나쁘다.** 파라미터·프로토콜 통제 상태이므로 구조 자체의 문제.
  2. **채널을 섞은 대조군이 unshuffled보다 33% 좋다** — "순서 활용"이 아니라
     **위치 불변성(GAP + stride)이 문제**라는 직접 증거. 이 태스크의 정보는 fringe의
     파장축 **절대 위치·위상**에 실려 있는데 GAP가 파장축을 평균으로 붕괴시켜 버린다.
     MLP는 채널별 절대값을 그대로 쓴다. 섞인 입력은 국소 윈도우가 원거리 채널들의
     무작위 투영이 되어 GAP 후에도 살아남는 정보가 더 많다 (매끄러운 스펙트럼의
     이웃 채널은 강상관 → 국소 특징이 정보 빈약).
  3. 진단 (holdout 예측 기준): 예측-정답 상관 MLP 0.992~0.997 vs CNN 0.915~0.970;
     CNN 예측 std 76.7~82.0 < 타깃 83.7 (평균 회귀 = 정보 손실, L2 최악);
     얇은 두께(10~60 nm)에서 CNN MAE 28.6 vs MLP 6.5 — 무늬가 거의 없는 평평한
     스펙트럼(정보가 저주파·절대 레벨에 있음)에서 국소 conv+GAP가 가장 크게 무너진다.
     layer_2(SiO₂, 최저 SNR 10.3)가 CNN에서 상대적으로 가장 악화 — 약한 신호 층이
     구조 손실에 가장 취약.
  4. 학습 곡선: 둘 다 ep29~30에서 미미하게 개선 중 (수렴 직전). 동일 프로토콜이므로
     비교는 유효하나, "CNN에 30ep가 부족한가"는 flatten 결과를 본 뒤 판단.
  - **다음 변형: GAP → flatten 헤드** (위치 정보 보존). CNN이 MLP와 경쟁하려면
    이 축이 필수라는 가설. 스모크 산출물(-smoke)은 저장소에서 제거, push 셀도 제외 처리.
- **라운드 2 완료 — flatten·dilated·flatten-dilated (`2e50702`)**:
  **flatten-dilated 2.931 nm로 baseline(4.599) 대비 −36.3%, 첫 돌파.**
  분해: 지배 요인은 수용영역(dilated 단독 +13.2, RF 97→259 전 대역), 위치 보존
  (flatten 단독 +4.5)은 그 위에서 시너지 — 라운드 1 진단("GAP가 병목")은 절반만
  맞았다. 확정 구성은 전 두께 구간에서 baseline 우위 (얇은 10~60 nm: 4.37 vs 6.49).
  체크포인트 6종 로컬 재현 검증 완료. **취합 리포트: [reports/level1_cnn.md](../reports/level1_cnn.md)**.
  Level 2 백본은 flatten-dilated로 확정.
- 인프라: 세션 유실 대비 체계 구축 (`2cf209e`) — best 갱신 즉시 model.pt 저장,
  매 에폭 resume.pt(+RNG)·Drive 미러, 재실행 시 완료 run 스킵·진행 run 재개
  (무중단 실행과 동일 결과, 테스트 검증). 라운드 2에서 실전 작동 (세션 종료 후
  미러 기록으로 3개 run 스킵·복원 → push). push 셀 PAT 정적 소스화(Drive/Secrets),
  전 실험 완료 시 런타임 자동 반납. 노트북은 라운드별 1개로 구조화 (`0ac1fce`).
- **라운드 3 준비 — bound on/off (Task 5 마지막 축)**: `configs/level1_cnn/flatten-dilated-bound.yaml`
  (flatten-dilated 대비 조작 변인 output_bound 하나) + `notebooks/level1_cnn/round3_bound.ipynb`.
  로컬 조립 확인: 파라미터 662,020으로 flatten-dilated와 동일(bound 무파라미터),
  출력 [10, 300] 중앙 초기화 — 공정 비교 조건 충족. 다음: Colab에서 round3 Run-All.

---

## 확정 수치 (이후 작업의 기준선)

| 항목 | 값 | 근거 |
|---|---|---|
| 노이즈 | σ ≈ 0.0087, 균등분포에 가까움, 채널에 균일 | [reports/eda_notes.md](../reports/eda_notes.md) §4 |
| Stage A 게이트 | RMSE < 1.2σ ≈ 0.0105 **+** 잔차 백색성 (둘 다) | 08-09 결정 (`34b7072`) |
| baseline | holdout MAE **4.599 nm** (MLP 512×3, dropout 0, bare regression) | [reports/mlp_baseline.md](../reports/mlp_baseline.md) |
| 층별 최소 SNR | 10.3 (layer_2) — 사각지대 없음 | eda_notes §2 |
| 강건성 주입 노이즈 | 균등 ±0.015 기본, "기존 노이즈 위 추가분"으로 표기 | eda_notes §4 |

## TODO

작업 백로그 (순서 준수 — 상세 DoD는 CLAUDE.md):

- [ ] **Task 5 — Level 1 ablation**: MLP vs 1D CNN, 단일 vs 다중 스케일, bound on/off 비교표
  — CNN·다중스케일(dilated) 완료 ([reports/level1_cnn.md](../reports/level1_cnn.md),
  flatten-dilated 2.931 nm). **bound on/off만 남음** (flatten-dilated 기준으로 1 run)
- [ ] **Task 6 — Stage A 캘리브레이션**: 게이트 판정까지 (게이지: SiO₂ Cauchy freeze)
- [ ] **Task 7 — Stage B 물리 손실**: beta ablation + 신뢰도 지표 분석
- [ ] **Task 8 — 문서화**: README 결과·그림·한계 논의 갱신

백로그 외 열린 항목:

- [ ] 에폭 연장 실험 — cosine 스케줄 재설정 필요 (30ep 끝은 LR→0 플래토; 대회 1등은 100ep)
- [ ] strong baseline (1등 솔루션 축소 재현) — 0.65M vs 213M 격차 측정용
- [ ] holdout과 분리된 best-epoch 선택용 split 도입 여부 (현재는 문서 명시로 처리)
- [ ] 물리 손실 채널 가중 ablation (대역 오른쪽 정보량 3배 — 기본은 균등 가중)
- [ ] layer_4 40~60 nm 민감도 저하 구간의 오차 확인 (두께 구간별 오차 분석 시)
