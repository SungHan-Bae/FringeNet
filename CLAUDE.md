# CLAUDE.md — FringeNet

이 파일은 매 세션 자동 로드되는 프로젝트 작업 메모리다.
배경·설계는 @README.md, 진행 기록·발견·TODO 상세는 docs/week_N.md(주차별 실험 노트) 참고.
여기에는 작업에 필요한 계약과 스펙만 둔다.

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
  parquet 캐시는 `data/cache/`에 자동 생성 (최초 29초 → 이후 3.4초).
- **데이터 파일을 git에 커밋하거나 저장소에 재배포하지 않는다** (.gitignore: `/data/**`,
  구조는 .gitkeep으로만 추적). 앵커(`/`)가 없으면 `src/data/` 패키지까지 무시되니 주의.
  runs/·configs/·reports/는 전체 git 추적.
- `train.csv`: `layer_1..layer_4` = 두께[nm] 타깃, 컬럼 `"0".."225"` = 반사율.
- 헤더 0~225는 **비식별화된 파장 인덱스**다. 실제 nm로 단정하는 코드/서술 금지.
  단, 채널 순서는 연속 스펙트럼으로 의미 있음 (1D conv 유효).
- 검증 완료 (Task 2, `scripts/verify_data.py`, 종료코드 0):
  - [x] 행 수 = 810,000 / train shape (810000, 230) = layer 4 + 채널 226
  - [x] 각 layer 고유값 = {10, 20, ..., 300} (30개, 10 nm 격자)
  - [x] 결측 없음, 중복 조합 없음, **30⁴ = 810,000 전수 조합** (격자를 빠짐없이 채움)
  - [x] test 10,000행 (`id` + 채널 226), sample_submission의 id와 일치
  - [ ] ~~반사율 ⊂ [0, 1]~~ → **거짓**. 아래 참조.
- **반사율은 [0,1]을 벗어난다 — 데이터에 가산 노이즈가 있다 (중요)**:
  - train 범위 [-0.015117, 0.943648], test [-0.014998, 0.940716]. `R > 1`은 0건.
  - 음의 반사율이 train 값의 0.35%, **행의 46.9%**에 등장. 물리적으로 불가능하므로
    참 스펙트럼 위에 노이즈가 얹힌 것.
  - 노이즈 크기 → **σ ≈ 0.0087~0.0088**. 서로 독립인 추정 **두 가지**가 일치한다:
    (1) 2차 차분 `Var(y[i-1]-2y[i]+y[i+1]) ≈ 6σ²` — 곡률이 섞여 **상한**이라
    무작위 표본 0.009122, 곡률이 적은 평평한 행만 보면 0.008838으로 내려간다.
    (2) 음수 하한을 균등 노이즈 ±a로 본 a/√3 = **0.008728** (독립).
    음수값이 -0.0151에서 잘리고 1퍼센타일이 -0.0135인 것도 가우시안보다 유계 노이즈에 부합.
  - **주의**: 고주파 잔차 `y[i]-(y[i-1]+y[i+1])/2` 추정은 2차 차분의 -1/2배라
    **같은 추정량**이다(독립 확인이 아니다). 거기서 얻는 새 정보는 첨도(분포 모양)뿐.
  - 스크립트는 이제 확인된 사실(`R ≤ 1`, `R ≥ -0.02`)을 지킨다.
- 평가지표: MAE. 제출 파일은 `sample_submission.csv` 형식.

## EDA 확정 수치 (Task 3, `scripts/eda.py` 산출)

해석은 `reports/eda_notes.md`, 표는 `reports/eda_metrics.md`. 설계 결정에 쓰이는 값만 여기 둔다.

- **노이즈 σ ≈ 0.0087~0.0088, 균등분포에 가깝다.** 2차 차분 계열 추정은 스펙트럼 곡률이
  섞여 σ를 위로, 첨도를 0(가우시안) 쪽으로 민다. 곡률이 적은 행만 골라 재면 둘 다
  균등분포 예측으로 수렴한다 — 이 방향성이 근거다:

  | 표본 | σ | 초과 첨도 |
  |---|---|---|
  | 무작위 100,000행 (상한) | 0.009120 | −0.328 |
  | 그중 진폭 하위 10% | **0.008841** | **−0.519** |
  | *(독립)* 음수 하한 a/√3 | **0.008728** | — |
  | *(이론)* 균등 노이즈의 잔차 통계 | — | **−0.60** (분포 자체는 −1.2) |

  → 강건성 실험의 주입 노이즈는 **균등 ±0.015를 기본**으로 한다. 가우시안은 별개 질문으로 병기.
- **표본 추출 주의**: train 행은 (layer_1..layer_4) **사전식 정렬**이다. `x[:N]` 앞머리
  자르기는 layer_1 = 10 nm 구석만 보게 되어 표본이 아니다(σ가 0.00872로 낮게 나왔다).
  통계를 낼 때는 반드시 시드 고정 무작위 표집을 쓸 것.
- **층별 민감도** (한 층만 +10 nm, RMS ΔR / SNR): layer_1 0.1105/12.7, layer_2 0.0896/**10.3**,
  layer_3 0.1329/**15.3**, layer_4 0.1025/11.8.
  → 최소 SNR 10.3. **원리적 사각지대 없음.** 층별 MAE 격차는 관측 한계가 아니라 모델 문제로 해석한다.
  layer_4는 40~60 nm에서 민감도가 0.0760까지 꺼지는 국소 최소가 있다(최저 50 nm,
  eda.py "구간 최소" 산출 — 두께 구간별 오차 분석 시 확인).
- **채널별 정보량 불균등**: mean |ΔR|이 채널 0~9 대비 216~225에서 약 **3배**(2.86~3.10×).
  노이즈는 채널에 균일 — 3차 차분 채널 프로파일의 좌/우 끝 σ 비 1.006 (eda.py 산출;
  채널별 반사율 std 0.18~0.21은 신호 분산이라 균일성의 근거가 아니다) → 대역 오른쪽 끝
  SNR이 3배 좋다. 채널 자르기·downsample 주의.
  물리 손실의 채널 가중은 ablation 후보(기본은 균등 가중).
- **두께↑ → fringe 조밀** (정성 확인, fig1 히트맵). 같은 두께에서 SiN 층이 SiO₂ 층보다
  무늬가 촘촘하다 — 굴절률이 크면 광경로차 2·n·d가 커진다는 예상과 방향 일치
  (문헌 n(SiN) ≈ 2.0 vs n(SiO₂) ≈ 1.46). Stage A 캘리브레이션의 방향성 확인에 쓸 것.
- 죽은 채널 없음(채널별 표준편차 0.1816~0.2124로 고르다). 최대 R = 0.944로 **위쪽 절단 없음**.
- **하지 말 것**: 무늬 개수를 세어 "두께에 비례한다"는 식의 정량 법칙을 주장하기.
  파장축 비식별화(채널 간격이 파장에서 균일한지 불명) + 대역에 들어가는 무늬 수가 적음
  + 4층 beat + 노이즈 때문에 신뢰할 수준이 안 나온다. 특히 네 층 모두 10 nm면 스펙트럼이
  거의 평평해(진폭 ≈ 노이즈) 무늬를 세는 지표는 노이즈를 세고 **부호가 반대로** 나온다.
  두께↑ → fringe 조밀은 **정성 확인까지만** 한다.

## 코딩 규약

- Python ≥ 3.11, PyTorch ≥ 2.x. 의존성 최소 유지(순수 PyTorch; lightning/hydra 등 대형 프레임워크는 사용자 승인 없이 추가 금지).
- 타입힌트 필수. docstring에 텐서 shape 명시. shape 규약:
  `d: (B, L)` float, `n_layers: (L, W)` complex, `lam: (W,)`, 반환 `R: (B, W)` real.
- 층 수(L=4)에 대해서만 파이썬 루프 허용. B, W축은 반드시 벡터화.
- dtype: 검증·캘리브레이션은 complex128, 학습은 complex64. d는 real 유지(autograd가 d로 흐르게).
- 포맷/린트 ruff, 테스트 pytest, 시드 고정 유틸(`src/utils/seed.py`),
  설정은 `configs/<실험>/<변형>.yaml` (평가 규약의 실험 관리 구조 참조).
- **GPU가 필요한 모든 실험·태스크는 Colab에서 돌린다** (로컬 WSL2는 CPU 전용 —
  분석·검증·리포트 담당). 학습 엔트리는 `src/train_gpu.py` (holdout 전용). CPU 파이프라인(`src/train.py` —
  baseline 검증 경로)과 디커플, 수정 금지. 산출물 계약은 동일하며 체크포인트는 CPU
  텐서로 저장돼 로컬 evaluate.py와 호환. CPU↔GPU는 bit 단위 재현이 아니라 MAE 수준에서
  비교한다. 워크플로: Colab에서 학습·push → 로컬 pull·분석.
  **세션 유실 대비는 필수 규율** — Colab 런타임은 언제든 끊길 수 있다는 전제로,
  GPU에서 도는 모든 학습·장시간 스크립트와 노트북은 다음 4가지를 갖춰야 한다
  (train_gpu.py에 구현돼 있으니 재사용이 기본; Stage A 캘리브레이션 등 새 GPU
  스크립트를 만들면 같은 계약을 구현할 것):
  1. best(또는 진행분) 갱신 즉시 체크포인트 저장 — 종료를 기다리지 않는다
  2. 에폭(또는 그에 준하는) 단위 resume 상태 저장(+RNG) 및 Drive 미러 백업
  3. 재실행 시 자동 감지: 완료 run 스킵 + 진행 run 재개 (재개 결과는 무중단
     실행과 동일해야 한다 — RNG 복원, 테스트로 검증)
  4. 노트북은 Run-All이 입력 대기 없이 end-to-end로 돌고(PAT 등 비밀은 정적
     소스에서 로드), 전 작업 완료+push 성공 시 런타임 자동 반납
- **Colab 노트북은 라운드(학습 세션)별 1개**: `notebooks/<대실험>/roundN_<내용>.ipynb`.
  완료된 라운드의 노트북은 실행 로그 보존을 위해 수정·재실행하지 않는다. 새 라운드는
  직전 노트북을 복사해 헤더·CONFIGS 갱신 + 출력 비움으로 시작한다.
  **복사 전 원본이 디스크 최신인지 확인** — IDE의 옛 버퍼가 저장되면 커밋된 셀 fix를
  되돌린다 (실사례: `dbefab4`가 `38add7d`의 push 셀 fix를 덮어씀 → round3에 재이식).
  GPU 학습 노트북 필수 셀 규약:
  1. Drive 마운트는 항상 `drive.mount(..., force_remount=True)` — 이전 세션의 stale
     마운트를 배제하고 최신 상태로 다시 마운트한다
  2. push 셀 PAT는 정적 소스에서 자동 로드 (env GITHUB_PAT → Colab Secrets →
     Drive `FringeNet/secrets/github_pat.txt` → 없을 때만 프롬프트) — Run-All 무정지
  3. 런타임 자동 반납의 취소 대기 sleep은 **5초** (60초 등 긴 대기 금지 — 유휴 과금)
  4. **push 후 Drive 체크포인트 무결성 검증, 통과해야만 런타임 반납**:
     `drive.flush_and_unmount()`(대기 업로드 완료 보장) → 재마운트 → **미러의 model.pt를
     다시 로드해 holdout 재추론 → 기록된 val MAE 재현 확인**. Drive FUSE는 비동기
     업로드라 세션이 죽으면 대용량 파일이 구버전으로 남을 수 있고(실사례: 3.4GB
     resume.pt가 5에폭 뒤처짐), git 미추적 대형 model.pt는 Drive가 유일본이므로
     내용 검증 없이 반납하면 안 된다.
- 커밋 메시지: `feat|fix|refactor|test|docs|exp: ...`

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
- Level 1(구조·표현 bias): 1D CNN. 입력 `(B, 1, 226)` → conv 스택 → head(gap|flatten) → 4출력.
  각 요소(head/dilations/kernel_sizes/bound/셔플 대조군)는 config 플래그로 on/off (ablation용).
  **확정 (2026-08-11, reports/level1_cnn.md): flatten-dilated + output bound가 holdout
  MAE 2.346 nm로 baseline 대비 −49%. 국소 conv는 수용영역이 전 대역을 덮고(dilated
  RF 259) 파장축 위치가 보존될 때만(flatten) 유효 — 소박한 conv+GAP는 4배 나쁘고 채널
  셔플 대조군보다도 나쁘다. sigmoid bound는 격자 끝 오차를 지워 추가 −20%(MLP 때와
  반대 결론 — 강한 백본에서는 범위 밖 초과분이 남은 오차를 지배). Level 2 백본 =
  flatten-dilated-bound.**
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
  - **판정 게이트 (2026-08-09 확정: (a)+(c) 병행, 둘 다 통과해야 물리 디코더 채택)**:
    - (a) 재구성 RMSE < 1.2σ ≈ 0.0105 (R 단위, σ = 0.0087 노이즈 바닥 기준).
      원 기준 5e-3은 폐기 — 관측 R에 σ ≈ 0.0087의 가산 노이즈가 있어 **완벽한
      forward 모델도 관측 대비 RMSE가 σ 아래로 못 내려간다** (원리적 통과 불가,
      이대로 두면 맞는 TMM을 틀렸다고 기각하게 된다).
    - (c) 잔차 백색성 진단: 잔차 R_obs − R_TMM(d_true)가 두께·채널에 대해 구조 없이
      백색이고 크기가 σ ≈ 0.0087과 일치해야 한다. (a)를 통과해도 잔차에 두께 의존
      구조가 남으면 모델 오차로 보고 기각한다. (노이즈가 iid·균등에 가깝다는 EDA
      확인 덕에 "백색 + 크기 σ" 기준의 판별력이 좋다.)
    - (b) "스무딩 후 비교"는 채택하지 않음 — 노이즈와 함께 무늬 곡률도 뭉개져 판별력이 떨어진다.
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
- **노이즈 강건성**: 입력 R에 노이즈 주입 시 MAE 열화 곡선. EDA 확정에 따라 기본은
  데이터와 같은 종류인 **균등 ±0.015**, 가우시안(σ = 1e-3, 5e-3, 1e-2)은 "다른 종류의
  노이즈에 대한 강건성"이라는 별개 질문으로 병기한다.
  단, 데이터에 이미 σ ≈ 0.0087의 노이즈가 있으므로(위 데이터 계약 참조) 주입 노이즈는
  **추가분**이다. 리포트에 "기존 노이즈 위에 더한 양"임을 명시할 것.
- 위 셋은 Task 4 이후 평가 스크립트에 고정 포함한다. 좋은 숫자만 고르지 않는다.
- **실험 관리 구조 — 대실험(experiment) / 변형(run) 2단**:
  - 설정 `configs/<실험>/<변형>.yaml` (config에 `experiment`·`run_name` 키 필수).
  - 산출물 `runs/<실험>/<변형>/` = **model.pt + train.log + metrics.json 세 가지만**
    (metrics.json이 설정 스냅샷을 겸한다 — 시작 시 기록, 완료 시 결과 포함 덮어씀.
    train.log는 에폭마다 실시간 기록). 전부 git 추적.
    **예외: GitHub 파일당 100MB 한도를 넘는 model.pt는 git 대신 Drive 미러에 보관**하고
    .gitignore에 경로를 명시해 제외한다 (실사례: strong_baseline/winner-repro-asis 813MB —
    push가 pre-receive hook에서 거부된다). 로컬 분석이 필요하면 Drive에서 수동으로
    내려받는다 (.gitignore 덕에 커밋 위험 없음).
  - 변형 이름은 번호(sub_run_1)가 아니라 **무엇이 다른지 드러나는 서술형**으로
    (예: dropout0.0, layernorm, residual-on).
  - 대실험이 끝나면 모든 변형의 결과·분석·최종 결론을 **`reports/<실험>.md`**로 취합한다.
    README에는 성능 수치·진행 서사를 두지 않는다 — 리포트 목록은 docs/README.md에서 링크.
- **주차별 실험 노트 `docs/week_N.md`** (첫 커밋 2026-08-08 기준 7일 단위, week_1 = 08-08~08-14):
  날짜별 진행·결과·발견·결정을 그때그때 기록하고, **TODO 관리도 여기서** 한다
  (백로그 외 열린 항목 포함). 백로그 체크박스를 갱신할 때 해당 주 노트도 함께 갱신한다.
  노트 인덱스·로드맵은 docs/README.md.

## 작업 백로그 (순서 준수, 완료 시 체크 — 일일 진행·열린 항목은 docs/week_N.md)

- [x] **Task 0 — 스캐폴드**: 디렉토리(README §4), `.gitignore`(data/, runs/, .venv 등),
  `requirements.txt`(torch, numpy, pandas, pyarrow, matplotlib, scikit-learn, pytest, ruff, pyyaml, tqdm),
  ruff 설정, seed 유틸. DoD: `pytest -q` 통과(수집 0 허용), `ruff check .` 클린.
- [x] **Task 1 — TMM 모듈**: `src/physics/tmm.py` + 위 단위 테스트 6종. DoD: 전부 green.
  (7종 통과. 명세 6종은 층 적층 순서를 고정하지 못해 — 순서를 뒤집어도 전부 통과 —
  비대칭 2층을 재귀 프레넬 공식과 대조하는 7번 테스트를 보강했다.)
- [x] **Task 2 — 데이터 검증·로더**: `scripts/verify_data.py`(가설 체크 출력),
  `src/data/dataset.py`(parquet 캐시 권장). DoD: 검증 결과를 본 파일 체크박스와 README §2에 반영.
  (가설 중 "반사율 ⊂ [0,1]"만 거짓 — σ ≈ 0.0087 가산 노이즈 발견.
  Stage A 게이트는 이후 (a)+(c)로 재설정 완료 — Level 2 판정 게이트 참조.)
- [x] **Task 3 — EDA**: (a) 한 층만 변화시킨 스펙트럼 오버레이(두께↑ → fringe 조밀 확인),
  (b) 네 층을 각각 흔들었을 때 스펙트럼 민감도 비교(어느 층이 잘 보이는가),
  (c) 반사율 분포·범위. DoD: `reports/figures/` 그림 3종 + 관찰 메모.
  (`scripts/eda.py` → 그림 3종 + `reports/eda_metrics.md`(스크립트 산출) +
  `reports/eda_notes.md`(해석). 아래 "EDA 확정 수치" 참조.)
- [x] **Task 4 — Baseline 학습**: 90/10 val split(시드 고정), MAE 리포트. DoD: 재현 커맨드 README 반영.
  (확정 baseline: MLP 512×3, Linear→BatchNorm→GELU 블록, dropout 0, bare regression,
  입력 표준화 없음, batch 512, AdamW 1e-3/wd 1e-4, warmup 1000스텝+cosine.
  **holdout MAE 4.599 nm** — 리포트 `reports/mlp_baseline.md`.
  dropout 0.1은 6.645 nm로 순손실(전수 격자라 과적합 압력 약함). 2026-08-10 확정.)
- [x] **Task 5 — Level 1 ablation**: MLP vs 1D CNN, 단일 vs 다중 스케일, bound on/off 비교표.
  (전 축 완료 — `reports/level1_cnn.md`. 확정: flatten-dilated-bound **2.346 nm**
  (baseline −49%). bound는 격자 끝 오차 −60%대 절감, MLP 때와 반대 결론. 2026-08-11.)
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
python -m src.train --config configs/mlp_baseline/dropout0.0.yaml
```

## 세션 시작 체크리스트

1. 본 파일과 git log/status로 현재 상태 파악 → 백로그 최상단 미완료 Task 확인
2. 착수 전 계획을 2~4줄로 요약해 제시
3. 구현 → 테스트 → 커밋 → 체크박스·docs 주차 노트 갱신
