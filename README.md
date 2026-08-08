# FringeNet

**Physics-informed deep learning for thin-film thickness metrology**

반도체 다층 박막 두께 광학 계측 AI — 간섭무늬(fringe)에 인코딩된 두께를 읽는다.

반사율 스펙트럼(226개 파장 채널)으로부터 4층 박막(SiN/SiO₂/SiN/SiO₂ on Si)의 두께를 예측하는
역문제(inverse problem)를, 미분가능한 광학 물리 모델(Transfer Matrix Method, TMM)을
inductive bias로 결합한 딥러닝으로 푼다.

- 데이터: [월간 데이콘 — 반도체 박막 두께 분석 경진대회](https://dacon.io/competitions/official/235554/overview/description)
- 키워드: optical metrology, spectral reflectometry, inverse problem, physics-informed ML, differentiable TMM

---

## 1. 문제 배경

반도체 공정은 웨이퍼 위에 박막을 쌓고(증착), 깎고(식각), 패턴을 새기는 일의 수백 번 반복이다.
특히 3D NAND는 산화막(SiO₂)과 질화막(SiN)을 수십~수백 층 교대로 적층(ON stack)하는데,
층 두께의 오차와 불균일이 누적되면 이후 식각 및 셀 특성에 직접적인 수율 문제로 이어진다.
이 프로젝트의 SiN/SiO₂/SiN/SiO₂/Si(기판) 구조는 그 축소 모형에 해당한다.

양산 라인에서는 웨이퍼를 자르지 않고 두께를 재야 하므로 **비파괴 광학 계측**을 쓴다.
백색광을 쏘면 각 계면에서 부분 반사된 파들이 광경로차(≈ 2nd)에 따라 파장별로 보강·상쇄
간섭을 일으키고, 그 결과 반사율 스펙트럼 R(λ)에 두께 정보가 간섭무늬(fringe)로 인코딩된다.

두께 → 스펙트럼의 **순방향 계산은 TMM으로 정확하고 빠르다.** 병목은 역방향, 즉 측정된
스펙트럼에서 두께를 역산하는 과정이다. 전통적인 라이브러리 피팅은 층이 많아질수록
해의 다중성·국소최소·속도 문제를 겪는다. 본 프로젝트는 신경망으로 역매핑을 암묵적으로
학습(amortized inference)하되, 미분가능 TMM을 물리 디코더로 붙여 예측의 물리적 정합성을
강제하는 것이 핵심이다.

## 2. 데이터

| 파일 | 내용 |
|---|---|
| `train.csv` | `layer_1`~`layer_4` = 각 층 두께[nm] (타깃) + `0`~`225` = 파장 채널별 반사율 |
| `test.csv` | `0`~`225` 반사율만 제공 |
| `sample_submission.csv` | 제출 양식 (`layer_1`~`layer_4` 예측값) |

- 소자 구조: 공기 / SiN(layer_1) / SiO₂(layer_2) / SiN(layer_3) / SiO₂(layer_4) / Si 기판
- 평가지표: 층별 두께의 **MAE**
- **주의:** 파장 헤더 `0`~`225`는 비식별화된 인덱스로, 실제 파장[nm]이 아니다.
  다만 채널 순서는 연속 스펙트럼으로서 물리적 의미를 가진다.
- 학습셋은 각 층 두께 10–300 nm를 10 nm 간격으로 조합한 격자(30⁴ = 810,000행)로
  알려져 있으며, `scripts/verify_data.py`로 검증 후 본 문단을 확정한다.

**다운로드:** 대회 페이지에서 규정 동의 후 내려받아 `data/raw/`에 배치한다.
대회 규정상 데이터는 이 저장소에 포함하지 않는다(`.gitignore` 처리).

## 3. 방법

### 3.1 Level 1 — 구조·표현 수준 bias (representation-level)

파장축이 비식별화되어 실제 nm 값을 모르더라도, **컬럼 순서가 연속 스펙트럼이라는 구조**는 살아있다.
이 구조를 모델에 심는 것이 첫 번째 bias다.

- **1D CNN**: 226개를 순서 없는 독립 피처로 보는 MLP 대신, 인접 채널의 국소 패턴을 공유 커널로 읽는
  1D conv를 쓴다. "간섭무늬는 파장축을 따라 이어지는 국소 진동"이라는 물리 가정을 구조로 주입하는 것.
- **다중 스케일 수용영역**: fringe 주기는 두께에 따라 달라진다(두꺼운 층 → 조밀한 무늬).
  커널 크기를 섞거나 dilated conv를 써서 넓은 주기 대역을 함께 본다.
- **출력 bound**: `sigmoid`를 물리 범위 [10, 300] nm로 스케일링해, 물리적으로 불가능한 예측을
  구조적으로 배제한다.
- **ablation**: MLP vs 1D CNN, 단일 스케일 vs 다중 스케일, bound on/off로 각 요소의 기여를 분리 측정.

### 3.2 Level 2 — 미분가능 TMM 물리 디코더 (loss-level bias)

수직입사에서 층 j의 위상두께와 특성행렬(Abelès, Macleod 관례):

$$\delta_j=\frac{2\pi n_j d_j}{\lambda},\qquad
M_j=\begin{pmatrix}\cos\delta_j & i\sin\delta_j/n_j\\ i\,n_j\sin\delta_j & \cos\delta_j\end{pmatrix}$$

$$\binom{B}{C}=\Big(\prod_{j=1}^{4}M_j\Big)\binom{1}{n_s},\qquad
r=\frac{n_0B-C}{n_0B+C},\qquad R=|r|^2$$

인버스 네트워크가 예측한 두께 $\hat d$를 TMM에 통과시켜 스펙트럼을 재구성하고,
cycle-consistency 손실을 건다:

$$\mathcal{L}=\mathrm{MAE}(\hat d, d)+\beta\,\lVert R_{\mathrm{TMM}}(\hat d)-R_{\mathrm{obs}}\rVert_1$$

파장축 비식별화 문제는 2단계로 대응한다.

- **Stage A — 캘리브레이션(시스템 식별):** 학습셋의 (두께, 스펙트럼) 정답 쌍으로
  forward 모델의 미지수 — 단조 증가 파장 그리드 λ(i), 물질별 분산 곡선
  (SiN·SiO₂: Cauchy n(λ)=A+B/λ²+C/λ⁴, Si: 복소 굴절률 곡선) — 를 역으로 피팅한다.
  초기값은 refractiveindex.info의 문헌 분산을 사용한다.
  **게이지 축퇴 주의:** δ=2πnd/λ이므로 모든 n과 λ를 같은 배수로 스케일하면 결과가 불변이다.
  즉 n과 λ는 개별적으로 식별되지 않는다. 따라서 둘 중 하나를 고정해야 한다 —
  기본 전략은 **SiO₂의 분산 곡선을 문헌값으로 고정(freeze)하고 λ 그리드와 나머지 물질만 학습**하는 것.
  이 게이지를 걸지 않으면 최적화는 수렴해도 물리적으로 무의미한 n·λ 조합에 안착한다.
- **Stage B — 물리 손실 학습:** 캘리브레이션된 디코더를 동결하고 위 손실로 인버스넷을
  학습한다. β=0 대비 ablation으로 물리 손실의 기여를 정량화한다.
- **판정 게이트:** Stage A 재구성 RMSE < 5×10⁻³ (R 단위)이면 물리 디코더 채택.
  실패 시 데이터가 TMM과 다른 방식으로 생성됐다는 뜻이므로, 지도학습된
  d→R forward emulator(NN)를 동결 디코더로 쓰는 fallback으로 전환하고 그 사실을 기록한다.

### 3.3 물리 검증 (forward 모델 단위 테스트)

| 테스트 | 기대 결과 |
|---|---|
| 무층 극한 (d=0) | r = (n₀−n_s)/(n₀+n_s) — 맨 기판 프레넬 반사 회복 |
| 에너지 보존 (무흡수 스택) | R + T = 1, T = 4n₀Re(n_s)/\|n₀B+C\|² |
| λ/4 무반사 | n₁=√(n₀n_s), d=λ/4n₁ → R ≈ 0 |
| Airy 대조 (단층) | 해석해 r=(r₀₁+r₁₂e^{−2iδ})/(1+r₀₁r₁₂e^{−2iδ})와 일치 |
| 미분가능성 | dR/dd가 유한차분과 일치 (autograd 검증) |
| 흡수 기판 | 복소 n_s에서 R<1, NaN/Inf 없음 |

### 3.4 부산물 — 계측 신뢰도 지표

추론 시 TMM 재구성 오차가 큰 샘플은 모델이 확신하지 못하는 측정으로 플래깅할 수 있다.
이는 실제 fab의 계측 이상 감지(FDC) 관점과 맞닿아 있으며, 오차 분포 분석을 리포트에 포함한다.

### 3.5 평가 프로토콜 (정직성 규약)

학습 데이터가 두께 격자를 전수 조합한 **시뮬레이션 산출물**이라는 점은 평가를 왜곡할 수 있다.
포트폴리오로서 신뢰를 얻으려면 이 함정을 스스로 드러내고 통제하는 편이 낫다.

1. **격자 스냅은 분리 보고.** 정답이 10 nm 격자 위에 있으므로 예측을 최근접 격자로 반올림하면
   MAE가 인위적으로 떨어진다. 이는 계측 성능이 아니라 **데이터 생성 방식의 누설**이다.
   사용하더라도 raw MAE와 반드시 분리해 보고한다.
2. **격자 밖 일반화.** 실제 웨이퍼의 두께는 격자 위에 있지 않다. 캘리브레이션된 forward 모델로
   격자를 벗어난 연속 두께 샘플을 합성해 별도 평가한다.
3. **held-out 두께 값 split.** 전수 조합 데이터에서 무작위 split은 모든 두께 값이 학습에 등장하므로
   "조합 보간"만 측정한다. 특정 두께 값 자체를 통째로 빼는 split을 추가해 진짜 외삽 능력을 본다.
4. **노이즈 강건성.** 실측 스펙트럼에는 shot noise와 광원 드리프트가 있다. 입력에 노이즈를 주입한
   조건에서의 성능 열화를 함께 보고한다.

### 3.6 가정과 한계

- 수직입사·등방성·평행 평면층·표면 거칠기 없음을 가정한다. 실제 계측은 유한 개구각과 거칠기 보정을 포함한다.
- 데이터가 TMM으로 생성되었다는 보장은 없다. §3.2의 판정 게이트로 **검증한 뒤** 진행한다.
- 시뮬레이션 데이터라 노이즈가 거의 없다. 실측 환경 성능은 별도 논의가 필요하다.

## 4. 저장소 구조

```
.
├── README.md               # 본 문서
├── CLAUDE.md               # Claude Code 작업 메모리 (계약·테스트 스펙·백로그)
├── requirements.txt
├── configs/                # 실험 설정 (yaml)
├── data/raw/               # 데이콘 원본 (git 미포함)
├── notebooks/              # EDA (탐색용; 재사용 로직은 src/로 승격)
├── reports/figures/        # 산출 그림
├── scripts/verify_data.py  # 데이터 가설 검증
├── src/
│   ├── physics/tmm.py      # 미분가능 TMM — 프로젝트의 물리 코어
│   ├── data/dataset.py
│   ├── models/cnn1d.py
│   ├── calibrate.py        # Stage A
│   ├── train.py            # Stage B 포함
│   └── evaluate.py
└── tests/test_tmm.py       # 3.4의 물리 단위 테스트
```

## 5. 시작하기

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest -q                          # 물리 단위 테스트
python scripts/verify_data.py      # 데이터 가설 검증
python -m src.train --config configs/baseline.yaml
```

(명령은 구현 진행에 따라 갱신)

## 6. 로드맵 (3주)

- **Week 1** — [ ] 스캐폴드 · [ ] TMM 모듈+테스트 통과 · [ ] 데이터 검증 · [ ] EDA · [ ] baseline 학습
- **Week 2** — [ ] 구조 ablation(MLP/CNN, 다중 스케일) · [ ] Stage A 캘리브레이션 + 게이트 판정 · [ ] Stage B 물리 손실 학습
- **Week 3** — [ ] 신뢰도 지표 분석 · [ ] 결과·그림 정리 · [ ] 문서화 마감

## 7. 결과

*(실험 완료 후 갱신 — 아래는 양식)*

| 모델 | val MAE (nm) | 비고 |
|---|---|---|
| MLP baseline | TBD | 구조 bias 대조군 |
| 1D CNN (Level 1) | TBD | 다중 스케일 수용영역 |
| + TMM 물리 손실 (Level 2) | TBD | β ablation 포함 |

## 8. 참고자료

- H. A. Macleod, *Thin-Film Optical Filters* — 특성행렬 정식화
- M. Born & E. Wolf, *Principles of Optics* — 다층막 간섭 이론
- [refractiveindex.info](https://refractiveindex.info) — Si / SiO₂ / Si₃N₄ 분산 데이터 (캘리브레이션 초기값)
- [대회 데이터 설명](https://dacon.io/competitions/official/235554/data)

## 라이선스

코드는 MIT. 데이터는 포함하지 않으며 데이콘 대회 규정을 따른다.
