# 실험 리포트: level1_cnn — Task 5 Level 1 구조 ablation (1D CNN)

2026-08-10~11. 대실험 주제: **"채널 순서 = 연속 스펙트럼" 구조 bias의 기여 측정** —
baseline MLP(구조 bias 없음, 4.599 nm) 대비 1D CNN이 무엇을 얻고 무엇을 잃는가.
산출물: `runs/level1_cnn/<변형>/` (model.pt, train.log, metrics.json).
라운드 3(bound on/off)까지 완료 — **Task 5 DoD 전 축 완료, 대실험 종결**.

## 변인 통제 (이 실험의 비교가 성립하는 근거)

- **학습 프로토콜 전부 동일**: baseline과 같은 seed 42 holdout 10%(81,000행 공통
  검증셋), AdamW lr 1e-3 / wd 1e-4, batch 512, 30 epochs, warmup 1,000스텝 + cosine,
  MAE 손실, bare regression + head bias 중앙(155 nm) 초기화, dropout 0.
- **파라미터 수 매칭**: gap/dilated 계열 646,340 (baseline 646,660 대비 **−0.05%**),
  flatten 계열 662,020 (**+2.4%**) — 전부 ±10% 이내 (테스트로 강제,
  `tests/test_models.py`). 용량이 아니라 연결 패턴이 조작 변인이다.
- **변형당 조작 변인 하나**: 기준(single-scale, GAP) 대비 shuffled는 채널 순서만,
  flatten은 head만, dilated는 수용영역만 바꾼다. dilation은 파라미터·연산량을
  바꾸지 않으므로 통제가 공짜다.
- **채널 셔플 대조군**: 고정 무작위 순열(seed 7)로 입력 순서만 파괴한 동일 CNN.
  MLP는 입력 순열에 불변(첫 Linear의 열 순서만 바뀜)이므로 이 대조군은 conv에만
  의미가 있다 — "순서 정보"의 기여를 직접 분리한다.
- 학습은 Colab GPU(`src/train_gpu.py` — CPU 파이프라인과 디커플, 산출물 계약 동일),
  분석·검증은 로컬 CPU. 라운드 1은 T4(에폭 ~67초), 라운드 2는 상위 GPU(에폭 ~12초).

## 결과 (holdout 81,000행, raw MAE [nm] — 격자 스냅 없음)

| run | 구조 (기준 대비 변경점) | overall | L1 | L2 | L3 | L4 | best ep | 학습시간 |
|---|---|---|---|---|---|---|---|---|
| [baseline](mlp_baseline.md) | MLP 512×3 (참조) | 4.599 | 3.562 | 5.390 | 4.782 | 4.662 | 27/30 | 13.8분 CPU |
| [`single-scale`](../runs/level1_cnn/single-scale/) | conv+GAP (기준) | 18.161 | 12.287 | 23.531 | 17.709 | 19.119 | 29/30 | 33.5분 |
| [`single-scale-shuffled`](../runs/level1_cnn/single-scale-shuffled/) | + 채널 순서 파괴 | 12.234 | 9.928 | 14.544 | 12.522 | 11.943 | 29/30 | 33.6분 |
| [`flatten`](../runs/level1_cnn/flatten/) | head: GAP→flatten | 13.677 | 9.782 | 17.200 | 13.209 | 14.518 | 30/30 | 6.0분 |
| [`dilated`](../runs/level1_cnn/dilated/) | dilations [1,2,4,4,2] (RF 97→259) | 4.976 | 3.594 | 5.955 | 5.092 | 5.263 | 28/30 | 6.0분 |
| [`flatten-dilated`](../runs/level1_cnn/flatten-dilated/) | flatten + dilated | 2.931 | 2.135 | 3.629 | 3.120 | 2.839 | 30/30 | 6.0분 |
| [`flatten-dilated-bound`](../runs/level1_cnn/flatten-dilated-bound/) | + output bound (sigmoid [10, 300]) | **2.346** | 1.594 | 2.961 | 2.730 | 2.096 | 29/30 | 6.0분 |

수렴 궤적 (val MAE, 에폭 1 → 10 → 20 → 30):

- single-scale: 41.97 → 23.16 → 19.04 → 18.17 / shuffled: 35.71 → 15.81 → 12.98 → 12.27
- flatten: 31.65 → 17.34 → 14.52 → 13.68 / dilated: 22.03 → 8.05 → 5.66 → 4.98
- flatten-dilated: 17.71 → 5.44 → 3.38 → **2.93**
- flatten-dilated-bound: 17.54 → 4.55 → 2.78 → **2.35** (best ep 29: 2.3455, ep 30: 2.3463)

검증: 7개 체크포인트 전부 로컬 CPU에서 holdout 재예측으로 기록 수치 재현 확인
(오차 < 1e-3 nm). 셔플 순열·head·dilations·bound 구성이 체크포인트 `model_cfg`에
저장된 대로임을 확인.

## 분석

### 1. 라운드 1 — 소박한 CNN(GAP)은 MLP보다 4배 나쁘고, 순서를 파괴하면 오히려 좋아진다

동일 용량·동일 프로토콜에서 conv+GAP는 18.16 nm로 baseline의 4배였다. 더 중요한
발견은 **채널 셔플 대조군(12.23)이 정상 입력(18.16)보다 33% 좋았다**는 것 —
"순서 정보를 활용해서 이득"은커녕, 순서 있는 입력에서 최적화가 더 안 됐다
(train_l1 17.6에서 정체 vs 셔플 11.6).

해석: 매끄러운 스펙트럼의 이웃 채널은 강하게 상관되어 국소 윈도우(k=7) 특징의
정보량이 빈약한데, GAP가 파장축 위치마저 평균으로 붕괴시킨다. 셔플된 입력의
윈도우는 원거리 채널들의 무작위 투영처럼 작동해 GAP 후에도 살아남는 정보가
더 많다. 진단이 이를 지지한다: gap 모델은 예측-정답 상관 0.915~0.970,
예측 std 76.7~82.0(타깃 83.7 — 평균 회귀), 얇은 두께(10~60 nm) 구간
MAE 28.6(baseline 6.5) — 무늬가 거의 없어 정보가 저주파·절대 레벨에 있는
영역에서 가장 크게 무너진다.

### 2. 라운드 2 — 병목의 분해: 수용영역이 지배 요인, 위치 보존이 그 위에 얹힌다

기준(gap 18.16) 대비 개선폭:

| 변형 | MAE | 개선폭 Δ |
|---|---|---|
| flatten (위치 보존만) | 13.68 | +4.5 |
| dilated (전 대역 수용영역만) | 4.98 | **+13.2** |
| flatten-dilated (둘 다) | **2.93** | +15.2 |

라운드 1의 진단("GAP 위치 불변성이 병목")은 절반만 맞았다. **지배 요인은
수용영역이다**: 기준 CNN의 RF는 97채널로 226채널 대역의 절반도 못 봐서, 어떤
국소 특징도 전체 간섭 패턴(4층 두께의 결합 정보)을 인코딩할 수 없었다.
dilation으로 RF를 259(전 대역)로 넓히자 GAP를 그대로 두고도 4.98 nm —
baseline에 근접했다. 그 위에 flatten(위치 보존)을 더하자 2.93 nm로 baseline을
36% 넘어섰다: 전 대역을 보는 특징이 "파장축 어디에" 있는지까지 살려야 시너지가
난다 (dilated 단독 대비 flatten의 한계 기여 +2.0, flatten 단독 대비 dilated의
한계 기여 +10.7 — 두 축은 상호 보완이되 비대칭).

### 3. flatten-dilated의 오차 구조 — 전 영역에서 baseline 우위 (라운드 2 시점)

- 예측-정답 상관 0.9973~0.9988 (baseline 0.9924~0.9968), 예측 std 86.3~86.7로
  평균 회귀 없음.
- 두께 구간별 MAE가 **모든 구간에서 baseline보다 낮다**. 특히 가장 어려운
  얇은 구간(10~60 nm): 4.37 vs baseline 6.49 — 라운드 1 CNN이 가장 크게
  무너지던 영역이 우위 영역으로 반전됐다.
- 격자 끝 편향도 작다 (d=10에서 4.91 / d=300에서 3.46 / 내부 2.84;
  baseline 6.97 / 5.99 / 4.46).
- layer_2가 최약인 순위는 모든 모델 공통 — EDA 층별 민감도 SNR 최저(10.3)와
  일치하는 물리적 패턴이지 특정 구조의 문제가 아니다.

### 4. 라운드 3 — output bound: 이득은 격자 끝에 집중된다 (MLP와 반대 결론)

flatten-dilated에 sigmoid bound(출력을 물리 범위 [10, 300] nm에 가둠, 무파라미터)
하나만 켠 비교: **2.931 → 2.346 nm (−20.0%)**. MLP baseline에서는 bare regression이
채택됐었는데(`99fe78e`), 백본이 강해지자 결론이 뒤집혔다.

오차 구조 분해 (동일 holdout, 로컬 재예측):

| 지표 | flatten-dilated | + bound |
|---|---|---|
| 범위 밖 예측 (<10 / >300 nm) | 1.82% / 1.53% | **0 / 0** (구조적으로 불가능) |
| 격자 끝 MAE (d=10 / d=300) | 4.91 / 3.46 | **1.82 / 1.39** (−63% / −60%) |
| 얇은 구간(10~60 nm) MAE | 4.37 | **3.14** |
| 내부(70~240 nm) MAE | 2.55 | 2.25 |
| 예측-정답 상관 (min~max) | 0.9973~0.9988 | 0.9984~0.9993 |

이득이 격자 끝에 집중된다: unbound 모델은 예측의 ~3.4%가 물리적으로 불가능한
범위 밖 값이었고 그 잔차가 끝 구간 MAE를 지배했는데, bound는 이를 구조적으로
차단한다. 우려했던 sigmoid 포화(끝 구간 gradient 소실)로 인한 열화는 관측되지
않았다 — 오히려 끝 구간이 가장 크게 좋아졌고, 내부 구간도 소폭 개선(2.55→2.25)
이라 순손실 구간이 없다. MLP와의 결론 역전에 대한 가설: 백본이 강할수록 남은
오차에서 격자 끝 범위 밖 초과분이 차지하는 비중이 커져 같은 제약의 한계 기여가
커진다. 단, MLP bound-on 산출물은 runs/에 보존되지 않아(`99fe78e`는 config 변경
커밋만 남음) 정량 대조는 불가 — 가설 수준으로만 기록한다.

### 5. 한계·주의

- **에폭 연장 여지**: flatten·flatten-dilated는 best가 30/30 — cosine 스케줄
  끝까지 하강 중이었다. 30 epochs는 baseline과의 통제 비교용이고, 절대 성능은
  연장 시 더 내려갈 여지가 있다 (백로그 "에폭 연장 실험"과 합류).
- lr 1e-3은 baseline(MLP) 기준으로 고른 값 — CNN 쪽에 미세조정하지 않은 채로도
  이겼으므로 결론에는 영향 없지만, 절대 수치는 lr 조정으로 더 좋아질 수 있다.
- flatten 계열은 파라미터 +2.4% — ±10% 통제 이내지만 0은 아니다. dilated(+0%)
  단독 비교가 이 우려를 상쇄한다: 파라미터 동일 조건에서 이미 +13.2 nm 개선.
- 학습은 GPU, 재현 검증은 CPU — bit 단위가 아니라 MAE 수준 비교다(전부 재현 확인).

## 결론

**Level 1 확정: flatten-dilated + output bound 1D CNN, holdout MAE 2.346 nm
(baseline 대비 −49.0%)** — `configs/level1_cnn/flatten-dilated-bound.yaml`.

구조 bias의 기여에 대한 답: "1D conv를 쓰면 좋다"가 아니라 —
**국소 conv 특징은 (1) 수용영역이 스펙트럼 전 대역을 덮고 (2) 파장축 위치가
보존될 때만 유효하며, 그 조건이 갖춰지면 동급 용량 MLP를 36% 능가한다.**
소박한 conv+GAP 이식은 오히려 4배 나쁘고, 그 사실은 채널 셔플 대조군이 정상
입력을 이기는 역전으로 가장 선명하게 드러났다. 그 위에 물리 범위 제약(output
bound)이 격자 끝 오차를 지워 추가 −20% — 구조 bias(연결 패턴)와 물리 bias(출력
범위)는 독립적으로 기여한다. Level 2(물리 손실)의 백본은 flatten-dilated-bound를
기본으로 한다.

## 재현

```bash
# Colab (권장): notebooks/level1_cnn/round1_gap-vs-shuffled.ipynb (기록 보존),
#               round2_flatten-dilated.ipynb, round3_bound.ipynb
# CLI (GPU 또는 CPU — 오래 걸림):
python -m src.train_gpu --config configs/level1_cnn/flatten-dilated-bound.yaml
python -m src.evaluate --run runs/level1_cnn/flatten-dilated-bound
```

부기: 라운드 2는 세션 유실 대비 체계(에폭 단위 resume.pt + Drive 미러 + 완료 run
스킵, `src/train_gpu.py`)가 처음 적용된 라운드다. 실제로 라운드 2 재실행 시 완료된
3개 run이 미러 기록으로부터 스킵·복원되어 push까지 이어졌다 (train.log에는 중단
흔적이 없다 — 세 run 모두 무중단 완주였고, 복구가 필요했던 것은 세션 종료 후의
산출물 회수였다).
