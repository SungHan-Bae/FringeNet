# mlp_baseline_dropout0 — 확정 baseline (Task 4)

2026-08-10 실행. **이 run이 Task 4의 확정 baseline이다** — 현재 `configs/baseline.yaml`
기본값과 동일한 설정이다 (당시 run_name만 `mlp_baseline_dropout0`으로 달랐다.
현재 config를 재실행하면 `runs/mlp_baseline`에 새로 쓰인다).

## 설정

- 모델: MLP 512×3 (약 0.66M 파라미터), 블록 = Linear → BatchNorm1d → GELU
  (dropout 0, residual off), head = Linear 4출력 **bare regression** (output_bound off)
- 입력: 반사율 226채널 원값 — 입력 표준화 없음 (첫 블록 BatchNorm이 대체)
- 학습: AdamW lr 1e-3 / weight_decay 1e-4, batch 512, 30 epochs,
  linear warmup 1,000스텝 + cosine 감쇠 (스텝 단위), MAE 손실, seed 42
- 데이터: train 810,000행 → 학습 729,000 + holdout 81,000 (10%, seed 42 고정 —
  프로젝트 공통 검증셋. 이후 모든 실험은 이 셋의 raw MAE로 비교한다)

## 결과 (holdout 81,000행, raw 예측 — 격자 스냅 없음)

| overall | layer_1 | layer_2 | layer_3 | layer_4 |
|---|---|---|---|---|
| **4.599 nm** | 3.562 | 5.390 | 4.782 | 4.662 |

best epoch 27/30, 학습 13.8분 (CPU-only torch).

수렴 궤적 (val MAE): epoch 1 → 26.41, 10 → 7.90, 20 → 5.27, 30 → 4.62
(에폭별 전체 기록은 `history_model.csv`, `train.log`)

## Ablation — dropout 0.1 대비 ([../mlp_baseline_dropout01/report.md](../mlp_baseline_dropout01/report.md))

| 설정 (나머지 동일) | holdout MAE (nm) |
|---|---|
| dropout 0.0 (본 run) | **4.599** |
| dropout 0.1 | 6.645 |

810k 전수 격자 데이터에서는 과적합 압력이 약해 dropout이 순손실이다 (31% 차이).
train_l1과 val_mae가 마지막 에폭까지 동반 하강한 것이 과적합 부재를 뒷받침한다.
대회 1등 수상자도 메인 모델 forward에서 dropout을 쓰지 않았다.

## 관찰

- **layer_2가 최약(5.39 nm)** — EDA 층별 민감도 SNR 최저(10.3, `reports/eda_metrics.md`)와
  방향 일치. 단 원리적 사각지대는 아니므로(최소 SNR 10.3) 모델 개선으로 줄일 대상.
- best가 27에폭에서 나왔고 막판까지 개선 중 — **에폭 연장 시 추가 하락 여지** 있음
  (대회 1등은 100 epochs).
- 참고 스케일: 대회 1등 단일 모델 val MAE ≈ 0.42 nm (약 213M 파라미터
  skip-connection MLP). 본 baseline은 0.66M — 이 격차를 재는 것이 strong baseline
  (수상자 재현) 행의 역할이다.

## 재현

```bash
python -m src.train --config configs/baseline.yaml   # runs/mlp_baseline에 기록됨
python -m src.evaluate --run runs/mlp_baseline_dropout0   # 본 run 재평가
```

산출물: `model.pt`(best 체크포인트), `history_model.csv`, `train.log`, `metrics.json`,
`config.yaml`(실행 시점 설정 스냅샷). report.md 외에는 git 미추적.
