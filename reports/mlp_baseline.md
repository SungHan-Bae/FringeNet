# 실험 리포트: mlp_baseline — Task 4 baseline 확정

2026-08-10. 대실험 주제: **구조 bias 없는 MLP 대조군(baseline)의 확정**.
1D CNN(Level 1)·물리 손실(Level 2)의 기여를 잴 기준선을 세우는 것이 목적이다.
산출물: `runs/mlp_baseline/<변형>/` (model.pt, train.log, metrics.json).

## 공통 설정

- 모델: MLP 512×3 (약 0.66M 파라미터), 블록 = Linear → BatchNorm1d → GELU
  (residual off), head = Linear 4출력 **bare regression** (output_bound off)
- 입력: 반사율 226채널 원값 — 입력 표준화 없음 (첫 블록 BatchNorm이 대체)
- 학습: AdamW lr 1e-3 / weight_decay 1e-4, batch 512, 30 epochs, MAE 손실,
  linear warmup 1,000스텝 + cosine 감쇠 (스텝 단위), seed 42
- 데이터: train 810,000행 → 학습 729,000 + holdout 81,000 (10%, seed 42 고정 —
  프로젝트 공통 검증셋. 이후 모든 실험은 이 셋의 raw MAE로 비교한다. 격자 스냅 없음)

## sub-run 결과 (holdout 81,000행, raw MAE [nm])

| run | 변형 | overall | layer_1 | layer_2 | layer_3 | layer_4 | best epoch | 학습시간 |
|---|---|---|---|---|---|---|---|---|
| [`dropout0.0`](../runs/mlp_baseline/dropout0.0/) | dropout 0.0 | **4.599** | 3.562 | 5.390 | 4.782 | 4.662 | 27/30 | 13.8분 (CPU) |
| [`dropout0.1`](../runs/mlp_baseline/dropout0.1/) | dropout 0.1 | 6.645 | 4.813 | 8.206 | 7.154 | 6.408 | 27/30 | 21.0분 (CPU) |

수렴 궤적 (val MAE, 에폭 1 → 10 → 20 → 30):

- dropout 0.0: 26.41 → 7.90 → 5.27 → 4.62
- dropout 0.1: 29.20 → 9.59 → 7.33 → 6.76

## 분석

1. **dropout은 순손실 (−31%)**. 810k 전수 격자 데이터에서는 과적합 압력이 약해
   dropout 0.1이 정규화 이득 없이 수렴만 늦춘다. 두 run 모두 train_l1과 val_mae가
   마지막 에폭까지 동반 하강한 것이 과적합 부재를 뒷받침한다. 대회 1등 수상자도
   메인 모델 forward에서 dropout을 쓰지 않았다.
2. **layer_2가 일관되게 최약** (양쪽 run에서 층별 순위 동일). EDA 층별 민감도
   SNR 최저(10.3, `reports/eda_metrics.md`)와 방향이 일치한다. 단 원리적
   사각지대는 아니므로(최소 SNR 10.3) 모델 개선으로 줄일 대상이다.
3. **에폭 연장 여지**. best가 27에폭에서 나왔고 막판까지 개선 중이었다
   (대회 1등은 100 epochs).
4. 참고 스케일: 대회 1등 단일 모델 val MAE ≈ 0.42 nm (약 213M 파라미터
   skip-connection MLP). 본 baseline은 0.66M — 이 격차를 재는 것이
   strong baseline(수상자 축소 재현) 실험의 역할이다.

## 결론

**baseline 확정: dropout 0.0 구성 (holdout MAE 4.599 nm)** —
`configs/mlp_baseline/dropout0.0.yaml`. 이후 Task 5(1D CNN ablation)·Level 2
(물리 손실)의 기준선은 이 수치다.

## 재현

```bash
python -m src.train --config configs/mlp_baseline/dropout0.0.yaml
python -m src.train --config configs/mlp_baseline/dropout0.1.yaml
python -m src.evaluate --run runs/mlp_baseline/dropout0.0
```

부기: 이 실험은 원래 `runs/mlp_baseline_dropout0{,1}` 평면 구조로 실행됐고,
2026-08-10 리포트 체계 개편 때 현 구조로 이관했다 (metrics.json의
experiment/run_name 필드도 새 이름으로 갱신). 결과 수치는 실행 당시 그대로다.
