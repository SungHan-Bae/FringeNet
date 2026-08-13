# 실험 리포트: mlp_baseline — Task 4 baseline 확정

2026-08-10. 대실험 주제: **구조 bias 없는 MLP 대조군(baseline)의 확정**.
1D CNN(Level 1)·물리 손실(Level 2)의 기여를 잴 기준선을 세우는 것이 목적이다.
산출물: `runs/mlp_baseline/<변형>/` (model.pt, train.log, metrics.json).

## 공통 설정

- 모델: MLP 512×3 (0.65M 파라미터 = 646,660), 블록 = Linear → BatchNorm1d → GELU
  (residual off), head = Linear 4출력 **bare regression** (output_bound off)
- 입력: 반사율 226채널 원값 — 입력 표준화 없음 (첫 블록 BatchNorm이 대체)
- 학습: AdamW lr 1e-3 / weight_decay 1e-4, batch 512, 30 epochs, MAE 손실,
  linear warmup 1,000스텝 + cosine 감쇠 (스텝 단위), seed 42
- 데이터: train 810,000행 → 학습 729,000 + holdout 81,000 (10%, seed 42 고정 —
  프로젝트 공통 검증셋. 이후 모든 실험은 이 셋의 raw MAE로 비교한다. 격자 스냅 없음)
- best 체크포인트 선택도 같은 holdout의 val MAE로 한다 — 즉 보고 수치는 30에폭 중
  최소값이다. 81k 행에서 min-선택 편향은 ~0.01 nm 규모로 무시 가능하지만, holdout이
  모델 선택에도 쓰인다는 점은 명시해 둔다 (k-fold 모드는 fold별 OOF로 선택, holdout 미사용)

## sub-run 결과 (holdout 81,000행, raw MAE [nm])

| run | 변형 | overall | layer_1 | layer_2 | layer_3 | layer_4 | best epoch | 학습시간 |
|---|---|---|---|---|---|---|---|---|
| [`dropout0.0`](../runs/mlp_baseline/dropout0.0/) | dropout 0.0 | **4.599** | 3.562 | 5.390 | 4.782 | 4.662 | 27/30 | 13.8분 (CPU) |
| [`dropout0.1`](../runs/mlp_baseline/dropout0.1/) | dropout 0.1 | 6.645 | 4.813 | 8.206 | 7.154 | 6.408 | 27/30 | 21.0분 (CPU) |

수렴 궤적 (val MAE, 에폭 1 → 10 → 20 → 30):

- dropout 0.0: 26.41 → 7.90 → 5.27 → 4.62
- dropout 0.1: 29.20 → 9.59 → 7.33 → 6.76

## 분석

1. **dropout 0.1은 순손실** — holdout MAE 4.599 → 6.645 nm (+44% 악화; 제거 방향으로
   읽으면 −31%). 810k 전수 격자 데이터에서는 과적합 압력이 약해 dropout 0.1이 정규화
   이득 없이 수렴만 늦춘다. 두 run 모두 train_l1이 내리는 동안 val_mae가 되오르는
   괴리(과적합 신호)가 없었다는 것이 근거다 — 단 마지막까지 단조 하강한 것은 아니고,
   best(27에폭) 이후 3에폭은 LR→0 구간이라 개선 없이 소폭 요동한다(train.log 참조).
   대회 1등 수상자도 메인 모델 forward에서 dropout을 쓰지 않았다(출처: 분석 4).
2. **layer_2가 일관되게 최약** (양쪽 run에서 층별 순위 동일). EDA 층별 민감도
   SNR 최저(10.3, `reports/eda_metrics.md`)와 방향이 일치한다. 단 원리적
   사각지대는 아니므로(최소 SNR 10.3) 모델 개선으로 줄일 대상이다.
3. **에폭 연장은 별도 실험 대상**. best가 27에폭이고 이후 3에폭은 개선이 없었지만,
   이는 cosine 스케줄이 LR을 0으로 보낸 구간이라 "수렴 완료"의 근거가 못 된다.
   연장하려면 늘어난 total step에 맞춰 스케줄을 다시 잡아 재학습해야 한다
   (대회 1등은 100 epochs).
4. 참고 스케일: 대회 1등 단일 모델 val MAE ≈ 0.42 nm (약 213M 파라미터
   skip-connection MLP). 본 baseline은 0.65M — 이 격차를 재는 것이
   strong baseline(수상자 **원본 충실 재현**) 실험의 역할이다 — 재현 성공,
   holdout MAE 0.3955 nm ([strong_baseline.md](strong_baseline.md)).
   (출처: [\[1등\]\[Context_KKP\] Skipconnection MLP with Ensemble — 데이콘 코드 공유](https://dacon.io/competitions/official/235554/codeshare/651).
   skip-connection MLP·앙상블·단일 모델 val 0.42는 페이지 본문에서 확인.
   파라미터 수·epochs·dropout 미사용은 페이지 본문이 아니라 원문 첨부 코드 기준.)

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

부기: 이 실험은 실험 관리 2단 구조 도입 전에 실행돼 `runs/` 경로를 이관했다 —
`train.log`는 실행 당시 원문 그대로라 옛 run 이름이 보인다. 두 체크포인트를 현재 코드의
`src.evaluate`로 재평가해 holdout MAE 4.5990 / 6.6453 nm이 재현됨을 확인했다.
