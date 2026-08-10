# mlp_baseline_dropout01 — dropout 0.1 ablation 변형

2026-08-10 실행. 확정 baseline([../mlp_baseline_dropout0/report.md](../mlp_baseline_dropout0/report.md))과
**dropout만 다른**(0.1 vs 0.0) 대조 run. 당시 기본값(dropout 0.1)으로 돌았고,
이 결과를 근거로 baseline 기본값을 dropout 0.0으로 확정했다.
(디렉토리명은 원래 `mlp_baseline`이었으나 이후 baseline 재실행과의 충돌을 피해 개명 —
내부 `config.yaml`/`metrics.json`의 run_name은 `mlp_baseline`으로 남아 있다.)

## 결과 (holdout 81,000행, raw 예측)

| overall | layer_1 | layer_2 | layer_3 | layer_4 |
|---|---|---|---|---|
| 6.645 nm | 4.813 | 8.206 | 7.154 | 6.408 |

best epoch 27/30, 학습 21.0분 (CPU). 설정은 dropout 0.1 외에 확정 baseline과 동일
(MLP 512×3, BatchNorm+GELU, bare regression, batch 512, warmup+cosine, seed 42).

## 결론

dropout 0.0 대비 전 층에서 열세 (overall 4.599 → 6.645, +44%). 층별 순위(layer_2 최약)는
두 run에서 동일. 810k 전수 격자 데이터에서는 dropout이 정규화 이득 없이 수렴만 늦춘다.
