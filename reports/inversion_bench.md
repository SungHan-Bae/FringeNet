# 역해 LM 추론 비용 — 장치 · dtype · 반복수

`scripts/bench_invert.py` 산출 — 재실행 시 덮어쓴다. 해석은 리포트 본문에서 한다.

- 표본 4,096행 (holdout 무작위) · 디코더 `runs/stage_a/joint-lam3-sin2-si2-schinke/model.pt`
- torch 2.11.0+cu128 · CPU 스레드 6 · x86_64
- **skip-MLP는 시간만 잰다** — 가중치 값은 지연에 무관하므로 무작위 초기화다 (813 MB 체크포인트는 Drive 전용). MAE는 `reports/strong_baseline.md`가 정본이다.
- LM은 CNN 예측 `d_hat`에서 출발한다 (실제 워크로드와 같다).

## CPU

| 무엇 | 파라미터 | ms/행 | skip-MLP 대비 | holdout MAE [nm] |
|---|---|---|---|---|
| CNN forward | 0.66M | 0.172 | 0.26× | 2.4482 |
| skip-MLP forward | 213.21M | 0.652 | 1.00× | — |
| LM 30회 (complex128) | 7 | 8.406 | 12.90× | 0.6752 |
| LM 30회 (complex64) | 7 | 5.829 | 8.94× | 0.6679 |
| LM 10회 (complex64) | 7 | 1.911 | 2.93× | 0.6874 |

**cnn + LM 30회(complex128) 합계 = 8.578 ms/행** — skip-MLP의 13.16배

## cuda — NVIDIA L4

| 무엇 | 파라미터 | ms/행 | skip-MLP 대비 | holdout MAE [nm] |
|---|---|---|---|---|
| CNN forward | 0.66M | 0.007 | 0.14× | 2.4481 |
| skip-MLP forward | 213.21M | 0.048 | 1.00× | — |
| LM 30회 (complex128) | 7 | 0.814 | 17.11× | 0.6630 |
| LM 30회 (complex64) | 7 | 0.238 | 5.01× | 0.6653 |
| LM 10회 (complex64) | 7 | 0.080 | 1.69× | 0.6962 |

**cnn + LM 30회(complex128) 합계 = 0.821 ms/행** — skip-MLP의 17.26배

