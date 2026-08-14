# 역해 LM 추론 비용 — 장치 · dtype · 반복수

`scripts/bench_invert.py` 산출 — 재실행 시 덮어쓴다. 해석은 리포트 본문에서 한다.

- 표본 4,096행 (holdout 무작위) · 디코더 `runs/stage_a/joint-lam3-sin2-si2-schinke/model.pt`
- torch 2.13.0+cpu · CPU 스레드 8 · x86_64
- **skip-MLP는 시간만 잰다** — 가중치 값은 지연에 무관하므로 무작위 초기화다 (813 MB 체크포인트는 Drive 전용). MAE는 `reports/strong_baseline.md`가 정본이다.
- LM은 CNN 예측 `d_hat`에서 출발한다 (실제 워크로드와 같다).

## CPU

| 무엇 | 파라미터 | ms/행 | skip-MLP 대비 | holdout MAE [nm] |
|---|---|---|---|---|
| CNN forward | 0.66M | 0.317 | 0.13× | 2.4482 |
| skip-MLP forward | 213.21M | 2.384 | 1.00× | — |
| LM 30회 (complex128) | 7 | 14.905 | 6.25× | 0.6752 |
| LM 30회 (complex64) | 7 | 10.778 | 4.52× | 0.6503 |
| LM 10회 (complex64) | 7 | 3.292 | 1.38× | 0.6744 |

**cnn + LM 30회(complex128) 합계 = 15.222 ms/행** — skip-MLP의 6.38배

