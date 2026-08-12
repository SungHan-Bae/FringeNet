# Stage A 게이트 진단 (`scripts/diagnose_calibration.py` 산출 — 손으로 고치지 말 것)

- run: `stage_a/sio2-freeze-adachi`, best step 3000, 진단 표본 20,000행 (피팅과 분리)
- 학습된 λ: 285.5–800.1 nm, 내림차순 / n(SiN) 2.020–2.185, n(SiO₂ freeze) 1.454–1.492, n(Si) 3.17–7.11, k(Si) 0.0172–5.0773

## 게이트 (a) — 재구성 RMSE

| 항목 | 값 | 기준 | 판정 |
|---|---|---|---|
| 진단 표본 RMSE | 0.01442 | < 0.0105 (= 1.2σ) | ✗ |

## 게이트 (c) — 잔차 백색성 (수치 진단; 최종 판정은 그림 육안 확인과 함께)

| 진단 | 값 | 판정 |
|---|---|---|
| |bias| < 0.001 | +0.00014 | ✓ |
| RMSE/σ ∈ [0.9, 1.2] | 1.657 | ✗ |
| 채널 RMSE max/min < 1.3 | 3.088 | ✗ |
| 두께 bin RMS max/min < 1.3 (전 층) | 1.156 / 1.033 / 1.106 / 1.067 | ✓ |
| |lag-1 자기상관| < 0.1 | +0.4604 | ✗ |
| RMSE/σ_hf < 1.15 | 1.620 (σ_hf 0.00890) | ✗ |

그림: `figures/fig_stage_a1_dispersion_adachi.png` (분산 곡선 육안 확인), `figures/fig_stage_a2_residuals_adachi.png` (백색성 4패널)
