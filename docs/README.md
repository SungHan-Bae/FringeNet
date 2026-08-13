# 실험 노트 (docs/)

주차별 실험 노트. **커밋 날짜 기준, 첫 커밋(2026-08-08)부터 7일 단위**로 나눈다.
그 주에 한 일·결과·발견·결정을 날짜별로 기록하고, **TODO 관리도 여기서** 한다.
실험별 취합 리포트는 [reports/](../reports/)에 있고 노트에서 링크한다.

| 노트 | 기간 | 요약 |
|---|---|---|
| [week_1.md](week_1.md) | 2026-08-08 ~ 08-14 | 스캐폴드 → TMM(테스트 7종) → 데이터 검증(**노이즈 σ ≈ 0.0087 발견**) → EDA → Stage A 게이트 확정 → **baseline 4.599 nm** → GPU(Colab) 전환 → CNN ablation 3라운드 (**flatten-dilated-bound 2.346 nm**, −49%) → **1등 단일 모델 재현 0.3955 nm** (상한 기준선) → **Stage A 캘리브레이션** — 물리 제약 자유도 7 + Si 표 Schinke 2015, RMSE **0.009573 (1.106σ)**로 게이트 (a) 통과 · λ 절대 스케일 검정 통과 / **유계 노이즈 게이트 (b)는 미통과 (9.99%)**, TMM 조건부 채택 |

## 로드맵 (3주)

| 주차 | 계획 | 상태 |
|---|---|---|
| Week 1 (08-08~08-14) | 스캐폴드 · TMM 모듈+테스트 · 데이터 검증 · EDA · baseline 학습 | **완료** (08-10) |
| Week 2 (08-15~08-21) | 구조 ablation(MLP/CNN, 다중 스케일) · Stage A 캘리브레이션+게이트 판정 · Stage B 물리 손실 학습 | 구조 ablation은 **08-11**, Stage A는 **08-13 종결** (Week 1로 당겨짐) · Stage B 예정 |
| Week 3 (08-22~08-28) | 신뢰도 지표 분석 · 결과·그림 정리 · 문서화 마감 | 예정 |

## 실험 리포트 목록 (reports/)

| 리포트 | 내용 |
|---|---|
| [eda_notes.md](../reports/eda_notes.md) | **Task 3 EDA 관찰·해석** — 노이즈 σ(균등에 가까움) 유도, 층별 민감도 SNR(사각지대 없음), 채널 정보량 3배 |
| [eda_metrics.md](../reports/eda_metrics.md) | EDA 측정값 표 — `scripts/eda.py` 산출물 (재실행 시 덮어씀) |
| [mlp_baseline.md](../reports/mlp_baseline.md) | **Task 4 baseline 확정** — MLP 대조군 + dropout ablation, holdout MAE 4.599 nm |
| [level1_cnn.md](../reports/level1_cnn.md) | **Task 5 Level 1 구조 ablation** — CNN 6변형(셔플 대조군·flatten·dilated·bound), **flatten-dilated-bound 2.346 nm (−49%)**. 수용영역+위치 보존이 결합돼야 conv가 유효, bound가 격자 끝 오차를 지움 |
| [strong_baseline.md](../reports/strong_baseline.md) | **1등 단일 모델 원본 충실 재현** — 213M skip-MLP, **0.3955 nm** (보고 0.42 재현 성공). 0.66M vs 213M 격차 확정 — Task 7 물리 손실의 상한 기준선 |
| [stage_a.md](../reports/stage_a.md) | **Task 6 Stage A 캘리브레이션** — 물리 제약 LM 피팅, 자유 파라미터 **7개**(Si 표 = Schinke 2015)로 RMSE **0.009573 (1.106σ)**, 두께 역해 MAE **0.340 nm**, **연속블록 채널 홀드아웃 예측 성립**(한계 효과 +14.0%), **λ 절대 스케일 검정 통과**, 물성 전부 문헌 정합. 실측 원본표 + 에너지축 스플라인이 단일 최대 기여. 유계 노이즈 게이트는 미통과(9.99%) — 남은 오차가 c-Si 임계점 E1·E2에 집중 |
| [stage_a_gate.md](../reports/stage_a_gate.md) | Stage A 게이트 수치 표 — `scripts/diagnose_calibration.py` 산출물 (재실행 시 덮어씀) |
