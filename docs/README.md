# 실험 노트 (docs/)

주차별 실험 노트. **커밋 날짜 기준, 첫 커밋(2026-08-08)부터 7일 단위**로 나눈다.
그 주에 한 일·결과·발견·결정을 날짜별로 기록하고, **TODO 관리도 여기서** 한다.
실험별 취합 리포트는 [reports/](../reports/)에 있고 노트에서 링크한다.

| 노트 | 기간 | 요약 |
|---|---|---|
| [week_1.md](week_1.md) | 2026-08-08 ~ 08-14 | 스캐폴드 → TMM(테스트 7종) → 데이터 검증(**노이즈 σ ≈ 0.0087 발견**) → EDA → Stage A 게이트 확정 → **baseline 4.599 nm** → GPU(Colab) 전환 → CNN ablation 라운드 1·2 (**flatten-dilated 2.931 nm**) |

## 로드맵 (3주)

| 주차 | 계획 | 상태 |
|---|---|---|
| Week 1 (08-08~08-14) | 스캐폴드 · TMM 모듈+테스트 · 데이터 검증 · EDA · baseline 학습 | **완료** (08-10) |
| Week 2 (08-15~08-21) | 구조 ablation(MLP/CNN, 다중 스케일) · Stage A 캘리브레이션+게이트 판정 · Stage B 물리 손실 학습 | 예정 |
| Week 3 (08-22~08-28) | 신뢰도 지표 분석 · 결과·그림 정리 · 문서화 마감 | 예정 |

## 실험 리포트 목록 (reports/)

| 리포트 | 내용 |
|---|---|
| [mlp_baseline.md](../reports/mlp_baseline.md) | **Task 4 baseline 확정** — MLP 대조군 + dropout ablation, holdout MAE 4.599 nm |
| [level1_cnn.md](../reports/level1_cnn.md) | **Task 5 Level 1 구조 ablation** — CNN 5변형(셔플 대조군·flatten·dilated), **flatten-dilated 2.931 nm (−36%)**. 수용영역+위치 보존이 결합돼야 conv가 유효 |
| [strong_baseline.md](../reports/strong_baseline.md) | **1등 단일 모델 원본 충실 재현** — 213M skip-MLP, **0.3955 nm** (보고 0.42 재현 성공). 0.66M vs 213M 격차 확정 — Task 7 물리 손실의 상한 기준선 |
