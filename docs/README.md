# 실험 노트 (docs/)

주차별 실험 노트 — 날짜별 진행·발견·결정의 연표이고 **TODO 관리도 여기서** 한다
(첫 커밋 2026-08-08부터 7일 단위). 수치의 근거는 [reports/](../reports/)에 있다.

| 노트 | 기간 | 요약 |
|---|---|---|
| [week_1.md](week_1.md) | 08-08 ~ 08-14 | 스캐폴드 → TMM(테스트 7종) → 데이터 검증(**가산 노이즈 발견**) → EDA → **baseline 4.599 nm** → GPU(Colab) 전환 → CNN ablation 3라운드(**2.346 nm**, −49%) → **1등 단일 모델 재현 0.3955 nm**(상한 기준선) → **Stage A 종결**(자유도 7 + Si 표 Schinke, RMSE 0.009573 = 1.106σ로 게이트 (a) 통과 / 게이트 (b) 미통과 9.99% — TMM 조건부 채택) |

## 로드맵 (3주)

| 주차 | 계획 | 상태 |
|---|---|---|
| Week 1 (08-08~14) | 스캐폴드 · TMM+테스트 · 데이터 검증 · EDA · baseline | **완료** (08-10) |
| Week 2 (08-15~21) | 구조 ablation · Stage A 캘리브레이션+게이트 · Stage B 물리 손실 | 구조 ablation **08-11**, Stage A **08-13** 종결 (Week 1로 당겨짐) · **Stage B 예정** |
| Week 3 (08-22~28) | 신뢰도 지표 분석 · 결과·그림 정리 · 문서화 마감 | 예정 |

## 실험 리포트 목록 (reports/)

| 리포트 | 내용 |
|---|---|
| [eda_notes.md](../reports/eda_notes.md) | **EDA 관찰·해석** — 노이즈 σ 유도(균등에 가까움), 층별 민감도 SNR(사각지대 없음), 채널 정보량 3배 |
| [eda_metrics.md](../reports/eda_metrics.md) | EDA 측정값 표 — `scripts/eda.py` 산출물 (재실행 시 덮어씀) |
| [mlp_baseline.md](../reports/mlp_baseline.md) | **Task 4 baseline 확정** — MLP 대조군 + dropout ablation, holdout MAE **4.599 nm** |
| [level1_cnn.md](../reports/level1_cnn.md) | **Task 5 Level 1 구조 ablation** — CNN 6변형(셔플 대조군·flatten·dilated·bound) → **2.346 nm (−49%)**. 수용영역과 위치 보존이 결합돼야 conv가 유효하고, bound가 격자 끝 오차를 지운다 |
| [strong_baseline.md](../reports/strong_baseline.md) | **1등 단일 모델 원본 재현** — 213M skip-MLP, **0.3955 nm**(보고 0.42 재현 성공). 0.66M vs 213M 격차가 Task 7의 상한 기준선 |
| [stage_a.md](../reports/stage_a.md) | **Task 6 Stage A 캘리브레이션** — 물리 제약 최소제곱, 자유 파라미터 **7개** + Si 표 Schinke 2015로 RMSE **0.009573 (1.106σ)**, 두께 역해 MAE **0.340 nm**, 연속블록 채널 홀드아웃 성립(+19.7%), λ 절대 스케일 검정 통과, 물성 전부 문헌 정합. 유계 노이즈 게이트 미통과(9.99%) — 잔차가 **대역 단파장 끝(Luke 유효범위 밖 외삽)** 에 몰리고 그 구간엔 문헌표 불일치가 없다(모델 부족) |
| [stage_a_gate.md](../reports/stage_a_gate.md) | Stage A 게이트 수치 표 — `scripts/diagnose_calibration.py` 산출물 (재실행 시 덮어씀). 게이트 (a)~(f) 종합 · 잔차 국소화 · 채널 홀드아웃 한계효과 · λ 절대 스케일 · λ 이동 · Si 문헌표 계통 |
