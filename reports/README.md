# reports/ — 실험 리포트 인덱스

![헤드라인 — 1.5M CNN + 물리 보정이 213M 단독을 넘는다](figures/fig_headline.png)

## 진행 비교표

수치의 정본은 각 행의 리포트다 — 수치가 갱신되면 이 표와 `fig_headline.png`·
`fig_offgrid.png`(`scripts/make_headline_figure.py` 산출)를 함께 갱신한다. 전부 같은
holdout 81,000행, 같은 split(val_frac 0.1, seed 42), 결정론적 추론이다.

| 단계 | holdout MAE [nm] | 정본 |
|---|---|---|
| MLP baseline (0.65M) | 4.5990 | [mlp_baseline.md](mlp_baseline.md) |
| 1D CNN flatten-dilated-bound (0.66M) | 2.3455 | [level1_cnn.md](level1_cnn.md) |
| + 학습 예산 100에폭 | 1.7185 | [cnn_recipe.md](cnn_recipe.md) |
| + LM 역산 + 되돌림 규칙 (budget100 파이프라인) | 0.3880 | [cnn_recipe_judge.md](cnn_recipe_judge.md) |
| + 잔차 · 깊이 ×2 · rFFT 백본 (task8/d2-fft, 1.52M) | 0.3589 | [task8_judge.md](task8_judge.md) |
| **+ LM 역산 + 되돌림 규칙 (채택 파이프라인)** | **0.3396** | [task8_judge.md](task8_judge.md) |
| (상한 기준선) 213M skip-MLP 단독 | 0.3955 | [strong_baseline.md](strong_baseline.md) |

리더보드 제출(test, 격자 밖 연속 두께)의 기계가독 기록은
[leaderboard.json](leaderboard.json)이다. 채택 파이프라인은 **test MAE 0.33895
(2026-08-20, 4위)** — holdout 0.3396이 열화 없이 전이됐다(−0.2%). 같은 날 판정 목적으로
제출한 raw 신경망 셋(d2-se · d2-fft · 213M)은 전부 +20~85% 열화했다 — **물리 파이프라인만
격자 밖으로 전이된다**는 것이 Task 8의 최종 발견이다 ([task8.md](task8.md) 발견 ②).

![격자 밖 반전 — raw는 전부 열화, 물리 파이프라인만 전이](figures/fig_offgrid.png)

물리 손실(Stage B β)은 사전등록 ablation 세 축 전부에서 **기각**됐다 — 물리의 값어치는
손실이 아니라 추론에 있다는 것이 이 프로젝트의 핵심 결론이다
([stage_b.md](stage_b.md)).

## 읽기 순서 (핵심 서사 8편)

1. [eda_notes.md](eda_notes.md) — 데이터·노이즈 모델 (σ = 0.008658, 유계 |ε| ≤ 0.0152)
2. [level1_cnn.md](level1_cnn.md) — 구조 bias ablation: 어떤 conv가 왜 통하는가 (−49%)
3. [strong_baseline.md](strong_baseline.md) — 상한 기준선: 리더보드 1등 213M 재현
4. [stage_a.md](stage_a.md) — 물리 디코더 캘리브레이션: 자유도 7, 게이트 (b) 미통과 명시
5. [stage_b.md](stage_b.md) — 물리 **손실**의 기각: 사전등록 예측이 어긋난 기록까지
6. [inversion_refine.md](inversion_refine.md) — 반전: 같은 물리를 **추론 후 보정**으로 (−74%)
7. [cnn_recipe.md](cnn_recipe.md) — 학습 예산 + 되돌림 규칙 → 0.3880 nm, 213M 단독을 넘다
8. [task8.md](task8.md) — 구조·용량·모듈 최적화 → **0.3396 / test 0.33895 (4위)** +
   격자 밖 반전: 물리 보정의 값어치는 정밀도가 아니라 **분포 이동에 대한 강건성**

주의 한 가지: [inversion_refine.md](inversion_refine.md)는 30에폭 백본(2.3455 → 0.6110)
기준 **사전등록 2 판정의 정본**이라 재생성하지 않는다 — 확정 모델들의 post-LM·되돌림
수치는 [cnn_recipe_judge.md](cnn_recipe_judge.md)(budget100)와
[task8_judge.md](task8_judge.md)(task8 8-run)가 정본이다 (같은 함수, 다른 run).

![역산 refinement — CNN 출발 vs 격자 중앙 출발](figures/fig_inversion_refine.png)

## 파일 구분 — 취합(사람) vs 산출 정본(스크립트, 재실행 시 덮어씀)

| 취합 리포트 (판단·서사) | 산출 정본 | 생성 스크립트 |
|---|---|---|
| [eda_notes.md](eda_notes.md) | [eda_metrics.md](eda_metrics.md) | `scripts/eda.py` |
| [mlp_baseline.md](mlp_baseline.md) · [level1_cnn.md](level1_cnn.md) | [level1_cnn_diagnostics.md](level1_cnn_diagnostics.md) | `scripts/diagnose_predictions.py` |
| [strong_baseline.md](strong_baseline.md) | — | — |
| [stage_a.md](stage_a.md) | [stage_a_gate.md](stage_a_gate.md) | `scripts/diagnose_calibration.py` |
| [stage_b.md](stage_b.md) | [stage_b_curves.md](stage_b_curves.md) · [_heldout-thickness](stage_b_curves_heldout-thickness.md) · [_ft-heldout](stage_b_curves_ft-heldout.md) | `scripts/analyze_stage_b_curves.py` |
| [cnn_recipe.md](cnn_recipe.md) | [cnn_recipe_judge.md](cnn_recipe_judge.md) (+ `.json`) · [inversion_refine.md](inversion_refine.md) · [inversion_bench.md](inversion_bench.md) · [cnn_recipe_axes.md](cnn_recipe_axes.md) | `scripts/judge_recipe.py` · `refine_inversion.py` · `bench_invert.py` · `evaluate_axes.py` |
| [task8.md](task8.md) | [task8_judge.md](task8_judge.md) (+ `.json`) · [task8_bench.md](task8_bench.md) · [leaderboard.json](leaderboard.json)(손기록 — 외부 측정) | `scripts/judge_recipe.py` · `bench_invert.py` |

취합 리포트는 산출 정본의 수치를 복제하지 않는다 — 참조와 판단 근거만 쓴다 (기록 규약).
그림 원본은 [figures/](figures/)에 있고 각 리포트·본 인덱스가 임베드한다.
