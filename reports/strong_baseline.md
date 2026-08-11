# strong_baseline — 대회 1등 단일 모델 원본 충실 재현

**결론: 재현 성공. holdout MAE 0.3955 nm — 1등 수상자가 보고한 단일 모델 val MAE ≈ 0.42와
같은 수준(소폭 우수).** 이로써 0.42는 신뢰 가능한 수치이며, 이 저장소의 상한 기준선으로
확정한다. 0.66M 모델(flatten-dilated-bound 2.346 nm) 대비 322배 파라미터로 −83% —
Task 7 물리 손실의 목표는 작은 모델이 이 격차를 얼마나 좁히는가로 자리매김된다.

## 목적

- 리더보드 1등 솔루션([\[1등\]\[Context_KKP\] Skipconnection MLP with Ensemble](https://dacon.io/competitions/official/235554/codeshare/651),
  코드 원문 [wikibook/dacon ch02](https://github.com/wikibook/dacon))의 **단일 모델**을
  원본 프로토콜 그대로 학습해 보고 수치가 재현되는지 확인한다.
- 축소 재현이 아니라 원본 재현을 택한 이유: 축소판은 격차가 스케일 탓인지 구현 실수
  탓인지 구분하지 못한다. 원본 수치가 재현돼야 상한 기준선으로 쓸 수 있다.
- 앙상블은 제외 — 0.42는 단일 모델 수치다. split 시드가 원본과 달라 bit 재현이 아닌
  **MAE 수준 재현**을 본다.

## 재현 충실도

원본 코드 대응표는 `configs/strong_baseline/winner-repro-asis.yaml` 헤더에 전문 기록.
요약: SkipConnectionModel(up 226→2000→4000→7000→10000, down →300, GELU tanh 근사
+BatchNorm, down 입구 LayerNorm, 덧셈 skip, bare regression, 입력 표준화 없음) —
파라미터 **213,208,104** (테스트로 고정, `src/models/winner_skip_mlp.py`). AdamW lr 1e-3
eps 1e-6 wd 0, L1, batch 2048, 100 epochs, warmup 2000스텝+cosine, 9:1 random split.

원본 코드의 **버그성 특이점 2건까지 그대로 재현**했다 (0.42가 나온 조건의 일부이므로):

1. 에폭 1 평가 후 `model.train()` 복귀 누락 → **에폭 2부터 eval 모드 학습**
   (BatchNorm 통계가 에폭 1 상태로 동결) — `eval_mode_after_first_epoch: true`
2. train DataLoader에 shuffle 없음(미리 섞은 CSV 순서 고정) — `shuffle: once`

## 결과 (runs/strong_baseline/winner-repro-asis, seed 42)

| run | 파라미터 | holdout MAE [nm] | L1 / L2 / L3 / L4 |
|---|---|---|---|
| mlp_baseline/dropout0.0 (Task 4) | 0.65M | 4.599 | 3.56 / 5.39 / 4.78 / 4.66 |
| level1_cnn/flatten-dilated-bound (Task 5) | 0.66M | 2.346 | 1.59 / 2.96 / 2.73 / 2.10 |
| **winner-repro-asis (이 실험)** | **213.2M** | **0.3955** | 0.328 / 0.554 / 0.372 / 0.328 |
| *(참조)* 1등 보고 단일 모델 val | 213.2M | ≈ 0.42 | — |

- best epoch **100/100** — cosine이 LR을 0으로 보내는 마지막 에폭까지 단조에 가깝게
  개선 (선택 편향 이슈 사실상 무관).
- 학습 곡선 마일스톤 (train.log): ep5 4.31 (**baseline 4.599를 5에폭에 추월**),
  ep11 2.30 (**CNN 2.346을 ~11에폭에 추월**), ep15 1.70, ep16~19 1.8~2.2로 출렁인 뒤
  ep30 1.21 → ep50 0.97 → ep70 0.59 → ep90 0.41 → **ep100 0.3955**.
- 비용: 에폭 ~120초(Colab GPU), 총 wall 14,788초 ≈ 4.1시간 (세션 유실 재계산 포함,
  metrics.json 기준). train.log에는 resume 이음새(초반 에폭 중복 기록)가 있다 —
  세션 5회에 걸친 학습의 정상 흔적.

## 분석

1. **재현 판정: 성공.** 0.3955 vs 보고 0.42 — split·시드·GPU가 다른 조건에서 MAE 수준
   일치(오히려 −6%). 아키텍처·프로토콜 포팅에 성능을 좌우하는 누락이 없다는 뜻이다.
2. **층별 패턴이 우리 모델들과 동일하다** — layer_2(SiO₂)가 최약(0.554, 전체 평균의
   1.4배). EDA 층별 민감도 SNR 최저(10.3)와 일치. 모델 용량을 328배 올려도 상대적
   서열은 그대로 → 층별 격차는 관측 물리(민감도)가 만드는 구조다.
3. **스케일 격차의 자리매김**: 0.66M → 213.2M (322×)로 MAE 2.346 → 0.3955 (−83%).
   파라미터당 효율은 작은 CNN이 압도적이지만, 이 데이터(전수 격자 81만 행, 노이즈
   σ≈0.0087)에서는 브루트포스 용량이 계속 이득을 본다. Task 7 물리 손실의 서사는
   "0.66M + 물리 사전지식이 213M 브루트포스와의 격차를 얼마나 좁히나"가 된다.
4. **버그까지 재현한 것의 함의**: 에폭 2부터 eval 모드(BN 동결)로도 0.3955가 나온다.
   BN이 사실상 "에폭 1 통계로 고정된 아핀 정규화"로 동작해도 학습에 지장이 없었다는
   것. 이 특이점을 고친 변형(정상 train 모드)은 라운드 2 후보로 남겨둔다 — 우선순위는
   낮다 (재현 목적은 달성, 마감 대비 비용 큼).

## 산출물·검증

- `runs/strong_baseline/winner-repro-asis/`: metrics.json, train.log (git 추적).
  **model.pt(813MB)는 GitHub 파일당 100MB 한도 초과로 git 미추적** — Drive 미러
  (`FringeNet/runs_mirror/strong_baseline/winner-repro-asis/`)가 보관처다
  (CLAUDE.md 실험 관리 구조의 예외 조항).
- **Drive 사본 무결성 검증 통과**: push 후 `flush_and_unmount` → 재마운트 → Drive의
  model.pt 재로드 → holdout 81,000행 재추론 **0.3955 = 기록 0.3955** (노트북 셀 8 출력).
- 세션 유실 대비 인프라가 실전 검증됨: 세션 6회에 걸쳐 학습, 재개 5회 (ep 2·15·50·
  70·95에서 재개 — ep 2만 5에폭 미러 격자 밖인데, 이것이 아래 미러 지연 건이다).
  이 실험이 드러낸 인프라 이슈 3건과 수정 — GPU resume 시
  CPU 계약 텐서가 cuda로 올라가는 버그(`a9519ed`), Drive FUSE 비동기 업로드로 대형
  resume.pt 미러 지연(`1033c2d`, mirror_resume_every), 100MB 한도·무결성 검증 규약
  (`2a599ae`) — 상세는 docs/week_1.md.

## 재현

```bash
# Colab (GPU): notebooks/strong_baseline/round1_winner-repro.ipynb Run-All
# 로컬 CPU 스모크:
python -m src.train_gpu --config configs/strong_baseline/winner-repro-asis.yaml \
  --device cpu --subset 1500 --epochs 1 --run-name winner-repro-asis-smoke --no-resume
```
