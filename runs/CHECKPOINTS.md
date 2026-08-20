# 체크포인트 아카이브

산출: notebooks/checkpoint_archive.ipynb — **일회성 이관 도구라 이 파일을 만든 뒤 삭제했다**
(히스토리 `2a2ba56`에 있다). 앞으로의 체크포인트는 `train_gpu.py`의 `_mirror_copy`가 학습
중에 Drive로 보내므로 이관 작업 자체가 생기지 않는다.

`runs/`에는 텍스트 산출물(`train.log` · `metrics.json`)만 두고 학습 체크포인트는 Drive 미러에 보관한다. **미러에는 3종을 다 둔다** — run 하나가 자기완결적이어야 새 VM 에서 복원이 되고, 학습 중 `train_gpu.py`가 이미 그 형태로 쓰기 때문이다. 텍스트 2종은 **git이 정본**이고 미러는 사본이다.

미러 루트: `/content/drive/MyDrive/FringeNet/runs_mirror/<실험>/<run>/`

**예외 — `runs/stage_a/*/model.pt`는 git에 남긴다** (합계 약 172 KB): `scripts/diagnose_calibration.py`가 직접 로드해 `reports/stage_a_gate.md`를 재생성하므로 Drive로 보내면 문서의 재현 커맨드가 깨진다.

- 원본 커밋: `2a2ba5673279dae9aa9600036b07c432cde8440d`
- 복구: `git show <원본 커밋>:runs/<실험>/<run>/model.pt > model.pt` — 히스토리에 blob이 남아 있으므로 **Drive는 편의 사본이다**. 단 `drive-only` 항목은 커밋된 적이 없어 Drive가 유일본이다.
- 되돌리기: Drive 미러에서 `model.pt`를 그대로 복사하면 된다 (위 sha256으로 대조).
  텍스트 2종은 **git이 정본**이므로 미러 사본으로 덮어쓰지 말 것 — 과거 미러의 `train.log`가
  마지막 줄이 빠진 채 남아 있던 이력이 있다.

| 실험 | run | model.pt | sha256 (model.pt) | 상태 | 미러 3종 검증 |
|---|---|---|---|---|---|
| level1_cnn | `dilated` | 2.5 MB | `958d8847f7821867…` | uploaded | 3/3 OK |
| level1_cnn | `flatten` | 2.5 MB | `cca33b6838607985…` | uploaded | 3/3 OK |
| level1_cnn | `flatten-dilated` | 2.5 MB | `7608cd376e444710…` | uploaded | 3/3 OK |
| level1_cnn | `flatten-dilated-bound` | 2.5 MB | `3542afb008b538e5…` | uploaded | 3/3 OK |
| level1_cnn | `single-scale` | 2.5 MB | `e4c739d9c2693ed3…` | uploaded | 3/3 OK |
| level1_cnn | `single-scale-shuffled` | 2.5 MB | `47a8a2d6cd4e7060…` | uploaded | 3/3 OK |
| mlp_baseline | `dropout0.0` | 2.5 MB | `b6c3b88fa5d34611…` | uploaded | 3/3 OK |
| mlp_baseline | `dropout0.1` | 2.5 MB | `532cba4b6a75ab17…` | uploaded | 3/3 OK |
| strong_baseline | `winner-repro-asis` | 813.6 MB | `dbab5d6b1e51958c…` | drive-only | 3/3 OK |

model.pt 합계 833.7 MB / run 9개 (미러 파일 27개)

## Stage B (Task 7) — 라운드 3개, 12 run

학습 중 `train_gpu.py`의 `_mirror_copy`가 미러에 쓴 것이라 이관 작업이 없었다. 그래서 위 표와
달리 **sha256을 기록하지 않는다** — 대신 각 라운드 노트북의 마지막 검증 셀이 `flush_and_unmount`
→ 재마운트 후 **미러의 model.pt를 다시 로드해 holdout 재추론**으로 기록된 val MAE를 재현하는지
확인한다 (sha256 대조보다 강하다). 라운드 1·2는 노트북에 그 출력이 남아 있고, 라운드 3은 검증
통과가 런타임 자동 반납의 조건이며 반납이 실행됐다(출력은 로컬에 저장되지 않았다).

| 라운드 | run 4개 | split | 비고 |
|---|---|---|---|
| 1 | `beta{0,30,100,300}` | 무작위 (val_frac 0.1) | `beta0`은 `level1_cnn/flatten-dilated-bound`와 비트 동일 |
| 2 | `heldout-thickness-beta{0,30,100,300}` | 두께 70·150·230 제외 | `beta0`이 라운드 3의 warm start 출처 |
| 3 | `ft-heldout-beta{0,30,100,300}` | 라운드 2와 동일 | 라운드 2 대조군에서 40에폭 fine-tune |

- 미러 경로: `MyDrive/FringeNet/runs_mirror/stage_b/<run>/` (model.pt 각 2.5 MB, 합계 약 30 MB)
- **회수가 필요한 작업**: `scripts/evaluate_axes.py`(노이즈 강건성·신뢰도 지표)는 model.pt를
  읽는다. `scripts/analyze_stage_b_curves.py`는 커밋된 `train.log`만 쓰므로 회수가 필요 없다.
- 미러에서 가져올 때 **model.pt만** 복사한다 — `metrics.json`·`train.log`는 git이 정본이다.

## cnn_recipe (Task 7) — 라운드 1, 5 run

Stage B와 같이 `_mirror_copy`가 학습 중에 쓴 것이라 sha256을 기록하지 않는다 — 라운드 1 노트북의
Drive 무결성 검증 셀이 **미러의 model.pt를 다시 로드해 holdout 재추론**으로 기록된 val MAE를
재현했고(5/5 OK), 그 출력이 노트북에 남아 있다.

| run | 변인 (부모 대비) | val MAE [nm] |
|---|---|---|
| `budget100` | 에폭 30 → 100 | 1.7185 |
| `budget100-std` | 입력 채널별 표준화 | 1.7525 |
| `budget100-std-ema` | 가중치 EMA 0.999 | 1.7476 |
| `budget100-std-ema-noise` | 균등 ±0.015 주입 | 1.7073 |
| `budget100-std-ema-tail` | L1 + 0.1·MSE | 1.9189 |

- 미러 경로: `MyDrive/FringeNet/runs_mirror/cnn_recipe/<run>/` (model.pt 각 2.5 MB)
- **회수가 필요한 작업**: `scripts/judge_recipe.py`(post-LM·분지 실패율 판정) ·
  `scripts/refine_inversion.py` · `python -m src.evaluate --refine`(제출)이 model.pt를 읽는다.
  판정과 제출은 체크포인트 없이는 재현되지 않는다.
- `budget100`은 **git에 완료 기록이 있는데 model.pt는 없다.** `run_config`는 미러에만 완료 기록이
  있을 때 3종을 되가져오므로, git 기록이 있는 이 run은 자동 회수 경로에 걸리지 않는다 —
  미러에서 `model.pt`만 직접 복사한다 (텍스트 2종은 git이 정본이라 덮으면 안 된다).
- **회수 후 재추론으로 무결성을 확인한다**: `python -m src.evaluate --run runs/cnn_recipe/budget100`
  이 기록된 val MAE와 층별까지 재현해야 한다 (budget100은 1.7185 / 1.175·2.183·1.986·1.530).
  sha256을 기록하지 않은 run에는 이것이 대조 수단이다.

## task8 (Task 8) — 라운드 1~4, 7 run

`_mirror_copy`가 학습 중에 쓴 것이라 sha256을 기록하지 않는다 — 검증은 각 라운드 노트북의
Drive 무결성 셀(미러 model.pt 재로드 → holdout 재추론 = 기록 val 재현) 그리고 로컬 회수 후
전체 holdout 판정(`reports/task8_judge.md`)의 CNN MAE 재현으로 한다. **7 run 전부 재추론
재현 확인 완료** (2026-08-19~20). 예외 하나: 라운드 3은 세션이 push 셀 전에 중단돼(범위
축소 결정) 노트북 검증 셀이 돌지 않았다 — 텍스트 2종은 미러 사본을 별도 커밋(e3c59f5)했고,
미러 model.pt 검증은 로컬 판정 재현으로 갈음했다.

| 라운드 | run | 변인 | val MAE [nm] |
|---|---|---|---|
| 1 (구조) | `resnet-match` | 잔차 연결, 파라미터 매칭 (budget100 대비) | 1.6483 |
| 1 (구조) | `convnext-match` | ConvNeXt-1D 블록 교체 | 1.8108 |
| 2 (용량) | `resnet-d2` | 블록 5→10 (stride-1 복제, ×2.3) | 0.3647 |
| 2 (용량) | `resnet-w2` | 전 블록 채널 ×2 (×3.9) | 1.1515 |
| 3 (모듈) | `d2-fft` | rFFT 입력 분기 (+0.5%) | 0.3589 |
| 3 (모듈) | `d2-se` | SE 채널 어텐션 r=8 (+4.1%) | 0.2954 |
| 4 (결합) | `d2-se-fft` | d2-se + rFFT 분기 | 0.2960 |

- 미러 경로: `MyDrive/FringeNet/runs_mirror/task8/<run>/`
- **채택 모델은 `task8/d2-fft`다** (`reports/task8.md`) — 제출 재현에는 이 run의
  model.pt가 필요하다.
- `resnet-d4`는 에폭 6에서 중단·범위 제외 — 부분 산출물은 git에서 제거했고 미러의
  잔여 상태(resume.pt)도 삭제 대상이다 (남으면 d4 config 포함 세션에서 조용히 재개된다).
