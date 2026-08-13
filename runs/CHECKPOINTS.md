# 체크포인트 아카이브

산출: notebooks/checkpoint_archive.ipynb — **일회성 이관 도구라 이 파일을 만든 뒤 삭제했다**
(히스토리 `2a2ba56`에 있다). 앞으로의 체크포인트는 `train_gpu.py`의 `_mirror_copy`가 학습
중에 Drive로 보내므로 이관 작업 자체가 생기지 않는다.

`runs/`에는 텍스트 산출물(`train.log` · `metrics.json`)만 두고 학습 체크포인트는 Drive 미러에 보관한다. **미러에는 3종을 다 둔다** — run 하나가 자기완결적이어야 새 VM 에서 복원이 되고, 학습 중 `train_gpu.py`가 이미 그 형태로 쓰기 때문이다. 텍스트 2종은 **git이 정본**이고 미러는 사본이다.

미러 루트: `/content/drive/MyDrive/FringeNet/runs_mirror/<실험>/<run>/`

**예외 — `runs/stage_a/*/model.pt`는 git에 남긴다** (합계 44 KB): `scripts/diagnose_calibration.py`가 직접 로드해 `reports/stage_a_gate.md`를 재생성하므로 Drive로 보내면 문서의 재현 커맨드가 깨진다.

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
