"""GPU(CUDA) 학습 엔트리포인트 — Colab 등 GPU 런타임용. CPU 파이프라인과 디커플.

src/train.py는 baseline(Task 4)을 만든 CPU 검증 경로 그대로 보존하고 수정하지 않는다.
이 모듈은 같은 학습 프로토콜·같은 산출물 계약을 따르는 별도 GPU 경로다:

- 산출물: runs/<experiment>/<run_name>/{model.pt, train.log, metrics.json} — CPU와 동일.
  metrics.json에 "device" 필드가 추가된다는 것만 다르다.
- 데이터 전체를 GPU에 상주시킨다 (train x 810k×226 float32 ≈ 0.69 GB — T4 16 GB에 여유).
- 체크포인트 state_dict는 **CPU 텐서로 변환해 저장**한다 — 로컬(CPU 전용)에서
  evaluate.py / load_model_checkpoint로 바로 분석할 수 있게.
- holdout 단일 모드만 지원한다. k-fold가 필요해지면 그때 추가한다.

세션 유실 대비 (Colab 런타임이 언제든 끊길 수 있다는 전제):
- **best 체크포인트(model.pt)는 val MAE가 갱신되는 즉시 저장**한다 — 학습 종료를
  기다리지 않으므로 어느 시점에 죽어도 best-so-far 모델이 남는다.
- **매 에폭 resume.pt** 저장: 모델·옵티마이저·스케줄러·best 상태·RNG 상태 전부.
  재실행 시 자동 감지해 다음 에폭부터 재개하며, RNG까지 복원하므로 중단 없이
  돌린 실행과 (같은 기종·cudnn deterministic 하에) 동일한 결과를 낸다.
- **mirror_dir**(예: Google Drive)를 주면 train.log를 매 에폭, resume.pt를
  mirror_resume_every 에폭마다(기본 5), model.pt를 best 갱신 에폭마다 미러에 복사한다.
  새 VM에서 재실행하면 미러에서 상태를 복원해 이어 달린다. 저장·복사는
  원자적(임시파일→교체)이다. 대형 모델은 resume.pt가 수 GB라 Drive 비동기 업로드가
  에폭 속도를 못 따라갈 수 있다 — 그때 mirror_resume_every를 올려 업로드량을 줄인다
  (로컬 resume.pt는 항상 매 에폭 저장이라 같은 VM 재개는 무손실).
- 완료된 run(metrics.json에 결과 존재)은 run_config가 통째로 건너뛴다 —
  여러 실험을 순차 실행하다 끊겨도 재실행하면 끝난 것은 스킵, 하던 것은 재개.
- 재현성: 같은 시드·같은 GPU 기종에서는 cudnn deterministic(set_seed)으로 재현된다.
  단 CPU와 GPU는 부동소수 연산 순서가 달라 bit 단위로 같지 않다 — CPU baseline과의
  비교는 MAE 수준에서 한다.

사용 (Colab 노트북 notebooks/<대실험>/roundN_*.ipynb가 이 함수를 호출한다):
    from src.train_gpu import run_config
    metrics = run_config("configs/level1_cnn/flatten.yaml",
                         mirror_dir="/content/drive/MyDrive/FringeNet/runs_mirror")

    # CLI로도 동작한다
    python -m src.train_gpu --config configs/level1_cnn/flatten.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn.functional import l1_loss

from src.data.dataset import REPO_ROOT, prepare_from_config
from src.evaluate import format_mae, mae_per_layer
from src.losses import build_physics_loss
from src.models import build_model
from src.train import RUNS_DIR, build_lr_scheduler, log_line
from src.utils.io import atomic_save
from src.utils.seed import set_seed

RESUME_NAME = "resume.pt"


def resolve_device(device: str | None = None) -> torch.device:
    """device 문자열을 torch.device로 푼다. None이면 cuda 우선 자동 선택."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda를 요청했지만 사용할 수 없다 — 런타임에 GPU가 붙어 있는지 확인")
    return resolved


@torch.no_grad()
def predict_on_device(model: torch.nn.Module, x: Tensor, batch_size: int = 8192) -> np.ndarray:
    """디바이스 상주 텐서 x (N, 226)를 배치 추론해 (N, 4) float32 numpy로 돌려준다."""
    model.eval()
    outs: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        outs.append(model(x[start : start + batch_size]).cpu().numpy())
    return np.concatenate(outs, axis=0)


def _mirror_copy(run_dir: Path, mirror_dir: Path | None, names: tuple[str, ...]) -> None:
    """run_dir의 파일들을 미러 디렉토리로 원자적으로 복사한다 (있는 것만)."""
    if mirror_dir is None:
        return
    mirror_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = run_dir / name
        if not src.exists():
            continue
        tmp = mirror_dir / (name + ".tmp")
        shutil.copy2(src, tmp)
        tmp.replace(mirror_dir / name)


def _fingerprint(cfg: dict[str, Any], seed: int, n_train: int) -> str:
    """resume 호환성 판별용 설정 지문 — 다른 설정의 resume.pt를 이어받지 않도록.

    data 블록도 넣는다: 분할 옵션(holdout_thickness 등)이 바뀌면 학습 집합의 **내용**이
    달라지는데 크기(n_train)는 같을 수 있어(값만 바꾼 경우) 크기만으로는 못 거른다.
    """
    return json.dumps(
        {
            "model": cfg["model"],
            "train": cfg["train"],
            "data": cfg.get("data"),
            "seed": seed,
            "n_train": n_train,
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def _init_from_checkpoint(model: nn.Module, cfg: dict[str, Any], device: torch.device) -> str:
    """`train.init_from` 체크포인트의 가중치를 model에 적재한다 (warm start).

    수렴된 모델에서 출발해 물리 항을 켜는 실험용이다 — 랜덤 초기화에서 물리 gradient에
    끌려가면 잘못된 fringe 차수 분지에 안착할 수 있으므로, 올바른 분지에 이미 들어간
    지점에서 물리 항을 시험한다.

    **분할이 같아야 한다.** 다른 split에서 학습된 체크포인트로 warm start하면 그 모델이
    이미 본 행이 이번 run의 holdout에 들어가 누수가 된다. 출처 run의 `metrics.json`이
    옆에 있으면 `data` 블록을 대조해 막고, 없으면(미러에서 model.pt만 받은 경우) 대조가
    불가능하다는 사실을 로그에 남긴다.

    Returns:
        로그에 남길 한 줄 (출처·성능·분할 대조 결과).

    Raises:
        FileNotFoundError: 체크포인트가 없는 경우.
        ValueError: 모델 구조 또는 분할이 다른 경우.
    """
    path = Path(cfg["train"]["init_from"])
    if not path.exists():
        raise FileNotFoundError(
            f"init_from 체크포인트가 없다: {path}\n"
            "  runs/는 텍스트 산출물만 git 추적한다 — Drive 미러에서 복사할 것"
            " (runs/CHECKPOINTS.md)"
        )
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if dict(ckpt["model_cfg"]) != dict(cfg["model"]):
        raise ValueError(
            f"init_from의 model 블록이 이번 config와 다르다: {path}\n"
            "  구조가 다르면 warm start가 성립하지 않는다 (state_dict 키·shape 불일치)"
        )
    model.load_state_dict(ckpt["state_dict"])

    split_note = "분할 대조 불가 (출처 metrics.json 없음 — 같은 split인지 직접 확인할 것)"
    metrics_path = path.parent / "metrics.json"
    if metrics_path.exists():
        src_data = json.loads(metrics_path.read_text()).get("config", {}).get("data") or {}
        this_data = cfg.get("data") or {}
        if src_data != this_data:
            raise ValueError(
                f"init_from 출처의 분할이 이번 run과 다르다 — **누수**다: {path}\n"
                f"  출처 data: {src_data}\n  이번 data: {this_data}\n"
                "  출처 모델이 이미 본 행이 이번 holdout에 들어간다"
            )
        split_note = "분할 일치 확인"
    return (
        f"warm start: {path} (출처 val MAE {ckpt['val_mae']:.4f} nm, "
        f"best epoch {ckpt['best_epoch']}) / {split_note}"
    )


def train_one_model_gpu(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
    run_dir: Path,
    tag: str,
    device: torch.device,
    mirror_dir: Path | None = None,
    resume: bool = True,
    mirror_resume_every: int = 5,
    _abort_after_epoch: int | None = None,
) -> dict[str, Any]:
    """모델 하나를 device에서 학습하고 best(val MAE) 체크포인트를 저장한다.

    src/train.py의 train_one_model과 같은 입출력 계약(반환 키·체크포인트 포맷·로그
    형식)을 따른다. 차이: (1) 데이터·모델이 device에 상주, (2) 체크포인트는 CPU 텐서,
    (3) best 갱신 즉시 {tag}.pt 저장, (4) 매 에폭 resume.pt 저장(+미러)로 세션 유실 대비,
    (5) cfg["train"]["physics"] 블록이 있으면 Stage B 물리 손실을 더한다 (src/losses.py —
    CPU 경로에는 없다). 블록이 없으면 손실 계산 경로가 물리 항 도입 전과 동일하다.

    Args:
        x_train: (N, 226) float32 반사율. y_train: (N, 4) float32 두께 [nm].
        x_val: (M, 226) float32 holdout 반사율. y_val: (M, 4) float32 두께 [nm].
        mirror_dir: 지정하면 train.log를 매 에폭, resume.pt를 mirror_resume_every
            에폭마다, {tag}.pt를 best 갱신 에폭마다 이 디렉토리에 복사한다. 시작 시
            로컬에 resume.pt가 없고 미러에 있으면 미러에서 복원해 이어 달린다.
        resume: False면 resume.pt를 무시하고 처음부터 학습한다.
        mirror_resume_every: resume.pt 미러 복사 간격(에폭). 로컬 저장은 항상 매 에폭
            이라 같은 VM 재개는 무손실이고, 이 값은 새 VM 복구의 최대 재계산 폭만
            정한다. 대형 모델(resume.pt 수 GB)에서 Drive 비동기 업로드가 에폭 생산
            속도를 못 따라가 밀리는 것을 완화하는 용도 — config가 아닌 함수 인자라
            fingerprint에 영향이 없어 진행 중 run에도 적용할 수 있다.
        _abort_after_epoch: 테스트 전용 — 해당 에폭의 저장·미러까지 끝낸 뒤 일부러
            RuntimeError를 던져 세션 중단을 흉내 낸다.

    Raises:
        ValueError: resume.pt의 설정 지문이 현재 설정과 다른 경우.

    Returns:
        {"tag", "seed", "ckpt_path", "best_epoch", "val_mae", "val_mae_per_layer",
         "val_pred" (M, 4) np.ndarray, "wall_sec"}. 물리 손실을 쓴 run은 "physics"
        (설정·디코더 출처 스냅샷)와 "val_phys_l1"(best 에폭의 holdout 재구성 L1)이 붙는다.
    """
    train_cfg = cfg["train"]
    epochs = int(train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])
    if epochs < 1:
        raise ValueError(f"epochs는 1 이상이어야 한다 (받은 값: {epochs})")
    if mirror_resume_every < 1:
        raise ValueError(
            f"mirror_resume_every는 1 이상이어야 한다 (받은 값: {mirror_resume_every})"
        )

    # 1등 솔루션 충실 재현용 프로토콜 플래그 (configs/strong_baseline/ 참조) —
    # 기본값은 기존 실험과 동일한 동작이라 이전 config·fingerprint에 영향이 없다.
    #   shuffle: "epoch" = 매 에폭 재셔플(기존 동작) | "once" = 시드 고정 순열 하나를
    #     전 에폭 재사용 (원본이 미리 섞어둔 CSV를 shuffle 없는 DataLoader로 돌린 것과 등가).
    #   eval_mode_after_first_epoch: True면 에폭 2부터 eval 모드로 학습한다 — 원본
    #     train.py가 평가 후 model.train() 복귀를 빠뜨려 BatchNorm 통계가 에폭 1 이후
    #     동결된 채 0.42가 나왔으므로, 재현에서는 이 동작까지 그대로 따른다.
    shuffle_mode = str(train_cfg.get("shuffle", "epoch"))
    if shuffle_mode not in ("epoch", "once"):
        raise ValueError(f'shuffle은 "epoch" | "once" 여야 한다 (받은 값: {shuffle_mode!r})')
    eval_after_first = bool(train_cfg.get("eval_mode_after_first_epoch", False))

    set_seed(seed)

    x_t = torch.from_numpy(x_train).to(device)
    y_t = torch.from_numpy(y_train).to(device)
    x_v = torch.from_numpy(x_val).to(device)
    model = build_model(cfg["model"]).to(device)
    if train_cfg.get("init_from"):
        note = _init_from_checkpoint(model, cfg, device)
        log_line(run_dir, f"[{tag}] {note}")
    # 물리 손실은 설정을 먼저 검증해 잘못된 config가 학습 시작 전에 걸리게 한다.
    # RNG를 소모하지 않으므로 beta=0 대조군의 학습 경로는 물리 항 도입 전과 같다.
    physics = build_physics_loss(train_cfg, device=device)
    if physics is not None:
        prov = physics.decoder.provenance
        log_line(
            run_dir,
            f"[{tag}] 물리 손실 beta {physics.beta:g} (워밍업 {physics.warmup_steps} 스텝)"
            f" / 동결 디코더 {prov['decoder']} (자유도 {len(prov['free'])},"
            f" Stage A RMSE {prov['stage_a_rmse']:.6f}, 위반율"
            f" {prov['stage_a_violation_rate']:.4%})",
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        eps=float(train_cfg.get("adam_eps", 1e-8)),
    )

    n = len(x_t)
    steps_per_epoch = math.ceil(n / batch_size)
    scheduler = build_lr_scheduler(
        optimizer,
        schedule=str(train_cfg.get("lr_schedule", "cosine")),
        warmup_steps=int(train_cfg.get("warmup_steps", 0)),
        total_steps=steps_per_epoch * epochs,
    )

    fingerprint = _fingerprint(cfg, seed, len(x_train))
    ckpt_path = run_dir / f"{tag}.pt"
    resume_path = run_dir / RESUME_NAME

    def save_best_checkpoint(
        best_state: dict[str, Tensor], best_epoch: int, best_mae: float
    ) -> None:
        atomic_save(
            {
                "model_cfg": dict(cfg["model"]),
                "state_dict": best_state,
                "seed": seed,
                "tag": tag,
                "best_epoch": best_epoch,
                "val_mae": best_mae,
            },
            ckpt_path,
        )

    best_mae = float("inf")
    best_state: dict[str, Tensor] | None = None
    best_epoch = -1
    best_metrics: dict[str, float] = {}
    best_pred: np.ndarray | None = None
    best_val_phys: float | None = None
    start_epoch = 1
    wall_prev = 0.0

    if resume:
        if not resume_path.exists() and mirror_dir is not None:
            # 새 VM: 미러에 남은 상태에서 복원
            restored: list[str] = []
            for name in (RESUME_NAME, "train.log", f"{tag}.pt"):
                src = mirror_dir / name
                if src.exists():
                    shutil.copy2(src, run_dir / name)
                    restored.append(name)
            if restored:
                log_line(run_dir, f"[{tag}] 미러에서 복원: {restored}")
        if resume_path.exists():
            try:
                # resume.pt는 이 모듈이 만든 자기 산출물 — RNG 상태 등 비텐서 객체 포함
                state = torch.load(resume_path, map_location=device, weights_only=False)
            except Exception as err:  # 저장 도중 죽어 깨진 파일 등
                log_line(run_dir, f"[{tag}] resume.pt 로드 실패({err!r}) — 처음부터 학습")
                state = None
            if state is not None:
                if state["fingerprint"] != fingerprint:
                    raise ValueError(
                        "resume.pt의 설정이 현재 config와 다르다 — "
                        "run_name을 바꾸거나 resume.pt를 지우고 다시 실행할 것"
                    )
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                if scheduler is not None and state["scheduler"] is not None:
                    scheduler.load_state_dict(state["scheduler"])
                best_mae = state["best_mae"]
                # map_location=device가 저장 시 CPU였던 텐서도 device로 올리므로,
                # CPU 계약인 것들(best_state = 체크포인트용, best_pred = numpy)은 되돌린다
                best_state = state["best_state"]
                if best_state is not None:
                    best_state = {k: v.detach().to("cpu", copy=True) for k, v in best_state.items()}
                best_epoch = state["best_epoch"]
                best_metrics = state["best_metrics"]
                best_pred = (
                    state["best_pred"].cpu().numpy() if state["best_pred"] is not None else None
                )
                best_val_phys = state.get("best_val_phys")
                torch.set_rng_state(state["torch_rng"].cpu())
                if device.type == "cuda" and state.get("cuda_rng") is not None:
                    torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
                np.random.set_state(state["numpy_rng"])  # noqa: NPY002 (seed.py와 동일 이유)
                random.setstate(state["py_rng"])
                start_epoch = state["epoch"] + 1
                wall_prev = state["wall_sec"]
                log_line(
                    run_dir,
                    f"[{tag}] resume: epoch {state['epoch']}까지 완료 상태에서 재개"
                    f" (best {best_mae:.4f} @ ep {best_epoch})",
                )

    t_start = time.perf_counter()

    for epoch in range(start_epoch, epochs + 1):
        # eval_mode_after_first_epoch=True면 에폭 2부터 eval 모드로 학습 (위 플래그 주석).
        # epoch 번호만으로 결정되므로 resume 후에도 무중단 실행과 같은 모드가 된다.
        model.train(epoch == 1 or not eval_after_first)
        t_epoch = time.perf_counter()
        if shuffle_mode == "once":
            # 시드 고정 전용 generator — 전역 RNG를 소모하지 않아 매 에폭·resume 후에도
            # 항상 같은 순열이 나온다 (원본의 "한 번 섞은 순서 고정" 재현)
            perm_gen = torch.Generator(device=device)
            perm_gen.manual_seed(seed)
            perm = torch.randperm(n, generator=perm_gen, device=device)
        else:
            perm = torch.randperm(n, device=device)
        loss_sum = 0.0
        phys_sum = 0.0
        beta_now = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            if physics is None:
                loss = l1_loss(model(x_t[idx]), y_t[idx])
                sup_value = loss.item()
            else:
                # 전역 스텝을 epoch에서 유도한다 — resume 후에도 워밍업 위치가 이어진다.
                step = (epoch - 1) * steps_per_epoch + start // batch_size
                parts = physics(model(x_t[idx]), y_t[idx], x_t[idx], step)
                loss, sup_value, beta_now = parts.total, parts.sup.item(), parts.beta
                phys_sum += parts.phys.item() * len(idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            # train_l1은 물리 항을 뺀 지도 항만 — run 사이 비교 가능해야 한다
            loss_sum += sup_value * len(idx)

        val_pred = predict_on_device(model, x_v)
        val_metrics = mae_per_layer(val_pred, y_val)
        val_phys: float | None = None
        if physics is not None:
            val_phys = float(
                physics.decoder.residual_l1(torch.from_numpy(val_pred).to(device), x_v).mean()
            )
        row = {
            "epoch": epoch,
            "train_l1": loss_sum / n,
            "val_mae": val_metrics["overall"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.perf_counter() - t_epoch,
        }
        marker = ""
        improved = val_metrics["overall"] < best_mae
        if improved:
            best_mae = val_metrics["overall"]
            # copy=True: CPU 텐서여도 참조가 아닌 복사본을 남긴다 (이후 학습이 덮어쓰지 않게)
            state_now = model.state_dict()
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in state_now.items()}
            best_epoch = epoch
            best_metrics = val_metrics
            best_pred = val_pred
            best_val_phys = val_phys
            marker = " *"
            save_best_checkpoint(best_state, best_epoch, best_mae)  # 갱신 즉시 저장
        # 물리 항이 없으면 빈 문자열 — 기존 run과 로그 형식이 같다
        phys_note = (
            ""
            if physics is None or val_phys is None
            else f"train_phys {phys_sum / n:.6f}  val_phys {val_phys:.6f}  beta {beta_now:g}  "
        )
        log_line(
            run_dir,
            f"[{tag}] epoch {epoch:3d}/{epochs}  train_l1 {row['train_l1']:.4f}  "
            f"val_mae {row['val_mae']:.4f}  {phys_note}"
            f"lr {row['lr']:.2e}  {row['sec']:.1f}s{marker}",
        )

        atomic_save(
            {
                "fingerprint": fingerprint,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "best_mae": best_mae,
                "best_state": best_state,
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
                "best_pred": torch.from_numpy(best_pred) if best_pred is not None else None,
                "best_val_phys": best_val_phys,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "numpy_rng": np.random.get_state(),  # noqa: NPY002
                "py_rng": random.getstate(),
                "wall_sec": wall_prev + time.perf_counter() - t_start,
            },
            resume_path,
        )
        mirror_names = (
            ("train.log",)
            + ((RESUME_NAME,) if epoch % mirror_resume_every == 0 else ())
            + ((f"{tag}.pt",) if improved else ())
        )
        _mirror_copy(run_dir, mirror_dir, mirror_names)

        if _abort_after_epoch is not None and epoch >= _abort_after_epoch:
            raise RuntimeError(f"테스트용 세션 중단 흉내 (epoch {epoch})")

    if best_state is None or best_pred is None:  # epochs >= 1 이므로 도달 불가
        raise RuntimeError("best 체크포인트가 만들어지지 않았다")

    # 완료: 재개용 상태는 로컬·미러 모두 정리 (best 체크포인트·로그는 남는다)
    resume_path.unlink(missing_ok=True)
    if mirror_dir is not None:
        (mirror_dir / RESUME_NAME).unlink(missing_ok=True)
        _mirror_copy(run_dir, mirror_dir, ("train.log", f"{tag}.pt"))

    out_path = ckpt_path
    if out_path.is_relative_to(REPO_ROOT):  # metrics.json에 로컬 절대경로가 남지 않게
        out_path = out_path.relative_to(REPO_ROOT)
    result: dict[str, Any] = {
        "tag": tag,
        "seed": seed,
        "ckpt_path": str(out_path),
        "best_epoch": best_epoch,
        "val_mae": best_mae,
        "val_mae_per_layer": best_metrics,
        "val_pred": best_pred,
        "wall_sec": wall_prev + time.perf_counter() - t_start,
    }
    if physics is not None:
        result["physics"] = physics.config
        result["val_phys_l1"] = best_val_phys
    return result


def _load_completed_metrics(path: Path) -> dict[str, Any] | None:
    """완료된 run의 metrics.json이면 그 내용을, 아니면 None을 돌려준다."""
    if not path.exists():
        return None
    try:
        metrics = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return metrics if "model" in metrics else None


def stale_config_keys(stored: Any, current: dict[str, Any], prefix: str = "") -> list[str]:
    """완료 기록의 설정 스냅샷과 현재 config가 다른 지점의 점 표기 키 목록.

    `run_name`은 뺀다 (같은 run을 가리키는 이름이므로 항상 같다). 스냅샷이 없으면
    (`stored is None`) 대조가 불가능하므로 빈 목록을 돌려준다 — 판정 불가를 불일치로
    취급하면 스냅샷 이전 run의 재실행이 막힌다.
    """
    if stored is None:
        return []
    if not isinstance(stored, dict) or not isinstance(current, dict):
        return [] if stored == current else [prefix.rstrip(".") or "config"]
    stale: list[str] = []
    for key in sorted(set(stored) | set(current)):
        if not prefix and key == "run_name":
            continue
        path = f"{prefix}{key}"
        if key not in stored or key not in current:
            stale.append(path)
        elif isinstance(stored[key], dict) and isinstance(current[key], dict):
            stale.extend(stale_config_keys(stored[key], current[key], f"{path}."))
        elif stored[key] != current[key]:
            stale.append(path)
    return stale


def run_config(
    config_path: str | Path,
    *,
    device: str | None = None,
    run_name: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    subset: int | None = None,
    physics_warmup_steps: int | None = None,
    resume: bool = True,
    mirror_dir: str | Path | None = None,
    mirror_resume_every: int = 5,
    runs_root: str | Path | None = None,
) -> dict[str, Any]:
    """config 하나를 GPU(가능하면)에서 학습하고 metrics dict를 돌려준다.

    노트북에서 호출하는 진입점. 키워드 인자는 config 값을 덮어쓴다
    (src/train.py의 CLI 오버라이드와 같은 의미).

    세션 유실 대비 (resume=True 기본):
    - 이미 완료된 run(metrics.json에 결과 존재 — 로컬 또는 미러)은 학습 없이
      기존 metrics를 돌려준다. 여러 실험 순차 실행 중 끊겨도 재실행이 싸다.
    - 진행 중이던 run은 resume.pt에서 다음 에폭부터 재개한다 (train_one_model_gpu).

    Args:
        mirror_dir: 세션 유실 대비 미러 루트 (예: Drive 경로). 실제 미러는
            <mirror_dir>/<experiment>/<run_name>/에 쌓인다.
        mirror_resume_every: resume.pt 미러 복사 간격(에폭) — train_one_model_gpu 참조.
        physics_warmup_steps: 물리 항 워밍업 스텝 수 덮어쓰기. **스모크 전용 손잡이**다 —
            서브셋 스모크는 총 스텝이 본 학습의 1/600이라 기본 워밍업(3,000)이면 유효
            beta가 목표의 2%에 그쳐 물리 항이 사실상 꺼진 채로 지나간다.
        runs_root: 산출물 루트 (기본 runs/ — 테스트용 오버라이드).

    Raises:
        ValueError: config가 k-fold 모드(num_folds >= 2)인 경우 (GPU 경로는 holdout 단일
            모드만 지원), 또는 physics 블록이 없는 config에 physics_warmup_steps를 준 경우.
    """
    cfg: dict[str, Any] = yaml.safe_load(Path(config_path).read_text())
    if run_name is not None:
        cfg["run_name"] = run_name
    if epochs is not None:
        cfg["train"]["epochs"] = epochs
    if batch_size is not None:
        cfg["train"]["batch_size"] = batch_size
    if lr is not None:
        cfg["train"]["lr"] = lr
    if weight_decay is not None:
        cfg["train"]["weight_decay"] = weight_decay
    if subset is not None:
        cfg.setdefault("data", {})["subset"] = subset
    if physics_warmup_steps is not None:
        if "physics" not in cfg["train"]:
            raise ValueError(
                "physics 블록이 없는 config에 physics_warmup_steps를 줄 수 없다 — "
                f"{config_path}에 train.physics를 두거나 인자를 빼라"
            )
        cfg["train"]["physics"]["warmup_steps"] = int(physics_warmup_steps)

    if int(cfg["train"].get("num_folds", 0)) >= 2:
        raise ValueError("GPU 경로는 holdout 단일 모드만 지원한다 — k-fold는 src/train.py 참조")

    experiment = cfg.get("experiment")
    if not experiment:
        raise ValueError(
            'config에 "experiment" 키(대실험 주제)가 필요하다 — runs/<experiment>/<run_name> 구조'
        )

    run_dir = Path(runs_root) if runs_root is not None else RUNS_DIR
    run_dir = run_dir / str(experiment) / str(cfg["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    mirror_run: Path | None = None
    if mirror_dir is not None:
        mirror_run = Path(mirror_dir) / str(experiment) / str(cfg["run_name"])

    if resume:
        done = _load_completed_metrics(run_dir / "metrics.json")
        if done is None and mirror_run is not None:
            done = _load_completed_metrics(mirror_run / "metrics.json")
            if done is not None:  # 미러에만 완료 기록이 있으면 산출물을 로컬로 되가져온다
                _mirror_copy(mirror_run, run_dir, ("metrics.json", "train.log", "model.pt"))
        if done is not None:
            # **설정이 같을 때만** 건너뛴다. metrics.json이 설정 스냅샷을 겸하므로 대조가
            # 가능하고, 하지 않으면 config를 고쳐 재실행해도 옛 결과를 조용히 돌려준다
            # (epochs만 늘린 run이 이전 예산의 수치를 그대로 받는 형태로 걸렸다).
            stale = stale_config_keys(done.get("config"), cfg)
            if stale:
                raise ValueError(
                    f"run {experiment}/{cfg['run_name']}: 완료 기록의 설정이 현재 config와"
                    f" 다르다 — 건너뛰면 옛 결과를 반환한다.\n"
                    f"  다른 키: {', '.join(stale)}\n"
                    f"  같은 이름으로 다시 돌리려면 {run_dir}"
                    + (f" 와 미러 {mirror_run}" if mirror_run is not None else "")
                    + "를 지우고, 둘을 함께 남기려면 run_name을 바꿀 것"
                )
            print(
                f"run {experiment}/{cfg['run_name']}: 이미 완료 — 건너뜀"
                f" (holdout MAE {done['model']['val_mae']:.4f} nm)"
            )
            return done

    dev = resolve_device(device)
    seed = int(cfg["seed"])
    set_seed(seed)
    x, y, train_idx, holdout_idx = prepare_from_config(cfg)

    log_line(
        run_dir,
        f"run {experiment}/{cfg['run_name']}: 행 {len(x):,} = 학습 {len(train_idx):,}"
        f" + holdout {len(holdout_idx):,} / mode=holdout / device={dev}",
    )

    metrics: dict[str, Any] = {
        "experiment": experiment,
        "run_name": cfg["run_name"],
        "seed": seed,
        "mode": "holdout",
        "device": str(dev),
        "rows": {"train": int(len(train_idx)), "holdout": int(len(holdout_idx))},
        "config": cfg,
    }
    # 시작 시점 설정 스냅샷 (중단돼도 남는다) — 완료 시 결과를 합쳐 덮어쓴다
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    result = train_one_model_gpu(
        x[train_idx],
        y[train_idx],
        x[holdout_idx],
        y[holdout_idx],
        cfg,
        seed,
        run_dir,
        "model",
        dev,
        mirror_dir=mirror_run,
        resume=resume,
        mirror_resume_every=mirror_resume_every,
    )
    result.pop("val_pred")
    metrics["model"] = result
    log_line(
        run_dir,
        f"\n[model] holdout {format_mae(result['val_mae_per_layer'])}"
        f" (best epoch {result['best_epoch']})",
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    # 로그를 **먼저** 다 쓴 뒤 미러로 복사한다 — 순서가 뒤집히면 미러의 train.log에서
    # 마지막 줄이 항상 빠진다 (미러 복원 경로를 탄 run의 로그가 잘려 있었다).
    log_line(run_dir, f"\n산출물: {run_dir}/ (metrics.json, train.log, model.pt)")
    _mirror_copy(run_dir, mirror_run, ("metrics.json", "train.log"))
    return metrics


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FringeNet GPU 학습 (holdout 단일 모드)")
    parser.add_argument("--config", required=True, help="configs/*.yaml 경로")
    parser.add_argument("--device", default=None, help='"cuda" | "cpu" (기본: cuda 우선 자동)')
    parser.add_argument("--run-name", default=None, help="runs/ 아래 저장 이름 (config 덮어씀)")
    parser.add_argument("--epochs", type=int, default=None, help="config 덮어씀")
    parser.add_argument("--batch-size", type=int, default=None, help="config 덮어씀")
    parser.add_argument("--lr", type=float, default=None, help="config 덮어씀")
    parser.add_argument("--weight-decay", type=float, default=None, help="config 덮어씀")
    parser.add_argument(
        "--subset", type=int, default=None, help="시드 고정 무작위 서브셋 크기 (스모크용)"
    )
    parser.add_argument("--mirror-dir", default=None, help="세션 유실 대비 미러 루트 (선택)")
    parser.add_argument(
        "--mirror-resume-every",
        type=int,
        default=5,
        help="resume.pt 미러 복사 간격(에폭) — 대형 모델의 Drive 업로드 밀림 완화용",
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="resume.pt·완료 기록을 무시하고 처음부터 학습"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_config(
        args.config,
        device=args.device,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        subset=args.subset,
        resume=not args.no_resume,
        mirror_dir=args.mirror_dir,
        mirror_resume_every=args.mirror_resume_every,
    )


if __name__ == "__main__":
    main()
