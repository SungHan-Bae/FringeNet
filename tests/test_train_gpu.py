"""GPU 학습 경로(src/train_gpu.py) 단위 테스트 — 대회 데이터·GPU 없이 전부 돈다.

train_one_model_gpu는 device 인자만 다를 뿐 CPU에서도 동작해야 한다 (device="cpu"로
스모크). CUDA가 있으면 같은 테스트가 cuda로도 돈다 — Colab에서 pytest로 확인 가능.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from src.evaluate import load_model_checkpoint, predict
from src.train_gpu import predict_on_device, resolve_device, run_config, train_one_model_gpu

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _tiny_cnn_cfg() -> dict:
    return {
        "model": {
            "name": "cnn",
            "channels": [8, 16],
            "strides": [1, 2],
            "output_bound": False,
        },
        "train": {
            "epochs": 2,
            "batch_size": 64,
            "lr": 1.0e-3,
            "weight_decay": 1.0e-4,
            "lr_schedule": "cosine",
            "warmup_steps": 2,
        },
    }


@pytest.mark.parametrize("device", _DEVICES)
def test_train_one_model_gpu_smoke_and_checkpoint_roundtrip(device: str, tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.random((256, 226), dtype=np.float32)
    y = rng.uniform(10.0, 300.0, size=(256, 4)).astype(np.float32)

    result = train_one_model_gpu(
        x[:192],
        y[:192],
        x[192:],
        y[192:],
        _tiny_cnn_cfg(),
        seed=0,
        run_dir=tmp_path,
        tag="model",
        device=torch.device(device),
    )

    assert result["best_epoch"] in (1, 2)
    assert np.isfinite(result["val_mae"])
    assert result["val_pred"].shape == (64, 4)
    assert (tmp_path / "train.log").read_text().count("epoch") == 2

    # 체크포인트는 device와 무관하게 CPU 텐서여야 한다 — 로컬 CPU 분석 호환성의 핵심
    ckpt = torch.load(tmp_path / "model.pt", map_location=None, weights_only=True)
    assert all(v.device.type == "cpu" for v in ckpt["state_dict"].values())

    # 왕복: CPU 파이프라인의 load_model_checkpoint + predict로 best 예측이 재현돼야 한다
    model = load_model_checkpoint(tmp_path / "model.pt")
    again = predict(model, x[192:])
    assert np.allclose(again, result["val_pred"], atol=1e-4)


def test_predict_on_device_matches_cpu_predict() -> None:
    from src.models import build_model

    torch.manual_seed(0)
    model = build_model({"name": "cnn", "channels": [8], "strides": [1]})
    x = torch.rand(50, 226)
    out = predict_on_device(model, x, batch_size=16)
    assert out.shape == (50, 4)
    assert np.allclose(out, predict(model, x.numpy()), atol=1e-6)


def test_resolve_device_defaults_and_rejects_missing_cuda() -> None:
    assert resolve_device("cpu").type == "cpu"
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device(None).type == expected
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            resolve_device("cuda")


def test_run_config_rejects_kfold(tmp_path: Path) -> None:
    import yaml

    cfg = {
        "seed": 0,
        "experiment": "exp",
        "run_name": "run",
        "model": {"name": "mlp", "hidden_dims": [8]},
        "train": {"epochs": 1, "batch_size": 64, "lr": 1e-3, "num_folds": 5},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="holdout"):
        run_config(path, device="cpu")


# ---------------------------------------------------------------------------
# 세션 유실 대비 — resume / best 즉시 저장 / 미러 / 완료 run 스킵
# ---------------------------------------------------------------------------
def _synthetic() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.random((256, 226), dtype=np.float32)
    y = rng.uniform(10.0, 300.0, size=(256, 4)).astype(np.float32)
    return x, y


def _train(run_dir: Path, cfg: dict | None = None, device: str = "cpu", **kwargs: object) -> dict:
    x, y = _synthetic()
    return train_one_model_gpu(
        x[:192],
        y[:192],
        x[192:],
        y[192:],
        cfg or _tiny_cnn_cfg(),
        seed=0,
        run_dir=run_dir,
        tag="model",
        device=torch.device(device),
        **kwargs,  # type: ignore[arg-type]
    )


# device 파라미터화 필수 — resume 로드는 map_location=device로 CPU 계약 텐서(best_pred·
# best_state)까지 device로 올리는 회귀가 있었다 (GPU에서만 재현, Colab pytest로 검증).
@pytest.mark.parametrize("device", _DEVICES)
def test_resume_after_interrupt_matches_uninterrupted(device: str, tmp_path: Path) -> None:
    cfg = _tiny_cnn_cfg()
    cfg["train"]["epochs"] = 3

    full_dir = tmp_path / "full"
    full_dir.mkdir()
    full = _train(full_dir, cfg, device)

    # epoch 1 직후 세션 중단을 흉내 낸 뒤 재개 — RNG까지 복원되므로 결과가 같아야 한다
    part_dir = tmp_path / "interrupted"
    part_dir.mkdir()
    with pytest.raises(RuntimeError, match="중단"):
        _train(part_dir, cfg, device, _abort_after_epoch=1)
    assert (part_dir / "resume.pt").exists()
    assert (part_dir / "model.pt").exists()  # best는 갱신 즉시 저장 — 중단 시점에도 존재
    resumed = _train(part_dir, cfg, device)

    assert resumed["best_epoch"] == full["best_epoch"]
    assert resumed["val_mae"] == pytest.approx(full["val_mae"], abs=1e-6)
    assert np.allclose(resumed["val_pred"], full["val_pred"], atol=1e-5)
    assert not (part_dir / "resume.pt").exists()  # 완료 시 재개 상태는 정리된다
    # 로그에 재개 기록이 남는다
    assert "resume" in (part_dir / "train.log").read_text()
    # resume 직후 저장되는 best 체크포인트는 device와 무관하게 CPU 텐서 계약을 지켜야 한다
    ckpt = torch.load(part_dir / "model.pt", map_location=None, weights_only=True)
    assert all(v.device.type == "cpu" for v in ckpt["state_dict"].values())


@pytest.mark.parametrize("device", _DEVICES)
def test_resume_restores_from_mirror_on_fresh_vm(device: str, tmp_path: Path) -> None:
    cfg = _tiny_cnn_cfg()
    cfg["train"]["epochs"] = 3

    full_dir = tmp_path / "full"
    full_dir.mkdir()
    full = _train(full_dir, cfg, device)

    run_dir = tmp_path / "run"
    mirror = tmp_path / "mirror"
    run_dir.mkdir()
    # 매 에폭 미러(K=1)의 복구 의미론을 검증하는 테스트 — 기본값(5)이면 3에폭짜리
    # 실행에서 resume.pt가 미러에 아예 안 가므로 K를 명시한다
    with pytest.raises(RuntimeError, match="중단"):
        _train(run_dir, cfg, device, mirror_dir=mirror, mirror_resume_every=1, _abort_after_epoch=2)
    # 미러에 에폭 단위 백업이 남아 있어야 한다
    assert (mirror / "resume.pt").exists()
    assert (mirror / "train.log").exists()
    assert (mirror / "model.pt").exists()

    # 세션 유실로 VM 디스크가 날아간 상황: run_dir를 비우고 미러만으로 재개
    shutil.rmtree(run_dir)
    run_dir.mkdir()
    resumed = _train(run_dir, cfg, device, mirror_dir=mirror, mirror_resume_every=1)

    assert resumed["val_mae"] == pytest.approx(full["val_mae"], abs=1e-6)
    assert np.allclose(resumed["val_pred"], full["val_pred"], atol=1e-5)
    assert not (mirror / "resume.pt").exists()  # 완료 시 미러의 재개 상태도 정리
    assert (mirror / "model.pt").exists()  # best 체크포인트·로그는 미러에 남는다


def test_resume_rejects_config_mismatch(tmp_path: Path) -> None:
    cfg = _tiny_cnn_cfg()
    cfg["train"]["epochs"] = 2
    with pytest.raises(RuntimeError, match="중단"):
        _train(tmp_path, cfg, _abort_after_epoch=1)
    changed = _tiny_cnn_cfg()
    changed["train"]["epochs"] = 2
    changed["train"]["lr"] = 5e-4  # 설정이 달라졌으면 이어받으면 안 된다
    with pytest.raises(ValueError, match="config"):
        _train(tmp_path, changed)


def test_run_config_skips_completed_run(tmp_path: Path) -> None:
    import yaml

    cfg = {
        "seed": 0,
        "experiment": "exp",
        "run_name": "done",
        "model": {"name": "mlp", "hidden_dims": [8]},
        "train": {"epochs": 1, "batch_size": 64, "lr": 1e-3},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    completed = {"experiment": "exp", "run_name": "done", "model": {"val_mae": 1.23}}
    run_dir = tmp_path / "runs" / "exp" / "done"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps(completed))

    out = run_config(cfg_path, device="cpu", runs_root=tmp_path / "runs")
    assert out == completed  # 데이터 로드·학습 없이 기존 결과를 그대로 돌려준다


def test_run_config_restores_completed_run_from_mirror(tmp_path: Path) -> None:
    import yaml

    cfg = {
        "seed": 0,
        "experiment": "exp",
        "run_name": "done",
        "model": {"name": "mlp", "hidden_dims": [8]},
        "train": {"epochs": 1, "batch_size": 64, "lr": 1e-3},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    completed = {"experiment": "exp", "run_name": "done", "model": {"val_mae": 1.23}}
    mirror_run = tmp_path / "mirror" / "exp" / "done"
    mirror_run.mkdir(parents=True)
    (mirror_run / "metrics.json").write_text(json.dumps(completed))
    (mirror_run / "model.pt").write_bytes(b"dummy")
    (mirror_run / "train.log").write_text("log")

    out = run_config(
        cfg_path, device="cpu", runs_root=tmp_path / "runs", mirror_dir=tmp_path / "mirror"
    )
    assert out == completed
    # 미러의 산출물이 로컬 runs/로 되돌아온다 (push 대상 복원)
    run_dir = tmp_path / "runs" / "exp" / "done"
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "model.pt").exists()
    assert (run_dir / "train.log").exists()


# ---------------------------------------------------------------------------
# 1등 재현 프로토콜 플래그 (strong_baseline) — shuffle "once" / eval 모드 학습 quirk.
# 기본값 경로(기존 실험)는 위 테스트들이 새 키 없는 config로 그대로 검증한다.
# ---------------------------------------------------------------------------
def _tiny_winner_cfg() -> dict:
    return {
        "model": {"name": "winner_skip_mlp", "up_dims": [16, 32], "head_dim": 8},
        "train": {
            "epochs": 3,
            "batch_size": 64,
            "lr": 1.0e-3,
            "weight_decay": 0.0,
            "adam_eps": 1.0e-6,
            "lr_schedule": "cosine",
            "warmup_steps": 2,
            "shuffle": "once",
            "eval_mode_after_first_epoch": True,
        },
    }


def test_invalid_shuffle_mode_raises(tmp_path: Path) -> None:
    cfg = _tiny_cnn_cfg()
    cfg["train"]["shuffle"] = "never"
    with pytest.raises(ValueError, match="shuffle"):
        _train(tmp_path, cfg)


def test_winner_protocol_resume_matches_uninterrupted(tmp_path: Path) -> None:
    # shuffle "once"(시드 고정 전용 generator)와 에폭 2부터 eval 모드 학습이 resume을
    # 가로질러도 무중단 실행과 동일해야 한다 (재개 = 무중단 계약이 새 플래그에도 성립)
    cfg = _tiny_winner_cfg()

    full_dir = tmp_path / "full"
    full_dir.mkdir()
    full = _train(full_dir, cfg)

    part_dir = tmp_path / "interrupted"
    part_dir.mkdir()
    with pytest.raises(RuntimeError, match="중단"):
        _train(part_dir, cfg, _abort_after_epoch=1)
    resumed = _train(part_dir, cfg)

    assert resumed["best_epoch"] == full["best_epoch"]
    assert resumed["val_mae"] == pytest.approx(full["val_mae"], abs=1e-6)
    assert np.allclose(resumed["val_pred"], full["val_pred"], atol=1e-5)


def _bn_running_stats(state: dict) -> dict:
    return {k: v.clone() for k, v in state.items() if k.endswith(("running_mean", "running_var"))}


@pytest.mark.parametrize("flag", [True, False])
def test_eval_mode_after_first_epoch_freezes_bn_stats(flag: bool, tmp_path: Path) -> None:
    # 원본 train.py의 quirk 재현 검증 — 플래그 on이면 에폭 1 이후 BatchNorm running
    # 통계가 동결되고(에폭 2부터 eval 모드 학습), off면 매 에폭 갱신되어야 한다.
    cfg = _tiny_winner_cfg()
    cfg["train"]["eval_mode_after_first_epoch"] = flag

    with pytest.raises(RuntimeError, match="중단"):
        _train(tmp_path, cfg, _abort_after_epoch=1)
    state = torch.load(tmp_path / "resume.pt", map_location="cpu", weights_only=False)
    after_ep1 = _bn_running_stats(state["model"])
    assert after_ep1  # BN이 실제로 존재해야 테스트가 의미 있다

    with pytest.raises(RuntimeError, match="중단"):
        _train(tmp_path, cfg, _abort_after_epoch=3)  # ep1 상태에서 재개 -> ep3까지
    state = torch.load(tmp_path / "resume.pt", map_location="cpu", weights_only=False)
    after_ep3 = _bn_running_stats(state["model"])

    frozen = all(torch.equal(after_ep1[k], after_ep3[k]) for k in after_ep1)
    assert frozen == flag


# ---------------------------------------------------------------------------
# resume.pt 미러 간격 (mirror_resume_every) — 대형 모델의 Drive 업로드 밀림 완화
# ---------------------------------------------------------------------------
def test_mirror_resume_every_rejects_below_one(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mirror_resume_every"):
        _train(tmp_path, mirror_resume_every=0)


def test_mirror_resume_interval_skips_off_epochs(tmp_path: Path) -> None:
    # K=2: 에폭 1(간격 미도달)에서는 미러에 train.log만 가고 resume.pt는 안 간다
    mirror = tmp_path / "mirror"
    with pytest.raises(RuntimeError, match="중단"):
        _train(tmp_path, mirror_dir=mirror, mirror_resume_every=2, _abort_after_epoch=1)
    assert (mirror / "train.log").exists()
    assert not (mirror / "resume.pt").exists()
    assert (tmp_path / "resume.pt").exists()  # 로컬 저장은 매 에폭 그대로


def test_lagged_mirror_resume_matches_uninterrupted(tmp_path: Path) -> None:
    # Drive 비동기 업로드 지연 시나리오의 재현: K=2라 미러 resume.pt가 마지막 에폭(3)보다
    # 뒤처진(2) 상태에서 세션이 죽고 VM 디스크가 날아가도, 미러에서 재개한 최종 결과는
    # 무중단 실행과 동일해야 한다 (에폭 3을 같은 궤적으로 재계산)
    cfg = _tiny_cnn_cfg()
    cfg["train"]["epochs"] = 4

    full_dir = tmp_path / "full"
    full_dir.mkdir()
    full = _train(full_dir, cfg)

    run_dir = tmp_path / "run"
    mirror = tmp_path / "mirror"
    run_dir.mkdir()
    with pytest.raises(RuntimeError, match="중단"):
        _train(run_dir, cfg, mirror_dir=mirror, mirror_resume_every=2, _abort_after_epoch=3)
    lagged = torch.load(mirror / "resume.pt", map_location="cpu", weights_only=False)
    assert lagged["epoch"] == 2  # 미러는 에폭 2에 머물러 있다 (로컬은 3까지 갔지만 유실)

    shutil.rmtree(run_dir)
    run_dir.mkdir()
    resumed = _train(run_dir, cfg, mirror_dir=mirror, mirror_resume_every=2)

    assert resumed["best_epoch"] == full["best_epoch"]
    assert resumed["val_mae"] == pytest.approx(full["val_mae"], abs=1e-6)
    assert np.allclose(resumed["val_pred"], full["val_pred"], atol=1e-5)


# ---------------------------------------------------------------------------
# Stage B 물리 손실 (train.physics 블록) — 손실 자체는 tests/test_losses.py가 검증하고
# 여기서는 배선을 건다: 대조군 동등성 · 진단 기록 · 워밍업이 resume을 가로질러 이어짐.
# ---------------------------------------------------------------------------
def _physics_cfg(beta: float, *, epochs: int = 2) -> dict:
    cfg = _tiny_cnn_cfg()
    cfg["train"]["epochs"] = epochs
    cfg["train"]["physics"] = {"beta": beta, "warmup_steps": 2}
    return cfg


def test_physics_run_records_decoder_provenance_and_diagnostics(tmp_path: Path) -> None:
    result = _train(tmp_path, _physics_cfg(100.0))
    assert result["physics"]["beta"] == 100.0
    assert result["physics"]["decoder"].endswith("joint-lam3-sin2-si2-schinke/model.pt")
    assert len(result["physics"]["free"]) == 7  # Stage A 확정 자유도
    assert np.isfinite(result["val_phys_l1"])
    log = (tmp_path / "train.log").read_text()
    assert "물리 손실 beta 100" in log
    assert "train_phys" in log and "val_phys" in log


def test_physics_beta_zero_reproduces_run_without_physics(tmp_path: Path) -> None:
    """대조군(beta=0)의 학습 경로가 물리 항 도입 전과 같아야 차이를 물리 항에 귀속할 수 있다.

    CPU 전용 — 결정성이 보장되는 쪽에서 비트 동일성을 건다 (CPU↔GPU는 MAE 수준 비교 규약).
    """
    plain_dir = tmp_path / "plain"
    phys_dir = tmp_path / "phys"
    plain_dir.mkdir()
    phys_dir.mkdir()
    cfg_plain = _tiny_cnn_cfg()
    cfg_plain["train"]["epochs"] = 2

    plain = _train(plain_dir, cfg_plain)
    zero = _train(phys_dir, _physics_cfg(0.0))

    assert zero["val_mae"] == plain["val_mae"]
    assert np.array_equal(zero["val_pred"], plain["val_pred"])
    assert zero["physics"]["beta"] == 0.0
    assert np.isfinite(zero["val_phys_l1"])  # 대조군도 진단은 기록한다
    # 대조군의 로그는 물리 항이 켜진 run과 같은 형식이어야 나란히 읽을 수 있다
    assert "beta 0" in (phys_dir / "train.log").read_text()


def test_physics_resume_matches_uninterrupted(tmp_path: Path) -> None:
    cfg = _physics_cfg(100.0, epochs=3)

    full_dir = tmp_path / "full"
    full_dir.mkdir()
    full = _train(full_dir, cfg)

    part_dir = tmp_path / "interrupted"
    part_dir.mkdir()
    with pytest.raises(RuntimeError, match="중단"):
        _train(part_dir, cfg, _abort_after_epoch=1)
    resumed = _train(part_dir, cfg)

    assert resumed["best_epoch"] == full["best_epoch"]
    assert resumed["val_mae"] == pytest.approx(full["val_mae"], abs=1e-6)
    assert resumed["val_phys_l1"] == pytest.approx(full["val_phys_l1"], abs=1e-6)


def test_physics_config_typo_raises_before_training(tmp_path: Path) -> None:
    cfg = _tiny_cnn_cfg()
    cfg["train"]["physics"] = {"beta": 1.0, "warmpu_steps": 2}  # 오타
    with pytest.raises(ValueError, match="physics"):
        _train(tmp_path, cfg)
    assert not (tmp_path / "model.pt").exists()


def test_physics_warmup_override_requires_physics_block(tmp_path: Path) -> None:
    import yaml

    cfg = {
        "seed": 0,
        "experiment": "exp",
        "run_name": "no-physics",
        "model": {"name": "cnn", "channels": [8], "strides": [1]},
        "train": {"epochs": 1, "batch_size": 64, "lr": 1e-3},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="physics 블록이 없는"):
        run_config(path, device="cpu", physics_warmup_steps=10, runs_root=tmp_path / "runs")


def test_physics_warmup_override_reaches_training(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """스모크 손잡이가 실제 학습 설정까지 전달되는지 — 데이터·GPU 없이 검증한다.

    서브셋 스모크는 총 스텝이 본 학습의 1/600이라 기본 워밍업(3,000)이면 유효 beta가
    목표의 2%에 그친다. 그 상태로 "물리 손실 경로를 확인했다"고 넘어가면 안 된다.
    """
    import yaml

    from src import train_gpu as tg

    cfg = {
        "seed": 0,
        "experiment": "exp",
        "run_name": "phys",
        "model": {"name": "cnn", "channels": [8], "strides": [1]},
        "train": {
            "epochs": 1,
            "batch_size": 64,
            "lr": 1e-3,
            "physics": {"beta": 100.0, "warmup_steps": 3000},
        },
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))

    captured: dict = {}

    def fake_prepare(_cfg: dict) -> tuple:
        x = np.zeros((8, 226), dtype=np.float32)
        y = np.zeros((8, 4), dtype=np.float32)
        return x, y, np.arange(6), np.arange(6, 8)

    def fake_train(*args: object, **kwargs: object) -> dict:
        captured["cfg"] = args[4]  # train_one_model_gpu(x_tr, y_tr, x_v, y_v, cfg, ...)
        per_layer = {f"layer_{i}": 1.0 for i in range(1, 5)} | {"overall": 1.0}
        return {
            "tag": "model",
            "seed": 0,
            "ckpt_path": "model.pt",
            "best_epoch": 1,
            "val_mae": 1.0,
            "val_mae_per_layer": per_layer,
            "val_pred": np.zeros((2, 4), dtype=np.float32),
            "wall_sec": 0.1,
        }

    monkeypatch.setattr(tg, "prepare_from_config", fake_prepare)
    monkeypatch.setattr(tg, "train_one_model_gpu", fake_train)
    tg.run_config(path, device="cpu", physics_warmup_steps=10, runs_root=tmp_path / "runs")

    assert captured["cfg"]["train"]["physics"]["warmup_steps"] == 10
    assert captured["cfg"]["train"]["physics"]["beta"] == 100.0  # beta는 건드리지 않는다


# --- train.init_from (warm start) -------------------------------------------------
# 라운드 3의 전제: 수렴된 CNN에서 출발해 물리 항을 켠다. 랜덤 초기화에서 물리 gradient에
# 끌려가면 잘못된 fringe 차수 분지에 안착할 수 있어, 올바른 분지에 이미 든 지점에서 시험한다.


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _seed_checkpoint(tmp_path: Path, *, data: dict | None = None) -> Path:
    """warm start 출처가 될 run 하나를 실제로 학습해 만든다."""
    src = tmp_path / "src_run"
    src.mkdir()
    cfg = _tiny_cnn_cfg()
    if data is not None:
        cfg["data"] = data
    _train(src, cfg)
    metrics = {"config": {"data": cfg.get("data", {})}}
    (src / "metrics.json").write_text(json.dumps(metrics))
    return src / "model.pt"


def test_init_from_loads_weights_and_logs_provenance(tmp_path: Path) -> None:
    ckpt_path = _seed_checkpoint(tmp_path)
    source = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    cfg = _tiny_cnn_cfg()
    cfg["train"]["init_from"] = str(ckpt_path)
    cfg["train"]["epochs"] = 1
    cfg["train"]["lr"] = 0.0  # LR 0이면 가중치가 그대로여야 한다 — 적재 자체를 본다
    out = _train(_mkdir(tmp_path / "warm"), cfg)

    # 파라미터만 대조한다 — BatchNorm 러닝 통계는 버퍼라 lr=0에서도 forward가 갱신한다
    warm_params = dict(load_model_checkpoint(tmp_path / "warm" / "model.pt").named_parameters())
    src_model = load_model_checkpoint(ckpt_path)
    for key, value in src_model.named_parameters():
        torch.testing.assert_close(warm_params[key], value)
    assert out["val_mae"] == pytest.approx(source["val_mae"], abs=1e-2)
    log = (tmp_path / "warm" / "train.log").read_text()
    assert "warm start" in log and "분할 일치 확인" in log


def test_init_from_rejects_split_mismatch(tmp_path: Path) -> None:
    """다른 split에서 학습된 체크포인트로 warm start하면 누수다 — 학습 전에 막는다."""
    ckpt_path = _seed_checkpoint(tmp_path, data={"holdout_thickness": [70, 150, 230]})
    cfg = _tiny_cnn_cfg()
    cfg["data"] = {"val_frac": 0.1}
    cfg["train"]["init_from"] = str(ckpt_path)
    with pytest.raises(ValueError, match="누수"):
        _train(_mkdir(tmp_path / "warm"), cfg)


def test_init_from_rejects_architecture_mismatch(tmp_path: Path) -> None:
    ckpt_path = _seed_checkpoint(tmp_path)
    cfg = _tiny_cnn_cfg()
    cfg["model"] = {"name": "mlp", "hidden_dims": [8]}
    cfg["train"]["init_from"] = str(ckpt_path)
    with pytest.raises(ValueError, match="model 블록"):
        _train(_mkdir(tmp_path / "warm"), cfg)


def test_init_from_missing_checkpoint_names_recovery(tmp_path: Path) -> None:
    cfg = _tiny_cnn_cfg()
    cfg["train"]["init_from"] = str(tmp_path / "nope.pt")
    with pytest.raises(FileNotFoundError, match="Drive 미러"):
        _train(_mkdir(tmp_path / "warm"), cfg)


def test_init_from_warns_when_source_metrics_absent(tmp_path: Path) -> None:
    """미러에서 model.pt만 받아온 경우 — 대조가 불가능하다는 사실을 로그에 남긴다."""
    ckpt_path = _seed_checkpoint(tmp_path)
    (ckpt_path.parent / "metrics.json").unlink()
    cfg = _tiny_cnn_cfg()
    cfg["train"]["init_from"] = str(ckpt_path)
    cfg["train"]["epochs"] = 1
    _train(_mkdir(tmp_path / "warm"), cfg)
    assert "분할 대조 불가" in (tmp_path / "warm" / "train.log").read_text()


def test_init_from_changes_resume_fingerprint(tmp_path: Path) -> None:
    """warm start run이 cold start run의 resume.pt를 이어받으면 안 된다."""
    ckpt_path = _seed_checkpoint(tmp_path)
    cold = _tiny_cnn_cfg()
    cold["train"]["epochs"] = 2
    with pytest.raises(RuntimeError, match="중단"):
        _train(_mkdir(tmp_path / "run"), cold, _abort_after_epoch=1)
    warm = _tiny_cnn_cfg()
    warm["train"]["epochs"] = 2
    warm["train"]["init_from"] = str(ckpt_path)
    with pytest.raises(ValueError, match="config"):
        _train(tmp_path / "run", warm)


# --- 완료 run 스킵의 설정 대조 -----------------------------------------------------
# metrics.json이 설정 스냅샷을 겸하므로 대조가 가능하다. 하지 않으면 config를 고쳐
# 재실행해도 옛 결과를 조용히 돌려준다 (epochs만 늘린 run이 이전 예산의 수치를 받는 형태).


def _completed_run(tmp_path: Path, cfg: dict, snapshot: dict | None) -> Path:
    """완료 기록만 있는 run 디렉토리를 만든다 (학습하지 않는다)."""
    import yaml

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    run_dir = tmp_path / "runs" / cfg["experiment"] / cfg["run_name"]
    run_dir.mkdir(parents=True)
    done: dict = {
        "experiment": cfg["experiment"],
        "run_name": cfg["run_name"],
        "model": {"val_mae": 1.23},
    }
    if snapshot is not None:
        done["config"] = snapshot
    (run_dir / "metrics.json").write_text(json.dumps(done, ensure_ascii=False))
    return cfg_path


def _skip_cfg(epochs: int = 1) -> dict:
    return {
        "seed": 0,
        "experiment": "exp",
        "run_name": "done",
        "model": {"name": "mlp", "hidden_dims": [8]},
        "train": {"epochs": epochs, "batch_size": 64, "lr": 1e-3},
    }


def test_completed_skip_requires_matching_config(tmp_path: Path) -> None:
    """설정이 같으면 기존대로 건너뛴다 (학습·데이터 로드 없이 옛 결과 반환)."""
    cfg = _skip_cfg()
    cfg_path = _completed_run(tmp_path, cfg, snapshot=cfg)
    out = run_config(cfg_path, device="cpu", runs_root=tmp_path / "runs")
    assert out["model"]["val_mae"] == 1.23


def test_completed_skip_raises_on_changed_config(tmp_path: Path) -> None:
    """epochs만 늘려 재실행하면 옛 예산의 수치를 돌려주면 안 된다."""
    cfg_path = _completed_run(tmp_path, _skip_cfg(epochs=8), snapshot=_skip_cfg(epochs=8))
    changed = _skip_cfg(epochs=40)
    cfg_path.write_text(__import__("yaml").safe_dump(changed, allow_unicode=True))
    with pytest.raises(ValueError, match="train.epochs"):
        run_config(cfg_path, device="cpu", runs_root=tmp_path / "runs")


def test_completed_skip_allows_missing_snapshot(tmp_path: Path) -> None:
    """스냅샷 이전 run은 대조가 불가능하다 — 막으면 재실행이 못 된다."""
    cfg = _skip_cfg()
    cfg_path = _completed_run(tmp_path, cfg, snapshot=None)
    out = run_config(cfg_path, device="cpu", runs_root=tmp_path / "runs")
    assert out["model"]["val_mae"] == 1.23


def test_stale_config_keys_reports_nested_paths_and_ignores_run_name() -> None:
    from src.train_gpu import stale_config_keys

    a = {"run_name": "x", "seed": 0, "train": {"epochs": 8, "physics": {"beta": 30.0}}}
    b = {"run_name": "y", "seed": 0, "train": {"epochs": 40, "physics": {"beta": 30.0}}}
    assert stale_config_keys(a, b) == ["train.epochs"]
    assert stale_config_keys(a, a) == []
    assert stale_config_keys(None, b) == []
    # 한쪽에만 있는 키도 잡는다 (물리 항을 뺀 재실행 등)
    c = {"run_name": "x", "seed": 0, "train": {"epochs": 8}}
    assert stale_config_keys(a, c) == ["train.physics"]


def test_mirror_restore_requires_resume_state(tmp_path: Path) -> None:
    """미러에 resume.pt가 없으면 train.log·model.pt를 끌어오지 않는다.

    이어 달릴 상태가 없는데 가져오면 **중단된 다른 run의 기록을 물려받는다** — 라운드 3에서
    8에폭 run의 로그 3줄이 40에폭 run의 train.log에 섞여 분석이 깨졌다.
    """
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "train.log").write_text("[model] epoch   3/8  train_l1 2.5  val_mae 4.1\n")
    (mirror / "model.pt").write_bytes(b"stale")  # 로드되면 안 되므로 내용은 쓰레기여도 된다

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = _tiny_cnn_cfg()
    cfg["train"]["epochs"] = 1
    _train(run_dir, cfg, mirror_dir=mirror)

    log = (run_dir / "train.log").read_text()
    assert "epoch   3/8" not in log, "중단된 다른 run의 로그를 물려받았다"
    assert "복원하지 않는다" in log
    assert "epoch   1/1" in log  # 처음부터 학습했다


def test_mirror_restore_still_works_with_resume_state(tmp_path: Path) -> None:
    """resume.pt가 있으면 기존 복원 계약은 그대로다 (새 VM 재개)."""
    cfg = _tiny_cnn_cfg()
    cfg["train"]["epochs"] = 3
    full_dir = tmp_path / "full"
    full_dir.mkdir()
    full = _train(full_dir, cfg)

    run_dir, mirror = tmp_path / "run", tmp_path / "mirror"
    run_dir.mkdir()
    with pytest.raises(RuntimeError, match="중단"):
        _train(run_dir, cfg, mirror_dir=mirror, mirror_resume_every=1, _abort_after_epoch=2)
    shutil.rmtree(run_dir)
    run_dir.mkdir()
    resumed = _train(run_dir, cfg, mirror_dir=mirror, mirror_resume_every=1)

    assert "미러에서 복원" in (run_dir / "train.log").read_text()
    assert resumed["val_mae"] == pytest.approx(full["val_mae"], abs=1e-6)


# ---------------------------------------------------------------------------
# 학습 레시피 손잡이 (configs/cnn_recipe/) — noise_aug · l2_weight · ema_decay · input_norm
# ---------------------------------------------------------------------------
def _train_l1_by_epoch(run_dir: Path) -> list[float]:
    """train.log에서 에폭별 train_l1을 뽑는다."""
    out = []
    for line in (run_dir / "train.log").read_text().splitlines():
        if "train_l1" in line:
            out.append(float(line.split("train_l1")[1].split()[0]))
    return out


def test_recipe_knobs_at_default_reproduce_run_without_them(tmp_path: Path) -> None:
    """손잡이를 0으로 **명시**한 run이 손잡이 없는 run과 비트 동일해야 한다.

    이게 깨지면 어떤 변형의 차이도 그 변형에 귀속할 수 없다 (beta=0 대조군과 같은 계약).
    """
    plain_dir, zeroed_dir = tmp_path / "plain", tmp_path / "zeroed"
    plain_dir.mkdir()
    zeroed_dir.mkdir()
    cfg = _tiny_cnn_cfg()
    cfg["train"].update({"noise_aug": 0.0, "l2_weight": 0.0, "ema_decay": 0.0})

    plain = _train(plain_dir, _tiny_cnn_cfg())
    zeroed = _train(zeroed_dir, cfg)

    assert zeroed["val_mae"] == plain["val_mae"]
    assert np.array_equal(zeroed["val_pred"], plain["val_pred"])


def test_input_norm_stats_come_from_train_and_travel_in_checkpoint(tmp_path: Path) -> None:
    """표준화 통계는 **학습 분할에서만** 재고 체크포인트에 함께 실려야 한다.

    전처리를 학습 스크립트에 두면 evaluate·역산 경로가 조용히 다른 입력을 본다 — 여기서
    거는 것은 체크포인트만으로 같은 예측이 재현된다는 계약이다.
    """
    x, y = _synthetic()
    cfg = _tiny_cnn_cfg()
    cfg["model"]["input_norm"] = True
    result = _train(tmp_path, cfg)

    ckpt = torch.load(tmp_path / "model.pt", map_location="cpu", weights_only=True)
    assert "x_mean" in ckpt["state_dict"] and "x_std" in ckpt["state_dict"]
    # 학습 분할(앞 192행)의 통계여야 한다 — 전체(256행)나 holdout이 섞이면 누수다
    expected = torch.from_numpy(x[:192]).mean(dim=0)
    assert torch.allclose(ckpt["state_dict"]["x_mean"], expected, atol=1e-6)
    assert not torch.allclose(
        ckpt["state_dict"]["x_mean"], torch.from_numpy(x).mean(dim=0), atol=1e-6
    )

    # 체크포인트만으로 학습 때와 같은 예측이 나와야 한다
    model = load_model_checkpoint(tmp_path / "model.pt")
    assert np.allclose(predict(model, x[192:]), result["val_pred"], atol=1e-6)


def test_input_norm_changes_the_answer(tmp_path: Path) -> None:
    """켰는데 아무것도 안 바뀌면 통계가 안 채워진 것이다 (버퍼가 항등값으로 남는다)."""
    off_dir, on_dir = tmp_path / "off", tmp_path / "on"
    off_dir.mkdir()
    on_dir.mkdir()
    cfg_on = _tiny_cnn_cfg()
    cfg_on["model"]["input_norm"] = True

    off = _train(off_dir, _tiny_cnn_cfg())
    on = _train(on_dir, cfg_on)

    assert on["val_mae"] != off["val_mae"]
    assert "입력 채널별 표준화" in (on_dir / "train.log").read_text()


def test_ema_checkpoint_holds_averaged_weights(tmp_path: Path) -> None:
    """EMA를 켜면 저장·평가되는 가중치가 마지막 스텝의 가중치와 달라야 한다."""
    cfg = _tiny_cnn_cfg()
    cfg["train"]["ema_decay"] = 0.9
    plain_dir, ema_dir = tmp_path / "plain", tmp_path / "ema"
    plain_dir.mkdir()
    ema_dir.mkdir()

    plain = _train(plain_dir, _tiny_cnn_cfg())
    ema = _train(ema_dir, cfg)

    assert ema["val_mae"] != plain["val_mae"]
    # EMA 가중치는 어느 스텝의 가중치와도 같지 않다 — 평균이므로 원본과 달라야 한다
    ema_ckpt = torch.load(ema_dir / "model.pt", map_location="cpu", weights_only=True)
    plain_ckpt = torch.load(plain_dir / "model.pt", map_location="cpu", weights_only=True)
    assert not torch.equal(
        ema_ckpt["state_dict"]["head.weight"], plain_ckpt["state_dict"]["head.weight"]
    )


def test_ema_resume_matches_uninterrupted(tmp_path: Path) -> None:
    """EMA 상태가 resume.pt에 실려야 재개 결과가 무중단 실행과 같다."""
    cfg = _tiny_cnn_cfg()
    cfg["train"].update({"epochs": 3, "ema_decay": 0.9})

    full_dir, part_dir = tmp_path / "full", tmp_path / "interrupted"
    full_dir.mkdir()
    part_dir.mkdir()
    full = _train(full_dir, cfg)
    with pytest.raises(RuntimeError, match="중단"):
        _train(part_dir, cfg, _abort_after_epoch=2)
    resumed = _train(part_dir, cfg)

    assert resumed["val_mae"] == pytest.approx(full["val_mae"], abs=1e-6)
    assert np.allclose(resumed["val_pred"], full["val_pred"], atol=1e-6)


def test_l2_weight_is_excluded_from_logged_train_l1(tmp_path: Path) -> None:
    """로그의 train_l1은 꼬리 항을 뺀 순수 지도 L1이어야 run 간 비교가 성립한다.

    배치가 하나뿐인 1에폭이면 train_l1은 **갱신 전** 가중치의 L1이므로, 손실이 달라도
    두 run의 값이 정확히 같아야 한다. 반대로 val은 한 스텝 갱신 차이로 갈라진다.
    """
    base_dir, tail_dir = tmp_path / "base", tmp_path / "tail"
    base_dir.mkdir()
    tail_dir.mkdir()
    cfg = _tiny_cnn_cfg()
    cfg["train"].update({"epochs": 1, "batch_size": 192})
    cfg_tail = _tiny_cnn_cfg()
    cfg_tail["train"].update({"epochs": 1, "batch_size": 192, "l2_weight": 0.1})

    base = _train(base_dir, cfg)
    tail = _train(tail_dir, cfg_tail)

    assert _train_l1_by_epoch(base_dir) == _train_l1_by_epoch(tail_dir)
    assert tail["val_mae"] != base["val_mae"]  # 손실은 실제로 달라졌다


def test_noise_aug_perturbs_training_inputs(tmp_path: Path) -> None:
    """증강이 실제로 켜지는지 — 같은 시드에서 결과가 달라져야 한다.

    기본 lr(1e-3)로 2에폭이면 섭동이 만든 가중치 차이가 float32 아래로 묻히므로, 학습이
    실제로 움직이는 lr에서 건다 (여기서 재는 것은 성능이 아니라 배선이다).
    """
    clean_dir, noisy_dir = tmp_path / "clean", tmp_path / "noisy"
    clean_dir.mkdir()
    noisy_dir.mkdir()
    base = _tiny_cnn_cfg()
    base["train"]["lr"] = 0.02
    cfg = _tiny_cnn_cfg()
    cfg["train"].update({"lr": 0.02, "noise_aug": 0.015})

    clean = _train(clean_dir, base)
    noisy = _train(noisy_dir, cfg)

    assert noisy["val_mae"] != clean["val_mae"]


@pytest.mark.parametrize(
    ("key", "value"),
    [("noise_aug", -0.1), ("l2_weight", -1.0), ("ema_decay", 1.0), ("ema_decay", -0.5)],
)
def test_recipe_knobs_reject_out_of_range(key: str, value: float, tmp_path: Path) -> None:
    cfg = _tiny_cnn_cfg()
    cfg["train"][key] = value
    with pytest.raises(ValueError, match=key):
        _train(tmp_path, cfg)
