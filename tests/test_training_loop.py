"""TorchDetector 학습 루프 테스트 (리뷰 P0 — DL 학습 부족 대응).

핵심 보장: 새 옵션을 쓰지 않으면 레거시 루프와 **같은 점수**가 나온다 (커밋된 DL
결과와 그 config_hash가 계속 유효). 새 옵션(검증 분할, 조기 종료, LR 스케줄,
윈도우 상한, weight decay)은 각각 의도대로 동작해야 한다.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="TorchDetector는 torch 필요 (extras: dl)")

from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from tsad_forge.models._window import window_scores_to_points  # noqa: E402
from tsad_forge.models.registry import get_model  # noqa: E402

RNG = np.random.default_rng(7)
TRAIN = RNG.normal(size=(400, 3))
TEST = RNG.normal(size=(160, 3))
TEST[80:90] += 5.0
SMALL = {"epochs": 2, "window": 16, "batch_size": 64}


def _legacy_fit_score(det, X_train, X_test):
    """옵션 도입 이전 TorchDetector.fit/score의 원문 그대로의 재현."""
    torch.manual_seed(det.seed)
    np.random.seed(det.seed)
    X = det._as_2d(X_train)
    det.n_features_ = X.shape[1]
    det._effective_window = min(det.window, len(X))
    wins = det._windows(X, stride=det.stride)
    if len(wins) > 10000:
        idx = np.random.default_rng(det.seed).choice(len(wins), 10000, replace=False)
        wins = wins[np.sort(idx)]
    det.model_ = det._build(det.n_features_).to(det.device)
    opt = torch.optim.Adam(det.model_.parameters(), lr=det.lr)
    loader = DataLoader(
        TensorDataset(wins),
        batch_size=det.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(det.seed),
    )
    det.model_.train()
    for epoch in range(det.epochs):
        det._epoch_hook(epoch)
        for (batch,) in loader:
            opt.zero_grad()
            loss = det._loss(batch.to(det.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(det.model_.parameters(), 5.0)
            opt.step()
    det._fitted = True
    with torch.no_grad():
        Xt = det._as_2d(X_test)
        tw = det._windows(Xt)
        det.model_.eval()
        out = [
            det._window_scores(tw[i : i + det.batch_size].to(det.device)).cpu().numpy()
            for i in range(0, len(tw), det.batch_size)
        ]
    ws = np.concatenate(out)
    return window_scores_to_points(ws.astype(np.float64), len(Xt), len(Xt) - len(ws) + 1)


@pytest.mark.parametrize("name", ["ae", "usad", "lstm_ad", "anomaly_transformer"])
def test_default_options_reproduce_legacy_loop(name):
    new = get_model(name, seed=0, device="cpu", **SMALL).fit(TRAIN).score(TEST)
    legacy_det = get_model(name, seed=0, device="cpu", **SMALL)
    legacy = _legacy_fit_score(legacy_det, TRAIN, TEST)
    np.testing.assert_array_equal(new, legacy)


def test_default_params_do_not_record_new_options():
    cfg = get_model("ae", seed=0, **SMALL).get_config()
    new_options = {"val_ratio", "patience", "lr_schedule", "amp", "max_train_windows"}
    assert not new_options & set(cfg)


def test_validation_split_is_temporal_and_recorded():
    det = get_model("ae", seed=0, val_ratio=0.25, **SMALL).fit(TRAIN)
    d = det.diagnostics_
    assert d["diag_epochs_run"] == 2
    assert np.isfinite(d["diag_val_loss_best"])
    # 앞 75%(300점)로만 학습 윈도우를 만든다: 300 - 16 + 1
    assert d["diag_n_train_windows"] == 285
    assert det.get_config()["val_ratio"] == 0.25


def test_early_stopping_stops_and_restores_best():
    det = get_model(
        "ae",
        seed=0,
        epochs=50,
        window=16,
        batch_size=64,
        lr=0.5,  # 일부러 불안정한 LR — 검증 손실이 곧 악화된다
        val_ratio=0.3,
        patience=2,
    ).fit(TRAIN)
    assert det.diagnostics_["diag_epochs_run"] < 50
    scores = det.score(TEST)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("schedule", ["cosine", "plateau"])
def test_lr_schedules_run(schedule):
    det = get_model("ae", seed=0, val_ratio=0.2, lr_schedule=schedule, **SMALL).fit(TRAIN)
    assert np.isfinite(det.score(TEST)).all()


def test_option_validation():
    with pytest.raises(ValueError, match="requires val_ratio"):
        get_model("ae", patience=3)
    with pytest.raises(ValueError, match="requires val_ratio"):
        get_model("ae", lr_schedule="plateau")
    with pytest.raises(ValueError, match="unknown lr_schedule"):
        get_model("ae", lr_schedule="step")
    with pytest.raises(ValueError, match="val_ratio"):
        get_model("ae", val_ratio=1.0)
    with pytest.raises(ValueError, match="unknown amp"):
        get_model("ae", amp="int8")


def test_max_train_windows_override():
    det = get_model("ae", seed=0, max_train_windows=50, **SMALL).fit(TRAIN)
    assert det.diagnostics_["diag_n_train_windows"] == 50
    det = get_model("ae", seed=0, max_train_windows=None, **SMALL).fit(TRAIN)
    assert det.diagnostics_["diag_n_train_windows"] == 400 - 16 + 1


def test_amp_is_noop_on_cpu():
    a = get_model("ae", seed=0, device="cpu", amp="bf16", **SMALL).fit(TRAIN).score(TEST)
    b = get_model("ae", seed=0, device="cpu", **SMALL).fit(TRAIN).score(TEST)
    np.testing.assert_array_equal(a, b)


def test_weight_decay_uses_adamw_and_changes_weights():
    a = get_model("ae", seed=0, weight_decay=0.1, **SMALL).fit(TRAIN).score(TEST)
    b = get_model("ae", seed=0, **SMALL).fit(TRAIN).score(TEST)
    assert not np.array_equal(a, b)


def test_mamba_rejects_torchdetector_options():
    # MambaTSAD는 공식 학습 프로토콜을 따른다 — 지원하지 않는 옵션은 조용히 무시하지 않는다
    with pytest.raises(TypeError, match="unsupported options"):
        get_model("mamba_tsad_fixed", seed=0, val_ratio=0.2)


def test_runner_stores_training_diagnostics(tmp_path):
    import json

    from tsad_forge.runner.experiment import run_experiment

    run_experiment(
        {
            "model": "ae",
            "model_params": {"epochs": 1, "window": 16, "val_ratio": 0.2},
            "results_dir": str(tmp_path),
            "device": "cpu",
        },
        verbose=False,
    )
    (js,) = tmp_path.glob("*.json")
    metrics = json.loads(js.read_text())["metrics"]
    assert metrics["diag_epochs_run"] == 1
    assert "diag_val_loss_best" in metrics
