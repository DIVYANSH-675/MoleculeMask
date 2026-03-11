import sys

import pytest

from moleculemask import cli


def test_property_weight_rejects_negative_values(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["moleculemask", "train", "--property-weight", "-1"])
    with pytest.raises(SystemExit):
        cli.main()
    assert "property-weight must" in capsys.readouterr().err


def test_learning_rate_rejects_nonfinite_values(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["moleculemask", "train", "--learning-rate", "inf"])
    with pytest.raises(SystemExit):
        cli.main()
    assert "learning-rate must" in capsys.readouterr().err


def test_batch_size_rejects_empty_batches(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["moleculemask", "train", "--batch-size", "0"])
    with pytest.raises(SystemExit):
        cli.main()
    assert "batch-size must" in capsys.readouterr().err


def test_best_checkpoint_matches_selected_validation_metric(monkeypatch, tmp_path):
    import json

    import torch

    output = tmp_path / "train"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "moleculemask",
            "train",
            "--count",
            "10",
            "--epochs",
            "2",
            "--batch-size",
            "3",
            "--learning-rate",
            ".002",
            "--property-weight",
            ".2",
            "--selection-metric",
            "property_mae",
            "--output",
            str(output),
        ],
    )
    cli.main()
    metrics = json.loads((output / "metrics.json").read_text())
    selection = json.loads((output / "selection.json").read_text())
    assert selection["value"] == min(row["property_mae"] for row in metrics)
    checkpoint = torch.load(output / "best.pt", weights_only=True)
    assert checkpoint["epoch"] == selection["epoch"]
    assert (output / "model.pt").exists()
