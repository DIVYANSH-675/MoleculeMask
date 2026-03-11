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
