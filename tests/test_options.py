import sys

import pytest

from moleculemask import cli


def test_property_weight_rejects_negative_values(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["moleculemask", "train", "--property-weight", "-1"])
    with pytest.raises(SystemExit):
        cli.main()
    assert "property-weight must" in capsys.readouterr().err
