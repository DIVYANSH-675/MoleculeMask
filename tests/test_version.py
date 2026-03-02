import subprocess
import sys

from moleculemask import __version__


def test_version_does_not_require_a_command():
    result = subprocess.run(
        [sys.executable, "-m", "moleculemask", "--version"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "moleculemask " + __version__
