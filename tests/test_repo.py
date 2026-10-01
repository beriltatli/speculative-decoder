import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_python_matches_pin() -> None:
    pinned = (ROOT / ".python-version").read_text().strip()
    assert f"{sys.version_info.major}.{sys.version_info.minor}" == pinned


def test_config_has_seed() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert isinstance(config["seed"], int)
