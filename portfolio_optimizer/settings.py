from __future__ import annotations
from pathlib import Path
import yaml


def load_config(
    path: str | Path = Path(__file__).parent.parent / "settings.yml",
) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


cfg = load_config()
