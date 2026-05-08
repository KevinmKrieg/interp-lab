from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path("configs/default.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dirs(config: dict[str, Any]) -> None:
    Path(config["outputs"]["artifacts_dir"]).mkdir(parents=True, exist_ok=True)
    reports = Path(config["outputs"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "assets").mkdir(parents=True, exist_ok=True)
    (reports / "examples").mkdir(parents=True, exist_ok=True)
