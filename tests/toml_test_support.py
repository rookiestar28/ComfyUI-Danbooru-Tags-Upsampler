"""Test-only TOML parsing across the declared Python versions."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


def load_toml(path: Path) -> dict[str, Any]:
    # CRITICAL: missing test tooling must fail, not skip metadata/privacy checks
    # on Python 3.10, which has no standard-library tomllib.
    try:
        parser = importlib.import_module("tomllib")
    except ModuleNotFoundError as exc:
        if exc.name != "tomllib":
            raise
        try:
            parser = importlib.import_module("tomli")
        except ModuleNotFoundError as fallback_exc:
            if fallback_exc.name != "tomli":
                raise
            raise RuntimeError(
                'TOML test parser unavailable. Install test tooling in the active '
                'test environment: python -m pip install "tomli==2.4.1"'
            ) from fallback_exc
    return parser.loads(path.read_text(encoding="utf-8"))
