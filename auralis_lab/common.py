from __future__ import annotations

import importlib
from pathlib import Path


AUDIO_SAMPLE_RATE = 16_000


def require_module(module_name: str, install_hint: str) -> object:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency: {module_name}\n"
            f"Original import error: {exc}\n"
            f"Install it with:\n  {install_hint}"
        ) from exc


def ensure_parent(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def require_file(path: str | Path) -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise SystemExit(f"File not found: {file_path}")
    return file_path
