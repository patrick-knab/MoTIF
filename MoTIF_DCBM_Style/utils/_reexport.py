from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reexport_public(source_module, target_globals: dict):
    for name in dir(source_module):
        if name.startswith("_"):
            continue
        target_globals[name] = getattr(source_module, name)
