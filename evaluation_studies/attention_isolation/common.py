from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONDITIONS = (
    "channel_reserved",
    "full_attention",
    "scrambled_concepts",
    "class_name_only",
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def checkpoint_path(root: Path, dataset: str, variant: str, seed: int) -> Path:
    safe_dataset = dataset.lower().replace(" ", "_")
    return (
        root
        / "attention_isolation"
        / "results"
        / "models"
        / f"{safe_dataset}_{variant}_seed{seed}.pkl"
    )

