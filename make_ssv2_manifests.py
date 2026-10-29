"""Build the ``class_names.json`` and JSONL manifests consumed by ``finetune_pe_l14_ssv2.py``
from the official Something-Something v2 label files.

Inputs (official split files, placed under ``Datasets/Something2/labels/`` by default):

* ``labels.json``: ``{"<template>": "<class index>", ...}``
* ``train.json`` / ``validation.json``: ``[{"id": "74225", "template": "Putting [something] next to [something]", ...}, ...]``

Videos are read from ``Datasets/Something2/Video_data/<id>.mp4`` (created by
``python save_videos.py --dataset Something2``). Class names are the templates with the
``[...]`` placeholders collapsed, exactly as ``utils/motif.py`` names SSv2 classes.

Example::

    python make_ssv2_manifests.py --out-dir pe_l14_ssv2_lora --train-per-class 50 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from utils.paths import DATASETS_DIR, resolve_repo_path


def class_name(template: str) -> str:
    """``"Putting [something] next to [something]"`` -> ``"Putting something next to something"``."""
    return re.sub(r"\[(.*?)\]", r"\1", template).strip()


def load_class_names(labels_dir: Path, train_entries: list[dict]) -> list[str]:
    labels_json = labels_dir / "labels.json"
    if labels_json.is_file():
        mapping = json.loads(labels_json.read_text(encoding="utf-8"))
        ordered = sorted(mapping.items(), key=lambda item: int(item[1]))
        return [class_name(template) for template, _ in ordered]
    print(f"{labels_json} not found; deriving class names from the training templates.")
    return sorted({class_name(entry["template"]) for entry in train_entries})


def build_rows(entries: list[dict], class_index: dict[str, int], video_dir: Path, per_class: int | None,
               rng: random.Random, split: str) -> list[dict]:
    by_class: dict[int, list[dict]] = defaultdict(list)
    missing = 0
    for entry in entries:
        name = class_name(entry["template"])
        if name not in class_index:
            raise KeyError(f"{split}: template {entry['template']!r} is not in class_names")
        video = video_dir / f"{entry['id']}.mp4"
        if not video.is_file():
            missing += 1
            continue
        by_class[class_index[name]].append({"video_path": str(video.resolve()), "class_idx": class_index[name]})
    if missing:
        print(f"{split}: skipped {missing} entries without an mp4 under {video_dir}")
    rows: list[dict] = []
    for idx in sorted(by_class):
        items = sorted(by_class[idx], key=lambda row: row["video_path"])
        if per_class is not None and len(items) > per_class:
            items = rng.sample(items, per_class)
        rows.extend(items)
    rng.shuffle(rows)
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-dir", default=str(DATASETS_DIR / "Something2" / "labels"))
    parser.add_argument("--video-dir", default=str(DATASETS_DIR / "Something2" / "Video_data"))
    parser.add_argument("--out-dir", required=True, help="Folder for class_names.json and the manifests.")
    parser.add_argument("--train-per-class", type=int, default=None,
                        help="Balanced subset: at most N training videos per class (default: all).")
    parser.add_argument("--val-per-class", type=int, default=None,
                        help="At most N validation videos per class (default: all).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    labels_dir = resolve_repo_path(args.labels_dir)
    video_dir = resolve_repo_path(args.video_dir)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_entries = json.loads((labels_dir / "train.json").read_text(encoding="utf-8"))
    val_entries = json.loads((labels_dir / "validation.json").read_text(encoding="utf-8"))
    names = load_class_names(labels_dir, train_entries)
    class_index = {name: i for i, name in enumerate(names)}

    rng = random.Random(args.seed)
    train_rows = build_rows(train_entries, class_index, video_dir, args.train_per_class, rng, "train")
    val_rows = build_rows(val_entries, class_index, video_dir, args.val_per_class, rng, "validation")

    (out_dir / "class_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    write_jsonl(train_rows, out_dir / "train_manifest.jsonl")
    write_jsonl(val_rows, out_dir / "val_manifest.jsonl")
    print(f"Wrote {len(names)} classes, {len(train_rows)} training and {len(val_rows)} validation rows to {out_dir}")


if __name__ == "__main__":
    main()
