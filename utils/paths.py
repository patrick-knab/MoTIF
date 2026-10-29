"""Repository-local data layout shared by the scripts and utilities.

All paths are resolved relative to the repository root (the folder that contains this
``utils/`` package), so the scripts work from any working directory:

    <repo>/Datasets/<DatasetName>/{Data,Video_data,Image_data,...}
    <repo>/Embeddings/Videos/<DatasetName>/<random>_<window>_clip_<backbone>.pkl
    <repo>/Models/checkpoint_<backbone>_<DatasetName>.pkl
    <repo>/concept_extraction_out_batch/<DatasetName>/<run>/<class>/<instance>/...
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "Datasets"
EMBEDDINGS_DIR = REPO_ROOT / "Embeddings"
MODELS_DIR = REPO_ROOT / "Models"
CONCEPT_RUNS_DIR = REPO_ROOT / "concept_extraction_out_batch"

# Keys accepted on the command line -> folder names under Datasets/.
DATASET_FOLDERS = {
    "breakfast": "Breakfast",
    "ucf101": "UCF101",
    "hmdb": "HMDB",
    "hmdb51": "HMDB",
    "ssv2": "Something2",
    "something2": "Something2",
    "jester": "Jester",
}


def dataset_folder(dataset_key: str) -> str:
    """Map a dataset key such as ``hmdb51`` to its folder name under ``Datasets/``."""
    try:
        return DATASET_FOLDERS[dataset_key.lower()]
    except KeyError as error:
        raise ValueError(
            f"Unknown dataset {dataset_key!r}; expected one of {sorted(DATASET_FOLDERS)}"
        ) from error


def dataset_dir(dataset_key: str) -> Path:
    return DATASETS_DIR / dataset_folder(dataset_key)


def video_data_dir(dataset_key: str) -> Path:
    return dataset_dir(dataset_key) / "Video_data"


def image_data_dir(dataset_key: str) -> Path:
    return dataset_dir(dataset_key) / "Image_data"


def embedding_path(dataset_key: str, clip_model: str, window_size: int, random: bool) -> Path:
    """Canonical location of a pickled ``VideoEmbedder`` for one dataset/backbone/window."""
    return (
        EMBEDDINGS_DIR
        / "Videos"
        / dataset_folder(dataset_key)
        / f"{random}_{window_size}_clip_{clip_model}.pkl"
    )


def resolve_repo_path(path_like: str | os.PathLike[str]) -> Path:
    """Resolve a user-supplied path: absolute paths are kept, relative ones are taken
    relative to the current working directory first and the repository root second."""
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (REPO_ROOT / path).resolve()
