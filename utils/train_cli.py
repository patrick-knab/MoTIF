"""Command-line overrides shared by ``train_MoTIF.py`` and ``train_MoTIF-ST.py``.

The training scripts keep an in-file ``search_space`` grid. This module lets the most
common settings be overridden from the command line without editing the file, e.g.::

    python train_MoTIF.py --dataset hmdb51 --clip-model pe-l14 \
        --agent-run-folder concept_extraction_out_batch/HMDB/8B

    python train_MoTIF.py --dataset hmdb51 --clip-model pe-l14 --manual-concepts
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

from utils.paths import CONCEPT_RUNS_DIR, REPO_ROOT, resolve_repo_path

DEFAULT_AGENT_RUN_FOLDER = os.path.join("concept_extraction_out_batch", "HMDB", "8B")

EXTRACTION_HINT = (
    "No VLM concept run was found at {concept_dir}.\n"
    "Either\n"
    "  (a) create one with the agentic extraction pipeline, e.g.\n"
    "        python utils/extract_cbm_concepts_from_image_data_windows.py \\\n"
    "          --image_data_dir Datasets/HMDB/Image_data \\\n"
    "          --output_root concept_extraction_out_batch/HMDB/8B \\\n"
    "          --dataset_name hmdb --instances_per_class 5 --shuffle_instances --seed 0 \\\n"
    "          --llm_server_url http://127.0.0.1:8001/v1 --llm_model Qwen/Qwen3-VL-8B-Thinking \\\n"
    "          --llm_api_key EMPTY --window_size 30 --max_windows 8 --max_window_overlap 0.25 \\\n"
    "          --max_concepts 6\n"
    "      (see slurm_cbm_extract_windows.sbatch) and pass its --output_root as --agent-run-folder, or\n"
    "  (b) run with --manual-concepts to use the built-in concept lists from utils/manual_concepts.py."
)


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["breakfast", "ucf101", "hmdb51", "something2"],
        default=None,
        help="Dataset key (overrides the in-file search space).",
    )
    parser.add_argument(
        "--clip-model",
        default=None,
        help="Backbone key, e.g. pe-l14, pe-g14, l14, b32, b16, res50, clip4clip, siglip, siglipl14.",
    )
    parser.add_argument("--test-split", default=None, help="Test split identifier, e.g. s1, s2, s3.")
    parser.add_argument("--window-size", type=int, default=None, help="Temporal window size used for the embeddings.")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    concept_group = parser.add_mutually_exclusive_group()
    concept_group.add_argument(
        "--agent-run-folder",
        default=None,
        help=(
            "Folder with VLM-extracted concepts (walked recursively for concepts.json). "
            f"Default: {DEFAULT_AGENT_RUN_FOLDER}, the --output_root of slurm_cbm_extract_windows.sbatch."
        ),
    )
    concept_group.add_argument(
        "--manual-concepts",
        action="store_true",
        help="Use the built-in manual concept lists (utils/manual_concepts.py) instead of a VLM concept run.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Cosine-similarity threshold used to drop near-duplicate concepts.",
    )
    parser.add_argument(
        "--embedding-path",
        default=None,
        help=(
            "Explicit path to a pickled VideoEmbedder (e.g. a file downloaded from Hugging Face). "
            "Overrides the canonical Embeddings/Videos/<Dataset>/<random>_<window>_clip_<backbone>.pkl location."
        ),
    )
    return parser


def apply_overrides(search_space: Dict[str, List[Any]], args: argparse.Namespace) -> Dict[str, List[Any]]:
    """Return a copy of ``search_space`` with the command-line overrides applied."""
    space = {key: list(values) for key, values in search_space.items()}
    simple = {
        "dataset": args.dataset,
        "clip_model": args.clip_model,
        "test_split": args.test_split,
        "window_size": args.window_size,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "similarity_threshold": args.similarity_threshold,
    }
    for key, value in simple.items():
        if value is not None:
            space[key] = [value]
    if args.manual_concepts:
        space["agentic-concepts"] = [False]
    elif args.agent_run_folder is not None:
        space["agentic-concepts"] = [True]
        space["agent_run_folder"] = [args.agent_run_folder]
    if args.embedding_path is not None:
        space["embedding_path"] = [args.embedding_path]
    return space


def resolve_concept_dir(agent_run_folder: str) -> str:
    """Resolve the concept run folder and fail with actionable instructions if it is missing."""
    concept_dir = resolve_repo_path(agent_run_folder)
    if not concept_dir.is_dir():
        raise FileNotFoundError(EXTRACTION_HINT.format(concept_dir=concept_dir))
    has_json = any(
        "concepts.json" in files for _, _, files in os.walk(concept_dir)
    )
    if not has_json:
        raise FileNotFoundError(
            f"{concept_dir} exists but contains no concepts.json files.\n" + EXTRACTION_HINT.format(concept_dir=concept_dir)
        )
    return str(concept_dir)


__all__ = [
    "CONCEPT_RUNS_DIR",
    "DEFAULT_AGENT_RUN_FOLDER",
    "REPO_ROOT",
    "apply_overrides",
    "build_parser",
    "resolve_concept_dir",
]
