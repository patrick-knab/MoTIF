#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


SUPPORTED_ABLATION_STAGES = [
    "json_only",
    "json+action",
    "json+action+visual",
    "json+action+visual+motion",
    "action_only",
    "visual_only",
]

DATASET_CONFIG = {
    "breakfast": {
        "dataset_name_arg": "breakfast",
        "dataset_folder": "Breakfast",
        "concept_subdir": "breakfast",
        "default_window_size": 80,
    },
    "hmdb51": {
        "dataset_name_arg": "hmdb",
        "dataset_folder": "HMDB",
        "concept_subdir": "hmdb",
        "default_window_size": 60,
    },
    "ucf101": {
        "dataset_name_arg": "ucf101",
        "dataset_folder": "UCF101",
        "concept_subdir": "ucf101",
        "default_window_size": 60,
    },
    "something2": {
        "dataset_name_arg": "ssv2",
        "dataset_folder": "Something2",
        "concept_subdir": "ssv2",
        "default_window_size": 60,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generic concept-extraction launcher for AgentMoTIF ablation runs."
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIG), required=True)
    parser.add_argument("--image-data-dir", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--run-folder", type=str, default=None)
    parser.add_argument("--instances-per-class", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-instances", action="store_true")
    parser.add_argument("--class-filter", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true")

    parser.add_argument("--llm-server-url", type=str, required=True)
    parser.add_argument("--llm-model", type=str, required=True)
    parser.add_argument("--llm-api-key", type=str, default=None)
    parser.add_argument("--llm-max-tokens", type=int, default=3500)

    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--max-window-overlap", type=float, default=1.0)
    parser.add_argument("--max-total-frames-to-scan-for-motion", type=int, default=1200)
    parser.add_argument("--max-concepts", type=int, default=10)
    parser.add_argument(
        "--ablation-stage",
        nargs="+",
        default=["all"],
        help="One or more stages, or 'all'. Stored as metadata for downstream runs.",
    )
    return parser.parse_args()


def normalize_stages(raw_stages: list[str]) -> list[str]:
    if raw_stages == ["all"] or "all" in raw_stages:
        return list(SUPPORTED_ABLATION_STAGES)

    invalid = sorted(set(raw_stages) - set(SUPPORTED_ABLATION_STAGES))
    if invalid:
        raise ValueError(
            f"Unsupported ablation stage(s): {invalid}. "
            f"Expected subset of {SUPPORTED_ABLATION_STAGES} or 'all'."
        )
    return raw_stages


def default_run_folder(llm_model: str, instances_per_class: int, seed: int, window_size: int) -> str:
    safe_model = llm_model.replace("/", "-")
    return f"vlm-{safe_model}_ipc{instances_per_class}_seed{seed}_ws{window_size}"


def main() -> int:
    args = parse_args()
    stages = normalize_stages(args.ablation_stage)

    toolkit_dir = Path(__file__).resolve().parent
    agentmotif_dir = toolkit_dir.parent
    videcbm_dir = agentmotif_dir.parent
    extractor_script = toolkit_dir / "extract_cbm_concepts_from_image_data_windows.py"

    cfg = DATASET_CONFIG[args.dataset]
    window_size = args.window_size or int(cfg["default_window_size"])

    image_data_dir = Path(args.image_data_dir) if args.image_data_dir else (
        videcbm_dir / "Datasets" / cfg["dataset_folder"] / "Image_data"
    )
    output_root = Path(args.output_root) if args.output_root else (
        videcbm_dir / "concept_extraction_out_batch" / cfg["concept_subdir"]
    )
    run_folder = args.run_folder or default_run_folder(
        llm_model=args.llm_model,
        instances_per_class=args.instances_per_class,
        seed=args.seed,
        window_size=window_size,
    )
    run_output_dir = output_root / run_folder
    run_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(extractor_script),
        "--image_data_dir",
        str(image_data_dir),
        "--output_root",
        str(run_output_dir),
        "--instances_per_class",
        str(args.instances_per_class),
        "--seed",
        str(args.seed),
        "--llm_server_url",
        args.llm_server_url,
        "--llm_model",
        args.llm_model,
        "--llm_max_tokens",
        str(args.llm_max_tokens),
        "--dataset_name",
        cfg["dataset_name_arg"],
        "--window_size",
        str(window_size),
        "--max_windows",
        str(args.max_windows),
        "--max_window_overlap",
        str(args.max_window_overlap),
        "--max_total_frames_to_scan_for_motion",
        str(args.max_total_frames_to_scan_for_motion),
        "--max_concepts",
        str(args.max_concepts),
    ]

    if args.llm_api_key:
        cmd.extend(["--llm_api_key", args.llm_api_key])
    if args.shuffle_instances:
        cmd.append("--shuffle_instances")
    if args.class_filter:
        cmd.extend(["--class_filter", args.class_filter])
    if args.skip_existing:
        cmd.append("--skip_existing")
    if args.dataset == "something2":
        cmd.extend(
            [
                "--ssv2_labels_json",
                str(videcbm_dir / "Datasets" / "Something2" / "labels" / "labels.json"),
            ]
        )

    print("Running extractor:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(toolkit_dir), check=True)

    manifest = {
        "dataset": args.dataset,
        "dataset_name_arg": cfg["dataset_name_arg"],
        "image_data_dir": str(image_data_dir.resolve()),
        "output_root": str(output_root.resolve()),
        "run_folder": run_folder,
        "run_output_dir": str(run_output_dir.resolve()),
        "ablation_stages": stages,
        "llm": {
            "model": args.llm_model,
            "server_url": args.llm_server_url,
            "max_tokens": args.llm_max_tokens,
        },
        "extraction": {
            "instances_per_class": args.instances_per_class,
            "seed": args.seed,
            "window_size": window_size,
            "max_windows": args.max_windows,
            "max_window_overlap": args.max_window_overlap,
            "max_total_frames_to_scan_for_motion": args.max_total_frames_to_scan_for_motion,
            "max_concepts": args.max_concepts,
            "shuffle_instances": args.shuffle_instances,
            "class_filter": args.class_filter,
            "skip_existing": args.skip_existing,
        },
        "note": (
            "The extractor writes one reusable concept bundle per window. "
            "These ablation stages are selected later by the training runner."
        ),
    }
    manifest_path = run_output_dir / "ablation_stage_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote ablation manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
