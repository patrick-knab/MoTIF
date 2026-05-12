#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

TOOLKIT_DIR = Path(__file__).resolve().parent
AGENTMOTIF_DIR = TOOLKIT_DIR.parent
if str(AGENTMOTIF_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTMOTIF_DIR))

from ablation_toolkit.utils.cbm_concept_extraction_utils import (
    gather_instances,
    load_frames,
    propose_concepts_with_vlm,
    select_windows,
    write_json,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract CBM concepts per window from Image_data folders."
    )
    p.add_argument("--image_data_dir", type=str, required=True)
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--instances_per_class", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle_instances", action="store_true")
    p.add_argument("--class_filter", type=str, default=None)
    p.add_argument("--skip_existing", action="store_true")

    p.add_argument("--llm_server_url", type=str, required=True)
    p.add_argument("--llm_model", type=str, required=True)
    p.add_argument("--llm_api_key", type=str, default=None)
    p.add_argument("--llm_max_tokens", type=int, default=3500)
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--ssv2_labels_json", type=str, default=None)

    p.add_argument("--window_size", type=int, default=60)
    p.add_argument("--max_windows", type=int, default=8)
    p.add_argument("--max_window_overlap", type=float, default=1.0)
    p.add_argument("--max_total_frames_to_scan_for_motion", type=int, default=1200)
    p.add_argument("--max_concepts", type=int, default=10)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    image_data_dir = os.path.abspath(args.image_data_dir)
    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    def send_generate_request(messages, max_tokens=None):
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=args.llm_api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
                base_url=args.llm_server_url,
            )
            resp = client.chat.completions.create(
                model=args.llm_model,
                messages=messages,
                max_completion_tokens=int(max_tokens) if max_tokens is not None else int(args.llm_max_tokens),
            )
            return resp.choices[0].message.content if resp.choices else None
        except Exception as e:
            print(f"[ERROR] LLM request failed: {e}")
            return None

    inst_map = gather_instances(
        image_data_dir,
        dataset_name=args.dataset_name,
        seed=int(args.seed),
        max_per_parent=int(args.instances_per_class),
        ssv2_labels_json=args.ssv2_labels_json,
    )
    classes = sorted(inst_map.keys())
    if args.class_filter:
        classes = [c for c in classes if c == args.class_filter]
    if not classes:
        print(f"No class folders found in {image_data_dir}", file=sys.stderr)
        return 2

    rnd = random.Random(int(args.seed))
    print(f"Found {len(classes)} class(es) under {image_data_dir}")

    for class_name in classes:
        instances = inst_map.get(class_name, [])
        if not instances:
            print(f"[WARN] no instances found for class '{class_name}'")
            continue

        if args.shuffle_instances:
            rnd.shuffle(instances)
        instances = instances[: max(0, int(args.instances_per_class))]
        print(f"\nClass '{class_name}': processing {len(instances)} instance(s)")

        for instance_dir in instances:
            instance_dir = os.path.abspath(instance_dir)
            instance_name = os.path.basename(instance_dir.rstrip("/"))

            out_dir = os.path.join(output_root, class_name, instance_name)
            windows_index_path = os.path.join(out_dir, "windows_index.json")
            windows_root = os.path.join(out_dir, "windows")

            if args.skip_existing and os.path.exists(windows_index_path):
                print(f"  - skip (exists): {windows_index_path}")
                continue

            all_frames = load_frames(instance_dir)
            if not all_frames:
                print(f"  - skip (no frames): {instance_dir}")
                continue

            os.makedirs(windows_root, exist_ok=True)
            print(f"  - instance: {instance_name} ({len(all_frames)} frames)")

            win_info = select_windows(
                all_frames=all_frames,
                window_size=int(args.window_size),
                max_windows=int(args.max_windows),
                max_total_frames_to_scan_for_motion=int(args.max_total_frames_to_scan_for_motion),
                max_window_overlap=float(args.max_window_overlap),
            )
            windows = win_info.get("windows") or []

            existing_windows_index = None
            if os.path.exists(windows_index_path):
                try:
                    import json

                    with open(windows_index_path, "r", encoding="utf-8") as f:
                        existing_windows_index = json.load(f)
                    print(f"  - resuming from existing: {windows_index_path}")
                except Exception as e:
                    print(f"  - warning: could not load existing windows_index.json: {e}")

            window_summaries: List[Dict[str, Any]] = []

            for wi, (s, e) in enumerate(windows):
                window_frames = all_frames[int(s) : int(e)]
                if not window_frames:
                    continue

                first_frame = window_frames[0]
                rep_idxs = win_info.get("rep_idxs") or []
                rep_frames = [all_frames[i] for i in rep_idxs] if rep_idxs else [first_frame]

                window_dir = os.path.join(windows_root, f"window_w{wi:02d}_s{int(s):06d}_e{int(e):06d}")
                os.makedirs(window_dir, exist_ok=True)

                vlm_reasoning_path = os.path.join(window_dir, "vlm_reasoning.json")
                concepts_json_path = os.path.join(window_dir, "concepts.json")

                if args.skip_existing and os.path.exists(concepts_json_path):
                    print(f"    - skip window {wi} (exists): {concepts_json_path}")
                    window_summaries.append(
                        {
                            "window_idx": int(wi),
                            "start": int(s),
                            "end": int(e),
                            "window_dir": window_dir,
                            "vlm_reasoning_path": vlm_reasoning_path,
                            "concepts_json_path": concepts_json_path,
                        }
                    )
                    continue

                concepts, temporal_concepts, action_concepts, reasoning = propose_concepts_with_vlm(
                    send_generate_request=send_generate_request,
                    rep_frames=rep_frames,
                    max_concepts=int(args.max_concepts),
                    action_class=class_name,
                    dataset_name=args.dataset_name,
                )

                write_json(
                    vlm_reasoning_path,
                    {
                        "class_name": class_name,
                        "instance_name": instance_name,
                        "instance_dir": instance_dir,
                        "window": {"window_idx": wi, "start": int(s), "end": int(e), "num_frames": len(window_frames)},
                        "first_frame": first_frame,
                        "rep_frames": rep_frames,
                        "window_selection": win_info,
                        "vlm": {
                            "server_url": args.llm_server_url,
                            "model": args.llm_model,
                            "max_tokens": int(args.llm_max_tokens),
                        },
                        **reasoning,
                    },
                )

                write_json(
                    concepts_json_path,
                    {
                        "class_name": class_name,
                        "instance_name": instance_name,
                        "instance_dir": instance_dir,
                        "window": {"window_idx": wi, "start": int(s), "end": int(e), "num_frames": len(window_frames)},
                        "first_frame": first_frame,
                        "concepts": concepts,
                        "temporal_concepts": temporal_concepts,
                        "action_concepts": action_concepts,
                        "params": {
                            "window_size": int(args.window_size),
                            "max_windows": int(args.max_windows),
                            "max_window_overlap": float(args.max_window_overlap),
                            "max_concepts": int(args.max_concepts),
                        },
                    },
                )

                window_summaries.append(
                    {
                        "window_idx": int(wi),
                        "start": int(s),
                        "end": int(e),
                        "window_dir": window_dir,
                        "vlm_reasoning_path": vlm_reasoning_path,
                        "concepts_json_path": concepts_json_path,
                    }
                )

            final_windows_index = {
                "class_name": class_name,
                "instance_name": instance_name,
                "instance_dir": instance_dir,
                "num_frames": len(all_frames),
                "window_selection": win_info,
                "windows": window_summaries,
            }
            if existing_windows_index:
                existing_windows_dict = {w["window_idx"]: w for w in existing_windows_index.get("windows", [])}
                new_windows_dict = {w["window_idx"]: w for w in window_summaries}
                existing_windows_dict.update(new_windows_dict)
                final_windows_index["windows"] = [
                    existing_windows_dict[i] for i in sorted(existing_windows_dict.keys())
                ]

            write_json(windows_index_path, final_windows_index)

    print("\nDONE")
    print(f"Outputs written under: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
