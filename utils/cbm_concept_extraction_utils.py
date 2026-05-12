"""
Reusable utilities for CBM concept extraction from frame folders.

Intended to be imported by notebooks and scripts.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image as PILImage
from PIL import ImageChops, ImageOps
import base64
import io

from openai import OpenAI


IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# Terms that suggest the string is describing motion rather than an object noun.
_MOTION_WORDS = {
    "motion",
    "movement",
    "moving",
    "action",
    "open",
    "opening",
    "close",
    "closing",
    "chew",
    "chewing",
    "clench",
    "tilt",
    "lift",
    "push",
    "pull",
    "swing",
    "wave",
    "turn",
    "rotate",
}


def safe_slug(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_.-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unnamed"


def clean_concept_list(xs: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    if not isinstance(xs, list):
        return out
    for c in xs:
        if not isinstance(c, str):
            continue
        c = c.strip().lower()
        c = re.sub(r"\s+", " ", c)
        c = re.sub(r"[^a-z0-9\s-]", "", c)
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def _filter_object_concepts(concepts: List[str]) -> List[str]:
    """Keep only noun-like object concepts by dropping entries that look motion-y."""
    kept: List[str] = []
    for c in concepts:
        tokens = set(c.split())
        if tokens & _MOTION_WORDS:
            continue
        kept.append(c)
    return kept


def extract_first_json_object(raw: Optional[str]) -> Tuple[Optional[dict], Optional[str]]:
    """
    Best-effort JSON object extraction.

    Handles common VLM formatting issues:
    - extra text before/after JSON
    - `<think>...</think>` blocks
    - JSON rendered with backslash-escaped quotes: {\"a\": 1}
    - duplicated JSON objects in the response
    - double-encoded JSON strings (VLM returns a JSON string that itself contains JSON)
    """
    if raw is None:
        return None, "raw is None"

    # If the model already returned a dict, pass it through.
    if isinstance(raw, dict):
        return raw, None

    raw_text = str(raw)

    # Strip common thinking tags to reduce brace noise.
    cleaned = re.sub(r"<(think|thought)>[\s\S]*?</\1>", "", raw_text)

    # Helper: attempt to parse a string into a dict, with unescape fallback.
    def _try_parse(s: str):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj, None
            if isinstance(obj, str):
                # Sometimes the model returns a JSON string containing JSON.
                return _try_parse(obj)
        except Exception as e:
            return None, f"json.loads failed: {e}"
        return None, "parsed value not a dict"

    # 0) direct attempt on the whole cleaned string (covers perfectly formatted replies)
    obj, err = _try_parse(cleaned)
    if obj is not None:
        return obj, None

    # 1) find candidate JSON objects (non-greedy), then try to parse each.
    matches = list(re.finditer(r"\{[\s\S]*?\}", cleaned))

    # 2) if still nothing, try to unescape common \" patterns and search again.
    if not matches and ('\\"' in cleaned or cleaned.startswith('{\\')):
        cleaned_unescaped = cleaned.replace('\\"', '"')
        matches = list(re.finditer(r"\{[\s\S]*?\}", cleaned_unescaped))
        cleaned = cleaned_unescaped

    last_err = err
    for m in matches:
        s = m.group(0)
        # direct parse
        obj, err2 = _try_parse(s)
        if obj is not None:
            return obj, None

        # handle {\"k\": ...} style inside the match
        if '\\"' in s or s.startswith('{\\'):
            s2 = s.replace('\\"', '"')
            obj, err3 = _try_parse(s2)
            if obj is not None:
                return obj, None
            last_err = err3
        else:
            last_err = err2

    return None, last_err or "json parse failed"


def _fallback_concepts_for_action_class(action_class: Optional[str]) -> Tuple[List[str], List[str]]:
    """
    Class-aware fallback so we don't always default to generic (person/legs/hands/floor/wall).
    Returns (concepts, temporal_concepts).
    """
    if not action_class:
        return (["person", "legs", "hands", "scene", "object"], ["person", "legs", "hands"])

    c = action_class.strip().lower()
    if c == "climb_stairs":
        concepts = [
            "stairs",
            "steps",
            "staircase",
            "handrail",
            "step edge",
            "feet",
            "legs",
            "shoes",
            "stair riser",
            "tread",
        ]
        temporal = ["feet", "legs", "steps"]
        return concepts, temporal

    # Generic but less useless than floor/wall.
    return (["person", "hands", "feet", "legs", "main object"], ["hands", "feet", "legs"])


def load_frames(instance_dir: str) -> List[str]:
    frames: List[str] = []
    for ext in IMG_EXTS:
        frames.extend(glob.glob(os.path.join(instance_dir, f"*{ext}")))
        frames.extend(glob.glob(os.path.join(instance_dir, f"*{ext.upper()}")))
    return sorted(set(frames))


def _dir_has_images(p: Path) -> bool:
    """True if the directory contains at least one image with an allowed extension."""
    if not p.is_dir():
        return False
    try:
        for f in os.listdir(p):
            fp = p / f
            if fp.is_file() and fp.suffix.lower() in IMG_EXTS:
                return True
    except FileNotFoundError:
        return False
    return False


def _gather_instances_default(image_data_dir: str) -> Dict[str, List[str]]:
    """
    HMDB / UCF-style layout:
      Image_data/<class>/<instance>/*.jpg
    Returns a mapping: class_name -> list[instance_dir_str]
    """
    out: Dict[str, List[str]] = {}
    root = Path(image_data_dir)
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        insts: List[str] = []
        for inst_dir in sorted(p for p in cls_dir.iterdir() if p.is_dir()):
            if _dir_has_images(inst_dir):
                insts.append(str(inst_dir))
        if insts:
            out[cls_dir.name] = insts
    return out


def _gather_instances_breakfast(
    image_data_dir: str,
    *,
    seed: int = 0,
    max_per_parent: int = 10,
    preferred_cams: Optional[Sequence[str]] = None,
) -> Dict[str, List[str]]:
    """
    Breakfast layout:
      Image_data/Pxx/cam01/<seq>/frame.jpg
    - Uses preferred_cams order if present; otherwise falls back to any cam folder.
    - Derives class name from sequence folder after the first underscore (P03_cereals -> cereals).
      Drops common channel suffixes (e.g., cereals_ch1 -> cereals) so class labels align with the
      Breakfast taxonomy.
    - Samples up to `max_per_parent` sequences per participant for balance.
    """
    cams = list(preferred_cams) if preferred_cams else ["cam01", "webcam01", "webcam02", "stereo"]
    out: Dict[str, List[str]] = {}
    rnd = random.Random(int(seed))
    root = Path(image_data_dir)
    for participant in sorted(p for p in root.iterdir() if p.is_dir()):
        cam_dirs = [participant / c for c in cams if (participant / c).is_dir()]
        if not cam_dirs:
            cam_dirs = [d for d in participant.iterdir() if d.is_dir()]
        seqs: List[Path] = []
        for cam in cam_dirs:
            for seq_dir in sorted(p for p in cam.iterdir() if p.is_dir()):
                if _dir_has_images(seq_dir):
                    seqs.append(seq_dir)
        if not seqs:
            continue
        rnd.shuffle(seqs)
        seqs = seqs[: max(0, int(max_per_parent))]
        for seq_dir in seqs:
            seq_name = seq_dir.name
            class_name = seq_name.split("_", 1)[1] if "_" in seq_name else seq_name
            # Normalize channel-suffixed labels: cereals_ch0, cereals_ch1 -> cereals
            class_name = re.sub(r"_ch\d+$", "", class_name)
            out.setdefault(class_name, []).append(str(seq_dir))
    return out


def _gather_instances_ssv2(
    image_data_dir: str,
    *,
    labels_json: Optional[str],
    seed: int = 0,
    max_per_class: int = 10,
) -> Dict[str, List[str]]:
    """
    Something-Something V2 layout:
      frames are under image_data_dir/<video_id>/frame.jpg
    Class labels are provided in a labels JSON (train/val) rather than folder names.
    We map class -> instance_dirs by reading the labels file.
    """
    if not labels_json:
        raise ValueError("labels_json is required for ssv2")
    labels_path = Path(labels_json)
    if not labels_path.exists():
        raise FileNotFoundError(f"ssv2 labels_json not found: {labels_json}")
    with open(labels_path, "r") as f:
        data = json.load(f)
    # Expect a list of dict entries with id/video_id and label/template/class fields.
    out: Dict[str, List[str]] = {}
    rnd = random.Random(int(seed))
    for entry in data if isinstance(data, list) else []:
        vid = (
            entry.get("id")
            or entry.get("video_id")
            or entry.get("video")
            or entry.get("uid")
        )
        # Prefer the template form (with placeholders) to reduce the number of distinct labels.
        label_raw = entry.get("template") or entry.get("label") or entry.get("class")
        if vid is None or label_raw is None:
            continue
        vid_str = str(vid)
        # Collapse any object/person placeholders like "[something]" to a generic token.
        label_template = re.sub(r"\[[^\]]+\]", "something", str(label_raw))
        cls = safe_slug(label_template)
        inst_dir = Path(image_data_dir) / vid_str
        if not _dir_has_images(inst_dir):
            continue
        out.setdefault(cls, []).append(str(inst_dir))
    # Shuffle and cap per class
    for cls, insts in out.items():
        rnd.shuffle(insts)
        out[cls] = insts[: max(0, int(max_per_class))]
    return out


def gather_instances(
    image_data_dir: str,
    *,
    dataset_name: Optional[str] = None,
    seed: int = 0,
    max_per_parent: int = 10,
    preferred_cams: Optional[Sequence[str]] = None,
    ssv2_labels_json: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Dataset-aware instance discovery for Image_data folders.
    Supports:
      - Default HMDB/UCF layout: Image_data/<class>/<instance>/*.jpg
      - Breakfast layout: Image_data/Pxx/cam01/<seq>/*.jpg
      - Something-Something V2: image_data_dir/<video_id>/*.jpg with class labels in a labels JSON.
    Returns mapping of class_name -> list of instance directory strings.
    """
    dn = (dataset_name or "").strip().lower()
    if dn == "breakfast":
        return _gather_instances_breakfast(
            image_data_dir, seed=seed, max_per_parent=max_per_parent, preferred_cams=preferred_cams
        )
    if dn in {"ssv2", "something-something", "something-something-v2", "something2"}:
        return _gather_instances_ssv2(
            image_data_dir, labels_json=ssv2_labels_json, seed=seed, max_per_class=max_per_parent
        )
    return _gather_instances_default(image_data_dir)


def dhash(img: PILImage.Image, hash_size: int = 8) -> int:
    im = img.convert("L").resize((hash_size + 1, hash_size), PILImage.BILINEAR)
    px = list(im.getdata())
    h = 0
    bit = 0
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            left = px[row_start + col]
            right = px[row_start + col + 1]
            if left > right:
                h |= 1 << bit
            bit += 1
    return h


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def box_xywh_norm_to_xyxy_px(box_xywh_norm, w: int, h: int) -> Tuple[int, int, int, int]:
    x, y, bw, bh = box_xywh_norm
    x0 = int(max(0, min(w - 1, round(x * w))))
    y0 = int(max(0, min(h - 1, round(y * h))))
    x1 = int(max(0, min(w, round((x + bw) * w))))
    y1 = int(max(0, min(h, round((y + bh) * h))))
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    return (x0, y0, x1, y1)


def select_windows(
    all_frames: Sequence[str],
    window_size: int,
    max_windows: int,
    max_total_frames_to_scan_for_motion: int,
    max_window_overlap: float = 1.0,
    motion_peak_threshold: float = 0.02,
    motion_top_k: int = 8,
) -> Dict[str, Any]:
    """Return dict with windows and diagnostics; covers video via motion peaks + uniform coverage."""
    N = len(all_frames)
    if N == 0:
        return {"windows": [], "scan_stride": 1, "peak_idxs": [], "rep_idxs": []}

    if not (0.0 <= float(max_window_overlap) <= 1.0):
        raise ValueError("max_window_overlap must be in [0, 1]")

    scan_stride = max(1, int(math.ceil(N / max_total_frames_to_scan_for_motion)))
    scan_idxs = list(range(0, N, scan_stride))

    prev = None
    motion_scores: List[Tuple[int, float]] = []
    for idx in scan_idxs:
        fp = all_frames[idx]
        im = PILImage.open(fp).convert("L").resize((96, 96))
        if prev is None:
            prev = im
            motion_scores.append((idx, 0.0))
            continue
        diff = ImageChops.difference(im, prev)
        score = sum(diff.getdata()) / (96 * 96 * 255.0)
        motion_scores.append((idx, float(score)))
        prev = im

    motion_scores_sorted = sorted(motion_scores, key=lambda t: t[1], reverse=True)
    top_k = min(motion_top_k, len(motion_scores_sorted))
    peak_idxs = [i for i, s in motion_scores_sorted[:top_k] if s > motion_peak_threshold]

    starts = set()
    for i in peak_idxs:
        s = max(0, min(max(0, N - window_size), i - window_size // 2))
        starts.add(int(s))

    if N <= window_size:
        starts.add(0)
    else:
        for j in range(max_windows):
            s = int(round(j * (N - window_size) / max(1, max_windows - 1)))
            starts.add(s)

    window_starts = sorted(starts)
    if len(window_starts) > max_windows:
        step = len(window_starts) / max_windows
        window_starts = [window_starts[int(k * step)] for k in range(max_windows)]

    # Enforce a maximum overlap between consecutive windows.
    # Overlap is approximated via start distance: overlap_frames ≈ window_size - (start_gap).
    # max_window_overlap is a fraction of window_size allowed to overlap (1.0 disables pruning).
    pruned_starts = []
    if window_size <= 0:
        pruned_starts = list(window_starts)
    else:
        min_start_gap = int(math.ceil(window_size * (1.0 - float(max_window_overlap))))
        # If overlap is allowed to be 100%, min_start_gap becomes 0 => no pruning.
        if min_start_gap <= 0:
            pruned_starts = list(window_starts)
        else:
            last_kept = None
            for s in window_starts:
                if last_kept is None or (int(s) - int(last_kept)) >= min_start_gap:
                    pruned_starts.append(int(s))
                    last_kept = int(s)
            # Always keep at least one window if candidates exist
            if not pruned_starts and window_starts:
                pruned_starts = [int(window_starts[0])]

    window_starts = pruned_starts

    windows = []
    for s in window_starts:
        e = min(N, s + window_size)
        windows.append((int(s), int(e)))

    rep_quantiles = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # Cover both uniform spread and top motion peaks so the VLM sees diverse + high-motion frames.
    rep_idxs = sorted(set([min(N - 1, int(q * (N - 1))) for q in rep_quantiles] + peak_idxs))

    return {
        "windows": windows,
        "scan_stride": scan_stride,
        "peak_idxs": peak_idxs,
        "rep_idxs": rep_idxs,
        "max_window_overlap": float(max_window_overlap),
        "window_starts_pruned": window_starts,
    }


def propose_concepts_with_vlm(
    *,
    send_generate_request,
    rep_frames: Sequence[str],
    max_concepts: int,
    action_class: Optional[str] = None,
    dataset_name: Optional[str] = None,
    fallback_concepts: Optional[List[str]] = None,
    fallback_temporal: Optional[List[str]] = None,
    max_retries: int = 2,
) -> Tuple[List[str], List[str], List[str], Dict[str, Any]]:
    class_hint = (
        f" The video is from the action class '{action_class}', so prioritize concepts that are specific/discriminative for this class."
        if isinstance(action_class, str) and action_class.strip()
        else ""
    )
    dataset_hint = (
        f" The dataset is '{dataset_name}', so align with its action taxonomy and typical visual content."
        if isinstance(dataset_name, str) and dataset_name.strip()
        else ""
    )
    vlm_query = (
        "You are helping build a Concept Bottleneck Model (CBM) for video action recognition."
        + class_hint
        + dataset_hint
        + "\n\nYou are given several frames sampled across a full video. "
        "Your job is to propose human-interpretable concepts that are useful to classify the video, "
        "with a strong bias toward concepts that distinguish this class from other HMDB actions.\n\n"
        "Return ONLY valid JSON with schema: "
        '{"concepts": [...], "temporal_concepts": [...], "action_concepts": [...]}.\n\n'
        "Hard requirements:\n"
        "- Provide EXACTLY "
        + str(max_concepts)
        + " entries in \"concepts\" (if unsure, make your best guess).\n"
        "- \"concepts\" MUST be segmentable objects/body parts/props (short noun phrases, lowercase, 1-2 words). "
        "They must be clearly visible in the PROVIDED FRAMES (no speculative/occluded items). If a candidate is not visible, do NOT include it. "
        "No verbs, no gerunds, no words like motion/movement/action/opening/closing/chewing; make them objects that can be segmented.\n"
        "- At least 60% of \"concepts\" must be class-specific/discriminative given the frames and the class label.\n"
        "- Include at least: 2 object/body-part concepts and 1 scene/prop concept; fill remaining slots with the most discriminative objects that are visibly present.\n"
        "Temporal concepts:\n"
        "- \"temporal_concepts\" must be a subset of \"concepts\" (still object nouns) that require motion across frames to be useful.\n"
        "\n"
        "Motion text concepts (for a video-text embedder):\n"
        "- \"action_concepts\" should list 3-6 short motion phrases (verbs allowed) that describe the visible motion involving those objects, "
        "e.g., \"opening mouth\", \"chewing\", \"turning head\", \"waving hand\". Keep lowercase; avoid duplicates.\n\n"
        "Avoid generic/low-information concepts unless clearly class-specific: floor, wall, room, background, lighting, blur, shadow, reflection.\n"
    )

    # Convert images to file:// URIs for vLLM's OpenAI-compatible API
    # This avoids huge base64 payloads and works with --allowed-local-media-path /
    content = []
    for fp in rep_frames:
        # Escape ? in file paths
        escaped_path = fp.replace("?", "%3F")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"file://{escaped_path}", "detail": "high"}
        })
    content.append({"type": "text", "text": vlm_query})
    base_messages = [
        {"role": "system", "content": "Return only JSON."},
        {"role": "user", "content": content},
    ]

    def _dedup_preserve(xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    attempts: List[Dict[str, Any]] = []
    concepts: List[str] = []
    temporal: List[str] = []
    action_concepts: List[str] = []
    validation_errors: List[str] = []

    retries = max(1, int(max_retries or 1))
    for attempt in range(retries):
        if attempt == 0:
            messages = base_messages
        else:
            # Add a concise corrective hint to force compliant JSON.
            hint = (
                "Previous response failed validation. Return EXACT JSON object only, "
                f"with exactly {max_concepts} noun concepts that are clearly visible in the provided frames. "
                "Do NOT escape quotes, do NOT repeat the JSON multiple times, "
                "keep temporal_concepts as a subset of concepts, and include 3-6 short action_concepts. "
                "No commentary, no extra text."
            )
            messages = base_messages + [{"role": "system", "content": hint}]

        raw = send_generate_request(messages)
        parsed_obj, parse_err = extract_first_json_object(raw)

        raw_concepts = clean_concept_list(parsed_obj.get("concepts") if isinstance(parsed_obj, dict) else None)
        action_concepts = clean_concept_list(parsed_obj.get("action_concepts") if isinstance(parsed_obj, dict) else None)
        raw_temporal = clean_concept_list(parsed_obj.get("temporal_concepts") if isinstance(parsed_obj, dict) else None)

        validation_errors = []
        if not isinstance(parsed_obj, dict):
            validation_errors.append(parse_err or "parsed JSON is not a dict")

        if not raw_concepts:
            validation_errors.append("VLM returned no concepts after cleaning")

        filtered_concepts = _filter_object_concepts(raw_concepts)
        if len(filtered_concepts) < len(raw_concepts):
            validation_errors.append(
                f"filtered out {len(raw_concepts) - len(filtered_concepts)} motion-like concept(s)"
            )

        concepts = _dedup_preserve(filtered_concepts)[:max_concepts]
        temporal_set = set(raw_temporal)
        temporal = [c for c in concepts if c in temporal_set]
        if raw_temporal and not temporal:
            validation_errors.append("no temporal concepts overlapped with provided temporal_concepts")

        if action_concepts is None:
            action_concepts = []
        else:
            action_concepts = _dedup_preserve(action_concepts)

        attempts.append(
            {
                "attempt": attempt,
                "raw_response": raw,
                "parsed_json": parsed_obj,
                "parse_error": parse_err,
                "validation_errors": list(validation_errors),
                "concepts_candidate": concepts,
                "temporal_candidate": temporal,
                "action_concepts_candidate": action_concepts,
            }
        )

        if not validation_errors:
            break

    if validation_errors:
        msg = "; ".join(validation_errors)
        print(f"[ERROR] propose_concepts_with_vlm validation failed after {retries} attempt(s): {msg}", file=sys.stderr)

    reasoning = {
        "vlm_query": vlm_query,
        "rep_frames": list(rep_frames),
        "raw_response": attempts[-1]["raw_response"] if attempts else None,
        "parsed_json": attempts[-1]["parsed_json"] if attempts else None,
        "parse_error": attempts[-1]["parse_error"] if attempts else None,
        "validation_errors": validation_errors,
        "concepts_final": concepts,
        "temporal_concepts_final": temporal,
        "action_concepts_final": action_concepts,
        "attempts": attempts,
        "max_retries": retries,
    }
    return concepts, temporal, action_concepts, reasoning


@dataclass
class Detection:
    concept: str
    window_idx: int
    window_start: int
    frame_path: str
    frame_global_idx: int
    score: Optional[float]
    box_xywh_norm: List[float]
    box_xyxy_px: List[int]
    dhash: int
    crop_path: Optional[str] = None


def collect_concept_detections(
    *,
    inference_fn,
    processor,
    all_frames: Sequence[str],
    windows: Sequence[Tuple[int, int]],
    concept: str,
    stride_in_window: int,
    max_unique: int,
    min_score: float,
    hash_size: int,
    max_hamming: int,
    save_crops: bool,
    crops_dir: str,
    lock_to_track: bool = False,
    track_dist_weight: float = 1.0,
    min_crop_area_px: int = 0,
    framewise: bool = True,
) -> List[Detection]:
    kept: List[Detection] = []
    seen_hashes: List[int] = []
    safe_concept = safe_slug(concept).replace(".", "_")

    # Lazily create output dirs only if we actually save a crop.
    concept_dir = ""

    # For simple "no switching" behavior: lock onto one instance by keeping continuity of bbox center.
    last_center: Optional[Tuple[float, float]] = None

    # Optional: force frame-wise processing (ignore provided window spans).
    if framewise:
        windows = [(i, i + 1) for i in range(len(all_frames))]

    for wi, (s, e) in enumerate(windows):
        if len(kept) >= max_unique:
            break
        window_frames = list(all_frames[s:e:stride_in_window])
        if len(window_frames) < 1:
            continue

        for local_t, fp in enumerate(window_frames):
            if len(kept) >= max_unique:
                break

            out = inference_fn(processor, fp, concept)
            boxes = out.get("pred_boxes", []) or []
            scores = out.get("pred_scores", []) or []
            if not boxes:
                continue

            # Choose which instance to use if multiple boxes exist:
            # - default: max score
            # - lock_to_track: first choose max score, then choose closest-to-previous center (with score as tie-break)
            cand_idxs = list(range(len(boxes)))
            if scores:
                cand_idxs = [i for i in cand_idxs if float(scores[i]) >= float(min_score)]
                if not cand_idxs:
                    continue
            else:
                # No scores provided; keep all candidates.
                pass

            def _center_for_idx(i: int) -> Tuple[float, float]:
                # boxes are normalized xywh
                x, y, bw, bh = boxes[i]
                return (float(x) + float(bw) / 2.0, float(y) + float(bh) / 2.0)

            def _score_for_idx(i: int) -> float:
                return float(scores[i]) if scores else 0.0

            best_i = cand_idxs[0]
            if not lock_to_track or last_center is None:
                # pick max score
                best_i = max(cand_idxs, key=lambda i: _score_for_idx(i))
            else:
                # pick closest center; use score to break ties / balance
                lx, ly = last_center

                def _key(i: int) -> Tuple[float, float]:
                    cx, cy = _center_for_idx(i)
                    dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
                    # Minimize dist, maximize score
                    return (dist * float(track_dist_weight), -_score_for_idx(i))

                best_i = min(cand_idxs, key=_key)

            score = float(scores[best_i]) if scores else None
            box = boxes[best_i]

            img = PILImage.open(fp).convert("RGB")
            W, H = img.size
            x0, y0, x1, y1 = box_xywh_norm_to_xyxy_px(box, W, H)
            crop = img.crop((x0, y0, x1, y1))
            if save_crops and int(min_crop_area_px) > 0:
                if int(crop.size[0]) * int(crop.size[1]) < int(min_crop_area_px):
                    # Too small to be useful; do not save and do not count this crop.
                    continue
            hsh = dhash(crop, hash_size=hash_size)

            if any(hamming(hsh, prev) <= max_hamming for prev in seen_hashes):
                continue
            seen_hashes.append(hsh)

            crop_path = None
            if save_crops:
                if concept_dir == "":
                    os.makedirs(crops_dir, exist_ok=True)
                    concept_dir = os.path.join(crops_dir, safe_concept)
                    os.makedirs(concept_dir, exist_ok=True)
                crop_name = f"{safe_concept}_f{(s + local_t):06d}.png"
                crop_path = os.path.join(concept_dir, crop_name)
                crop.save(crop_path)

            kept.append(
                Detection(
                    concept=concept,
                    window_idx=int(wi),
                    window_start=int(s),
                    frame_path=str(fp),
                    frame_global_idx=int(s + local_t),
                    score=score,
                    box_xywh_norm=[float(x) for x in box],
                    box_xyxy_px=[int(x0), int(y0), int(x1), int(y1)],
                    dhash=int(hsh),
                    crop_path=crop_path,
                )
            )

            # update tracking center in normalized coordinates
            if lock_to_track:
                last_center = _center_for_idx(best_i)
    return kept


def maybe_save_motion_gif(
    *,
    concept: str,
    detections: Sequence[Detection],
    gif_dir: str,
    max_frames: int,
    min_unique: int,
    min_span_frames: int,
    min_center_disp: float,
    min_avg_hash_step: float,
    duration_ms: int,
    crop_mode: str = "padded",
    pad_frac: float = 0.8,
) -> Tuple[Optional[str], Dict[str, Any]]:
    os.makedirs(gif_dir, exist_ok=True)
    safe_concept = safe_slug(concept).replace(".", "_")

    def _clamp_box_xyxy(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> Tuple[int, int, int, int]:
        x0 = int(max(0, min(w - 1, x0)))
        y0 = int(max(0, min(h - 1, y0)))
        x1 = int(max(1, min(w, x1)))
        y1 = int(max(1, min(h, y1)))
        if x1 <= x0:
            x1 = min(w, x0 + 1)
        if y1 <= y0:
            y1 = min(h, y0 + 1)
        return x0, y0, x1, y1

    def _pad_box_xyxy(x0: int, y0: int, x1: int, y1: int, w: int, h: int, frac: float) -> Tuple[int, int, int, int]:
        frac = float(frac or 0.0)
        if frac <= 0.0:
            return _clamp_box_xyxy(x0, y0, x1, y1, w, h)
        bw = max(1, int(x1 - x0))
        bh = max(1, int(y1 - y0))
        pad_x = int(round(bw * frac))
        pad_y = int(round(bh * frac))
        return _clamp_box_xyxy(x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y, w, h)

    def _union_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return _clamp_box_xyxy(min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1), w, h)


    if len(detections) < min_unique:
        return None, {"status": "skipped", "reason": f"not enough unique detections (<{min_unique})"}

    segs = sorted(detections, key=lambda d: int(d.frame_global_idx))
    span = int(segs[-1].frame_global_idx) - int(segs[0].frame_global_idx)
    if span < min_span_frames:
        return None, {"status": "skipped", "reason": f"span too short ({span} < {min_span_frames})", "span": span}

    sample_img = PILImage.open(segs[0].frame_path).convert("RGB")
    W, H = sample_img.size
    centers = []
    for s in segs:
        x0, y0, x1, y1 = s.box_xyxy_px
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        centers.append((cx / W, cy / H))
    (c0x, c0y), (c1x, c1y) = centers[0], centers[-1]
    center_disp = float(((c1x - c0x) ** 2 + (c1y - c0y) ** 2) ** 0.5)

    hash_steps = []
    for i in range(1, len(segs)):
        hash_steps.append(hamming(int(segs[i - 1].dhash), int(segs[i].dhash)))
    avg_hash_step = float(sum(hash_steps) / max(1, len(hash_steps)))

    if center_disp < min_center_disp and avg_hash_step < min_avg_hash_step:
        return None, {
            "status": "skipped",
            "reason": "seems static",
            "span": span,
            "center_disp": center_disp,
            "avg_hash_step": avg_hash_step,
        }

    segs_for_gif = segs[:max_frames]
    crop_imgs: List[PILImage.Image] = []
    used_crop_boxes_xyxy_px: List[List[int]] = []
    for s in segs_for_gif:
        img = PILImage.open(s.frame_path).convert("RGB")
        w, h = img.size
        x0, y0, x1, y1 = map(int, s.box_xyxy_px)
        box = _clamp_box_xyxy(x0, y0, x1, y1, w, h)

        mode = (crop_mode or "padded").strip().lower()
        if mode not in {"concept", "padded", "concept_plus_person"}:
            mode = "padded"

        if mode == "concept_plus_person":
            # Context box functionality removed (previously used SAM3)
            box = _pad_box_xyxy(*box, w, h, pad_frac)
        elif mode == "padded":
            box = _pad_box_xyxy(*box, w, h, pad_frac)
        else:
            box = _clamp_box_xyxy(*box, w, h)

        cx0, cy0, cx1, cy1 = box
        used_crop_boxes_xyxy_px.append([int(cx0), int(cy0), int(cx1), int(cy1)])
        crop_imgs.append(img.crop((cx0, cy0, cx1, cy1)))

    max_w = max(im.size[0] for im in crop_imgs)
    max_h = max(im.size[1] for im in crop_imgs)
    crop_imgs = [ImageOps.pad(im, (max_w, max_h), color=(0, 0, 0)) for im in crop_imgs]

    gif_path = os.path.join(gif_dir, f"{safe_concept}.gif")
    crop_imgs[0].save(
        gif_path,
        save_all=True,
        append_images=crop_imgs[1:],
        duration=duration_ms,
        loop=0,
    )

    return gif_path, {
        "status": "saved",
        "span": span,
        "center_disp": center_disp,
        "avg_hash_step": avg_hash_step,
        "frames_in_gif": len(segs_for_gif),
        "used_frame_paths": [s.frame_path for s in segs_for_gif],
        "used_frame_global_idxs": [int(s.frame_global_idx) for s in segs_for_gif],
        "used_box_xyxy_px": [list(map(int, s.box_xyxy_px)) for s in segs_for_gif],
        "used_crop_xyxy_px": used_crop_boxes_xyxy_px,
        "crop_mode": (crop_mode or "padded"),
        "pad_frac": float(pad_frac),
        "frame_dhashes": [int(dhash(im, hash_size=8)) for im in crop_imgs],
    }


def _pil_to_data_url(pil: PILImage.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def caption_motion_from_keyframes(
    *,
    server_url: str,
    api_key: str,
    model: str,
    keyframes: Sequence[PILImage.Image],
    action_class: Optional[str] = None,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    """
    Ask the VLM to provide a short action/motion concept describing the motion across keyframes.
    Returns parsed JSON (best effort) plus raw response.
    """
    class_hint = (
        f" The action class is '{action_class}'."
        if isinstance(action_class, str) and action_class.strip()
        else ""
    )
    prompt = (
        "You are helping build a Concept Bottleneck Model (CBM) for video action recognition."
        + class_hint
        + " You are given a few keyframes from a short motion clip (cropped around one tracked region). "
        "Describe the motion with ONE short action concept.\n\n"
        "Return ONLY valid JSON with schema: "
        "{\"action_concept\": \"...\", \"is_temporal\": true/false, \"why\": \"...\"}.\n"
        "Rules:\n"
        "- action_concept: 1-3 words, lowercase; verb or verb phrase allowed (e.g., \"stepping up\", \"arm swing\").\n"
        "- Focus on motion, not static appearance.\n"
    )

    content: List[dict] = []
    for im in keyframes:
        content.append({"type": "image_url", "image_url": {"url": _pil_to_data_url(im), "detail": "high"}})
    content.append({"type": "text", "text": prompt})

    if OpenAI is None:
        return {
            "raw_response": None,
            "error": "openai package not installed; install 'openai' to enable captioning",
        }
    client = OpenAI(api_key=api_key, base_url=server_url)
    raw = None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return only JSON."},
                {"role": "user", "content": content},
            ],
            max_completion_tokens=max_tokens,
            n=1,
        )
        raw = resp.choices[0].message.content if resp.choices else None
    except Exception as e:
        return {"raw_response": raw, "error": repr(e)}

    parsed, err = extract_first_json_object(raw)
    return {"raw_response": raw, "parsed": parsed, "parse_error": err}


def avg_frame_hamming_between_gifs(
    a_hashes: Sequence[int], b_hashes: Sequence[int], sample_k: int = 8
) -> Optional[float]:
    """
    Compare two GIFs by average hamming distance between per-frame dhashes.
    Returns None if either list is empty.
    """
    if not a_hashes or not b_hashes:
        return None
    ak = min(sample_k, len(a_hashes))
    bk = min(sample_k, len(b_hashes))
    k = min(ak, bk)
    if k <= 0:
        return None

    def _sample(xs: Sequence[int], k_: int) -> List[int]:
        if len(xs) <= k_:
            return list(xs)
        idxs = [int(round(i * (len(xs) - 1) / max(1, k_ - 1))) for i in range(k_)]
        return [int(xs[i]) for i in idxs]

    aa = _sample(a_hashes, k)
    bb = _sample(b_hashes, k)
    dists = [hamming(int(x), int(y)) for x, y in zip(aa, bb)]
    return float(sum(dists) / max(1, len(dists)))


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


