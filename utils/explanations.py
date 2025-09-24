"""
CBM models and utilities consolidated from the Video_cbm.ipynb notebook.
"""

from __future__ import annotations

import os
import random
import numpy as np
import torch

from typing import List, Optional, Dict, Tuple

import cv2
from PIL import Image

import torch.nn as nn

import torch.nn.functional as F

import math
from sklearn.preprocessing import LabelEncoder
import re
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl


@torch.no_grad()
def explain_instance(
    model: nn.Module,
    window_embeddings: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor] = None,
    channel_ids: Optional[Union[List[int], torch.Tensor]] = None,
    window_ids: Optional[Union[List[int], torch.Tensor]] = None,
    target_class: Optional[int] = None,
    window_spans: Optional[List[Tuple[int, int]]] = None,
    fps: Optional[float] = None,
):
    # device + shape
    device = next(model.parameters(), torch.empty(0)).device
    x = window_embeddings.to(device)
    if x.dim() == 2:
        x = x.unsqueeze(0)  # [1,T,C]
        if key_padding_mask is not None and key_padding_mask.dim() == 1:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # single forward, reuse its tau/masking behavior
    logits, concepts, concepts_t, sharpness = model(
        x,
        key_padding_mask=key_padding_mask,
        channel_ids=channel_ids,
        window_ids=window_ids,
    )  # logits:[B,K], concepts_t:[B,T,C]

    # per-time logits (not returned by forward)
    logits_t = model.classifier(concepts_t)  # [B,T,K]

    # choose class
    if target_class is None:
        target_class = int(logits[0].argmax().item())

    # pull first item (assumes B=1 for explanation)
    concepts_t_1 = concepts_t[0]  # [T,C]
    logits_t_1 = logits_t[0]  # [T,K]

    # class params
    w = model.classifier.weight[target_class]  # [C]
    b = (
        0.0
        if model.classifier.bias is None
        else float(model.classifier.bias[target_class].item())
    )

    # per-time contributions and scores
    contrib_t = concepts_t_1 * w.unsqueeze(0)  # [T,C]
    score_per_time = contrib_t.sum(dim=1) + b  # [T]

    # time importance consistent with forward (LSE/softmax with tau and mask)
    tau = float(model.lse_tau)
    time_scores = logits_t_1[:, target_class]  # [T]
    if key_padding_mask is not None:
        time_scores = time_scores.masked_fill(key_padding_mask[0], float("-inf"))
    time_importance = torch.softmax(time_scores / tau, dim=0)  # [T]

    # time-weighted global concept contributions
    contrib_global = (time_importance.unsqueeze(1) * contrib_t).sum(dim=0)  # [C]

    # package
    res = {
        "target_class": torch.tensor(target_class),
        "logits": logits[0].detach().cpu(),
        "logits_per_time": logits_t_1.detach().cpu(),
        "concepts": concepts[0].detach().cpu(),
        "concepts_per_time": concepts_t_1.detach().cpu(),
        "time_importance": time_importance.detach().cpu(),
        "score_per_time": score_per_time.detach().cpu(),
        "concept_contributions_per_time": contrib_t.detach().cpu(),
        "concept_contributions_global": contrib_global.detach().cpu(),
        "sharpness": {
            k: {m: v.detach().cpu() for m, v in d.items()} for k, d in sharpness.items()
        },
    }

    # optional spans
    if window_spans is not None and len(window_spans) == concepts_t_1.shape[0]:
        res["frame_spans"] = torch.tensor(window_spans, dtype=torch.long)
        if fps is not None and fps > 0:
            res["second_spans"] = torch.tensor(
                [(s / fps, e / fps) for (s, e) in window_spans], dtype=torch.float32
            )

    # optional per-layer attention if present
    attn = [getattr(layer, "attn_weights", None) for layer in model.layers]
    if any(a is not None for a in attn):
        res["attn_per_layer"] = [
            a[0].detach().cpu() if a is not None else None for a in attn
        ]

    return res


def _bar(x, width=20):
    # x in [0,1]
    n = int(round(x * width))
    return "█" * n + "·" * (width - n)


def print_explanation(
    res: dict, 
    fps_frame: dict, 
    concepts_list: Optional[List[str]] = None, 
    top_k_times: int = 3, 
    top_k_concepts: int = 8, 
    by_abs: bool = True,
    positive_only: bool = True,
):
    # pull & detach to CPU safely
    def td(x):
        return x.detach().cpu() if isinstance(x, torch.Tensor) else x

    ti = td(res["time_importance"]).flatten()  # [T]
    spt = td(res["score_per_time"]).flatten()  # [T]
    cpt = td(res["concept_contributions_per_time"])  # [T, C]
    cglob = td(res["concept_contributions_global"]).flatten()  # [C]
    tgt = res["target_class"]
    target_class = int(tgt.item()) if hasattr(tgt, "item") else int(tgt)

    T, C = ti.shape[0], cglob.shape[0]
    if concepts_list is None:
        concepts_list = [f"c{j}" for j in range(C)]

    # optional spans
    frame_spans = res.get("frame_spans", None)
    second_spans = res.get("second_spans", None)

    # normalizations
    ti_norm = (ti - ti.min()) / (ti.max() - ti.min() + 1e-8)
    spt_norm = (spt - spt.min()) / (spt.max() - spt.min() + 1e-8)

    # global concepts: choose ranking
    rank_vals = cglob.abs() if by_abs else cglob
    if positive_only:
        # Enforce positive-only based on original sign, even if by_abs=True
        rank_vals = torch.where(cglob > 0, rank_vals, torch.zeros_like(rank_vals))
        top_k_concepts = min(top_k_concepts, int((rank_vals > 0).sum().item()))
    topc_vals, topc_idx = torch.topk(rank_vals, k=min(top_k_concepts, C))
    print(f"Target class: {target_class}\n")
    print("Top concepts (global):")
    for _, j in zip(topc_vals, topc_idx):
        j = int(j)
        name = concepts_list[j] if j < len(concepts_list) else f"c{j}"
        val = float(cglob[j])  # signed value
        # bar by magnitude, show sign in number
        mag = abs(val)
        mag_norm = mag / (float(cglob.abs().max()) + 1e-8)
        print(f"  {name:30s} {val:+.3f}  {_bar(mag_norm)}")

    # top time steps
    _, topt_idx = torch.topk(ti, k=min(top_k_times, T))
    topt_idx = sorted(topt_idx.tolist(), key=lambda t: float(ti[t]), reverse=True)

    print("\nImportant time steps:")
    for t in topt_idx:
        t_imp = float(ti[t])
        extras = []
        if frame_spans is not None:
            fs = frame_spans[t]
            extras.append(f"frames=[{int(fs[0])},{int(fs[1])}]")
        if second_spans is not None:
            ss = second_spans[t]
            extras.append(f"sec=[{float(ss[0]):.2f},{float(ss[1]):.2f}]")
        extra_str = ("  " + "  ".join(extras)) if extras else ""
        start, end = fps_frame[t]
        print(
            f"  t=[{int(start//60):02d}:{start%60:05.2f} - {int(end//60):02d}:{end%60:05.2f}] time_importance={t_imp:.3f}  TI[{_bar(float(ti_norm[t]))}]  Score[{_bar(float(spt_norm[t]))}]"
            + extra_str
        )
        ct = cpt[t]  # [C]
        # per-time top concepts (by abs or signed)
        rank_vals_t = ct.abs() if by_abs else ct
        if positive_only:
            # Enforce positive-only based on original sign, even if by_abs=True
            rank_vals_t = torch.where(ct > 0, rank_vals_t, torch.zeros_like(rank_vals_t))
            k = min(top_k_concepts, int((rank_vals_t > 0).sum().item()), C)
        else:
            k = min(top_k_concepts, C)
        vals, idxs = torch.topk(rank_vals_t, k=k)
        # normalize bars by magnitude within this timestep for readability
        denom = float(ct.abs().max()) + 1e-8
        for j_rank in idxs:
            j = int(j_rank)
            name = concepts_list[j] if j < len(concepts_list) else f"c{j}"
            val = float(ct[j])  # signed
            print(f"     - {name:30s} {val:+.3f}  {_bar(abs(val)/denom)}")


def print_explanation_with_labels(
    res: dict, 
    fps_frame: dict, 
    label_decoder: LabelEncoder, 
    true_label_idx: int,
    positive_only: bool = True,
    **kwargs
):
    pred = res["target_class"]
    pred_idx = int(pred.item()) if hasattr(pred, "item") else int(pred)
    true_idx = int(true_label_idx)
    pred_name = label_decoder.inverse_transform([pred_idx])[0]
    true_name = label_decoder.inverse_transform([true_idx])[0]
    print(f"Predicted: {pred_idx} ({pred_name}) | True: {true_idx} ({true_name})")
    print_explanation(res, fps_frame, positive_only=positive_only, **kwargs)


# ---------------
# Batching utility
# ---------------
def pad_batch_sequences(
    seqs: List[torch.Tensor], device: torch.device
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Pad a list of [T_i, C] tensors into a batch [B, T_max, C] and return
    a key_padding_mask [B, T_max] with True for padded positions.
    """
    if len(seqs) == 0:
        raise ValueError("pad_batch_sequences received empty sequence list")
    lengths = [int(s.shape[0]) for s in seqs]
    C = int(seqs[0].shape[1])
    T_max = int(max(lengths))
    B = len(seqs)
    batch = torch.zeros((B, T_max, C), dtype=torch.float32, device=device)
    mask = torch.ones((B, T_max), dtype=torch.bool, device=device)  # True=padded
    for i, s in enumerate(seqs):
        t = lengths[i]
        batch[i, :t, :] = s.to(device)
        mask[i, :t] = False
    return batch, mask


def _cv_bar_img(frac: float, width: int = 160, height: int = 8) -> np.ndarray:
    frac = float(max(0.0, min(1.0, frac)))
    w = max(1, int(round(frac * width)))
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    bar[:, :w, :] = 255
    return bar


def _put_text_multiline(
    img,
    lines,
    org,
    line_h,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.40,
    thickness=1,
    color=(255, 255, 255),
):
    x, y = org
    for i, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (x, y + i * line_h),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def _safe_paste_bar(frame: np.ndarray, x: int, y: int, bar: np.ndarray) -> None:
    H, W = frame.shape[:2]
    bh, bw = bar.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + bw)
    y2 = min(H, y + bh)
    if x1 >= x2 or y1 >= y2:
        return
    bx1 = x1 - x
    by1 = y1 - y
    bx2 = bx1 + (x2 - x1)
    by2 = by1 + (y2 - y1)
    roi = frame[y1:y2, x1:x2]
    bar_crop = bar[by1:by2, bx1:bx2]
    np.maximum(roi, bar_crop, out=roi)


@torch.no_grad()
def render_explained_video_small_tl(
    vid_path: str,
    out_path: str,
    res: dict,  # from explain_instance(...)
    fps_frame_seconds: List[Tuple[float, float]],  # spans in SECONDS
    label_decoder,  # fitted LabelEncoder
    true_label_idx: int,
    concepts_list: Optional[List[str]] = None,
    top_k_times: int = 3,
    top_k_concepts: int = 4,
    by_abs: bool = True,
    up_scale: float = 2.0,  # upscale factor
    margin: int = 10,
    panel_w_px: int = 300,  # small box width
    panel_alpha: float = 0.70,
    font_scale: float = 0.40,
    thickness: int = 1,
    codec: str = "mp4v",
) -> str:
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {vid_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    F = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    outW = int(round(W * up_scale))
    outH = int(round(H * up_scale))
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*codec), fps, (outW, outH)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open writer for: {out_path}")

    # tensors -> CPU
    ti = res["time_importance"].detach().cpu().float()  # [T]
    cpt = res["concept_contributions_per_time"].detach().cpu()  # [T,C]
    tgt = res["target_class"]
    pred_idx = int(tgt.item()) if hasattr(tgt, "item") else int(tgt)

    T = ti.shape[0]
    C = cpt.shape[1]
    if concepts_list is None:
        concepts_list = [f"c{j}" for j in range(C)]

    try:
        pred_name = label_decoder.inverse_transform([pred_idx])[0]
        true_name = label_decoder.inverse_transform([int(true_label_idx)])[0]
    except Exception:
        pred_name = str(pred_idx)
        true_name = str(true_label_idx)

    # top-k windows
    if top_k_times == 0:
        top_k_times = T
    kT = min(top_k_times, T)
    _, topt_idx = torch.topk(ti, k=kT, largest=True, sorted=True)
    important_t = set(int(i) for i in topt_idx.tolist())

    # per-window top concepts (precompute)
    per_t_top = []
    for t in range(T):
        ct = cpt[t]
        rank_vals = ct.abs() if by_abs else ct
        kk = min(top_k_concepts, C)
        _, idxs = torch.topk(rank_vals, k=kk, largest=True, sorted=True)
        denom = float(ct.abs().max().item()) + 1e-8
        entries = []
        for j in idxs.tolist():
            name = concepts_list[j] if j < len(concepts_list) else f"c{j}"
            sval = float(ct[j].item())
            frac = min(1.0, abs(sval) / denom) if denom > 0 else 0.0
            entries.append((name, sval, frac))
        per_t_top.append(entries)

    # map sec->frames on original fps
    frame_to_t = [None] * F
    for t, (ss, es) in enumerate(fps_frame_seconds):
        fs = max(0, int(round(ss * fps)))
        fe = min(F - 1, int(round(es * fps)))
        for f in range(fs, fe + 1):
            frame_to_t[f] = t

    # small top-left panel geometry (after upscaling!)
    # keep it compact: header(2 lines) + k concepts
    line_h = 16
    rows = 2 + top_k_concepts
    panel_h_px = 18 + rows * line_h + 12
    x0, y0 = margin, margin
    panel_rect = (x0, y0, panel_w_px, panel_h_px)

    fidx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # upscale first, so overlay stays small proportionally
            frame = cv2.resize(frame, (outW, outH), interpolation=cv2.INTER_CUBIC)

            t = frame_to_t[fidx] if fidx < len(frame_to_t) else None
            if (t is not None) and (t in important_t):
                # translucent panel
                overlay = frame.copy()
                x, y, pw, ph = panel_rect
                cv2.rectangle(overlay, (x, y), (x + pw, y + ph), (0, 0, 0), -1)
                cv2.addWeighted(overlay, panel_alpha, frame, 1 - panel_alpha, 0, frame)

                # header (compressed)
                sec = fidx / fps
                ss, es = fps_frame_seconds[t]
                header = [
                    f"Pred:{pred_name} | True:{true_name}",
                    f"t={t} TI={float(ti[t]):.3f} [{ss:.2f}-{es:.2f}]s",
                ]
                _put_text_multiline(
                    frame,
                    header,
                    (x + 8, y + 18),
                    line_h,
                    font_scale=font_scale,
                    thickness=thickness,
                )

                # concepts (fewer, tight spacing)
                y_cursor = y + 18 + line_h * len(header) + 2
                for name, sval, frac in per_t_top[t]:
                    bar = _cv_bar_img(frac, width=120, height=8)
                    bx, by = x + 8, int(y_cursor - 8)
                    _safe_paste_bar(frame, bx, by, bar)
                    cv2.putText(
                        frame,
                        f"{name[:16]:16s} {sval:+.2f}",
                        (bx + bar.shape[1] + 8, int(y_cursor + 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA,
                    )
                    y_cursor += line_h

            writer.write(frame)
            fidx += 1
    finally:
        cap.release()
        writer.release()

    return out_path


@torch.no_grad()
def print_temporal_dependencies(
    res: dict,
    top_k_times: int = 5,  # how many query timesteps to print
    top_k_links: int = 5,  # how many strongest dependencies per timestep
    concept_idx: Optional[
        int
    ] = None,  # pick a concept for per-channel attention; None -> mean over concepts
    layer_agg: str = "mean",  # "mean" or "max" across layers
    head_or_concept_agg: str = "mean",  # how to aggregate heads (full-attn) or concepts (per-channel): "mean" or "max"
    focus_times: Optional[
        List[int]
    ] = None,  # if given, only print these query timesteps
    by_abs: bool = False,  # rank links by absolute weight (usually False)
):
    """
    Print temporal dependencies (attention) between timesteps.

    Handles both per-channel attention [C,T,T] and full-attention [H,T,T].

    Strategy:
      1) Load attention maps per layer.
      2) If shape is [C,T,T] (per-channel), either select 'concept_idx' or aggregate across concepts.
         If shape is [H,T,T] (full), aggregate across heads.
      3) Aggregate across layers via mean/max.
      4) Choose which timesteps to display:
           - 'focus_times' if given,
           - else top 'top_k_times' by res["time_importance"] (if available),
           - else first 'top_k_times'.
      5) For each chosen timestep t, print top 'top_k_links' target timesteps u with largest attention weight.
    """
    attn_layers = res.get("attn_per_layer", None)
    if not attn_layers or all(a is None for a in attn_layers):
        print(
            "[temporal] No attention maps available in 'res'. Ensure your model layers store 'attn_weights'."
        )
        return

    # Collect valid layers and ensure tensor type
    mats = []
    for a in attn_layers:
        if a is None:
            continue
        # a can be [C,T,T] (per-channel) OR [H,T,T] (full multi-head)
        if not torch.is_tensor(a):
            a = torch.as_tensor(a)
        mats.append(a.float())

    if len(mats) == 0:
        print("[temporal] No attention maps available after filtering.")
        return

    # Determine shape kind
    # Each layer mat has shape [G, T, T], where G = C (per-channel) or H (heads)
    G, T, T2 = mats[0].shape
    assert T == T2, f"Expected square attention [G,T,T], got {mats[0].shape}"

    # Aggregate across concepts/heads (dim 0)
    def agg_g(x: torch.Tensor) -> torch.Tensor:  # x: [G,T,T] -> [T,T]
        if concept_idx is not None and x.shape[0] > concept_idx:
            return x[concept_idx]
        if head_or_concept_agg == "max":
            return x.max(dim=0).values
        return x.mean(dim=0)

    mats_agg_g = [agg_g(a) for a in mats]  # list of [T,T]

    # Aggregate across layers -> [T,T]
    stack = torch.stack(mats_agg_g, dim=0)  # [L,T,T]
    if layer_agg == "max":
        A = stack.max(dim=0).values
    else:
        A = stack.mean(dim=0).values if hasattr(stack, "values") else stack.mean(dim=0)
        if isinstance(A, torch.return_types.max):
            A = A.values

    # Sanity: normalize rows (optional; attention should already be row-softmaxed)
    # A = A / (A.sum(dim=-1, keepdim=True) + 1e-9)

    # Decide which timesteps to print
    if focus_times is not None and len(focus_times) > 0:
        query_times = [t for t in focus_times if 0 <= t < T]
    else:
        ti = res.get("time_importance", None)
        if isinstance(ti, torch.Tensor) and ti.numel() == T:
            vals, idx = torch.topk(ti, k=min(top_k_times, T))
            query_times = idx.tolist()
            # Sort by decreasing importance
            query_times = sorted(query_times, key=lambda t: float(ti[t]), reverse=True)
        else:
            query_times = list(range(min(top_k_times, T)))

    # Optional second spans
    second_spans = res.get("second_spans", None)  # [T,2] if present

    def _fmt_time(ti_):
        if (
            second_spans is not None
            and hasattr(second_spans, "__len__")
            and len(second_spans) == T
        ):
            ss, es = second_spans[ti_]
            return f"t={ti_} [{float(ss):.2f}-{float(es):.2f}s]"
        return f"t={ti_}"

    # Print header context
    tgt = res.get("target_class", None)
    if tgt is not None:
        tc = int(tgt.item()) if hasattr(tgt, "item") else int(tgt)
        print(f"[temporal] Target class: {tc}")
    if concept_idx is not None:
        print(f"[temporal] Using per-channel attention for concept c={concept_idx}")
    else:
        print(
            f"[temporal] Aggregation over {'concepts' if G==A.shape[0] else 'heads'}: {head_or_concept_agg}, layers: {layer_agg}"
        )

    # For each chosen query timestep, print its strongest links
    for t in query_times:
        row = A[t]  # [T]
        # row = row.clone(); row[t] = 0.0

        rank_vals = row.abs() if by_abs else row
        k = min(top_k_links, T)
        vals, idxs = torch.topk(rank_vals, k=k, largest=True, sorted=True)

        # Pretty print
        print(f"\n{_fmt_time(t)}  (row-softmaxed attention to other timesteps)")
        # Normalize for bar length
        denom = float(rank_vals[idxs[0]] + 1e-12)
        for j, v in zip(idxs.tolist(), vals.tolist()):
            w = float(row[j])
            rel = max(0.0, min(1.0, float(abs(v) / denom)))
            bar = (
                _bar(rel)
                if " _bar" in globals() or "_bar" in locals()
                else f"{rel:.2f}"
            )
            if second_spans is not None and len(second_spans) == T:
                ss, es = second_spans[j]
                target_str = f"u={j} [{float(ss):.2f}-{float(es):.2f}s]"
            else:
                target_str = f"u={j}"
            print(f"  -> {target_str:18s} w={w:+.4f}  {bar}")


def _fmt_sec(sec: float) -> str:
    # 0:00.00 style for readability
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m}:{s:05.2f}s" if m else f"{s:.2f}s"


@torch.no_grad()
def plot_attention_heatmaps(
    res: dict,
    concept_idx: Optional[
        int
    ] = None,  # per-channel if set; else aggregate across concepts/heads
    concept_names: Optional[List[str]] = None,  # usually concepts.text_concepts
    layer_idxs: Optional[List[int]] = None,  # which layers to plot; None -> all
    layer_agg: Optional[str] = None,  # None | "mean" | "max"
    head_or_concept_agg: str = "mean",  # "mean" | "max"
    normalize_rows: bool = True,
    show_seconds: bool = True,
    cmap: str = "magma",
    figsize: Tuple[int, int] = (5, 4),
    savepath: Optional[str] = None,
    title_prefix: str = "Attention",
    ):
    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Liberation Serif"],
        "mathtext.fontset": "stix",
    }

    attn_layers = res.get("attn_per_layer", None)
    if not attn_layers or all(a is None for a in attn_layers):
        print("[heatmap] No attention maps in 'res'.")
        return

    mats = []
    for a in attn_layers:
        if a is None:
            continue
        a = torch.as_tensor(a, dtype=torch.float32)
        assert (
            a.ndim == 3 and a.shape[-1] == a.shape[-2]
        ), f"Expected [G,T,T], got {tuple(a.shape)}"
        mats.append(a)
    if not mats:
        print("[heatmap] No usable attention maps.")
        return

    if layer_idxs is not None:
        mats = [mats[i] for i in layer_idxs if 0 <= i < len(mats)]
        if not mats:
            print("[heatmap] Selected layer_idxs produced empty set.")
            return

    G, T, _ = mats[0].shape
    second_spans = res.get("second_spans", None)

    def agg_g(x: torch.Tensor) -> torch.Tensor:
        if concept_idx is not None:
            if not (0 <= concept_idx < x.shape[0]):
                raise IndexError(
                    f"concept_idx={concept_idx} out of range [0,{x.shape[0]-1}]."
                )
            return x[concept_idx]
        return x.max(dim=0).values if head_or_concept_agg == "max" else x.mean(dim=0)

    per_layer = [agg_g(L) for L in mats]

    plots = []
    if layer_agg in (None, ""):
        for Li, A in enumerate(per_layer):
            plots.append((Li, A))
    elif layer_agg == "mean":
        plots.append(("mean", torch.stack(per_layer, dim=0).mean(dim=0)))
    elif layer_agg == "max":
        plots.append(("max", torch.stack(per_layer, dim=0).max(dim=0).values))
    else:
        raise ValueError("layer_agg must be None, 'mean', or 'max'.")

    def row_norm(A: torch.Tensor) -> torch.Tensor:
        if not normalize_rows:
            return A
        denom = A.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return A / denom

    def make_ticks(T: int):
        step = max(1, T // 8)
        idxs = list(range(0, T, step))
        if idxs[-1] != T - 1:
            idxs.append(T - 1)
        if (
            show_seconds
            and isinstance(second_spans, torch.Tensor)
            and second_spans.shape[0] == T
        ):
            lbls = []
            for i in idxs:
                ss, es = second_spans[i].tolist()
                mid = 0.5 * (float(ss) + float(es))
                lbls.append(f"u={i} · {_fmt_sec(mid)}")
        else:
            lbls = [f"u={i}" for i in idxs]
        return idxs, lbls

    figs = []
    # apply Times New Roman only for the plotting block
    with mpl.rc_context(rc):
        for tag, A in plots:
            A = row_norm(A.detach().cpu())
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(
                A,
                origin="lower",
                interpolation="nearest",
                cmap=cmap,
                vmin=0.0,
                vmax=float(A.max().item()) or None,
            )

            ax.set_xlabel("Key time u (source/context)", fontsize=16)
            ax.set_ylabel("Query time t (target/current)", fontsize=16)

            xt, xl = make_ticks(T)
            yt, yl = make_ticks(T)
            yl = [lbl.replace("u=", "t=") for lbl in yl]

            ax.set_xticks(xt)
            ax.set_xticklabels(xl, rotation=45, ha="right", fontsize=13)
            ax.set_yticks(yt)
            ax.set_yticklabels(yl, fontsize=13)

            cname = None
            if (
                concept_idx is not None
                and concept_names
                and 0 <= concept_idx < len(concept_names)
            ):
                cname = concept_names[concept_idx]
            tag_str = f"layer={tag}" if isinstance(tag, (int, str)) else str(tag)
            if concept_idx is not None:
                title = f" ({cname})" if cname else ""
            else:
                title = f" (agg over {'concepts' if G==A.shape[0] else 'heads'})"
            ax.set_title(title, fontsize=18)

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Attention weight", fontsize=16)

            fig.tight_layout()
            if savepath:
                p = savepath
                if len(plots) > 1:
                    stem, ext = (savepath.rsplit(".", 1) + ["png"])[:2]
                    p = f"{stem}_{tag_str}.{ext}"
                fig.savefig(p, dpi=150, bbox_inches="tight")
            figs.append(fig)
            
    plt.show()

    return figs
