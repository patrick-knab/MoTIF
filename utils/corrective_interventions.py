import contextlib
import io
import math
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from utils.explanations import explain_instance
from utils.motif import init_repro


DEFAULT_CONFIG = {
    "dataset": "breakfast",
    "dataset_name": "Breakfast",
    "clip_model": "res50",
    "window_size": 32,
    "test_split": "s1",
    "model_path": None,
    "device": "cuda:0",
    "seed": 42,
    "n_samples": 100,
    "min_samples": 100,
    "max_samples": 100,
    "max_concepts_per_video": 4,
    "manual_annotation_csv": "manual_corrective_annotations.csv",
    "topk_filter": 5,
}


LOG_COLUMNS = [
    "edit_mode",
    "dataset_name",
    "clip_model",
    "window_size",
    "model_path",
    "sample_idx",
    "video_path",
    "true_label",
    "pred_before",
    "pred_after",
    "true_rank_before",
    "true_rank_after",
    "true_logit_before",
    "true_logit_after",
    "pred_before_logit_before",
    "pred_before_logit_after",
    "num_annotations",
    "repaired_top1",
    "edited_concepts",
    "edited_slots",
    "original_values",
    "target_values",
]


def default_root() -> Path:
    here = Path.cwd()
    if here.name == "NeurIPS26_MOTIF_supplement":
        return here
    return Path(__file__).resolve().parents[1]


def make_config(root: Path, overrides: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    config = dict(DEFAULT_CONFIG)
    if overrides:
        config.update({k: v for k, v in overrides.items() if v is not None})
    if not config.get("model_path"):
        config["model_path"] = str(
            root
            / "Models"
            / f"{config['dataset_name']}_{config['clip_model']}_testing_1.0_{config['window_size']}_motif.pkl"
        )
    return config


def init_runtime(seed: int) -> int:
    return init_repro(int(seed), deterministic=True)


def resolve_device(device_name: str) -> torch.device:
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA was requested but is not available; using CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


@contextlib.contextmanager
def torch_pickle_map_location(map_location="cpu"):
    old_loader = torch.storage._load_from_bytes

    def mapped_loader(storage_bytes):
        return torch.load(
            io.BytesIO(storage_bytes),
            map_location=map_location,
            weights_only=False,
        )

    torch.storage._load_from_bytes = mapped_loader
    try:
        yield
    finally:
        torch.storage._load_from_bytes = old_loader


def load_cbm_checkpoint(path: str, device: torch.device):
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    with torch_pickle_map_location("cpu"):
        with checkpoint_path.open("rb") as f:
            cbm_model = pickle.load(f)
    if getattr(cbm_model, "model", None) is None:
        raise ValueError("Loaded checkpoint does not contain cbm_model.model")
    try:
        cbm_model.model = cbm_model.model.to(device).eval()
        cbm_model._app_device = device
    except Exception as exc:
        if device.type != "cuda":
            raise
        warnings.warn(f"Could not move model to {device}: {exc}. Falling back to CPU.")
        fallback = torch.device("cpu")
        cbm_model.model = cbm_model.model.to(fallback).eval()
        cbm_model._app_device = fallback
    return cbm_model


def prepare_video(video, device: torch.device):
    x = video if torch.is_tensor(video) else torch.as_tensor(np.asarray(video), dtype=torch.float32)
    x = x.to(device=device, dtype=torch.float32)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    return x


def label_name(cbm_model, idx: int) -> str:
    try:
        return str(cbm_model.encoder.inverse_transform([int(idx)])[0])
    except Exception:
        return str(idx)


def concept_names_for_model(cbm_model) -> List[str]:
    n = int(getattr(cbm_model, "num_concepts", cbm_model.X_test[0].shape[-1]))
    names = None
    if getattr(cbm_model, "concepts", None) is not None:
        names = getattr(cbm_model.concepts, "text_concepts", None)
    if names is None:
        names = getattr(cbm_model, "text_concepts", None)
    if names is None or len(names) != n:
        return [f"c{i}" for i in range(n)]
    return [str(x) for x in names]


def concept_label(concept_names: Sequence[str], concept_idx: int) -> str:
    name = concept_names[concept_idx] if concept_idx < len(concept_names) else f"c{concept_idx}"
    return f"c{concept_idx} | {name}"


def class_rank(logits: torch.Tensor, class_idx: int) -> int:
    logits = logits.detach().flatten().cpu()
    order = torch.argsort(logits, descending=True)
    return int((order == int(class_idx)).nonzero(as_tuple=True)[0].item() + 1)


def logits_to_record(cbm_model, sample_idx: int, logits: torch.Tensor) -> Dict[str, object]:
    logits = logits.detach().flatten().cpu()
    true_idx = int(cbm_model.y_test[sample_idx])
    pred_idx = int(torch.argmax(logits).item())
    return {
        "sample_idx": int(sample_idx),
        "true_idx": true_idx,
        "pred_idx": pred_idx,
        "true_label": label_name(cbm_model, true_idx),
        "pred_label": label_name(cbm_model, pred_idx),
        "true_rank": class_rank(logits, true_idx),
        "true_logit": float(logits[true_idx].item()),
        "pred_logit": float(logits[pred_idx].item()),
        "top5": [int(i) for i in torch.argsort(logits, descending=True)[:5].tolist()],
    }


@torch.no_grad()
def forward_logits(cbm_model, sample_idx: int, device: torch.device) -> torch.Tensor:
    x = prepare_video(cbm_model.X_test[sample_idx], device)
    logits, _, _, _ = cbm_model.model(x)
    return logits[0].detach().cpu()


@torch.no_grad()
def compute_concepts_t(model, video: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
    x = video
    if x.dim() == 2:
        x = x.unsqueeze(0)
        if key_padding_mask is not None and key_padding_mask.dim() == 1:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    x = model.posenc(x)
    for layer in model.layers:
        x = layer(x, key_padding_mask=key_padding_mask)
    x = model.norm(x)
    return model.concept_predictor(x)


@torch.no_grad()
def pool_logits_from_concepts_t(model, concepts_t: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
    logits_t = model.classifier(concepts_t)
    tau = float(model.lse_tau)
    if key_padding_mask is not None:
        logits_t = logits_t.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf"))
    return (logits_t * tau).logsumexp(dim=1) / tau


@torch.no_grad()
def logits_with_concept_edits(
    cbm_model,
    sample_idx: int,
    edits: Sequence[Tuple[int, int, float]],
    device: torch.device,
) -> torch.Tensor:
    model = cbm_model.model
    x = prepare_video(cbm_model.X_test[sample_idx], device)
    concepts_t = compute_concepts_t(model, x).clone()
    _, num_windows, num_concepts = concepts_t.shape

    for window_idx, concept_idx, value in edits:
        w = int(window_idx)
        c = int(concept_idx)
        if 0 <= w < num_windows and 0 <= c < num_concepts and pd.notna(value):
            concepts_t[:, w, c] = float(value)

    return pool_logits_from_concepts_t(model, concepts_t)[0].detach().cpu()


@torch.no_grad()
def logits_with_global_concept_edits(
    cbm_model,
    sample_idx: int,
    edits: Sequence[Tuple[int, float]],
    device: torch.device,
) -> torch.Tensor:
    model = cbm_model.model
    x = prepare_video(cbm_model.X_test[sample_idx], device)
    concepts_t = compute_concepts_t(model, x).clone()
    _, _, num_concepts = concepts_t.shape

    for concept_idx, value in edits:
        c = int(concept_idx)
        if 0 <= c < num_concepts and pd.notna(value):
            concepts_t[:, :, c] = float(value)

    return pool_logits_from_concepts_t(model, concepts_t)[0].detach().cpu()


@torch.no_grad()
def original_concepts_for_sample(cbm_model, sample_idx: int, device: torch.device) -> torch.Tensor:
    model = cbm_model.model
    x = prepare_video(cbm_model.X_test[sample_idx], device)
    return compute_concepts_t(model, x)[0].detach().cpu().float()


def select_repair_candidates(cbm_model, device: torch.device, config: Dict[str, object]) -> pd.DataFrame:
    rows = []
    cbm_model.model.eval()
    for sample_idx in range(len(cbm_model.X_test)):
        logits = forward_logits(cbm_model, sample_idx, device)
        rec = logits_to_record(cbm_model, sample_idx, logits)
        if rec["pred_idx"] != rec["true_idx"] and rec["true_rank"] <= int(config["topk_filter"]):
            rec["video_path"] = (
                cbm_model.paths_test[sample_idx]
                if getattr(cbm_model, "paths_test", None) is not None and sample_idx < len(cbm_model.paths_test)
                else None
            )
            rows.append(rec)

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        warnings.warn("No eligible repair candidates found.")
        return candidates

    n_requested = min(int(config["n_samples"]), int(config["max_samples"]))
    if len(candidates) < int(config["min_samples"]):
        warnings.warn(
            f"Only {len(candidates)} eligible candidates found; requested minimum is {config['min_samples']}."
        )
    if len(candidates) > n_requested:
        candidates = candidates.sample(n=n_requested, random_state=int(config["seed"])).sort_values("sample_idx")

    return candidates.reset_index(drop=True)


@torch.no_grad()
def suggested_value_for_slot(
    concepts_t: torch.Tensor,
    window_idx: int,
    concept_idx: int,
    true_idx: int,
    pred_idx: int,
    model,
) -> float:
    values = concepts_t[:, int(concept_idx)].detach().cpu().float()
    current = float(values[int(window_idx)].item())
    w_true = float(model.classifier.weight[int(true_idx), int(concept_idx)].detach().cpu().item())
    w_pred = float(model.classifier.weight[int(pred_idx), int(concept_idx)].detach().cpu().item())
    delta_weight = w_true - w_pred

    if values.numel() <= 1:
        return current + (1.0 if delta_weight >= 0 else -1.0)

    q = 0.90 if delta_weight >= 0 else 0.10
    proposed = float(torch.quantile(values, torch.tensor(q)).item())
    if math.isclose(proposed, current, rel_tol=1e-6, abs_tol=1e-6):
        proposed = current + (1.0 if delta_weight >= 0 else -1.0)
    return proposed


@torch.no_grad()
def suggested_value_for_global_concept(
    concepts_t: torch.Tensor,
    concept_idx: int,
    true_idx: int,
    pred_idx: int,
    model,
) -> float:
    values = concepts_t[:, int(concept_idx)].detach().cpu().float()
    if values.numel() == 0:
        return 0.0
    w_true = float(model.classifier.weight[int(true_idx), int(concept_idx)].detach().cpu().item())
    w_pred = float(model.classifier.weight[int(pred_idx), int(concept_idx)].detach().cpu().item())
    delta_weight = w_true - w_pred
    q = 0.90 if delta_weight >= 0 else 0.10
    proposed = float(torch.quantile(values, torch.tensor(q)).item())
    current = float(values.mean().item())
    if math.isclose(proposed, current, rel_tol=1e-6, abs_tol=1e-6):
        proposed = current + (1.0 if delta_weight >= 0 else -1.0)
    return proposed


def build_local_edit_table(
    cbm_model,
    candidates: pd.DataFrame,
    device: torch.device,
    config: Dict[str, object],
    concept_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()

    names = list(concept_names) if concept_names is not None else concept_names_for_model(cbm_model)
    rows = []
    model = cbm_model.model.eval()
    max_rows = int(config["max_concepts_per_video"])

    for rec in candidates.to_dict("records"):
        sample_idx = int(rec["sample_idx"])
        true_idx = int(rec["true_idx"])
        pred_idx = int(rec["pred_idx"])
        video = prepare_video(cbm_model.X_test[sample_idx], device)

        exp_wrong = explain_instance(model, video, target_class=pred_idx)
        exp_true = explain_instance(model, video, target_class=true_idx)
        wrong = exp_wrong["concept_contributions_per_time"].detach().cpu().float()
        true = exp_true["concept_contributions_per_time"].detach().cpu().float()
        concepts_t = exp_wrong["concepts_per_time"].detach().cpu().float()
        priority = wrong - true

        flat = priority.reshape(-1)
        positive = torch.where(flat > 0, flat, torch.zeros_like(flat))
        ranked = torch.argsort(positive if int((positive > 0).sum().item()) > 0 else flat.abs(), descending=True)

        T, C = priority.shape
        used = set()
        for flat_idx in ranked.tolist():
            window_idx, concept_idx = np.unravel_index(flat_idx, (T, C))
            key = (int(window_idx), int(concept_idx))
            if key in used:
                continue
            used.add(key)
            suggested = suggested_value_for_slot(concepts_t, window_idx, concept_idx, true_idx, pred_idx, model)
            rows.append(
                {
                    "edit_mode": "local",
                    "sample_idx": sample_idx,
                    "video_path": rec.get("video_path"),
                    "true_label": rec["true_label"],
                    "pred_label": rec["pred_label"],
                    "window_idx": int(window_idx),
                    "concept_idx": int(concept_idx),
                    "concept_name": names[int(concept_idx)] if int(concept_idx) < len(names) else f"c{concept_idx}",
                    "true_rank_before": int(rec["true_rank"]),
                    "wrong_contribution": float(wrong[int(window_idx), int(concept_idx)].item()),
                    "true_contribution": float(true[int(window_idx), int(concept_idx)].item()),
                    "repair_priority": float(priority[int(window_idx), int(concept_idx)].item()),
                    "suggested_correction_value": float(suggested),
                    "apply_edit": False,
                    "correction_value": np.nan,
                }
            )
            if len(used) >= max_rows:
                break

    return pd.DataFrame(rows).sort_values(["sample_idx", "repair_priority"], ascending=[True, False]).reset_index(drop=True)


def build_global_edit_table(
    cbm_model,
    candidates: pd.DataFrame,
    device: torch.device,
    config: Dict[str, object],
    concept_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()

    names = list(concept_names) if concept_names is not None else concept_names_for_model(cbm_model)
    rows = []
    model = cbm_model.model.eval()
    max_rows = int(config["max_concepts_per_video"])

    for rec in candidates.to_dict("records"):
        sample_idx = int(rec["sample_idx"])
        true_idx = int(rec["true_idx"])
        pred_idx = int(rec["pred_idx"])
        video = prepare_video(cbm_model.X_test[sample_idx], device)

        exp_pred = explain_instance(model, video, target_class=pred_idx)
        exp_true = explain_instance(model, video, target_class=true_idx)
        pred_global = exp_pred["concept_contributions_global"].detach().cpu().float()
        true_global = exp_true["concept_contributions_global"].detach().cpu().float()
        concepts_t = exp_pred["concepts_per_time"].detach().cpu().float()
        priority = pred_global - true_global

        positive = torch.where(priority > 0, priority, torch.zeros_like(priority))
        ranked = torch.argsort(positive if int((positive > 0).sum().item()) > 0 else priority.abs(), descending=True)

        used = set()
        for concept_idx in ranked.tolist():
            concept_idx = int(concept_idx)
            if concept_idx in used:
                continue
            used.add(concept_idx)
            suggested = suggested_value_for_global_concept(concepts_t, concept_idx, true_idx, pred_idx, model)
            rows.append(
                {
                    "edit_mode": "global",
                    "sample_idx": sample_idx,
                    "video_path": rec.get("video_path"),
                    "true_label": rec["true_label"],
                    "pred_label": rec["pred_label"],
                    "window_idx": np.nan,
                    "concept_idx": concept_idx,
                    "concept_name": names[concept_idx] if concept_idx < len(names) else f"c{concept_idx}",
                    "true_rank_before": int(rec["true_rank"]),
                    "pred_contribution": float(pred_global[concept_idx].item()),
                    "true_contribution": float(true_global[concept_idx].item()),
                    "repair_priority": float(priority[concept_idx].item()),
                    "suggested_correction_value": float(suggested),
                    "apply_edit": False,
                    "correction_value": np.nan,
                }
            )
            if len(used) >= max_rows:
                break

    return pd.DataFrame(rows).sort_values(["sample_idx", "repair_priority"], ascending=[True, False]).reset_index(drop=True)


def get_window_spans(cbm_model, sample_idx: int):
    path = None
    if getattr(cbm_model, "paths_test", None) is not None and sample_idx < len(cbm_model.paths_test):
        path = cbm_model.paths_test[sample_idx]
    if path is None:
        return None
    spans = None
    if getattr(cbm_model, "video_spans", None) is not None:
        spans = cbm_model.video_spans.get(path)
    if spans is None and getattr(cbm_model, "video_window_spans", None) is not None:
        spans = cbm_model.video_window_spans.get(path)
    return spans


def resolve_video_path(root: Path, path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    raw = str(path)
    candidates = [Path(raw)]
    if raw.startswith("../"):
        candidates.append(Path(raw.replace("../", "../../Data/", 1)))
    candidates.extend([root / raw, root / raw.lstrip("./")])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def global_concept_table(
    cbm_model,
    sample_idx: int,
    device: torch.device,
    concept_names: Sequence[str],
) -> pd.DataFrame:
    logits = forward_logits(cbm_model, sample_idx, device)
    rec = logits_to_record(cbm_model, sample_idx, logits)
    video = prepare_video(cbm_model.X_test[sample_idx], device)

    exp_pred = explain_instance(cbm_model.model, video, target_class=rec["pred_idx"])
    exp_true = explain_instance(cbm_model.model, video, target_class=rec["true_idx"])

    pred_global = exp_pred["concept_contributions_global"].detach().cpu().float()
    true_global = exp_true["concept_contributions_global"].detach().cpu().float()
    rows = []
    for concept_idx in range(len(pred_global)):
        rows.append(
            {
                "sample_idx": int(sample_idx),
                "concept_idx": int(concept_idx),
                "concept_name": concept_names[concept_idx] if concept_idx < len(concept_names) else f"c{concept_idx}",
                "pred_label": rec["pred_label"],
                "true_label": rec["true_label"],
                "pred_contribution": float(pred_global[concept_idx].item()),
                "true_contribution": float(true_global[concept_idx].item()),
                "repair_priority": float((pred_global[concept_idx] - true_global[concept_idx]).item()),
            }
        )
    return pd.DataFrame(rows)


def local_concept_table(
    cbm_model,
    sample_idx: int,
    device: torch.device,
    concept_names: Sequence[str],
) -> pd.DataFrame:
    logits = forward_logits(cbm_model, sample_idx, device)
    rec = logits_to_record(cbm_model, sample_idx, logits)
    video = prepare_video(cbm_model.X_test[sample_idx], device)

    exp_pred = explain_instance(cbm_model.model, video, target_class=rec["pred_idx"])
    exp_true = explain_instance(cbm_model.model, video, target_class=rec["true_idx"])

    pred_local = exp_pred["concept_contributions_per_time"].detach().cpu().float()
    true_local = exp_true["concept_contributions_per_time"].detach().cpu().float()
    priority = pred_local - true_local

    rows = []
    T, C = priority.shape
    for window_idx in range(T):
        for concept_idx in range(C):
            rows.append(
                {
                    "sample_idx": int(sample_idx),
                    "window_idx": int(window_idx),
                    "concept_idx": int(concept_idx),
                    "concept_name": concept_names[concept_idx] if concept_idx < len(concept_names) else f"c{concept_idx}",
                    "pred_label": rec["pred_label"],
                    "true_label": rec["true_label"],
                    "pred_contribution": float(pred_local[window_idx, concept_idx].item()),
                    "true_contribution": float(true_local[window_idx, concept_idx].item()),
                    "repair_priority": float(priority[window_idx, concept_idx].item()),
                }
            )
    return pd.DataFrame(rows)


def plot_global_important_concepts(table: pd.DataFrame, top_k: int = 10):
    if table.empty:
        return None
    modes = ("pred", "true", "repair_priority")
    mode_to_column = {
        "pred": "pred_contribution",
        "true": "true_contribution",
        "repair_priority": "repair_priority",
    }
    titles = {
        "pred": "Concepts supporting predicted class",
        "true": "Concepts supporting true class",
        "repair_priority": "Wrong-vs-true repair priority",
    }
    colors = {
        "pred": "tab:red",
        "true": "tab:green",
        "repair_priority": "tab:orange",
    }
    fig, axes = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), max(3.2, 0.32 * top_k)))
    for ax, mode in zip(np.asarray(axes).reshape(-1), modes):
        column = mode_to_column[mode]
        ranked = table.sort_values(column, ascending=False).head(top_k).iloc[::-1]
        labels = [f"c{int(r.concept_idx)} {r.concept_name}" for r in ranked.itertuples(index=False)]
        ax.barh(labels, ranked[column], color=colors[mode])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(titles[mode])
        ax.set_xlabel(column)
    plt.tight_layout()
    return fig


def plot_top_window_concepts(table: pd.DataFrame, top_windows: int = 4, top_k: int = 10, rank_by: str = "repair_priority"):
    if table.empty:
        return None
    window_scores = (
        table.assign(_positive=table[rank_by].clip(lower=0.0))
        .groupby("window_idx", as_index=False)["_positive"]
        .sum()
        .sort_values("_positive", ascending=False)
        .head(top_windows)
    )
    if float(window_scores["_positive"].sum()) == 0.0:
        window_scores = (
            table.assign(_abs=table[rank_by].abs())
            .groupby("window_idx", as_index=False)["_abs"]
            .sum()
            .sort_values("_abs", ascending=False)
            .head(top_windows)
        )
    selected_windows = [int(w) for w in window_scores["window_idx"].tolist()]

    ncols = 2
    nrows = int(np.ceil(max(1, len(selected_windows)) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(3.2, 3.0 * nrows)))
    axes = np.asarray(axes).reshape(-1)
    for ax, window_idx in zip(axes, selected_windows):
        part = table[table["window_idx"].eq(window_idx)].sort_values(rank_by, ascending=False).head(top_k).iloc[::-1]
        labels = [f"c{int(r.concept_idx)} {r.concept_name}" for r in part.itertuples(index=False)]
        ax.barh(labels, part[rank_by], color="tab:purple")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"window {window_idx}: top {top_k} local concepts")
        ax.set_xlabel(rank_by)
    for ax in axes[len(selected_windows) :]:
        ax.axis("off")
    plt.tight_layout()
    return fig


def frame_preview_figure(
    cbm_model,
    sample_idx: int,
    video_path: Path,
    annotation_table: Optional[pd.DataFrame],
    num_frames: int = 8,
):
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    if frame_count <= 0:
        cap.release()
        return None

    spans = get_window_spans(cbm_model, sample_idx)
    annotated_windows = set()
    if annotation_table is not None and not annotation_table.empty and "window_idx" in annotation_table:
        annotated_windows = set(
            int(w)
            for w in annotation_table.loc[annotation_table["sample_idx"].eq(sample_idx), "window_idx"].dropna().tolist()
        )
    annotated_ranges = []
    if spans is not None:
        for w in annotated_windows:
            if 0 <= w < len(spans):
                annotated_ranges.append(tuple(spans[w]))

    frame_ids = np.linspace(0, frame_count - 1, num=min(num_frames, frame_count), dtype=int)
    fig, axes = plt.subplots(1, len(frame_ids), figsize=(2.2 * len(frame_ids), 2.4))
    axes = np.asarray([axes]).reshape(-1) if len(frame_ids) == 1 else np.asarray(axes).reshape(-1)

    for ax, frame_id in zip(axes, frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, frame = cap.read()
        if not ok:
            ax.axis("off")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t_sec = frame_id / fps
        highlighted = any(start <= t_sec <= end for start, end in annotated_ranges)
        ax.imshow(frame)
        ax.set_title(f"{t_sec:.1f}s", color=("red" if highlighted else "black"))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(highlighted)
            spine.set_edgecolor("red")
            spine.set_linewidth(3)
    cap.release()
    plt.tight_layout()
    return fig


def original_value_detail(concepts_t: torch.Tensor, edit_mode: str, window_idx: int, concept_idx: int) -> Tuple[float, str]:
    values = concepts_t[:, int(concept_idx)]
    if edit_mode == "global":
        return (
            float(values.mean()),
            f"mean={float(values.mean()):.6g}; min={float(values.min()):.6g}; max={float(values.max()):.6g}",
        )
    w = min(max(int(window_idx), 0), int(values.numel()) - 1)
    value = float(values[w])
    return value, f"w{w}={value:.6g}"


def evaluate_rows(
    cbm_model,
    sample_idx: int,
    edit_mode: str,
    rows: Sequence[Dict[str, object]],
    device: torch.device,
) -> List[Dict[str, object]]:
    base_logits = forward_logits(cbm_model, sample_idx, device)
    base = logits_to_record(cbm_model, sample_idx, base_logits)
    reports = []
    for k in range(1, len(rows) + 1):
        prefix = rows[:k]
        if edit_mode == "global":
            edits = [(int(row["concept_idx"]), float(row["correction_value"])) for row in prefix]
            after_logits = logits_with_global_concept_edits(cbm_model, sample_idx, edits, device)
        else:
            edits = [
                (int(row["window_idx"]), int(row["concept_idx"]), float(row["correction_value"]))
                for row in prefix
            ]
            after_logits = logits_with_concept_edits(cbm_model, sample_idx, edits, device)
        after = logits_to_record(cbm_model, sample_idx, after_logits)
        reports.append(
            {
                "k": k,
                "base_logits": base_logits,
                "after_logits": after_logits,
                "base": base,
                "after": after,
                "repaired_top1": bool(after["pred_idx"] == base["true_idx"]),
            }
        )
    return reports


def format_logit_report(
    cbm_model,
    sample_idx: int,
    edit_mode: str,
    rows: Sequence[Dict[str, object]],
    report: Dict[str, object],
) -> str:
    k = int(report["k"])
    base = report["base"]
    after = report["after"]
    base_logits = report["base_logits"]
    after_logits = report["after_logits"]
    class_ids = []
    for idx in [base["pred_idx"], base["true_idx"], after["pred_idx"]]:
        idx = int(idx)
        if idx not in class_ids:
            class_ids.append(idx)
    edited = "; ".join(
        f"c{int(row['concept_idx'])}={float(row['correction_value']):.6g}"
        if edit_mode == "global"
        else f"w{int(row['window_idx'])}:c{int(row['concept_idx'])}={float(row['correction_value']):.6g}"
        for row in rows[:k]
    )
    lines = [
        f"sample {sample_idx} | {edit_mode} intervention | k={k}",
        f"edited: {edited}",
        f"prediction: {base['pred_label']} -> {after['pred_label']} | true: {base['true_label']}",
        f"true rank: {int(base['true_rank'])} -> {int(after['true_rank'])}",
        "",
        "class logits:",
    ]
    for idx in class_ids:
        before = float(base_logits[idx].item())
        after_value = float(after_logits[idx].item())
        lines.append(f"  {label_name(cbm_model, idx)}: {before:.6g} -> {after_value:.6g} (delta {after_value - before:+.6g})")
    return "\n".join(lines)


def load_repair_log(log_csv: Path) -> pd.DataFrame:
    if not log_csv.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    if log_csv.stat().st_size == 0:
        return pd.DataFrame(columns=LOG_COLUMNS)
    try:
        log = pd.read_csv(log_csv)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=LOG_COLUMNS)
    for column in LOG_COLUMNS:
        if column not in log.columns:
            log[column] = np.nan
    return log


def append_or_replace_log(log_csv: Path, row: Dict[str, object]) -> pd.DataFrame:
    log = load_repair_log(log_csv)
    key = (row["edit_mode"], int(row["sample_idx"]), int(row["num_annotations"]))
    if not log.empty:
        mask = (
            log["edit_mode"].astype(str).eq(str(key[0]))
            & log["sample_idx"].astype(int).eq(key[1])
            & log["num_annotations"].astype(int).eq(key[2])
        )
        if row.get("video_path") is not None and "video_path" in log.columns:
            mask = mask & log["video_path"].astype(str).eq(str(row["video_path"]))
        for column in ("dataset_name", "clip_model", "window_size", "model_path"):
            if column in row and pd.notna(row[column]) and column in log.columns:
                existing = log[column]
                mask = mask & (existing.isna() | existing.astype(str).eq(str(row[column])))
        log = log.loc[~mask].copy()
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    log.to_csv(log_csv, index=False)
    return log


def top1_repair_summary(repair_log: pd.DataFrame, max_k: int) -> pd.DataFrame:
    if repair_log.empty:
        return pd.DataFrame(
            columns=[
                "edit_mode",
                "k",
                "n",
                "n_logged_at_k",
                "n_repaired_top1",
                "n_direct_repaired_at_k",
                "n_carried_repaired_from_lower_k",
                "top1_repair_rate",
                "mean_true_logit_delta",
            ]
        )

    rows = []
    log = repair_log.copy()
    log["num_annotations"] = log["num_annotations"].astype(int)
    log["repaired_top1"] = (
        log["repaired_top1"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )
    log["true_logit_before"] = pd.to_numeric(log["true_logit_before"], errors="coerce")
    log["true_logit_after"] = pd.to_numeric(log["true_logit_after"], errors="coerce")
    log["true_logit_delta"] = log["true_logit_after"] - log["true_logit_before"]

    for edit_mode, mode_part in log.groupby("edit_mode"):
        sample_ids = sorted(mode_part["sample_idx"].astype(int).unique().tolist())
        first_success = {}
        sample_deltas = {}
        sample_annotations = {}
        for sample_idx, sample_part in mode_part.groupby("sample_idx"):
            sample_part = sample_part.sort_values("num_annotations")
            successes = sample_part[sample_part["repaired_top1"]]
            first_success[int(sample_idx)] = int(successes["num_annotations"].iloc[0]) if not successes.empty else None
            sample_deltas[int(sample_idx)] = sample_part[["num_annotations", "true_logit_delta"]]
            sample_annotations[int(sample_idx)] = set(sample_part["num_annotations"].astype(int).tolist())

        for k in range(1, int(max_k) + 1):
            repaired = [first_success[s] is not None and first_success[s] <= k for s in sample_ids]
            logged_at_k = [k in sample_annotations[s] for s in sample_ids]
            direct_repaired = [
                first_success[s] is not None and first_success[s] == k
                for s in sample_ids
            ]
            carried_repaired = [
                first_success[s] is not None and first_success[s] < k
                for s in sample_ids
            ]
            deltas = []
            for sample_idx in sample_ids:
                delta_part = sample_deltas[sample_idx]
                observed = delta_part[delta_part["num_annotations"] <= k]
                if not observed.empty and pd.notna(observed["true_logit_delta"].iloc[-1]):
                    deltas.append(float(observed["true_logit_delta"].iloc[-1]))
            rows.append(
                {
                    "edit_mode": edit_mode,
                    "k": k,
                    "n": len(sample_ids),
                    "n_logged_at_k": int(np.sum(logged_at_k)),
                    "n_repaired_top1": int(np.sum(repaired)),
                    "n_direct_repaired_at_k": int(np.sum(direct_repaired)),
                    "n_carried_repaired_from_lower_k": int(np.sum(carried_repaired)),
                    "top1_repair_rate": float(np.mean(repaired)) if repaired else 0.0,
                    "mean_true_logit_delta": float(np.mean(deltas)) if deltas else np.nan,
                }
            )

    return pd.DataFrame(rows).sort_values(["edit_mode", "k"]).reset_index(drop=True)


def repair_log_row(
    candidate_by_sample: Dict[int, Dict[str, object]],
    sample_idx: int,
    edit_mode: str,
    rows: Sequence[Dict[str, object]],
    report: Dict[str, object],
) -> Dict[str, object]:
    k = int(report["k"])
    prefix = rows[:k]
    base = report["base"]
    after = report["after"]
    base_logits = report["base_logits"]
    after_logits = report["after_logits"]
    return {
        "edit_mode": edit_mode,
        "sample_idx": sample_idx,
        "video_path": candidate_by_sample.get(sample_idx, {}).get("video_path"),
        "true_label": base["true_label"],
        "pred_before": base["pred_label"],
        "pred_after": after["pred_label"],
        "true_rank_before": int(base["true_rank"]),
        "true_rank_after": int(after["true_rank"]),
        "true_logit_before": float(base_logits[int(base["true_idx"])].item()),
        "true_logit_after": float(after_logits[int(base["true_idx"])].item()),
        "pred_before_logit_before": float(base_logits[int(base["pred_idx"])].item()),
        "pred_before_logit_after": float(after_logits[int(base["pred_idx"])].item()),
        "num_annotations": k,
        "repaired_top1": bool(report["repaired_top1"]),
        "edited_concepts": "; ".join(str(row["concept_name"]) for row in prefix),
        "edited_slots": "; ".join(
            f"c{int(row['concept_idx'])}" if edit_mode == "global" else f"w{int(row['window_idx'])}:c{int(row['concept_idx'])}"
            for row in prefix
        ),
        "original_values": "; ".join(str(row["original_detail"]) for row in prefix),
        "target_values": "; ".join(str(row["correction_value"]) for row in prefix),
    }


def failure_log_row(
    candidate_by_sample: Dict[int, Dict[str, object]],
    cbm_model,
    sample_idx: int,
    edit_mode: str,
    rows: Sequence[Dict[str, object]],
    device: torch.device,
) -> Dict[str, object]:
    base_logits = forward_logits(cbm_model, sample_idx, device)
    base = logits_to_record(cbm_model, sample_idx, base_logits)
    return {
        "edit_mode": edit_mode,
        "sample_idx": sample_idx,
        "video_path": candidate_by_sample.get(sample_idx, {}).get("video_path"),
        "true_label": base["true_label"],
        "pred_before": base["pred_label"],
        "pred_after": base["pred_label"],
        "true_rank_before": int(base["true_rank"]),
        "true_rank_after": int(base["true_rank"]),
        "true_logit_before": float(base_logits[int(base["true_idx"])].item()),
        "true_logit_after": float(base_logits[int(base["true_idx"])].item()),
        "pred_before_logit_before": float(base_logits[int(base["pred_idx"])].item()),
        "pred_before_logit_after": float(base_logits[int(base["pred_idx"])].item()),
        "num_annotations": max(1, len(rows)),
        "repaired_top1": False,
        "edited_concepts": "; ".join(str(row["concept_name"]) for row in rows),
        "edited_slots": "; ".join(
            f"c{int(row['concept_idx'])}" if edit_mode == "global" else f"w{int(row['window_idx'])}:c{int(row['concept_idx'])}"
            for row in rows
        ),
        "original_values": "; ".join(str(row["original_detail"]) for row in rows),
        "target_values": "; ".join(str(row["correction_value"]) for row in rows),
    }
