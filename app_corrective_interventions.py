from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils.corrective_interventions import (
    append_or_replace_log,
    build_global_edit_table,
    build_local_edit_table,
    concept_label,
    concept_names_for_model,
    default_root,
    evaluate_rows,
    failure_log_row,
    format_logit_report,
    frame_preview_figure,
    global_concept_table,
    init_runtime,
    load_cbm_checkpoint,
    load_repair_log,
    local_concept_table,
    make_config,
    original_concepts_for_sample,
    original_value_detail,
    plot_global_important_concepts,
    plot_top_window_concepts,
    repair_log_row,
    resolve_device,
    resolve_video_path,
    select_repair_candidates,
    top1_repair_summary,
)


st.set_page_config(page_title="Corrective Interventions", layout="wide")


ROOT = default_root()

MODEL_PRESETS = {
    "Breakfast": {
        "dataset": "breakfast",
        "dataset_name": "Breakfast",
        "clip_model": "res50",
        "window_size": 32,
        "manual_annotation_csv": "manual_corrective_annotations.csv",
    },
    "HMDB": {
        "dataset": "hmdb51",
        "dataset_name": "HMDB",
        "clip_model": "b32",
        "window_size": 8,
        "manual_annotation_csv": "manual_corrective_annotations_hmdb_b32_w8.csv",
    },
}


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.75rem;
        padding-bottom: 0.75rem;
        max-width: 100%;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.35rem;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 0.45rem;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    input {
        min-height: 2rem;
    }
    button[kind="primary"],
    button[kind="secondary"] {
        min-height: 2rem;
        padding-top: 0.2rem;
        padding-bottom: 0.2rem;
    }
    [data-testid="stCaptionContainer"] p {
        font-size: 0.72rem;
        line-height: 1.05rem;
    }
    [data-testid="stSidebar"] .stExpander {
        border: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_runtime(model_path: str, device_name: str, seed: int):
    init_runtime(seed)
    device = resolve_device(device_name)
    model = load_cbm_checkpoint(model_path, device)
    device = getattr(model, "_app_device", device)
    names = concept_names_for_model(model)
    return model, device, names


@st.cache_data(show_spinner=False)
def build_tables(
    model_path: str,
    device_name: str,
    seed: int,
    n_samples: int,
    min_samples: int,
    max_samples: int,
    max_concepts_per_video: int,
    topk_filter: int,
):
    config = make_config(
        ROOT,
        {
            "model_path": model_path,
            "device": device_name,
            "seed": seed,
            "n_samples": n_samples,
            "min_samples": min_samples,
            "max_samples": max_samples,
            "max_concepts_per_video": max_concepts_per_video,
            "topk_filter": topk_filter,
        },
    )
    cbm_model, device, names = load_runtime(model_path, device_name, seed)
    candidates = select_repair_candidates(cbm_model, device, config)
    local_table = build_local_edit_table(cbm_model, candidates, device, config, names)
    global_table = build_global_edit_table(cbm_model, candidates, device, config, names)
    return config, candidates, local_table, global_table


@st.cache_data(show_spinner=False)
def cached_global_table(model_path: str, device_name: str, seed: int, sample_idx: int):
    cbm_model, device, names = load_runtime(model_path, device_name, seed)
    return global_concept_table(cbm_model, sample_idx, device, names)


@st.cache_data(show_spinner=False)
def cached_local_table(model_path: str, device_name: str, seed: int, sample_idx: int):
    cbm_model, device, names = load_runtime(model_path, device_name, seed)
    return local_concept_table(cbm_model, sample_idx, device, names)


@st.cache_data(show_spinner=False)
def cached_concepts(model_path: str, device_name: str, seed: int, sample_idx: int):
    cbm_model, device, _ = load_runtime(model_path, device_name, seed)
    return original_concepts_for_sample(cbm_model, sample_idx, device)


def reset_slots(max_slots: int):
    for i in range(max_slots):
        for name in ("apply", "window", "concept", "value"):
            st.session_state.pop(f"slot_{i}_{name}", None)
    st.session_state.pop("last_result_text", None)


def fill_suggested_slots(sample_idx: int, mode: str, table: pd.DataFrame, concept_options, max_slots: int):
    rows = table[table["sample_idx"].eq(sample_idx)].head(max_slots)
    for i in range(max_slots):
        if i >= len(rows):
            st.session_state[f"slot_{i}_apply"] = False
            st.session_state[f"slot_{i}_value"] = ""
            continue
        row = rows.iloc[i]
        concept_idx = int(row["concept_idx"])
        st.session_state[f"slot_{i}_apply"] = True
        st.session_state[f"slot_{i}_concept"] = concept_options[concept_idx]
        st.session_state[f"slot_{i}_window"] = 0 if mode == "global" or pd.isna(row["window_idx"]) else int(row["window_idx"])
        suggested = row.get("suggested_correction_value", np.nan)
        st.session_state[f"slot_{i}_value"] = "" if pd.isna(suggested) else f"{float(suggested):.6g}"


def parse_active_rows(sample_idx: int, mode: str, concepts_t, concept_options, concept_names, max_slots: int):
    label_to_idx = {label: i for i, label in enumerate(concept_options)}
    rows = []
    T, C = concepts_t.shape
    for i in range(max_slots):
        if not st.session_state.get(f"slot_{i}_apply", False):
            continue
        value_text = str(st.session_state.get(f"slot_{i}_value", "")).strip()
        if not value_text:
            raise ValueError(f"Slot {i + 1} is active but has no target value.")
        try:
            value = float(value_text)
        except ValueError as exc:
            raise ValueError(f"Slot {i + 1} has a non-numeric target value.") from exc
        concept_text = str(st.session_state.get(f"slot_{i}_concept", ""))
        if concept_text not in label_to_idx:
            raise ValueError(f"Slot {i + 1} has an unknown concept.")
        concept_idx = int(label_to_idx[concept_text])
        window_idx = int(st.session_state.get(f"slot_{i}_window", 0))
        if not 0 <= concept_idx < C:
            raise ValueError(f"Slot {i + 1} concept is outside range.")
        if mode == "local" and not 0 <= window_idx < T:
            raise ValueError(f"Slot {i + 1} window is outside range 0..{T - 1}.")
        original_value, original_detail = original_value_detail(concepts_t, mode, window_idx, concept_idx)
        rows.append(
            {
                "slot": i + 1,
                "edit_mode": mode,
                "window_idx": np.nan if mode == "global" else window_idx,
                "concept_idx": concept_idx,
                "concept_name": concept_names[concept_idx] if concept_idx < len(concept_names) else f"c{concept_idx}",
                "original_value": original_value,
                "original_detail": original_detail,
                "correction_value": value,
            }
        )
    return rows[:max_slots]


def advance_sample(sample_ids):
    pos = int(st.session_state.get("sample_pos", 0))
    st.session_state["sample_pos"] = min(pos + 1, len(sample_ids) - 1)
    reset_slots(int(st.session_state["max_concepts_per_video"]))


def with_run_metadata(row, dataset_name: str, clip_model: str, window_size: int, model_path: str):
    row = dict(row)
    row.update(
        {
            "dataset_name": dataset_name,
            "clip_model": clip_model,
            "window_size": int(window_size),
            "model_path": model_path,
        }
    )
    return row


st.subheader("Manual Corrective Interventions")

with st.sidebar:
    preset_name = st.selectbox("Model preset", list(MODEL_PRESETS.keys()))
    default_config = make_config(ROOT, MODEL_PRESETS[preset_name])
    preset_key = preset_name.lower()
    with st.expander("Configuration", expanded=False):
        dataset_name = st.text_input(
            "Dataset name",
            value=str(default_config["dataset_name"]),
            key=f"{preset_key}_dataset_name",
        )
        clip_model = st.text_input(
            "CLIP model",
            value=str(default_config["clip_model"]),
            key=f"{preset_key}_clip_model",
        )
        window_size = st.number_input(
            "Window size",
            min_value=1,
            value=int(default_config["window_size"]),
            step=1,
            key=f"{preset_key}_window_size",
        )
        default_model_path = make_config(
            ROOT,
            {"dataset_name": dataset_name, "clip_model": clip_model, "window_size": int(window_size)},
        )["model_path"]
        model_path = st.text_input(
            "Model path",
            value=str(default_model_path),
            key=f"{preset_key}_model_path_{dataset_name}_{clip_model}_{int(window_size)}",
        )
        device_name = st.text_input("Device", value=str(default_config["device"]))
        seed = st.number_input("Seed", min_value=0, value=int(default_config["seed"]), step=1)
        n_samples = st.number_input("Samples", min_value=1, value=int(default_config["n_samples"]), step=1)
        min_samples = st.number_input("Minimum candidates", min_value=1, value=int(default_config["min_samples"]), step=1)
        max_samples = st.number_input("Maximum candidates", min_value=1, value=int(default_config["max_samples"]), step=1)
        max_slots = st.number_input(
            "Max edits per video",
            min_value=1,
            max_value=12,
            value=int(default_config["max_concepts_per_video"]),
            step=1,
        )
        topk_filter = st.number_input("True-label top-k filter", min_value=1, value=int(default_config["topk_filter"]), step=1)
        log_csv = ROOT / st.text_input(
            "Annotation CSV",
            value=str(default_config["manual_annotation_csv"]),
            key=f"{preset_key}_annotation_csv_{dataset_name}_{clip_model}_{int(window_size)}",
        )
    if st.button("Clear app cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        reset_slots(int(max_slots))
        st.rerun()
    st.caption(f"{dataset_name} / {clip_model} / w{int(window_size)}")
    st.caption(f"log: {log_csv.name}")

st.session_state["max_concepts_per_video"] = int(max_slots)
run_state_key = f"{preset_name}:{model_path}:{device_name}:{int(seed)}:{int(n_samples)}:{int(max_samples)}:{int(max_slots)}:{int(topk_filter)}"
if st.session_state.get("run_state_key") != run_state_key:
    st.session_state["run_state_key"] = run_state_key
    st.session_state["sample_pos"] = 0
    reset_slots(int(max_slots))

try:
    with st.spinner("Loading model and building candidate/suggestion tables..."):
        config, candidates_df, local_edit_df, global_edit_df = build_tables(
            model_path,
            device_name,
            int(seed),
            int(n_samples),
            int(min_samples),
            int(max_samples),
            int(max_slots),
            int(topk_filter),
        )
        cbm_model, device, concept_names = load_runtime(model_path, device_name, int(seed))
except Exception as exc:
    st.error(str(exc))
    st.stop()

if candidates_df.empty:
    st.warning("No eligible repair candidates found.")
    st.stop()

sample_ids = [int(x) for x in candidates_df["sample_idx"].tolist()]
if "sample_pos" not in st.session_state:
    st.session_state["sample_pos"] = 0
st.session_state["sample_pos"] = min(max(int(st.session_state["sample_pos"]), 0), len(sample_ids) - 1)

top_bar = st.columns([0.75, 2.7, 0.75, 1.2])
with top_bar[0]:
    if st.button("Previous", use_container_width=True):
        st.session_state["sample_pos"] = max(int(st.session_state["sample_pos"]) - 1, 0)
        reset_slots(int(max_slots))
with top_bar[1]:
    labels = [
        f"{i + 1}/{len(sample_ids)} | sample {int(r.sample_idx)}: {r.pred_label} -> {r.true_label} (rank {int(r.true_rank)})"
        for i, r in enumerate(candidates_df.itertuples(index=False))
    ]
    selected_label = st.selectbox(
        "Sample",
        labels,
        index=int(st.session_state["sample_pos"]),
        label_visibility="collapsed",
    )
    new_pos = labels.index(selected_label)
    if new_pos != int(st.session_state["sample_pos"]):
        st.session_state["sample_pos"] = new_pos
        reset_slots(int(max_slots))
with top_bar[2]:
    if st.button("Next", use_container_width=True):
        st.session_state["sample_pos"] = min(int(st.session_state["sample_pos"]) + 1, len(sample_ids) - 1)
        reset_slots(int(max_slots))
with top_bar[3]:
    mode = st.radio("Edit mode", ["local", "global"], horizontal=True, label_visibility="collapsed")

sample_idx = sample_ids[int(st.session_state["sample_pos"])]
candidate_by_sample = {int(r["sample_idx"]): r for r in candidates_df.to_dict("records")}
candidate = candidate_by_sample[sample_idx]
suggestion_table = local_edit_df if mode == "local" else global_edit_df
concept_options = [concept_label(concept_names, i) for i in range(len(concept_names))]
state_key = f"{sample_idx}:{mode}:{int(max_slots)}"
if st.session_state.get("loaded_state_key") != state_key:
    reset_slots(int(max_slots))
    fill_suggested_slots(sample_idx, mode, suggestion_table, concept_options, int(max_slots))
    st.session_state["loaded_state_key"] = state_key

st.markdown(
    "#### "
    f"Sample {sample_idx}: {candidate['pred_label']} -> {candidate['true_label']} "
    f"(true rank {int(candidate['true_rank'])})"
)

left, right = st.columns([0.85, 1.35])

with left:
    video_path = resolve_video_path(ROOT, candidate.get("video_path"))
    if video_path is not None:
        st.video(str(video_path))
        fig = frame_preview_figure(cbm_model, sample_idx, video_path, suggestion_table, num_frames=8)
        if fig is not None:
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("Raw video file is unavailable; showing concept diagnostics only.")

    with st.expander("Concept diagnostics", expanded=False):
        global_table = cached_global_table(model_path, device_name, int(seed), sample_idx)
        global_fig = plot_global_important_concepts(global_table, top_k=10)
        if global_fig is not None:
            st.pyplot(global_fig)
            plt.close(global_fig)

        local_table = cached_local_table(model_path, device_name, int(seed), sample_idx)
        local_fig = plot_top_window_concepts(local_table, top_windows=4, top_k=10)
        if local_fig is not None:
            st.pyplot(local_fig)
            plt.close(local_fig)

with right:
    concepts_t = cached_concepts(model_path, device_name, int(seed), sample_idx)
    max_window = int(concepts_t.shape[0]) - 1

    if st.button("Fill suggested", use_container_width=True):
        fill_suggested_slots(sample_idx, mode, suggestion_table, concept_options, int(max_slots))
        st.rerun()

    with st.form(f"intervention_form_{sample_idx}_{mode}", clear_on_submit=False):
        action_cols = st.columns(3)
        with action_cols[0]:
            test_clicked = st.form_submit_button("Test checked", type="primary", use_container_width=True)
        with action_cols[1]:
            log_clicked = st.form_submit_button("Log checked", use_container_width=True)
        with action_cols[2]:
            failure_clicked = st.form_submit_button("Log failure + next", use_container_width=True)

        st.caption(f"{mode.title()} annotations: choose any concept and target value, up to {int(max_slots)} slots.")

        for i in range(int(max_slots)):
            cols = st.columns([0.45, 0.85, 3.6, 1.0, 1.7])
            with cols[0]:
                st.checkbox(str(i + 1), key=f"slot_{i}_apply")
            with cols[1]:
                st.number_input(
                    "w",
                    min_value=0,
                    max_value=max(0, max_window),
                    step=1,
                    key=f"slot_{i}_window",
                    disabled=mode == "global",
                    label_visibility="collapsed",
                )
            with cols[2]:
                default_idx = 0
                current_concept = st.session_state.get(f"slot_{i}_concept", concept_options[0])
                if current_concept in concept_options:
                    default_idx = concept_options.index(current_concept)
                st.selectbox(
                    "concept",
                    concept_options,
                    index=default_idx,
                    key=f"slot_{i}_concept",
                    label_visibility="collapsed",
                    format_func=lambda value: value if len(value) <= 52 else f"{value[:49]}...",
                )
            with cols[3]:
                st.text_input("target", key=f"slot_{i}_value", label_visibility="collapsed")
            with cols[4]:
                try:
                    concept_idx = concept_options.index(st.session_state.get(f"slot_{i}_concept", concept_options[0]))
                    _, detail = original_value_detail(
                        concepts_t,
                        mode,
                        int(st.session_state.get(f"slot_{i}_window", 0)),
                        concept_idx,
                    )
                    st.caption(detail)
                except Exception:
                    st.caption("unknown concept")

    if test_clicked or log_clicked or failure_clicked:
        try:
            rows = parse_active_rows(sample_idx, mode, concepts_t, concept_options, concept_names, int(max_slots))
            if (test_clicked or log_clicked) and not rows:
                raise ValueError("Activate at least one slot with a numeric target value.")
            if test_clicked:
                reports = evaluate_rows(cbm_model, sample_idx, mode, rows, device)
                result_text = []
                for report in reports:
                    result_text.append(format_logit_report(cbm_model, sample_idx, mode, rows, report))
                    if report["repaired_top1"]:
                        st.success(f"Repaired with k={int(report['k'])}.")
                        break
                else:
                    st.warning(f"Not repaired after {len(rows)} edits.")
                st.session_state["last_result_text"] = "\n\n".join(result_text)
            elif log_clicked:
                reports = evaluate_rows(cbm_model, sample_idx, mode, rows, device)
                result_text = []
                logged_k = []
                repaired_k = None
                for report in reports:
                    result_text.append(format_logit_report(cbm_model, sample_idx, mode, rows, report))
                    log_row = repair_log_row(candidate_by_sample, sample_idx, mode, rows, report)
                    log_row = with_run_metadata(log_row, dataset_name, clip_model, int(window_size), model_path)
                    append_or_replace_log(log_csv, log_row)
                    logged_k.append(int(report["k"]))
                    if report["repaired_top1"]:
                        repaired_k = int(report["k"])
                        break
                st.session_state["last_result_text"] = "\n\n".join(result_text)
                logged_text = ", ".join(str(k) for k in logged_k)
                if repaired_k is None:
                    st.warning(f"Logged checked intervention prefixes: k={logged_text}. Not repaired.")
                else:
                    st.success(f"Logged repaired intervention at k={repaired_k}. Checked prefixes: k={logged_text}.")
            elif failure_clicked:
                log_row = failure_log_row(candidate_by_sample, cbm_model, sample_idx, mode, rows, device)
                log_row = with_run_metadata(log_row, dataset_name, clip_model, int(window_size), model_path)
                append_or_replace_log(log_csv, log_row)
                advance_sample(sample_ids)
                st.rerun()
        except Exception as exc:
            st.error(str(exc))

    if st.session_state.get("last_result_text"):
        st.code(st.session_state["last_result_text"], language="text")

    st.divider()
    st.subheader("Repair Summary")
    repair_log_df = load_repair_log(log_csv)
    if repair_log_df.empty:
        logged_mode_instances = 0
        local_examples = 0
        global_examples = 0
    else:
        instance_log = repair_log_df.assign(sample_idx=repair_log_df["sample_idx"].astype(int))
        per_mode_examples = instance_log.groupby("edit_mode")["sample_idx"].nunique()
        local_examples = int(per_mode_examples.get("local", 0))
        global_examples = int(per_mode_examples.get("global", 0))
        logged_mode_instances = local_examples + global_examples
    counter_cols = st.columns(4)
    counter_cols[0].metric("Logged instances", logged_mode_instances)
    counter_cols[1].metric("Global instances", global_examples)
    counter_cols[2].metric("Local instances", local_examples)
    counter_cols[3].metric("k rows", int(len(repair_log_df)))
    summary_df = top1_repair_summary(repair_log_df, int(max_slots))
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    if not summary_df.empty:
        plot_cols = st.columns(2)
        with plot_cols[0]:
            st.line_chart(summary_df, x="k", y="top1_repair_rate", color="edit_mode")
        with plot_cols[1]:
            st.line_chart(summary_df, x="k", y="mean_true_logit_delta", color="edit_mode")
    with st.expander("Logged annotations"):
        st.dataframe(repair_log_df, use_container_width=True, hide_index=True)
