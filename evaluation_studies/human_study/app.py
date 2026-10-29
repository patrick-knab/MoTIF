from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import streamlit as st


HERE = Path(__file__).resolve().parent
BUNDLE_PATH = HERE / "study_bundle.jsonl"
ASSIGNMENTS_PATH = HERE / "assignments.csv"
RESPONSES_DIR = HERE / "responses"
PROTOCOL = "motif_human_usefulness"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@st.cache_data
def load_study() -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    trials = {row["trial_id"]: row for row in read_jsonl(BUNDLE_PATH)}
    with ASSIGNMENTS_PATH.open(encoding="utf-8", newline="") as handle:
        assignments = list(csv.DictReader(handle))
    return trials, assignments


def safe_rater_id(value: str) -> str:
    cleaned = "".join(character for character in value.strip() if character.isalnum() or character in "-_")
    if not cleaned or cleaned != value.strip():
        raise ValueError("Enter the anonymous code exactly as provided.")
    return cleaned


def response_path(rater_id: str) -> Path:
    return RESPONSES_DIR / f"{rater_id}.jsonl"


def completed_trials(rater_id: str, assigned_trial_ids: set[str]) -> set[str]:
    path = response_path(rater_id)
    if not path.exists():
        return set()
    return {
        row["trial_id"]
        for row in read_jsonl(path)
        if row.get("protocol") == PROTOCOL
        and row["trial_id"] in assigned_trial_ids
    }


def assignment_order(rows: list[dict[str, str]], rater_id: str) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{rater_id}|{row['trial_id']}|order".encode()
        ).hexdigest(),
    )


def render_condition(title: str, condition: dict[str, Any]) -> None:
    st.markdown(f"#### {title}")
    for section in condition["sections"]:
        heading = section["heading"]
        start_rank = 1
        if heading in {
            "Highest-ranked evidence across the whole video",
            "Concepts 1–4: complete video (same in both explanations)",
        }:
            heading = "Complete-video ranking: ranks 1–4 (shared by A and B)"
        elif heading in {
            "Additional evidence across the whole video",
            "Concepts 5–8: complete video",
        }:
            heading = "Complete-video ranking: ranks 5–8"
            start_rank = 5
        elif heading.startswith("Evidence localized to "):
            heading = heading.replace(
                "Evidence localized to ",
                "Cited-segment ranking: ranks 1–4 (",
                1,
            ) + ")"
        elif heading.startswith("Concepts 5–8: cited segment "):
            span = heading.removeprefix("Concepts 5–8: cited segment ")
            heading = f"Cited-segment ranking: ranks 1–4 ({span})"
        with st.container(border=True):
            st.caption(heading)
            for rank, concept in enumerate(
                section["concepts"], start=start_rank
            ):
                st.markdown(f"**{rank}.** {concept}")


def render_concept_list(title: str, concepts: list[str]) -> None:
    st.markdown(f"#### {title}")
    with st.container(border=True):
        st.caption(
            "Original cited-segment ranking (rank 1 = largest model contribution)"
        )
        for rank, concept in enumerate(concepts, start=1):
            st.markdown(f"**{rank}.** {concept}")


def save_response(record: dict[str, Any]) -> None:
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    path = response_path(record["rater_id"])
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


st.set_page_config(page_title="Video explanation study", layout="wide")
st.markdown(
    """
    <style>
    div[data-testid="stVideo"] {
        max-width: 640px;
    }
    div[data-testid="stVideo"] video {
        max-height: 60vh;
        object-fit: contain;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Video explanation study")
st.info(
    "No machine-learning expertise is required. Use ordinary visual judgment: "
    "watch what happens in the video and decide which explanation or concept "
    "list would be clearer and more useful to you."
)

with st.expander("Instructions and consent", expanded=True):
    st.markdown(
        """
This study contains two types of anonymized A/B comparison:

1. **Explanation usefulness:** decide which explanation better helps you
   understand why a video model made its displayed prediction.
2. **Local concept grounding:** watch a short clip and decide which concept list
   better matches what is visibly present in that clip.

The **model prediction** is the activity label produced by the video model.
A **concept** is a short object or action word that the model identified as
evidence for that prediction. A concept listed across the whole video may matter
somewhere in the video but does not say when. A localized concept is tied to the
displayed time interval.

Concepts are shown in ranked order according to their positive contribution to
the model prediction within the stated scope: rank 1 has the largest
contribution, followed by ranks 2, 3, and so on. The complete-video and
cited-segment sections are separate rankings, so each can begin at rank 1.

For example, if a model predicts “making tea,” a whole-video explanation might
list `kettle`, `cup`, and `pour`. A localized explanation might additionally say
that `pour` and `cup` are important specifically from `00:12` to `00:15`.

For usefulness trials, evidence “across the whole video” does not say when a
concept appears. Evidence “localized to X–Y” associates concepts with that
specific segment. One explanation provides more whole-video concepts; the other
uses the same number of entries but replaces some with localized evidence.

For grounding trials, judge only whether the concept words match the visible
clip. The model prediction is intentionally hidden. “Tie” is appropriate when
neither side is better.

There are no trick questions. The study stores only your anonymous participant
code, choices, confidence, and response time. Participation is voluntary; you
may stop at any time. Do not enter your name or other personal information.
"""
    )

rater_input = st.text_input("Anonymous participant code")
consent = st.checkbox("I am at least 18 and consent to participate.")
if not rater_input or not consent:
    st.stop()

try:
    rater_id = safe_rater_id(rater_input)
except ValueError as error:
    st.error(str(error))
    st.stop()

trials, all_assignments = load_study()
rater_assignments = [
    row for row in all_assignments if row["rater_id"] == rater_id
]
if not rater_assignments:
    st.error("This code is not in the assignment file. Please check the code.")
    st.stop()

rater_assignments = assignment_order(rater_assignments, rater_id)
assigned_trial_ids = {row["trial_id"] for row in rater_assignments}
completed = completed_trials(rater_id, assigned_trial_ids)
remaining = [row for row in rater_assignments if row["trial_id"] not in completed]
st.progress(min(1.0, len(completed) / len(rater_assignments)))
st.caption(f"Completed {len(completed)} of {len(rater_assignments)} trials")
if not remaining:
    st.success("Thank you — your assigned trials are complete.")
    st.stop()

assignment = remaining[0]
trial = trials[assignment["trial_id"]]
if st.session_state.get("active_trial") != trial["trial_id"]:
    st.session_state.active_trial = trial["trial_id"]
    st.session_state.trial_started_at = time.time()

localized_span = trial["localized_span_label"]
if assignment["task_type"] == "usefulness":
    st.subheader("Task: explanation usefulness")
    st.markdown(f"### Model prediction: `{trial['pred_label']}`")
    context_video, cited_video = st.columns(2)
    with context_video:
        st.markdown("#### 1. Complete video (context)")
        st.caption(
            "Use the entire video to understand the model's prediction."
        )
        st.video(trial["video_path"], width="stretch")
    with cited_video:
        st.markdown(f"#### 2. Cited segment ({localized_span})")
        st.caption(
            "This excerpt comes from the same video. Only this interval is "
            "associated with the localized concepts."
        )
        st.video(
            trial["video_path"],
            start_time=int(trial["localized_span_seconds"][0]),
            end_time=int(trial["localized_span_seconds"][1]),
            width="stretch",
        )
    st.info(
        "**What differs in this trial:** Both explanations share the first four "
        "whole-video concepts. One uses its remaining four entries for "
        "additional whole-video concepts; the other uses them for concepts "
        f"associated with **{localized_span}**. The same concept may appear in "
        "both sections because they summarize different temporal scopes."
    )
    left, right = st.columns(2)
    with left:
        render_condition(
            "Explanation A", trial["conditions"][assignment["left_condition"]]
        )
    with right:
        render_condition(
            "Explanation B", trial["conditions"][assignment["right_condition"]]
        )
    question = (
        "Considering the concepts and any cited segment, which explanation "
        "better helps you understand why the model made this prediction?"
    )
    choice_labels = ["Explanation A", "Tie", "Explanation B"]
    inspected_label = "I inspected the video and the cited time span."
elif assignment["task_type"] == "grounding":
    st.subheader("Task: local concept grounding")
    st.markdown(f"#### Clip to judge ({localized_span})")
    st.caption(
        "Only this displayed excerpt matters for this task. Watch the complete "
        "short clip, then decide which concept list better matches its visible content."
    )
    st.video(
        trial["video_path"],
        start_time=int(trial["localized_span_seconds"][0]),
        end_time=int(trial["localized_span_seconds"][1]),
        width=640,
    )
    left, right = st.columns(2)
    with left:
        render_concept_list(
            "Concept list A",
            trial["grounding_conditions"][assignment["left_condition"]][
                "concepts"
            ],
        )
    with right:
        render_concept_list(
            "Concept list B",
            trial["grounding_conditions"][assignment["right_condition"]][
                "concepts"
            ],
        )
    question = "Which concept list better matches what is visible in this clip?"
    choice_labels = ["Concept list A", "Tie", "Concept list B"]
    inspected_label = "I watched the complete displayed clip."
else:
    st.error(f"Unknown task type: {assignment['task_type']}")
    st.stop()

with st.form("rating"):
    preference = st.radio(
        question,
        choice_labels,
        index=None,
        horizontal=True,
    )
    confidence = st.radio(
        "How confident are you in this choice?",
        ["Low", "Medium", "High"],
        index=None,
        horizontal=True,
    )
    inspected = st.checkbox(inspected_label)
    submitted = st.form_submit_button("Submit and continue", type="primary")

if submitted:
    if preference is None or confidence is None or not inspected:
        st.error("Please answer both questions and confirm that you inspected the video.")
    else:
        save_response(
            {
                "protocol": PROTOCOL,
                "rater_id": rater_id,
                "trial_id": trial["trial_id"],
                "dataset": trial["dataset"],
                "backbone": trial["backbone"],
                "correctness": trial["correctness"],
                "task_type": assignment["task_type"],
                "left_condition": assignment["left_condition"],
                "right_condition": assignment["right_condition"],
                "preference": preference,
                "confidence": confidence,
                "response_seconds": round(
                    time.time() - st.session_state.trial_started_at, 3
                ),
            }
        )
        st.rerun()
