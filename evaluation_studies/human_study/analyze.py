from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = "motif_human_usefulness"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def decode_choice(row: pd.Series) -> str:
    if row["preference"] == "Tie":
        return "tie"
    side = (
        "left_condition"
        if str(row["preference"]).endswith(" A")
        else "right_condition"
    )
    return str(row[side])


def bootstrap_trial_mean(
    frame: pd.DataFrame, replicates: int, seed: int
) -> tuple[float, float, float]:
    per_trial = frame.groupby("trial_id", sort=False)["preference_score"].mean()
    values = per_trial.to_numpy(dtype=float)
    estimate = float(values.mean())
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return estimate, float(low), float(high)


def summarize_group(
    frame: pd.DataFrame,
    task_type: str,
    group_name: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    favorable, control = {
        "usefulness": ("compact_multilevel", "global_only"),
        "grounding": ("target_local", "matched_video_local"),
    }[task_type]
    counts = Counter(frame["decoded_preference"])
    estimate, low, high = bootstrap_trial_mean(frame, replicates, seed)
    decisive = counts[favorable] + counts[control]
    return {
        "task_type": task_type,
        "group": group_name,
        "trials": int(frame["trial_id"].nunique()),
        "ratings": int(len(frame)),
        "favorable_condition": favorable,
        "control_condition": control,
        "favorable_count": counts[favorable],
        "control_count": counts[control],
        "tie": counts["tie"],
        "favorable_rate_all": counts[favorable] / len(frame),
        "control_rate_all": counts[control] / len(frame),
        "tie_rate": counts["tie"] / len(frame),
        "favorable_rate_decisive": (
            counts[favorable] / decisive if decisive else float("nan")
        ),
        "mean_preference_score": estimate,
        "ci95_low": low,
        "ci95_high": high,
    }


def pairwise_agreement(frame: pd.DataFrame) -> dict[str, Any]:
    agreements: list[bool] = []
    for _, group in frame.groupby("trial_id"):
        choices = group["decoded_preference"].tolist()
        for left in range(len(choices)):
            for right in range(left + 1, len(choices)):
                agreements.append(choices[left] == choices[right])
    return {
        "rating_pairs": len(agreements),
        "exact_agreement": float(np.mean(agreements)) if agreements else float("nan"),
    }


def analyze(
    responses_dir: Path,
    assignments_path: Path,
    output_dir: Path,
    bootstrap_replicates: int,
    seed: int,
    rater_ids: list[str] | None = None,
    protocol: str = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    response_paths = sorted(responses_dir.glob("*.jsonl"))
    if not response_paths:
        raise FileNotFoundError(f"No response files found under {responses_dir}")
    all_rows = [
        row for path in response_paths for row in read_jsonl(path)
    ]
    protocol_rows = [
        row for row in all_rows if row.get("protocol") == protocol
    ]
    rows = protocol_rows
    selected_raters = set(rater_ids or [])
    if selected_raters:
        rows = [
            row for row in rows if row.get("rater_id") in selected_raters
        ]
    if not rows:
        raise ValueError(f"No responses found for protocol {protocol}")
    frame = pd.DataFrame(rows)
    required = {
        "rater_id",
        "trial_id",
        "dataset",
        "backbone",
        "correctness",
        "task_type",
        "left_condition",
        "right_condition",
        "preference",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Response fields are missing: {sorted(missing)}")
    frame = frame.drop_duplicates(["rater_id", "trial_id"], keep="last").copy()
    frame["decoded_preference"] = frame.apply(decode_choice, axis=1)
    favorable_by_task = {
        "usefulness": "compact_multilevel",
        "grounding": "target_local",
    }
    control_by_task = {
        "usefulness": "global_only",
        "grounding": "matched_video_local",
    }
    frame["preference_score"] = frame.apply(
        lambda row: (
            0
            if row["decoded_preference"] == "tie"
            else (
                1
                if row["decoded_preference"]
                == favorable_by_task[row["task_type"]]
                else -1
            )
        ),
        axis=1,
    )
    for _, row in frame.iterrows():
        allowed = {
            "tie",
            favorable_by_task[row["task_type"]],
            control_by_task[row["task_type"]],
        }
        if row["decoded_preference"] not in allowed:
            raise ValueError(
                f"Invalid decoded preference for {row['task_type']}: "
                f"{row['decoded_preference']}"
            )

    with assignments_path.open(encoding="utf-8", newline="") as handle:
        assignments = list(csv.DictReader(handle))
    if selected_raters:
        assignments = [
            row for row in assignments
            if row["rater_id"] in selected_raters
        ]
    assigned = {(row["rater_id"], row["trial_id"]) for row in assignments}
    observed = set(zip(frame["rater_id"], frame["trial_id"]))
    unexpected = observed - assigned
    if unexpected:
        raise ValueError(f"Found {len(unexpected)} responses outside the assignment")

    summaries: list[dict[str, Any]] = []
    for task_type, task_frame in frame.groupby("task_type"):
        summaries.append(
            summarize_group(
                task_frame,
                task_type,
                "ALL",
                bootstrap_replicates,
                seed + len(summaries),
            )
        )
        for column in ("dataset", "backbone", "correctness"):
            for value, group in task_frame.groupby(column):
                summaries.append(
                    summarize_group(
                        group,
                        task_type,
                        f"{column}={value}",
                        bootstrap_replicates,
                        seed + len(summaries),
                    )
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "responses_decoded.csv", index=False)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output_dir / "preference_summary.csv", index=False)
    agreement = pairwise_agreement(frame)
    coverage = {
        "protocol": protocol,
        "assigned_ratings": len(assigned),
        "completed_ratings": len(observed),
        "completion_rate": len(observed) / len(assigned),
        "unique_trials_completed": int(frame["trial_id"].nunique()),
        "unique_raters_completed": int(frame["rater_id"].nunique()),
        "ignored_other_protocol_rows": len(all_rows) - len(protocol_rows),
        "excluded_other_rater_rows": len(protocol_rows) - len(rows),
        "usefulness_ratings": int((frame["task_type"] == "usefulness").sum()),
        "grounding_ratings": int((frame["task_type"] == "grounding").sum()),
        **agreement,
    }
    with (output_dir / "analysis_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(coverage, handle, indent=2, sort_keys=True)
        handle.write("\n")

    pooled = {
        row["task_type"]: row
        for row in summaries
        if row["group"] == "ALL"
    }
    usefulness = pooled["usefulness"]
    grounding = pooled.get("grounding")
    agreement_text = (
        f"{coverage['exact_agreement']:.1%}"
        if coverage["rating_pairs"]
        else "not estimable (no trial has two evaluated ratings)"
    )
    grounding_section = ""
    if grounding is not None:
        grounding_section = f"""## Local concept grounding

- Target-clip concepts preferred: {grounding['favorable_rate_all']:.1%}
- Matched other-video concepts preferred: {grounding['control_rate_all']:.1%}
- Tie: {grounding['tie_rate']:.1%}
- Target concepts among decisive ratings: {grounding['favorable_rate_decisive']:.1%}
- Mean match score (+1 target, 0 tie, -1 donor):
  {grounding['mean_preference_score']:.3f}
  (trial-bootstrap 95% CI [{grounding['ci95_low']:.3f}, {grounding['ci95_high']:.3f}])

"""
    task_text = (
        "two blinded tasks on disjoint, balanced trials"
        if grounding is not None
        else "blinded usefulness task on balanced trials"
    )
    report = f"""# Human study results

Protocol: `{protocol}`; {task_text}.

- Completed: {coverage['completed_ratings']}/{coverage['assigned_ratings']} assigned ratings
- Raters: {coverage['unique_raters_completed']}
- Unique trials: {coverage['unique_trials_completed']}

## Explanation usefulness

- Compact multi-level preferred: {usefulness['favorable_rate_all']:.1%}
- Global-only preferred: {usefulness['control_rate_all']:.1%}
- Tie: {usefulness['tie_rate']:.1%}
- Compact multi-level among decisive ratings: {usefulness['favorable_rate_decisive']:.1%}
- Mean preference score (+1 compact, 0 tie, -1 global):
  {usefulness['mean_preference_score']:.3f}
  (trial-bootstrap 95% CI [{usefulness['ci95_low']:.3f}, {usefulness['ci95_high']:.3f}])

{grounding_section}- Exact agreement among raters assigned the same trial:
  {agreement_text}

This is an exploratory toy study. The confidence interval resamples trials and
treats the recruited raters as fixed; it does not support population-wide claims
about all possible users.
"""
    (output_dir / "RESULTS.md").write_text(report, encoding="utf-8")
    return coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, default=HERE / "responses")
    parser.add_argument("--assignments", type=Path, default=HERE / "assignments.csv")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--rater-id", action="append", dest="rater_ids")
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metadata = analyze(
        responses_dir=args.responses,
        assignments_path=args.assignments,
        output_dir=args.output,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        rater_ids=args.rater_ids,
        protocol=args.protocol,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
