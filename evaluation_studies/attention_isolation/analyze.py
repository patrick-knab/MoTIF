from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from attention_isolation.common import load_config, read_jsonl, write_json
from llm_judge.analyze import (
    agreement_tables as _original_agreement_tables,
    paired_differences,
    summarize_scores,
)


def _latest_responses(study_dir: Path) -> list[dict]:
    latest = {}
    response_root = study_dir / "responses"
    if not response_root.exists():
        return []
    for path in sorted(response_root.rglob("*.jsonl")):
        for row in read_jsonl(path):
            if row.get("status") != "dry_run":
                latest[row["request_id"]] = row
    return list(latest.values())


def decode(responses: list[dict]):
    score_rows, preference_rows, status_rows = [], [], []
    for response in responses:
        base = {
            "request_id": response["request_id"],
            "trial_id": response["trial_id"],
            "dataset": response["dataset"],
            "correctness": response["correctness"],
            "judge_model": response["judge_model"],
            "run_seed": int(response["run_seed"]),
            "panel_id": f"{response['judge_model']}|{response['run_seed']}",
            "stage": response["stage"],
        }
        status_rows.append({**base, "status": response["status"]})
        if response["status"] != "ok":
            continue
        mapping = response["condition_map"]
        for option, ratings in response["ratings"].items():
            for dimension, score in ratings.items():
                score_rows.append(
                    {
                        **base,
                        "condition": mapping[option],
                        "dimension": dimension,
                        "score": int(score),
                    }
                )
        raw = response["main_pair_preference"]
        preference_rows.append(
            {
                **base,
                "preference": "tie" if raw == "tie" else mapping[raw],
            }
        )
    return (
        pd.DataFrame(score_rows),
        pd.DataFrame(preference_rows),
        pd.DataFrame(status_rows),
    )


def preference_summary(preferences: pd.DataFrame, replicates: int, seed: int):
    rows = []
    scopes = [("ALL", "ALL", preferences)]
    for dataset, part in preferences.groupby("dataset"):
        scopes.append(("ALL", dataset, part))
    for judge, judge_part in preferences.groupby("judge_model"):
        scopes.append((judge, "ALL", judge_part))
        for dataset, part in judge_part.groupby("dataset"):
            scopes.append((judge, dataset, part))
    rng = np.random.default_rng(seed)
    for judge, dataset, scoped in scopes:
        for stage, part in scoped.groupby("stage"):
            trial = (
                part.groupby("trial_id")["preference"]
                .value_counts(normalize=True)
                .unstack(fill_value=0.0)
            )
            for name in ("channel_reserved", "full_attention", "tie"):
                if name not in trial:
                    trial[name] = 0.0
            row = {
                "judge_model": judge,
                "dataset": dataset,
                "stage": stage,
                "n_videos": len(trial),
            }
            for name in ("channel_reserved", "full_attention", "tie"):
                values = trial[name].to_numpy()
                estimates = [
                    float(values[rng.integers(0, len(values), len(values))].mean())
                    for _ in range(replicates)
                ]
                low, high = np.quantile(estimates, [0.025, 0.975])
                row[f"{name}_rate"] = float(values.mean())
                row[f"{name}_ci_low"] = float(low)
                row[f"{name}_ci_high"] = float(high)
            rows.append(row)
    return pd.DataFrame(rows)


def agreement(scores: pd.DataFrame, preferences: pd.DataFrame) -> pd.DataFrame:
    # The shared implementation is generic except for its display-only pair label.
    output = _original_agreement_tables(scores, preferences)
    output.loc[
        output["measure"].eq("main_pair_preference"), "condition"
    ] = "channel_reserved_vs_full_attention"
    return output


def analyze(
    config_path: Path,
    study_dir: Path,
    output_dir: Path,
    bootstrap_replicates: int | None = None,
) -> dict:
    config = load_config(config_path)
    replicates = int(
        bootstrap_replicates
        if bootstrap_replicates is not None
        else config["analysis"]["bootstrap_replicates"]
    )
    scores, preferences, statuses = decode(_latest_responses(study_dir))
    if scores.empty:
        raise ValueError(f"No valid responses under {study_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    score_summary = summarize_scores(scores, replicates)
    primary = paired_differences(
        scores, "channel_reserved", "full_attention", replicates
    )
    # The architecture-specific hypothesis is better semantic identity
    # coherence. Other explanation dimensions are guarded for noninferiority.
    primary["criterion"] = np.where(
        primary["dimension"].eq("concept_identity_coherence"),
        "superiority",
        "noninferiority",
    )
    primary["margin"] = np.where(
        primary["criterion"].eq("noninferiority"),
        float(config["analysis"]["isolation_noninferiority_margin"]),
        0.0,
    )
    primary["criterion_pass"] = primary["ci_low"] > primary["margin"]
    scrambled_dimensions = {
        "evidence_relevance",
        "concept_identity_coherence",
    }
    class_only_dimensions = {
        "temporal_localization",
        "diagnostic_usefulness",
        "error_diagnostic_usefulness",
        "corrective_actionability",
    }
    controls = pd.concat(
        [
            paired_differences(
                scores[scores["dimension"].isin(scrambled_dimensions)],
                "channel_reserved",
                "scrambled_concepts",
                replicates,
            ),
            paired_differences(
                scores[scores["dimension"].isin(class_only_dimensions)],
                "channel_reserved",
                "class_name_only",
                replicates,
            ),
        ],
        ignore_index=True,
    )
    controls["validity_pass"] = controls["ci_low"] > 0.0
    preferences_summary = preference_summary(
        preferences, replicates, int(config["study_seed"])
    )
    tables = {
        "scores_long.csv": scores,
        "preferences_long.csv": preferences,
        "response_status.csv": statuses,
        "score_summary.csv": score_summary,
        "channel_reserved_vs_full_attention.csv": primary,
        "channel_reserved_vs_scrambled_control.csv": controls,
        "preference_summary.csv": preferences_summary,
        "agreement.csv": agreement(scores, preferences),
    }
    for name, frame in tables.items():
        frame.to_csv(output_dir / name, index=False)
    pooled_primary = primary[
        primary["judge_model"].eq("ALL")
        & primary["dataset"].eq("ALL")
        & primary["correctness"].eq("ALL")
    ]
    pooled_control = controls[
        controls["judge_model"].eq("ALL")
        & controls["dataset"].eq("ALL")
        & controls["correctness"].eq("ALL")
    ]
    metadata = {
        "bootstrap_replicates": replicates,
        "valid_responses": int((statuses["status"] == "ok").sum()),
        "failed_responses": int((statuses["status"] == "failed").sum()),
        "all_pooled_primary_criteria_pass": bool(
            not pooled_primary.empty and pooled_primary["criterion_pass"].all()
        ),
        "all_pooled_control_checks_pass": bool(
            not pooled_control.empty and pooled_control["validity_pass"].all()
        ),
        "claim_boundary": (
            "LLM scores measure explanation quality and semantic coherence. "
            "Functional concept isolation is assessed by the separate perturbation test."
        ),
    }
    write_json(output_dir / "analysis_metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("results") / "analysis",
    )
    parser.add_argument("--bootstrap-replicates", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                args.config.resolve(),
                args.study_dir.resolve(),
                args.output.resolve(),
                args.bootstrap_replicates,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
