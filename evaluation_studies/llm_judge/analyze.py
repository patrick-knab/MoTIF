from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from llm_judge.common import load_config, read_jsonl, write_json


def _seed_for(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def load_latest_responses(study_dir: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    response_dir = study_dir / "responses"
    if not response_dir.exists():
        return []
    for path in sorted(response_dir.rglob("*.jsonl")):
        for row in read_jsonl(path):
            if row.get("status") == "dry_run":
                continue
            latest[str(row["request_id"])] = row
    return list(latest.values())


def decode_responses(
    responses: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, Any]] = []
    preference_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
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
        status_rows.append(
            {
                **base,
                "status": response["status"],
                "media_transport_ok": response.get(
                    "media_transport_ok",
                    response.get("video_observable"),
                ),
            }
        )
        if response["status"] != "ok":
            continue
        condition_map = response["condition_map"]
        for option_id, dimension_scores in response["ratings"].items():
            condition = condition_map[option_id]
            for dimension, score in dimension_scores.items():
                score_rows.append(
                    {
                        **base,
                        "condition": condition,
                        "dimension": dimension,
                        "score": int(score),
                    }
                )
        raw_preference = response["main_pair_preference"]
        preference = (
            "tie" if raw_preference == "tie" else condition_map[raw_preference]
        )
        if preference not in {"full_three_level", "global_only", "tie"}:
            raise ValueError(
                f"Invalid decoded main-pair preference: {preference}"
            )
        preference_rows.append({**base, "preference": preference})
    return (
        pd.DataFrame(score_rows),
        pd.DataFrame(preference_rows),
        pd.DataFrame(status_rows),
    )


def _scopes(frame: pd.DataFrame) -> Iterable[tuple[str, str, str, pd.DataFrame]]:
    judge_values = ["ALL"] + sorted(frame["judge_model"].unique().tolist())
    dataset_values = ["ALL"] + sorted(frame["dataset"].unique().tolist())
    correctness_values = ["ALL"] + sorted(frame["correctness"].unique().tolist())
    for judge_model in judge_values:
        for dataset in dataset_values:
            for correctness in correctness_values:
                part = frame
                if judge_model != "ALL":
                    part = part[part["judge_model"].eq(judge_model)]
                if dataset != "ALL":
                    part = part[part["dataset"].eq(dataset)]
                if correctness != "ALL":
                    part = part[part["correctness"].eq(correctness)]
                if not part.empty:
                    yield judge_model, dataset, correctness, part


def _cluster_bootstrap(
    trial_frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    if trial_frame.empty:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    strata = [
        part.copy()
        for _, part in trial_frame.groupby(
            ["dataset", "correctness"], sort=True, dropna=False
        )
    ]
    estimates = np.empty(replicates, dtype=np.float64)
    for bootstrap_index in range(replicates):
        sampled_parts = []
        for part in strata:
            indices = rng.integers(0, len(part), size=len(part))
            sampled_parts.append(part.iloc[indices])
        estimates[bootstrap_index] = statistic(
            pd.concat(sampled_parts, ignore_index=True)
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def summarize_scores(
    scores: pd.DataFrame, replicates: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for judge_model, dataset, correctness, scoped in _scopes(scores):
        for (stage, dimension, condition), part in scoped.groupby(
            ["stage", "dimension", "condition"], sort=True
        ):
            trial_means = (
                part.groupby(
                    ["trial_id", "dataset", "correctness"], as_index=False
                )["score"]
                .mean()
                .rename(columns={"score": "value"})
            )
            mean = float(trial_means["value"].mean())
            low, high = _cluster_bootstrap(
                trial_means,
                lambda data: float(data["value"].mean()),
                replicates,
                _seed_for(
                    "score", judge_model, dataset, correctness, stage, dimension, condition
                ),
            )
            rows.append(
                {
                    "judge_model": judge_model,
                    "dataset": dataset,
                    "correctness": correctness,
                    "stage": stage,
                    "dimension": dimension,
                    "condition": condition,
                    "n_videos": int(trial_means["trial_id"].nunique()),
                    "mean_score": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def paired_differences(
    scores: pd.DataFrame,
    left_condition: str,
    right_condition: str,
    replicates: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for judge_model, dataset, correctness, scoped in _scopes(scores):
        for (stage, dimension), part in scoped.groupby(
            ["stage", "dimension"], sort=True
        ):
            selected = part[
                part["condition"].isin([left_condition, right_condition])
            ]
            panel_pairs = selected.pivot_table(
                index=[
                    "trial_id",
                    "dataset",
                    "correctness",
                    "panel_id",
                ],
                columns="condition",
                values="score",
                aggfunc="first",
            ).dropna()
            if (
                panel_pairs.empty
                or left_condition not in panel_pairs
                or right_condition not in panel_pairs
            ):
                continue
            panel_pairs["difference"] = (
                panel_pairs[left_condition] - panel_pairs[right_condition]
            )
            trial_differences = (
                panel_pairs.reset_index()
                .groupby(
                    ["trial_id", "dataset", "correctness"], as_index=False
                )["difference"]
                .mean()
                .rename(columns={"difference": "value"})
            )
            mean = float(trial_differences["value"].mean())
            low, high = _cluster_bootstrap(
                trial_differences,
                lambda data: float(data["value"].mean()),
                replicates,
                _seed_for(
                    "delta",
                    left_condition,
                    right_condition,
                    judge_model,
                    dataset,
                    correctness,
                    stage,
                    dimension,
                ),
            )
            rows.append(
                {
                    "judge_model": judge_model,
                    "dataset": dataset,
                    "correctness": correctness,
                    "stage": stage,
                    "dimension": dimension,
                    "left_condition": left_condition,
                    "right_condition": right_condition,
                    "n_videos": int(trial_differences["trial_id"].nunique()),
                    "mean_difference": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def summarize_preferences(
    preferences: pd.DataFrame, replicates: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    outcomes = ("full_three_level", "global_only", "tie")
    for judge_model, dataset, correctness, scoped in _scopes(preferences):
        for stage, part in scoped.groupby("stage", sort=True):
            counts = (
                part.assign(value=1.0)
                .pivot_table(
                    index=["trial_id", "dataset", "correctness"],
                    columns="preference",
                    values="value",
                    aggfunc="sum",
                    fill_value=0.0,
                )
                .reset_index()
            )
            for outcome in outcomes:
                if outcome not in counts:
                    counts[outcome] = 0.0
            denominator = counts[list(outcomes)].sum(axis=1)
            for outcome in outcomes:
                counts[f"{outcome}_rate"] = counts[outcome] / denominator

            result = {
                "judge_model": judge_model,
                "dataset": dataset,
                "correctness": correctness,
                "stage": stage,
                "n_videos": int(counts["trial_id"].nunique()),
            }
            for outcome in outcomes:
                column = f"{outcome}_rate"
                result[f"{outcome}_rate"] = float(counts[column].mean())
                low, high = _cluster_bootstrap(
                    counts,
                    lambda data, name=column: float(data[name].mean()),
                    replicates,
                    _seed_for(
                        "preference", judge_model, dataset, correctness, stage, outcome
                    ),
                )
                result[f"{outcome}_ci_low"] = low
                result[f"{outcome}_ci_high"] = high
            rows.append(result)
    return pd.DataFrame(rows)


def krippendorff_alpha(
    units_by_panel: pd.DataFrame,
    level: str,
) -> float:
    categories = sorted(
        {
            value
            for value in units_by_panel.to_numpy().flatten().tolist()
            if pd.notna(value)
        }
    )
    if len(categories) < 2:
        return float("nan")
    category_index = {category: index for index, category in enumerate(categories)}
    coincidence = np.zeros((len(categories), len(categories)), dtype=np.float64)
    for values in units_by_panel.itertuples(index=False, name=None):
        observed = [value for value in values if pd.notna(value)]
        count = len(observed)
        if count < 2:
            continue
        weight = 1.0 / (count - 1)
        for left_position, left in enumerate(observed):
            for right_position, right in enumerate(observed):
                if left_position == right_position:
                    continue
                coincidence[category_index[left], category_index[right]] += weight
    marginals = coincidence.sum(axis=1)
    total = float(marginals.sum())
    if total <= 1:
        return float("nan")
    expected = np.outer(marginals, marginals) / (total - 1.0)
    expected[np.diag_indices_from(expected)] = (
        marginals * (marginals - 1.0) / (total - 1.0)
    )
    if level == "nominal":
        distance = np.ones_like(coincidence) - np.eye(len(categories))
    elif level == "ordinal":
        distance = np.zeros_like(coincidence)
        for left in range(len(categories)):
            for right in range(left + 1, len(categories)):
                cumulative = marginals[left : right + 1].sum()
                delta = cumulative - (marginals[left] + marginals[right]) / 2.0
                distance[left, right] = distance[right, left] = delta * delta
    else:
        raise ValueError(f"Unsupported measurement level: {level}")
    observed_disagreement = float((coincidence * distance).sum())
    expected_disagreement = float((expected * distance).sum())
    if expected_disagreement == 0:
        return float("nan")
    return 1.0 - observed_disagreement / expected_disagreement


def agreement_tables(
    scores: pd.DataFrame, preferences: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    judge_scopes = ["ALL"] + sorted(scores["judge_model"].unique().tolist())
    for judge_model in judge_scopes:
        score_part = (
            scores
            if judge_model == "ALL"
            else scores[scores["judge_model"].eq(judge_model)]
        )
        for (stage, dimension, condition), part in score_part.groupby(
            ["stage", "dimension", "condition"], sort=True
        ):
            matrix = part.pivot_table(
                index="trial_id",
                columns="panel_id",
                values="score",
                aggfunc="first",
            )
            rows.append(
                {
                    "judge_model": judge_model,
                    "stage": stage,
                    "measure": dimension,
                    "condition": condition,
                    "level": "ordinal",
                    "n_units": int(len(matrix)),
                    "n_panels": int(len(matrix.columns)),
                    "alpha": krippendorff_alpha(matrix, "ordinal"),
                }
            )

        preference_part = (
            preferences
            if judge_model == "ALL"
            else preferences[preferences["judge_model"].eq(judge_model)]
        )
        for stage, part in preference_part.groupby("stage", sort=True):
            matrix = part.pivot_table(
                index="trial_id",
                columns="panel_id",
                values="preference",
                aggfunc="first",
            )
            rows.append(
                {
                    "judge_model": judge_model,
                    "stage": stage,
                    "measure": "main_pair_preference",
                    "condition": "full_vs_global",
                    "level": "nominal",
                    "n_units": int(len(matrix)),
                    "n_panels": int(len(matrix.columns)),
                    "alpha": krippendorff_alpha(matrix, "nominal"),
                }
            )
    return pd.DataFrame(rows)


def control_checks(scores: pd.DataFrame, replicates: int) -> pd.DataFrame:
    comparisons = (
        ("fidelity", "temporal_localization", "full_three_level", "shuffled_windows"),
        ("fidelity", "evidence_relevance", "full_three_level", "random_concepts"),
        ("fidelity", "specificity", "full_three_level", "random_concepts"),
        ("fidelity", "specificity", "full_three_level", "class_name_only"),
        (
            "fidelity",
            "diagnostic_usefulness",
            "full_three_level",
            "class_name_only",
        ),
        (
            "correction",
            "error_diagnostic_usefulness",
            "full_three_level",
            "class_name_only",
        ),
        (
            "correction",
            "corrective_actionability",
            "full_three_level",
            "class_name_only",
        ),
    )
    outputs = []
    for stage, dimension, left, right in comparisons:
        subset = scores[
            scores["stage"].eq(stage) & scores["dimension"].eq(dimension)
        ]
        delta = paired_differences(subset, left, right, replicates)
        pooled = delta[
            delta["judge_model"].eq("ALL")
            & delta["dataset"].eq("ALL")
            & delta["correctness"].eq("ALL")
        ].copy()
        pooled["validity_pass"] = pooled["ci_low"] > 0.0
        outputs.append(pooled)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def primary_endpoints(
    full_vs_global: pd.DataFrame, relevance_margin: float
) -> pd.DataFrame:
    pooled = full_vs_global[
        full_vs_global["judge_model"].eq("ALL")
        & full_vs_global["dataset"].eq("ALL")
        & full_vs_global["correctness"].eq("ALL")
    ].copy()
    pooled["criterion"] = "superiority"
    pooled["margin"] = 0.0
    relevance = pooled["dimension"].eq("evidence_relevance")
    pooled.loc[relevance, "criterion"] = "noninferiority"
    pooled.loc[relevance, "margin"] = float(relevance_margin)
    pooled["criterion_pass"] = pooled["ci_low"] > pooled["margin"]
    return pooled


def _paper_table(
    primary: pd.DataFrame,
    preferences: pd.DataFrame,
    controls: pd.DataFrame,
) -> str:
    sections = ["% Generated by python -m llm_judge.analyze"]
    if not primary.empty:
        table = primary[
            [
                "stage",
                "dimension",
                "mean_difference",
                "ci_low",
                "ci_high",
                "criterion_pass",
            ]
        ].copy()
        sections.append(table.to_latex(index=False, float_format="%.3f"))
    pooled_preferences = preferences[
        preferences["judge_model"].eq("ALL")
        & preferences["dataset"].eq("ALL")
        & preferences["correctness"].eq("ALL")
    ]
    if not pooled_preferences.empty:
        table = pooled_preferences[
            [
                "stage",
                "full_three_level_rate",
                "global_only_rate",
                "tie_rate",
            ]
        ].copy()
        sections.append(table.to_latex(index=False, float_format="%.3f"))
    if not controls.empty:
        table = controls[
            [
                "stage",
                "dimension",
                "right_condition",
                "mean_difference",
                "ci_low",
                "ci_high",
                "validity_pass",
            ]
        ].copy()
        sections.append(table.to_latex(index=False, float_format="%.3f"))
    return "\n\n".join(sections) + "\n"


def analyze(
    config_path: Path,
    study_dir: Path,
    output_dir: Path,
    bootstrap_replicates: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    replicates = int(
        bootstrap_replicates
        if bootstrap_replicates is not None
        else config["analysis"]["bootstrap_replicates"]
    )
    responses = load_latest_responses(study_dir)
    scores, preferences, statuses = decode_responses(responses)
    if scores.empty:
        raise ValueError(f"No valid judge scores found under {study_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    score_summary = summarize_scores(scores, replicates)
    full_vs_global = paired_differences(
        scores, "full_three_level", "global_only", replicates
    )
    preference_summary = summarize_preferences(preferences, replicates)
    agreement = agreement_tables(scores, preferences)
    controls = control_checks(scores, replicates)
    primary = primary_endpoints(
        full_vs_global,
        float(config["analysis"]["relevance_noninferiority_margin"]),
    )

    tables = {
        "scores_long.csv": scores,
        "preferences_long.csv": preferences,
        "response_status.csv": statuses,
        "score_summary.csv": score_summary,
        "full_vs_global.csv": full_vs_global,
        "preference_summary.csv": preference_summary,
        "agreement.csv": agreement,
        "control_checks.csv": controls,
        "primary_endpoints.csv": primary,
    }
    for filename, frame in tables.items():
        frame.to_csv(output_dir / filename, index=False)
    with (output_dir / "paper_summary.tex").open("w", encoding="utf-8") as handle:
        handle.write(_paper_table(primary, preference_summary, controls))

    metadata = {
        "input_response_records": len(responses),
        "valid_score_rows": len(scores),
        "valid_preference_rows": len(preferences),
        "failed_responses": int((statuses["status"] == "failed").sum()),
        "media_transport_failures": int(
            (
                statuses["status"].eq("ok")
                & statuses["media_transport_ok"].eq(False)
            ).sum()
        ),
        "bootstrap_replicates": replicates,
        "all_primary_criteria_pass": bool(
            not primary.empty and primary["criterion_pass"].all()
        ),
        "all_control_checks_pass": bool(
            not controls.empty and controls["validity_pass"].all()
        ),
    }
    write_json(output_dir / "analysis_metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze blinded MoTIF LLM-judge responses."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
    )
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=Path(__file__).with_name("results") / "study",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("results") / "analysis",
    )
    parser.add_argument("--bootstrap-replicates", type=int)
    args = parser.parse_args()
    metadata = analyze(
        args.config.resolve(),
        args.study_dir.resolve(),
        args.output.resolve(),
        args.bootstrap_replicates,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
