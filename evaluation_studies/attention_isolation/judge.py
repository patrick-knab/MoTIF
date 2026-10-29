from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from jsonschema import validate
from openai import OpenAI

from attention_isolation.common import CONDITIONS, load_config, read_jsonl


STAGE_DIMENSIONS = {
    "fidelity": (
        "evidence_relevance",
        "concept_identity_coherence",
        "temporal_localization",
        "diagnostic_usefulness",
    ),
    "correction": (
        "error_diagnostic_usefulness",
        "corrective_actionability",
    ),
}


def _blind(stimulus: dict, judge_model: str, seed: int, stage: str) -> dict:
    stable = int(
        hashlib.sha256(
            f"{stimulus['trial_id']}|{judge_model}|{seed}|{stage}".encode()
        ).hexdigest()[:16],
        16,
    )
    rng = random.Random(stable)
    conditions = list(CONDITIONS)
    rng.shuffle(conditions)
    option_ids = [f"Explanation {chr(65 + i)}" for i in range(len(conditions))]
    condition_map = dict(zip(option_ids, conditions))
    pair = [
        next(k for k, v in condition_map.items() if v == "channel_reserved"),
        next(k for k, v in condition_map.items() if v == "full_attention"),
    ]
    rng.shuffle(pair)
    return {
        "condition_map": condition_map,
        "pair": pair,
        "candidates": [
            {"option_id": option, "explanation": stimulus["conditions"][condition]}
            for option, condition in condition_map.items()
        ],
    }


def _schema(option_ids: list[str], pair: list[str], dimensions: tuple[str, ...]):
    rating = {
        "type": "object",
        "properties": {
            dimension: {"type": "integer", "minimum": 1, "maximum": 5}
            for dimension in dimensions
        },
        "required": list(dimensions),
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "ratings": {
                "type": "object",
                "properties": {option: rating for option in option_ids},
                "required": option_ids,
                "additionalProperties": False,
            },
            "main_pair_preference": {
                "type": "string",
                "enum": pair + ["tie"],
            },
        },
        "required": ["ratings", "main_pair_preference"],
        "additionalProperties": False,
    }


def build_prompt(stimulus: dict, blinded: dict, stage: str) -> str:
    candidates = "\n\n".join(
        f"{item['option_id']}:\n"
        + json.dumps(item["explanation"], indent=2, sort_keys=True)
        for item in blinded["candidates"]
    )
    common = f"""You are evaluating explanations of a video classifier.
Watch the complete video. The model prediction is: {stimulus['pred_label']}.
The explanations are anonymized and randomized. Do not guess their source.
Score 1 (absent/contradicted) to 5 (precise and visibly grounded). Do not
reward verbosity or numeric precision by itself."""
    if stage == "fidelity":
        dimensions = """Score:
- evidence_relevance: named concepts are visibly relevant.
- concept_identity_coherence: each named concept consistently denotes its own
  visible semantic evidence, rather than acting as a generic/correlated proxy.
- temporal_localization: stated spans locate the named evidence.
- diagnostic_usefulness: the explanation clarifies why the prediction occurred."""
    else:
        dimensions = f"""The prediction is incorrect; reference class:
{stimulus['true_label']}.
Score:
- error_diagnostic_usefulness: evidence exposes the prediction/reference mismatch.
- corrective_actionability: evidence identifies concrete checks or corrections."""
    left, right = blinded["pair"]
    return f"""{common}

{dimensions}

{candidates}

After rating every candidate, compare only {left} and {right}. Return the better
one or tie as main_pair_preference. Return only the required JSON."""


def _output_path(study_dir: Path, served: str, seed: int, stage: str) -> Path:
    safe = "".join(x if x.isalnum() or x in "._-" else "_" for x in served)
    return study_dir / "responses" / safe / f"seed_{seed}_{stage}.jsonl"


def run(
    config_path: Path,
    study_dir: Path,
    judge_index: int,
    seed: int,
    stage: str,
    base_url: str,
    limit: int | None,
    dry_run: bool,
    retry_failures: bool,
) -> dict:
    config = load_config(config_path)
    judge = config["judges"][judge_index]
    stimuli = read_jsonl(study_dir / "stimuli.internal.jsonl")
    if stage == "correction":
        stimuli = [x for x in stimuli if x["correctness"] == "incorrect"]
    if limit is not None:
        stimuli = stimuli[:limit]
    output = _output_path(study_dir, judge["served_name"], seed, stage)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = read_jsonl(output) if output.exists() else []
    completed = {
        x["request_id"]
        for x in previous
        if x.get("status") == "ok" or not retry_failures
    }
    client = None if dry_run else OpenAI(
        api_key="EMPTY",
        base_url=base_url,
        timeout=float(config["generation"]["request_timeout_seconds"]),
        max_retries=0,
    )
    counts = {"ok": 0, "failed": 0, "skipped": 0, "dry_run": 0}
    with output.open("a", encoding="utf-8") as handle:
        for stimulus in stimuli:
            request_id = hashlib.sha256(
                f"{stimulus['trial_id']}|{judge['model']}|{seed}|{stage}".encode()
            ).hexdigest()[:24]
            if request_id in completed:
                counts["skipped"] += 1
                continue
            blinded = _blind(stimulus, judge["model"], seed, stage)
            prompt = build_prompt(stimulus, blinded, stage)
            option_ids = [x["option_id"] for x in blinded["candidates"]]
            schema = _schema(option_ids, blinded["pair"], STAGE_DIMENSIONS[stage])
            base = {
                "request_id": request_id,
                "trial_id": stimulus["trial_id"],
                "dataset": stimulus["dataset"],
                "correctness": stimulus["correctness"],
                "judge_model": judge["model"],
                "run_seed": seed,
                "stage": stage,
                "condition_map": blinded["condition_map"],
                "main_pair_order": blinded["pair"],
            }
            if dry_run:
                record = {**base, "status": "dry_run", "prompt": prompt}
                counts["dry_run"] += 1
            else:
                parsed = None
                errors = []
                raw = ""
                for attempt in range(int(config["generation"]["max_retries"]) + 1):
                    try:
                        response = client.chat.completions.create(
                            model=judge["served_name"],
                            messages=[{
                                "role": "user",
                                "content": [
                                    {
                                        "type": "video_url",
                                        "video_url": {
                                            "url": Path(stimulus["video_path"])
                                            .resolve()
                                            .as_uri()
                                        },
                                    },
                                    {"type": "text", "text": prompt},
                                ],
                            }],
                            temperature=float(config["generation"]["temperature"]),
                            top_p=float(config["generation"]["top_p"]),
                            seed=seed,
                            max_completion_tokens=int(
                                config["generation"]["max_completion_tokens"]
                            ),
                            response_format={
                                "type": "json_schema",
                                "json_schema": {
                                    "name": f"attention_isolation_{stage}",
                                    "strict": True,
                                    "schema": schema,
                                },
                            },
                            extra_body={"chat_template_kwargs": {
                                "enable_thinking": bool(
                                    config["generation"]["enable_thinking"]
                                )
                            }},
                        )
                        raw = response.choices[0].message.content or ""
                        parsed = json.loads(raw)
                        validate(parsed, schema)
                        break
                    except Exception as exc:
                        errors.append(f"{type(exc).__name__}: {exc}")
                        if attempt < int(config["generation"]["max_retries"]):
                            time.sleep(
                                float(config["generation"]["retry_backoff_seconds"])
                                * (attempt + 1)
                            )
                if parsed is None:
                    record = {
                        **base,
                        "status": "failed",
                        "errors": errors,
                        "raw_response": raw,
                    }
                    counts["failed"] += 1
                else:
                    record = {
                        **base,
                        "status": "ok",
                        "ratings": parsed["ratings"],
                        "main_pair_preference": parsed["main_pair_preference"],
                        "raw_response": raw,
                    }
                    counts["ok"] += 1
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--judge-index", type=int, required=True)
    parser.add_argument("--run-seed", type=int, required=True)
    parser.add_argument("--stage", choices=tuple(STAGE_DIMENSIONS), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config.resolve(),
                args.study_dir.resolve(),
                args.judge_index,
                args.run_seed,
                args.stage,
                args.base_url,
                args.limit,
                args.dry_run,
                args.retry_failures,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
