#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

AGENTMOTIF_DIR = Path(__file__).resolve().parent.parent
if str(AGENTMOTIF_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTMOTIF_DIR))
if str(AGENTMOTIF_DIR / "utils") not in sys.path:
    sys.path.insert(0, str(AGENTMOTIF_DIR / "utils"))
if str(AGENTMOTIF_DIR.parent) not in sys.path:
    sys.path.insert(0, str(AGENTMOTIF_DIR.parent))

DATASET_CONFIG = {
    "breakfast": {
        "dataset_name": "Breakfast",
        "concept_subdir": "breakfast",
        "default_window_size": 32,
        "default_test_splits": ["s1", "s2", "s3", "s4"],
    },
    "hmdb51": {
        "dataset_name": "HMDB",
        "concept_subdir": "hmdb",
        "default_window_size": 32,
        "default_test_splits": ["s1", "s2", "s3"],
    },
    "ucf101": {
        "dataset_name": "UCF101",
        "concept_subdir": "ucf101",
        "default_window_size": 32,
        "default_test_splits": ["s1", "s2", "s3"],
    },
    "something2": {
        "dataset_name": "Something2",
        "concept_subdir": "ssv2",
        "default_window_size": 32,
        "default_test_splits": ["s1"],
    },
}

ABLATION_STAGES = [
    "json_only",
    "json+action",
    "json+action+visual",
    "json+action+visual+motion",
    "action_only",
    "visual_only",
]


def load_runtime_dependencies():
    from ablation_toolkit.utils.motif import init_repro

    import clip
    import wandb
    from transformers import (
        AutoModel,
        AutoProcessor,
        CLIPModel,
        CLIPProcessor,
        CLIPTextModelWithProjection,
        CLIPTokenizer,
        CLIPVisionModelWithProjection,
    )

    from ablation_toolkit.utils.concept_handling import (
        get_test_split_instances,
        load_agentsam_concepts,
        move_model_to_cpu,
        process_concepts,
    )
    from ablation_toolkit.utils.motif import CBMTransformer, MoTIF, mean_cbm
    from ablation_toolkit.utils.video_embedder import Create_Concepts, VideoEmbedder

    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as pe_transformer

    return {
        "init_repro": init_repro,
        "clip": clip,
        "wandb": wandb,
        "AutoModel": AutoModel,
        "AutoProcessor": AutoProcessor,
        "CLIPModel": CLIPModel,
        "CLIPProcessor": CLIPProcessor,
        "CLIPTextModelWithProjection": CLIPTextModelWithProjection,
        "CLIPTokenizer": CLIPTokenizer,
        "CLIPVisionModelWithProjection": CLIPVisionModelWithProjection,
        "get_test_split_instances": get_test_split_instances,
        "load_agentsam_concepts": load_agentsam_concepts,
        "move_model_to_cpu": move_model_to_cpu,
        "process_concepts": process_concepts,
        "CBMTransformer": CBMTransformer,
        "MoTIF": MoTIF,
        "mean_cbm": mean_cbm,
        "Create_Concepts": Create_Concepts,
        "VideoEmbedder": VideoEmbedder,
        "pe": pe,
        "pe_transformer": pe_transformer,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one AgentMoTIF ablation with configurable dataset and stage."
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIG), required=True)
    parser.add_argument("--clip-model", required=True)
    parser.add_argument("--ablation-stage", choices=ABLATION_STAGES, required=True)
    parser.add_argument("--agentsam-run-folder", required=True)
    parser.add_argument("--test-split", required=True)

    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--similarity-threshold", type=float, default=0.9)

    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lse-tau", type=float, default=1.0)
    parser.add_argument("--l1-lambda", type=float, default=1e-3)
    parser.add_argument("--lambda-sparse", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument("--classifier-layers", type=int, default=1)
    parser.add_argument("--enforce-nonneg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagonal-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="agent-motif")
    parser.add_argument("--wandb-mode", type=str, default=None)
    parser.add_argument("--concept-root", type=str, default=None)
    return parser.parse_args()


def get_ablation_stage_display_name(ablation_stage: str) -> str:
    mapping = {
        "json_only": "text",
        "json+action": "text+action",
        "json+action+visual": "text+action+visual",
        "json+action+visual+motion": "text+action+visual+motion",
    }
    return mapping.get(ablation_stage, ablation_stage)


def build_models(deps: dict, clip_model: str, dataset_name: str, random_flag: bool, window_size: int):
    model_text = None
    tokenizer = None
    if clip_model == "b32":
        model = deps["CLIPModel"].from_pretrained("openai/clip-vit-base-patch32").eval()
        processor = deps["CLIPProcessor"].from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
        embedd_path = f"../Embeddings/Videos/{dataset_name}/{random_flag}_{window_size}_clip_b32_42.pkl"
        clip_name = "clip"
    elif clip_model == "b16":
        model = deps["CLIPModel"].from_pretrained("openai/clip-vit-base-patch16").eval()
        processor = deps["CLIPProcessor"].from_pretrained("openai/clip-vit-base-patch16", use_fast=False)
        embedd_path = f"../Embeddings/Videos/{dataset_name}/{random_flag}_{window_size}_clip_b16.pkl"
        clip_name = "clip"
    elif clip_model == "l14":
        model = deps["CLIPModel"].from_pretrained("openai/clip-vit-large-patch14").eval()
        processor = deps["CLIPProcessor"].from_pretrained("openai/clip-vit-large-patch14", use_fast=False)
        embedd_path = f"../Embeddings/Videos/{dataset_name}/{random_flag}_{window_size}_clip_l14.pkl"
        clip_name = "clip"
    elif clip_model == "res50":
        model, preprocess = deps["clip"].load("RN50", device="cpu")
        processor = preprocess
        embedd_path = f"../Embeddings/Videos/{dataset_name}/{random_flag}_{window_size}_clip_res50_42.pkl"
        clip_name = "res50"
    elif clip_model == "clip4clip":
        model = deps["CLIPVisionModelWithProjection"].from_pretrained("Searchium-ai/clip4clip-webvid150k").eval()
        model_text = deps["CLIPTextModelWithProjection"].from_pretrained("Searchium-ai/clip4clip-webvid150k")
        processor = deps["CLIPTokenizer"].from_pretrained("Searchium-ai/clip4clip-webvid150k")
        embedd_path = f"../Embeddings/Videos/{dataset_name}/{random_flag}_{window_size}_clip_clip4clip.pkl"
        clip_name = "clip4clip"
    elif clip_model == "siglip":
        model = deps["AutoModel"].from_pretrained("google/siglip-base-patch16-224")
        processor = deps["AutoProcessor"].from_pretrained("google/siglip-base-patch16-224")
        embedd_path = f"../Embeddings/Videos/{dataset_name}/{random_flag}_{window_size}_clip_siglip.pkl"
        clip_name = "siglip"
    elif clip_model == "siglipl14":
        model = deps["AutoModel"].from_pretrained("google/siglip-so400m-patch14-384")
        processor = deps["AutoProcessor"].from_pretrained("google/siglip-so400m-patch14-384")
        embedd_path = f"../Embeddings/Videos/{dataset_name}/{random_flag}_{window_size}_clip_siglipl14.pkl"
        clip_name = "siglipl14"
    elif clip_model == "pe-l14":
        model = deps["pe"].CLIP.from_config("PE-Core-L14-336", pretrained=True)
        processor = deps["pe_transformer"].get_image_transform(model.image_size)
        tokenizer = deps["pe_transformer"].get_text_tokenizer(model.context_length)
        embedd_path = f"../Embeddings/Videos/{dataset_name}/{random_flag}_{window_size}_clip_pe-l14.pkl"
        clip_name = "pe-l14"
    elif clip_model == "pe-g14":
        model = deps["pe"].CLIP.from_config("PE-Core-G14-448", pretrained=True)
        processor = deps["pe_transformer"].get_image_transform(model.image_size)
        tokenizer = deps["pe_transformer"].get_text_tokenizer(model.context_length)
        embedd_path = f"../Embeddings/Videos/{dataset_name}/{random_flag}_{window_size}_clip_pe-g14_42.pkl"
        clip_name = "pe-g14"
    else:
        raise ValueError(f"Unknown clip_model {clip_model}")

    return model, processor, model_text, tokenizer, embedd_path, clip_name


def run_experiment(hparams: dict) -> None:
    deps = load_runtime_dependencies()
    dataset = hparams["dataset"]
    cfg = DATASET_CONFIG[dataset]
    dataset_name = cfg["dataset_name"]
    clip_model = hparams["clip_model"]
    window_size = hparams["window_size"]
    random_flag = hparams["random"]
    ablation_stage = hparams["ablation_stage"]
    agentsam_run_folder = hparams["agentsam_run_folder"]
    seed = hparams["seed"]

    deps["init_repro"](seed, deterministic=True)

    model, processor, model_text, tokenizer, embedd_path, clip_name = build_models(
        deps=deps,
        clip_model=clip_model,
        dataset_name=dataset_name,
        random_flag=random_flag,
        window_size=window_size,
    )

    embedder = deps["VideoEmbedder"](clip_name, model, processor)
    embedder.dataset_name = dataset

    if os.path.exists(embedd_path):
        with open(embedd_path, "rb") as f:
            embedder = pickle.load(f)
            embedder.attach_backends(model=model, tokenizer=processor, clip_model=None)
            print("Loaded existing embedder from", embedd_path)
    else:
        folder_path = [f"../Datasets/{dataset_name}/Video_data"]
        embedder.process_data(folder_path, window_size=window_size, output_path="../Embeddings/Datasets")
        with open(embedd_path, "wb") as f:
            pickle.dump(embedder, f)

    if clip_model == "clip4clip":
        concepts = deps["Create_Concepts"](clip_name, model_text, processor)
    elif clip_model in {"pe-l14", "pe-g14"}:
        concepts = deps["Create_Concepts"](clip_name, model, tokenizer)
    else:
        concepts = deps["Create_Concepts"](clip_name, model, processor)

    concept_root = hparams.get("concept_root")
    if concept_root:
        concept_dir = os.path.abspath(os.path.join(concept_root, agentsam_run_folder))
    else:
        concept_dir = os.path.abspath(
            os.path.join("..", "concept_extraction_out_batch", cfg["concept_subdir"], agentsam_run_folder)
        )
    if not os.path.exists(concept_dir):
        raise ValueError(f"Concept directory not found: {concept_dir}")

    test_split_instances = deps["get_test_split_instances"](dataset, hparams["test_split"])
    print(
        f"Found {len(test_split_instances)} instances in test split "
        f"'{hparams['test_split']}' to exclude from concepts"
    )

    print(f"Loading agentSAM concepts from {concept_dir} with ablation stage: {ablation_stage}")
    agentsam_data = deps["load_agentsam_concepts"](
        concept_dir,
        ablation_stage=ablation_stage,
        test_split_instances=test_split_instances,
    )

    concept_result = deps["process_concepts"](
        embedder=embedder,
        concepts=concepts,
        agentsam_data=agentsam_data,
        ablation_stage=ablation_stage,
        clip_model=clip_model,
        similarity_threshold=hparams["similarity_threshold"],
    )

    concepts = concept_result["concepts"]
    concept_metadata_list = concept_result["concept_metadata_list"]
    raw_counts = concept_result.get("raw_counts", agentsam_data.get("concept_counts", {}))
    filtered_counts = concept_result.get("filtered_counts", {})
    timing = concept_result["timing"]

    cbm_model = deps["MoTIF"](embedder, concepts)
    cbm_model.preprocess(dataset, info=hparams["test_split"], random_state=seed)
    cbm_model.model = deps["CBMTransformer"](
        cbm_model.num_concepts,
        num_classes=cbm_model.num_classes,
        transformer_layers=hparams["transformer_layers"],
        classifier_layers=hparams["classifier_layers"],
        lse_tau=hparams["lse_tau"],
        dimension=hparams["d"],
        diagonal_attention=hparams["diagonal_attention"],
    )

    time_now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    ablation_display_name = get_ablation_stage_display_name(ablation_stage)
    run_name = f"{dataset}_{clip_model}_{ablation_stage}_{time_now}"

    if hparams.get("wandb_mode"):
        os.environ["WANDB_MODE"] = hparams["wandb_mode"]
    wandb_run = deps["wandb"].init(project=hparams["wandb_project"], name=run_name, config=hparams)

    total_embed_time = (
        timing["text_embed"]
        + timing["visual_embed"]
        + timing["motion_embed"]
        + timing["filter"]
    )
    wandb_run.log(
        {
            "test_split": hparams["test_split"],
            "ablation_stage": ablation_stage,
            "similarity_threshold": hparams["similarity_threshold"],
            "num_concepts": len(concepts.text_concepts),
            "num_json_concepts": filtered_counts.get("json_concepts", 0),
            "num_action_concepts": filtered_counts.get("action_concepts", 0),
            "num_motion_concepts": filtered_counts.get("motion_concepts", 0),
            "num_visual_concepts": filtered_counts.get("visual_concepts", 0),
            "num_action_concepts_before_filter": raw_counts.get("action_concepts", 0),
            "num_motion_concepts_before_filter": raw_counts.get("motion_concepts", 0),
            "num_visual_concepts_before_filter": raw_counts.get("visual_concepts", 0),
            "num_json_concepts_before_filter": raw_counts.get("json_concepts", 0),
            "agentsam_run_folder": agentsam_run_folder,
            "embed_time_text_seconds": timing["text_embed"],
            "embed_time_visual_seconds": timing["visual_embed"],
            "embed_time_motion_seconds": timing["motion_embed"],
            "filter_time_seconds": timing["filter"],
            "total_concept_embed_time_seconds": total_embed_time,
        }
    )

    cbm_model.train_model(
        num_epochs=hparams["num_epochs"],
        l1_lambda=hparams["l1_lambda"],
        lambda_sparse=hparams["lambda_sparse"],
        lr=hparams["lr"],
        batch_size=hparams["batch_size"],
        enforce_nonneg=hparams["enforce_nonneg"],
        class_weights=hparams["class_weights"],
        wandb_run=wandb_run,
        random_seed=seed,
    )
    cbm_model.zero_shot(concepts, wandb_run=wandb_run)
    deps["mean_cbm"](cbm_model, wandb_run=wandb_run)
    wandb_run.finish()

    cbm_model.concept_metadata = concept_metadata_list
    cbm_model.agentsam_run_folder = agentsam_run_folder
    cbm_model.ablation_stage = ablation_stage

    ablation_short = (
        ablation_stage.replace("+", "_")
        .replace("json_only", "txt")
        .replace("action", "act")
        .replace("motion", "mot")
        .replace("visual", "vis")
    )
    agentsam_short = (
        agentsam_run_folder.replace("vlm-", "")
        .replace("Qwen-Qwen3-VL-", "Q3VL")
        .replace("-Thinking", "T")
        .replace("_ipc", "_i")
        .replace("_seed", "_s")
        .replace("_ws", "_w")
        .replace("/", "_")
        .replace("\\", "_")
    )
    if len(agentsam_short) > 25:
        agentsam_short = agentsam_short[:25]

    if hasattr(cbm_model, "model") and cbm_model.model is not None:
        cbm_model.model = deps["move_model_to_cpu"](cbm_model.model)

    model_name = f"./Models/{dataset_name}_{clip_model}_{ablation_short}_{hparams['test_split']}_{agentsam_short}.pkl"
    os.makedirs(os.path.dirname(model_name), exist_ok=True)
    with open(model_name, "wb") as f:
        pickle.dump(cbm_model, f)

    metadata_dict = {meta.get("concept_name", ""): meta for meta in concept_metadata_list}
    concepts_with_metadata = []
    for concept_name in concepts.text_concepts:
        metadata = metadata_dict.get(
            concept_name,
            {"concept_name": concept_name, "type": "text", "paths": []},
        )
        concepts_with_metadata.append(
            {
                "concept_name": concept_name,
                "type": metadata.get("type", "text"),
                "paths": metadata.get("paths", []) if metadata.get("paths") else [],
            }
        )
    concepts_with_metadata.sort(key=lambda x: x["concept_name"].lower())

    concepts_json = {
        "run_info": {
            "dataset": dataset,
            "clip_model": clip_model,
            "ablation_stage": ablation_stage,
            "ablation_stage_display": ablation_display_name,
            "test_split": hparams["test_split"],
            "agentsam_run_folder": agentsam_run_folder,
            "similarity_threshold": hparams["similarity_threshold"],
            "num_concepts": len(concepts.text_concepts),
            "concept_counts": {
                "json_concepts": filtered_counts.get("json_concepts", 0),
                "action_concepts": filtered_counts.get("action_concepts", 0),
                "motion_concepts": filtered_counts.get("motion_concepts", 0),
                "visual_concepts": filtered_counts.get("visual_concepts", 0),
            },
        },
        "concepts": concepts_with_metadata,
    }
    json_name = model_name.replace(".pkl", "_concepts.json")
    with open(json_name, "w", encoding="utf-8") as f:
        json.dump(concepts_json, f, indent=2, ensure_ascii=False)

    print("Saved model to", model_name)
    print("Saved concepts to", json_name)


def main() -> int:
    os.chdir(AGENTMOTIF_DIR)
    args = parse_args()

    valid_splits = DATASET_CONFIG[args.dataset]["default_test_splits"]
    if args.test_split not in valid_splits:
        raise ValueError(
            f"Unsupported test split '{args.test_split}' for dataset '{args.dataset}'. "
            f"Expected one of {valid_splits}."
        )

    hparams = {
        "dataset": args.dataset,
        "clip_model": args.clip_model,
        "window_size": args.window_size or DATASET_CONFIG[args.dataset]["default_window_size"],
        "random": args.random,
        "ablation_stage": args.ablation_stage,
        "agentsam_run_folder": args.agentsam_run_folder,
        "test_split": args.test_split,
        "similarity_threshold": args.similarity_threshold,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "lse_tau": args.lse_tau,
        "l1_lambda": args.l1_lambda,
        "lambda_sparse": args.lambda_sparse,
        "lr": args.lr,
        "transformer_layers": args.transformer_layers,
        "classifier_layers": args.classifier_layers,
        "enforce_nonneg": args.enforce_nonneg,
        "class_weights": args.class_weights,
        "diagonal_attention": args.diagonal_attention,
        "weight_decay": args.weight_decay,
        "d": args.d,
        "seed": args.seed,
        "wandb_project": args.wandb_project,
        "wandb_mode": args.wandb_mode,
        "concept_root": args.concept_root,
    }
    print("Running single ablation with:")
    print(json.dumps(hparams, indent=2))
    run_experiment(hparams)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
