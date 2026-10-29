from utils.motif import init_repro

seed = 42
init_repro(seed, deterministic=True)

import sys, os, time, pickle, copy, math

# make `core.*` (Perception Encoder) importable regardless of the working directory
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))

import torch
from transformers import (
    CLIPProcessor, CLIPModel, 
    CLIPVisionModelWithProjection, 
    CLIPTokenizer, CLIPTextModelWithProjection,
    AutoProcessor, AutoModel
)
import clip
import wandb

from utils.video_embedder import VideoEmbedder, Create_Concepts
from utils.motif import MoTIF, CBMTransformer, mean_cbm
from utils.motif_spacetime import CBMTransformerST

from utils.concept_handling import (
    get_test_split_instances,
    load_concepts,
    process_concepts,
    move_model_to_cpu
)
from utils.manual_concepts import get_manual_concepts
from utils.paths import EMBEDDINGS_DIR, MODELS_DIR, embedding_path, resolve_repo_path, video_data_dir
from utils.train_cli import DEFAULT_AGENT_RUN_FOLDER, apply_overrides, build_parser, resolve_concept_dir

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as pe_transformer

def run_experiment(hparams):
    """Run one CBM training experiment with given hyperparameters."""
    dataset = hparams["dataset"]
    clip_model = hparams["clip_model"]
    window_size = hparams["window_size"]
    random = hparams["random"]

    # dataset name mapping
    dataset_map = {
        "breakfast": "Breakfast",
        "ucf101": "UCF101",
        "hmdb51": "HMDB",
        "something2": "Something2"
    }
    dataset_name = dataset_map[dataset]

    # load CLIP or related models
    if clip_model == "b32":
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
        embedd_path = str(embedding_path(dataset, "b32", window_size, random))
        clip_name = "clip"
    elif clip_model == "b16":
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", use_fast=False)
        embedd_path = str(embedding_path(dataset, "b16", window_size, random))
        clip_name = "clip"
    elif clip_model == "l14":
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=False)
        embedd_path = str(embedding_path(dataset, "l14", window_size, random))
        clip_name = "clip"
    elif clip_model == "res50":
        model, preprocess = clip.load("RN50", device="cpu")
        processor = preprocess
        embedd_path = str(embedding_path(dataset, "res50", window_size, random))
        clip_name = "res50"
    elif clip_model == "clip4clip":
        model = CLIPVisionModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k").eval()
        model_text = CLIPTextModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k")
        processor = CLIPTokenizer.from_pretrained("Searchium-ai/clip4clip-webvid150k")
        embedd_path = str(embedding_path(dataset, "clip4clip", window_size, random))
        clip_name = "clip4clip"
    elif clip_model == "siglip":
        model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        clip_name = "siglip"
        embedd_path = str(embedding_path(dataset, "siglip", window_size, random))
        
    elif clip_model == "siglipl14":
        model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
        processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        clip_name = "siglipl14"
        embedd_path = str(embedding_path(dataset, "siglipl14", window_size, random))
    
    elif clip_model == "pe-l14":
        model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)

        processor = pe_transformer.get_image_transform(model.image_size)
        tokenizer = pe_transformer.get_text_tokenizer(model.context_length)
        clip_name = "pe-l14"
        embedd_path = str(embedding_path(dataset, "pe-l14", window_size, random))
    elif clip_model == "pe-g14":
        model = pe.CLIP.from_config("PE-Core-G14-448", pretrained=True)

        processor = pe_transformer.get_image_transform(model.image_size)
        tokenizer = pe_transformer.get_text_tokenizer(model.context_length)
        embedd_path = str(embedding_path(dataset, "pe-g14", window_size, random))
        clip_name = "pe-g14"
    else:
        model = None
        processor = None
        model_text = None

        raise ValueError(f"Unknown clip_model {clip_model}")

    # embedder
    embedder = VideoEmbedder(clip_name, model, processor)
    embedder.dataset_name = dataset

    if hparams.get("embedding_path"):
        # explicit file, e.g. a pickle downloaded from Hugging Face
        embedd_path = str(resolve_repo_path(hparams["embedding_path"]))
    elif not os.path.exists(embedd_path):
        # earlier releases stored PE-G/14 embeddings with a "_42" seed suffix
        legacy_path = embedd_path.replace(".pkl", "_42.pkl")
        if os.path.exists(legacy_path):
            embedd_path = legacy_path

    if os.path.exists(embedd_path):
        with open(embedd_path, "rb") as f:
            embedder = pickle.load(f)
            print("Loaded existing embedder from", embedd_path)
    else:
        folder_path = [str(video_data_dir(dataset))]
        if not os.path.isdir(folder_path[0]):
            raise FileNotFoundError(
                f"No embeddings at {embedd_path} and no derived videos at {folder_path[0]}. "
                "Run `python save_videos.py --dataset <Name>` and `python embedding.py --dataset ... --clip-model ...` "
                "first, or pass --embedding-path to an existing pickle."
            )
        print(f"No embeddings at {embedd_path}; computing them from {folder_path[0]} (this is slow).")
        embedder.process_data(folder_path, window_size=window_size, output_path=str(EMBEDDINGS_DIR / "Datasets"))
        os.makedirs(os.path.dirname(embedd_path), exist_ok=True)
        with open(embedd_path, "wb") as f:
            pickle.dump(embedder, f)


    # concepts
    if clip_model == "clip4clip":
        concepts = Create_Concepts(clip_name, model_text, processor)
    elif clip_model == "pe-l14" or clip_model == "pe-g14":
        concepts = Create_Concepts(clip_name, model, tokenizer)
    else:
        concepts = Create_Concepts(clip_name, model, processor)
    
    if hparams.get("agentic-concepts", False):
        ablation_stage = "json+action"
        # Resolved against the working directory first and the repository root second;
        # raises with the exact extraction command if the run folder is missing.
        concept_dir = resolve_concept_dir(hparams.get("agent_run_folder", DEFAULT_AGENT_RUN_FOLDER))
        
        # Get test split instances to exclude
        test_split_instances = get_test_split_instances(dataset, hparams["test_split"])
        print(f"Found {len(test_split_instances)} instances in test split '{hparams['test_split']}' to exclude from concepts")
        
        print(f"Loading concepts from {concept_dir} with ablation stage: {ablation_stage}")
        concept_data = load_concepts(
            concept_dir, 
            ablation_stage=ablation_stage, 
            test_split_instances=test_split_instances,
        )
        
        # Process concepts: embed, filter, and build final concept list
        concept_result = process_concepts(
            embedder=embedder,
            concepts=concepts,
            concept_data=concept_data,
            ablation_stage=ablation_stage,
            clip_model=clip_model,
            similarity_threshold=hparams.get("similarity_threshold", 0.95)
        )
        
        concepts = concept_result["concepts"]

    else:
        # Fallback without a VLM concept run: the hand-written vocabularies from the
        # original release, one list entry per concept (see utils/manual_concepts.py).
        text_concepts = get_manual_concepts(dataset)
        print(f"Using {len(text_concepts)} manual concepts for dataset '{dataset}'")
        concepts.embedd_text(text_concepts)

    # model
    cbm_model = MoTIF(embedder, concepts)
    cbm_model.preprocess(dataset, info=hparams["test_split"], random_state=seed)

    cbm_model.model = CBMTransformerST(
        cbm_model.num_concepts,
        num_classes=cbm_model.num_classes,
        transformer_layers=hparams["transformer_layers"],
        lse_tau=hparams["lse_tau"],
        temporal_d=hparams["d"],
        spatial_d=hparams.get("spatial_d", 1),
        dropout=hparams.get("dropout", 0.1),
        nonneg_classifier=hparams.get("enforce_nonneg", False),
        spatial_gate=hparams.get("spatial_gate", 0.1),
        identity_bias=hparams.get("identity_bias", 1.0),
    )

    # wandb
    time_now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    run_name = f'{dataset}_{clip_model}_{time_now}'
    
    wandb_mode = os.environ.get("WANDB_MODE", "online")
    if not os.environ.get("WANDB_API_KEY") and wandb_mode != "disabled":
        print("WANDB_API_KEY not found. Setting WANDB_MODE to disabled.")
        wandb_mode = "disabled"
        
    wandb_run = wandb.init(project="motif", name=run_name, config=hparams, mode=wandb_mode)
    
    if wandb_mode != "disabled":
        wandb_run.log({
            'test_split': hparams["test_split"],
        })

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
    mean_cbm(cbm_model, wandb_run=wandb_run)
    wandb_run.finish()


    model_name = str(MODELS_DIR / f"checkpoint_{clip_model}_{dataset_name}.pkl")
    os.makedirs(os.path.dirname(model_name), exist_ok=True)
    with open(model_name, "wb") as f:
        pickle.dump(cbm_model, f)
    print("cbm_model and class saved to", model_name)


if __name__ == "__main__":
    # define hyperparameter grid with descriptions
    # Hyperparameter descriptions:
    # num_epochs: Number of training epochs.
    # batch_size: Number of samples per training batch.
    # lse_tau: Temperature parameter for log-sum-exp pooling.
    # l1_lambda: L1 regularization strength.
    # lambda_sparse: Sparsity regularization strength.
    # lr: Learning rate for optimizer.
    # transformer_layers: Number of transformer layers in the CBM model.
    # diagonal_attention: If True, restricts attention to diagonal (self-attention only).
    # enforce_nonneg: If True, enforces non-negative concept activations.
    # class_weights: If True, uses class weights to balance loss.
    # weight_decay: Weight decay (L2 regularization) for optimizer.
    # d: Model dimension. Always 1, can be set higher to express more representations after Conv1d.
    # test_split: Which test split to use (e.g., "s1").
    # window_size: Temporal window size for video embedding.
    # dataset: Dataset to use (e.g., "hmdb51", "breakfast", "something2").
    # random: If True, uses random seed for image selection in window.
    # clip_model: CLIP model variant to use (e.g., "pe-l14", "b16", "res50", "clip4clip").
    
    search_space = {
        "num_epochs": [100],
        "batch_size": [512],
        "lse_tau": [1.0],
        "l1_lambda": [1e-4],
        "lambda_sparse": [1e-4],
        "lr": [1e-4],
        "transformer_layers": [1],
        "dropout": [0.1],
        "enforce_nonneg": [True],
        "spatial_d": [1],
        "class_weights": [True],
        "weight_decay": [1e-2],
        "d": [1],
        "test_split": ["s1"],
        "window_size": [8],
        "dataset": ["hmdb51"],  
        "random": [True],
        "clip_model": ["pe-l14"],
        "agentic-concepts": [True],
        "agent_run_folder": [DEFAULT_AGENT_RUN_FOLDER],  # written by slurm_cbm_extract_windows.sbatch
        "similarity_threshold": [0.9],
        "spatial_gate": [0.1],
        "identity_bias": [1.0],

    }

    # command-line overrides (see `python train_MoTIF-ST.py --help`)
    args = build_parser("Train the space-time MoTIF-ST concept bottleneck model.").parse_args()
    search_space = apply_overrides(search_space, args)

    import itertools
    keys, values = zip(*search_space.items())
    for v in itertools.product(*values):
        hparams = dict(zip(keys, v))
        run_experiment(hparams)
