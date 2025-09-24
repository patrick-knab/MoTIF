import sys
import os

# Ensure current directory is in sys.path for local imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))

import math
import time
import pickle
import wandb
import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from transformers import CLIPProcessor, CLIPModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPTokenizer,
    CLIPTextModelWithProjection,
)
from transformers import AutoProcessor, AutoModel  # siglip

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as pe_transformer

import clip


from video_embedder import VideoEmbedder


# Set random seed
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

# Map dataset keys to readable names
DATASET_MAP = {
    "breakfast": "Breakfast",
    "ucf101": "UCF101",
    "hmdb": "HMDB",
    "ssv2": "Something2",
    "jester": "Jester",
}


def process_dataset(dataset_key, clip_model, window_size=16, random=True,
                   batch_size: int = 256,
                   pe_video_batch_size: Optional[int] = None,
                   pe_target_T: Optional[int] = None,
                   enable_tf32: bool = True):
    dataset_name = DATASET_MAP.get(dataset_key.lower())
    if dataset_name is None:
        raise ValueError(f"Unknown dataset: {dataset_key}")

    folder_path = [f"../Datasets/{dataset_name}/Video_data"]
    output_dir = "../Embeddings/Datasets"
    embedd_path = f"../Embeddings/Videos/{dataset_name}/{random}_{window_size}_clip_{clip_model}.pkl"

    # Optional: enable TF32 for faster matmul on Ampere+
    if enable_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load CLIP model & processor
    if clip_model == "b32":
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", use_fast=True
        )
    elif clip_model == "b16":
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").eval()
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch16", use_fast=True
        )
    elif clip_model == "l14":
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14", use_fast=True
        )
    elif clip_model == "res50":
        # use the 'clip' module (imported at top) to load RN50
        model, processor = clip.load("RN50", device="cuda")
    elif clip_model == "clip4clip":
        # avoid naming any variable `clip` that would shadow the imported module
        model = CLIPVisionModelWithProjection.from_pretrained(
            "Searchium-ai/clip4clip-webvid150k"
        )
        model = model.eval()
        clip_full = CLIPModel.from_pretrained(
            "Searchium-ai/clip4clip-webvid150k"
        )  # renamed to avoid shadowing

        model_text = CLIPTextModelWithProjection.from_pretrained(
            "Searchium-ai/clip4clip-webvid150k"
        )  # for text
        processor = CLIPTokenizer.from_pretrained(
            "Searchium-ai/clip4clip-webvid150k"
        )  # for text
    elif clip_model == "siglip":
        model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    elif clip_model == "siglip2":
        model = AutoModel.from_pretrained("google/siglip2-base-patch32-256")
        processor = AutoProcessor.from_pretrained("google/siglip2-base-patch32-256")
    elif clip_model == "pe-l14":
        model = pe.CLIP.from_config("PE-Core-L14-336")

        processor = pe_transformer.get_image_transform(model.image_size)
        tokenizer = pe_transformer.get_text_tokenizer(model.context_length)
    else:
        raise ValueError(f"Unknown CLIP model: {clip_model}")

    # Create embedder
    if (
        clip_model == "clip4clip"
        or clip_model == "siglip"
        or clip_model == "siglip2"
        or clip_model == "res50"
        or clip_model == "pe-l14"
    ):
        embedder = VideoEmbedder(
            clip_model, model, processor,
            pe_video_batch_size=pe_video_batch_size,
            pe_target_T=pe_target_T,
        )
    else:
        embedder = VideoEmbedder("clip", model, processor)
    embedder.dataset_name = dataset_key

    if os.path.exists(embedd_path):
        try:
            with open(embedd_path, "rb") as f:
                embedder = pickle.load(f)
                print(f"Loaded existing embedder from {embedd_path}")
        except FileNotFoundError:
            print("Embedder file not found, creating a new one.")
    else:
        embedder.process_data(
            folder_path,
            window_size=window_size,
            output_path=output_dir,
            random=random,
            save_intermediate=True,
            batch_size=batch_size,
        )
        os.makedirs(os.path.dirname(embedd_path), exist_ok=True)
        with open(embedd_path, "wb") as f:
            pickle.dump(embedder, f)


# Example usage

window_size = 32  # example: match your window size
clip_model = "pe-l14"
# Faster defaults for video models: smaller T and larger video batch
process_dataset(
    "breakfast",
    clip_model,
    window_size=window_size,
    random=True,
    batch_size=256,            # ignored for PE path except as upper bound
    pe_video_batch_size=24,    # try 8–16 depending on VRAM
    pe_target_T=8,             # uniformly sample each window to 8 frames
    enable_tf32=True,
)
