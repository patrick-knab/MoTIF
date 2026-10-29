"""Generate window-level video embeddings for one dataset and backbone.

The result is a pickled ``VideoEmbedder`` at the location the training scripts expect::

    Embeddings/Videos/<DatasetName>/<random>_<window_size>_clip_<backbone>.pkl

Examples::

    python embedding.py --dataset hmdb51 --clip-model pe-l14 --window-size 8
    python embedding.py --dataset breakfast --clip-model pe-l14 --window-size 32 \
        --pe-video-batch-size 24 --pe-target-t 8
    # SSv2 with a LoRA adapter trained by finetune_pe_l14_ssv2.py --adapter lora
    python embedding.py --dataset something2 --clip-model pe-l14 --window-size 8 \
        --lora-adapter pe_l14_ssv2_lora/best.pt --output-path Embeddings/Videos/Something2/True_8_clip_pe-l14-lora.pkl

All paths are relative to the repository root, regardless of the working directory.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
for path in (ROOT, ROOT / "utils"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

import clip
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as pe_transformer

from utils.lora import load_lora_adapter, merge_lora
from utils.paths import EMBEDDINGS_DIR, dataset_folder, embedding_path, resolve_repo_path, video_data_dir
from utils.video_embedder import VideoEmbedder


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Keys understood by VideoEmbedder.extract_labels
EMBEDDER_DATASET_KEYS = {
    "breakfast": "breakfast",
    "ucf101": "ucf101",
    "hmdb": "hmdb",
    "hmdb51": "hmdb",
    "ssv2": "something2",
    "something2": "something2",
    "jester": "jester",
}

BACKBONES = ("b32", "b16", "l14", "res50", "clip4clip", "siglip", "siglip2", "siglipl14", "pe-l14", "pe-g14")


def load_backbone(clip_model: str, device: torch.device, lora_adapter: Optional[str] = None):
    """Return ``(embedder_name, model, processor)`` for a backbone key."""
    if clip_model == "b32":
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        name = "clip"
    elif clip_model == "b16":
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", use_fast=True)
        name = "clip"
    elif clip_model == "l14":
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
        name = "clip"
    elif clip_model == "res50":
        model, processor = clip.load("RN50", device=device)
        name = "res50"
    elif clip_model == "clip4clip":
        model = CLIPVisionModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k").eval()
        processor = CLIPTokenizer.from_pretrained("Searchium-ai/clip4clip-webvid150k")
        name = "clip4clip"
    elif clip_model == "siglip":
        model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        name = "siglip"
    elif clip_model == "siglip2":
        model = AutoModel.from_pretrained("google/siglip2-base-patch32-256")
        processor = AutoProcessor.from_pretrained("google/siglip2-base-patch32-256")
        name = "siglip2"
    elif clip_model == "siglipl14":
        model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
        processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        name = "siglipl14"
    elif clip_model in ("pe-l14", "pe-g14"):
        config_name = "PE-Core-L14-336" if clip_model == "pe-l14" else "PE-Core-G14-448"
        model = pe.CLIP.from_config(config_name, pretrained=True)
        if lora_adapter:
            adapter_path = resolve_repo_path(lora_adapter)
            config = load_lora_adapter(model, str(adapter_path))
            merged = merge_lora(model)
            print(f"Loaded LoRA adapter {adapter_path} (rank={config.rank}, alpha={config.alpha}, "
                  f"targets={config.targets}); merged {merged} weights into the backbone.")
        model = model.eval()
        processor = pe_transformer.get_image_transform(model.image_size)
        name = clip_model
    else:
        raise ValueError(f"Unknown CLIP model: {clip_model}; choose from {BACKBONES}")

    if lora_adapter and not clip_model.startswith("pe-"):
        raise ValueError("--lora-adapter is only supported for the Perception Encoder backbones (pe-l14, pe-g14)")
    return name, model, processor


def process_dataset(
    dataset_key: str,
    clip_model: str,
    window_size: int = 16,
    random: bool = True,
    batch_size: int = 256,
    pe_video_batch_size: Optional[int] = None,
    pe_target_T: Optional[int] = None,
    enable_tf32: bool = True,
    lora_adapter: Optional[str] = None,
    output_path: Optional[str] = None,
    overwrite: bool = False,
):
    dataset_name = dataset_folder(dataset_key)
    folder_path = [str(video_data_dir(dataset_key))]
    output_dir = str(EMBEDDINGS_DIR / "Datasets")
    embedd_path = str(resolve_repo_path(output_path)) if output_path else str(
        embedding_path(dataset_key, clip_model, window_size, random)
    )

    if os.path.exists(embedd_path) and not overwrite:
        print(f"Embeddings already exist at {embedd_path}; pass --overwrite to recompute.")
        return embedd_path
    if not os.path.isdir(folder_path[0]):
        raise FileNotFoundError(
            f"{folder_path[0]} does not exist. Run `python save_videos.py --dataset {dataset_name}` first."
        )

    if enable_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name, model, processor = load_backbone(clip_model, device, lora_adapter=lora_adapter)

    if name == "clip":
        embedder = VideoEmbedder("clip", model, processor)
    else:
        embedder = VideoEmbedder(
            name, model, processor,
            pe_video_batch_size=pe_video_batch_size,
            pe_target_T=pe_target_T,
        )
    embedder.dataset_name = EMBEDDER_DATASET_KEYS[dataset_key.lower()]

    print(f"Embedding {folder_path[0]} with {clip_model} (window_size={window_size}, random={random}) -> {embedd_path}")
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
    print(f"Saved embedder to {embedd_path}")
    return embedd_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate window-level video embeddings for MoTIF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="hmdb51", choices=sorted(EMBEDDER_DATASET_KEYS),
                        help="Dataset key (hmdb51/hmdb, ucf101, breakfast, something2/ssv2).")
    parser.add_argument("--clip-model", default="pe-l14", choices=BACKBONES, help="Backbone key.")
    parser.add_argument("--window-size", type=int, default=8,
                        help="Frames per temporal window; must match the training script's window_size.")
    parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=True,
                        help="Random (vs. uniform) frame selection inside each window; part of the output file name.")
    parser.add_argument("--batch-size", type=int, default=256, help="Frame batch size for image backbones.")
    parser.add_argument("--pe-video-batch-size", type=int, default=24,
                        help="Windows per forward pass for the Perception Encoder video path (reduce on small GPUs).")
    parser.add_argument("--pe-target-t", type=int, default=8,
                        help="Frames each window is uniformly sampled to for the Perception Encoder video path.")
    parser.add_argument("--no-tf32", action="store_true", help="Disable TF32 matmuls on Ampere+ GPUs.")
    parser.add_argument("--lora-adapter", default=None,
                        help="Adapter file written by finetune_pe_l14_ssv2.py --adapter lora (PE backbones only).")
    parser.add_argument("--output-path", default=None,
                        help="Where to write the pickle; default is the canonical Embeddings/Videos/... location.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if the output pickle exists.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    process_dataset(
        args.dataset,
        args.clip_model,
        window_size=args.window_size,
        random=args.random,
        batch_size=args.batch_size,
        pe_video_batch_size=args.pe_video_batch_size,
        pe_target_T=args.pe_target_t,
        enable_tf32=not args.no_tf32,
        lora_adapter=args.lora_adapter,
        output_path=args.output_path,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
