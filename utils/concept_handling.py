"""
Concept handling utilities for MoTIF experiments.

This module provides functions for loading, embedding, and filtering concepts
from concept extraction outputs.
"""

import os
import json
import re
import time
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_test_split_instances(dataset, test_split):
    """
    Get set of instance names/paths that are in the test split.
    
    Args:
        dataset: Dataset name (e.g., "hmdb51", "breakfast", "ucf101")
        test_split: Test split identifier (e.g., "s1" for HMDB)
    
    Returns:
        set of instance identifiers (basenames or paths) that are in test split
    """
    test_instances = set()
    
    if dataset == "hmdb51":
        # HMDB uses split files
        import glob
        index = int(test_split.replace("s", "")) if test_split.startswith("s") else int(test_split)
        labels_path = "./Datasets/HMDB/testTrainMulti_7030_splits/"
        path_text_dirs = glob.glob(os.path.join(labels_path, "*.txt"))
        path_text_dirs_idx = [p for p in path_text_dirs if f"split{index}" in p]
        path_text_dirs_idx.sort()
        
        for txt_path in path_text_dirs_idx:
            with open(txt_path, "r") as fh:
                for line in fh:
                    name, flag = line.strip().split()
                    if flag == "2":  # Test split
                        # Store both .avi and .mp4 variants, and just the basename
                        test_instances.add(name)
                        test_instances.add(name.replace(".avi", ".mp4"))
                        test_instances.add(name.replace(".avi", ""))
    
    elif dataset == "breakfast":
        # Breakfast uses participant-based splits
        # Test split participants are determined by the split number
        RANGES = {
            "s1": range(3, 16),   # P03-P15
            "s2": range(16, 29),  # P16-P28
            "s3": range(29, 42),  # P29-P41
            "s4": range(42, 54),  # P42-P53
        }
        
        if test_split not in RANGES:
            print(f"Warning: Unknown Breakfast split '{test_split}'. Expected one of {list(RANGES.keys())}")
            return test_instances
        
        target_range = RANGES[test_split]
        # Add all participant numbers in the test range
        for num in target_range:
            # Add various formats: P03, P03_cereals, P03_cereals_ch0, etc.
            test_instances.add(f"P{num:02d}")
            # Also add without leading zero for flexibility
            test_instances.add(f"P{num}")
    
    elif dataset == "ucf101":
        # UCF101 uses split files similar to HMDB
        import glob
        index = int(test_split.replace("s", "")) if test_split.startswith("s") else int(test_split)
        labels_path = "./Datasets/UCF101/ucfTrainTestlist/"
        path_text_dirs = glob.glob(os.path.join(labels_path, f"*testlist{index:02d}.txt"))
        
        for txt_path in path_text_dirs:
            with open(txt_path, "r") as fh:
                for line in fh:
                    name = line.strip()
                    if name:
                        # Extract class and video name
                        parts = name.split("/")
                        if len(parts) >= 2:
                            video_name = parts[-1].replace(".avi", "").replace(".mp4", "")
                            test_instances.add(video_name)
                            test_instances.add(name)
    
    elif dataset == "something2":
        # SSV2 uses JSON files for splits
        val_json = "./Datasets/Something2/labels/validation.json"
        test_json = "./Datasets/Something2/labels/test.json"
        
        import json as json_lib
        with open(test_json, "r") as f:
            test_data = json_lib.load(f)
            if isinstance(test_data, list):
                for entry in test_data:
                    vid = entry.get("id") or entry.get("video_id") or entry.get("video") or entry.get("uid")
                    if vid is not None:
                        test_instances.add(str(vid))
    
    return test_instances


def load_concepts(concept_dir, ablation_stage="json_only", test_split_instances=None):
    """
    Load concepts from JSON files.
    
    Args:
        concept_dir: Path to the concept extraction output directory (e.g., vlm-Qwen-Qwen3-VL-8B-Thinking_ipc5_seed0)
        ablation_stage: One of ["json_only", "json+action", "action_only"]
        test_split_instances: Optional set of instance identifiers to exclude (instances in test split)
    
    Returns:
        dict with keys:
            - text_concepts: list of text concept names
            - concept_counts: dict with counts per category (json_concepts, action_concepts)
            - concept_category: dict mapping concept_name -> category ("json", "action")
            - source_category: dict mapping concept_name -> source category
    """
    text_concepts_set = set()
    
    # Track concepts separately by category
    json_concepts_set = set()
    action_concepts_set = set()
    
    # Map concept names to their categories and original source
    concept_category = {}
    source_category = {}  # stable source bucket: json/action
    
    skipped_test_count = 0
    total_count = 0
    
    def _resolve_path(path_value, base_dir):
        """Resolve potentially relative paths against the JSON's directory."""
        if not path_value:
            return None
        if os.path.isabs(path_value):
            return path_value
        return os.path.normpath(os.path.join(base_dir, path_value))


    # Walk through all concept JSON files
    for root, dirs, files in os.walk(concept_dir):
        if "concepts.json" in files:
            json_path = os.path.join(root, "concepts.json")
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                
                # Skip if this instance is in the test split
                if test_split_instances is not None and len(test_split_instances) > 0:
                    instance_name = data.get("instance_name", "")
                    instance_dir = data.get("instance_dir", "")
                    # Check if instance is in test split
                    is_test = False
                    
                    # For Breakfast, check if participant ID matches test split range
                    # Breakfast uses participant-based splits (P03-P15, P16-P28, etc.)
                    participant_match = re.search(r"P(\d{2})", instance_name or instance_dir or "")
                    if participant_match:
                        participant_id = participant_match.group(1)
                        # Check if this participant ID (as P##) is in test split
                        if f"P{participant_id}" in test_split_instances:
                            is_test = True
                    
                    # Try matching instance_name
                    if not is_test and instance_name:
                        # Direct match
                        if instance_name in test_split_instances:
                            is_test = True
                        else:
                            # Try basename
                            basename = os.path.basename(instance_name)
                            if basename in test_split_instances:
                                is_test = True
                            else:
                                # Try without extensions
                                name_no_ext = basename.replace(".avi", "").replace(".mp4", "").replace(".jpg", "").replace(".png", "")
                                if name_no_ext in test_split_instances:
                                    is_test = True
                    
                    # Try matching instance_dir if not already matched
                    if not is_test and instance_dir:
                        dir_basename = os.path.basename(instance_dir.rstrip("/"))
                        if dir_basename in test_split_instances:
                            is_test = True
                        else:
                            dir_no_ext = dir_basename.replace(".avi", "").replace(".mp4", "").replace(".jpg", "").replace(".png", "")
                            if dir_no_ext in test_split_instances:
                                is_test = True
                            
                            # Also check if any part of the path matches
                            if not is_test:
                                path_parts = instance_dir.split(os.sep)
                                for part in path_parts:
                                    if part in test_split_instances or part.replace(".avi", "").replace(".mp4", "") in test_split_instances:
                                        is_test = True
                                        break
                    
                    if is_test:
                        skipped_test_count += 1
                        continue  # Skip this instance
                
                total_count += 1
                
                # Helper function to add concept with optional class context
                def add_concept(concept, category_set, source_label="json"):
                    # Remove class suffix if present (format: "concept (class_name)")
                    # This ensures clean concept names without class context
                    cleaned_concept = concept
                    if " (" in concept and concept.endswith(")"):
                        cleaned_concept = concept.rsplit(" (", 1)[0]
                    
                    # Always use cleaned concept name (without class suffix)
                    text_concepts_set.add(cleaned_concept)
                    category_set.add(cleaned_concept)
                    # Stable source category: only set the first time we see it
                    if cleaned_concept not in source_category:
                        source_category[cleaned_concept] = source_label
                
                # Stage 1: JSON concepts (text only)
                # For json_only: include basic concepts + temporal concepts (persistent objects)
                # This gives richer semantic information while staying text-only
                if ablation_stage in ["json_only", "json+action"]:     
                    if "concepts" in data:
                        for concept in data["concepts"]:
                            add_concept(concept, json_concepts_set, source_label="json")
                    
                    # Add temporal concepts to json_only for better concept naming
                    # Temporal concepts indicate objects that persist across frames, adding temporal context
                    if ablation_stage in ["json_only", "json+action"] and "temporal_concepts" in data:
                        for concept in data["temporal_concepts"]:
                            add_concept(concept, json_concepts_set, source_label="json")
                    
                # Stage 2: Add action concepts
                if ablation_stage in ["json+action", "action_only"]:
                    if "action_concepts" in data:
                        for concept in data["action_concepts"]:
                            add_concept(concept, action_concepts_set, source_label="action")
                            
            except Exception as e:
                print(f"Error loading {json_path}: {e}")
                continue
    
    if test_split_instances and len(test_split_instances) > 0:
        print(f"Filtered out {skipped_test_count} instances from test split (kept {total_count} training instances)")
    
    # Determine category for each concept based on what's actually available
    # Priority: action > json
    num_action_concepts_total = 0
    num_json_concepts_total = 0
    for concept_name in text_concepts_set:
        if concept_name in action_concepts_set:
            concept_category[concept_name] = "action"
            num_action_concepts_total += 1
        else:
            concept_category[concept_name] = "json"
            num_json_concepts_total += 1        

    concept_counts = {
        "json_concepts": num_json_concepts_total,
        "action_concepts": num_action_concepts_total
    }
    
    return {
        "text_concepts": sorted(list(text_concepts_set)),
        "concept_counts": concept_counts,
        "concept_category": concept_category,
        "source_category": source_category
    }


def filter_redundant_concepts(embeddings_dict, similarity_threshold=0.98):
    """
    Filter redundant concepts based on cosine similarity.
    
    Args:
        embeddings_dict: dict mapping concept_name -> embedding tensor
        similarity_threshold: threshold for considering concepts redundant
    
    Returns:
        list of concept names to keep (non-redundant)
    """
    if len(embeddings_dict) == 0:
        return []
    
    concept_names = list(embeddings_dict.keys())
    embeddings_matrix = torch.stack([embeddings_dict[name] for name in concept_names]).detach().numpy()
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    embeddings_normalized = embeddings_matrix / (norms + 1e-8)
    
    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(embeddings_normalized)
    
    # Find redundant concepts
    to_keep = []
    to_remove = set()
    
    for i, name_i in enumerate(concept_names):
        if name_i in to_remove:
            continue
        
        # Check similarity with concepts we've already decided to keep
        is_redundant = False
        for j, name_j in enumerate(concept_names[:i]):
            if name_j in to_keep and similarity_matrix[i, j] > similarity_threshold:
                is_redundant = True
                break
        
        if not is_redundant:
            to_keep.append(name_i)
        else:
            to_remove.add(name_i)
    
    return to_keep


def process_concepts(
    embedder,
    concepts,
    concept_data,
    ablation_stage,
    clip_model,
    similarity_threshold=0.95,
    ):
    """
    Process concepts: embed text/action concepts, filter redundant ones, and build final concept list.
    
    Args:
        embedder: VideoEmbedder instance
        concepts: Create_Concepts instance
        concept_data: dict from load_concepts
        ablation_stage: One of ["json_only", "json+action", "action_only"]
        clip_model: CLIP model name (for determining embedding dimension)
        similarity_threshold: Threshold for filtering redundant concepts

    Returns:
        dict with keys:
            - concepts: Updated Create_Concepts instance with filtered embeddings
            - concept_metadata_list: List of metadata dicts for each concept
            - filtered_counts: Dict with counts per category after filtering
            - before_filter_counts: Dict with counts per category before filtering
            - raw_counts: Counts directly from JSON (as loaded)
            - timing: Dict with timing information
    """
    timing = {}
    
    # Separate text concepts from action concepts based on category
    # For json_only stage, we only have text concepts
    # For json+action stages, we need to separate them
    concept_category = concept_data.get("concept_category", {})
    source_category = concept_data.get("source_category", {})
    all_concepts = list(concept_data["text_concepts"])
    
    # Separate into text (json) and action concepts
    text_only_concepts = []
    action_only_concepts = []
    
    for concept_name in all_concepts:
        category = concept_category.get(concept_name, "json")
        if category == "action":
            action_only_concepts.append(concept_name)
        else:
            text_only_concepts.append(concept_name)

    before_filter_counts = {
        "json_concepts": len(text_only_concepts),
        "action_concepts": len(action_only_concepts)
    }

    filtered_counts = {
        "json_concepts": 0,
        "action_concepts": 0
    }
 
    original_concept_category = {}

    concepts_using_action = set()
    
    # Embed text concepts first (skip for action_only)
    # Initialize text_concepts list with text-only concepts
    text_concepts = list(text_only_concepts)
    concept_name_to_embedding = {}
    text_embed_time = 0.0
    if text_only_concepts:
        print(f"Embedding {len(text_only_concepts)} text concepts...")
        start_time = time.time()
        concepts.embedd_text(text_only_concepts)
        text_embed_time = time.time() - start_time
        print(f"Text embedding time: {text_embed_time:.2f} seconds")
        text_embeddings = concepts.text_embeddings.cpu()
        for i, concept_name in enumerate(text_only_concepts):
            new_emb = text_embeddings[i]
            # For the very first one, concept_name_to_embedding is empty, so always add it
            max_similarity = -1.0
            for existing_name, existing_emb in concept_name_to_embedding.items():
                similarity = F.cosine_similarity(
                    new_emb.unsqueeze(0),
                    existing_emb.unsqueeze(0),
                    dim=1
                ).item()
                if similarity > max_similarity:
                    max_similarity = similarity
            # Only add if sufficiently different from all previous (or if none exist)
            # Skip if too similar (consistent with action logic)
            if max_similarity < similarity_threshold:
                concept_name_to_embedding[concept_name] = new_emb
                filtered_counts["json_concepts"] += 1
    
    timing["text_embed"] = text_embed_time
        
    # Embed action concepts if in ablation stage, with similarity checking
    action_embed_time = 0.0
    if ablation_stage in ["json+action", "action_only"] and action_only_concepts:
        print(f"Embedding {len(action_only_concepts)} action concepts...")
        start_time = time.time()
        concepts.embedd_text(action_only_concepts)
        action_embed_time = time.time() - start_time
        print(f"Action embedding time: {action_embed_time:.2f} seconds")
        action_embeddings = concepts.text_embeddings.cpu()
        
        # Check each action concept against ALL existing text concepts
        # For action_only, skip similarity checking since there are no text concepts
        for i, action_concept in enumerate(action_only_concepts):
            action_emb = action_embeddings[i]
            use_action = True
            max_similarity = -1.0

            for existing_name, existing_emb in concept_name_to_embedding.items():
                similarity = F.cosine_similarity(
                    action_emb.unsqueeze(0),
                    existing_emb.unsqueeze(0),
                    dim=1
                ).item()
                if similarity > max_similarity:
                    max_similarity = similarity
            if max_similarity >= similarity_threshold:
                use_action = False

            if use_action:
                concept_name_to_embedding[action_concept] = action_emb
                concepts_using_action.add(action_concept)
                original_concept_category[action_concept] = source_category.get(
                    action_concept, concept_category.get(action_concept, "action")
                )
                if action_concept not in text_concepts:
                    text_concepts.append(action_concept)
                filtered_counts["action_concepts"] += 1
    
    print(f"Filtered {filtered_counts['json_concepts']} text concepts and {filtered_counts['action_concepts']} action concepts.")
    timing["action_embed"] = action_embed_time
    
    # Filter time is essentially the time spent on similarity checking and filtering
    # This is already included in the individual embed times, so set to 0 or sum
    timing["filter"] = 0.0
    
    # Filter text_concepts to only include concepts that have embeddings
    # This ensures we don't have KeyError when stacking embeddings
    filtered_text_concepts = [name for name in text_concepts if name in concept_name_to_embedding]
    
    # Assign final concept list to concepts object
    concepts.text_concepts = filtered_text_concepts
    concepts.text_embeddings = torch.stack([concept_name_to_embedding[name] for name in filtered_text_concepts])
    
    # Build metadata aligned with the final concept list
    concept_metadata_list = []
    for concept_name in concepts.text_concepts:
        concept_type = "text"
        concept_paths = []
        
        concept_metadata_list.append({
            "concept_name": concept_name,
            "unique_id": None,
            "type": concept_type,
            "paths": concept_paths
        })
    
    return {
        "concepts": concepts,
        "concept_metadata_list": concept_metadata_list,
        "filtered_counts": filtered_counts,
        "raw_counts": concept_data.get("concept_counts", {}),
        "timing": timing,
    }


def move_model_to_cpu(model):
    """
    Move a PyTorch model to CPU to avoid CUDA device errors when loading on CPU-only machines.
    
    This function moves the model's parameters, buffers, and the model itself to CPU.
    This is useful when saving models that were trained on GPU but need to be loaded
    on machines without CUDA support.
    
    Args:
        model: A PyTorch nn.Module or an object with a 'model' attribute containing a nn.Module.
    
    Returns:
        The model with all tensors moved to CPU.
    """
    if hasattr(model, 'model') and model.model is not None:
        model.model = model.model.cpu()
    elif isinstance(model, torch.nn.Module):
        model = model.cpu()
    
    return model

