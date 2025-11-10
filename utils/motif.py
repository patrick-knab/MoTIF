from __future__ import annotations

import os
import random
import numpy as np
import torch
import copy


from typing import List, Optional, Dict, Tuple

import cv2
from PIL import Image
import tqdm

import torch.nn as nn
import gc

import torch.nn.functional as F
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    InterpolationMode,
)
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import wandb
import re
import pandas as pd
import glob

def init_repro(seed: int = 42, deterministic: bool = True):
    """Call this at the very top of your notebook/script BEFORE creating any model/processor/device context."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
        ":16:8"  # deterministic cuBLAS on Ampere+, nice default
    )
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism knobs (do this before any CUDA ops)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # older torch may not support signature
            torch.set_deterministic(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # Reduce threading nondeterminism
    torch.set_num_threads(1)

    return seed

def get_torch_device(prefer: Optional[str] = None) -> torch.device:
    if prefer is not None:
        pref = prefer.lower()
        if pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if (
            pref == "mps"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        if pref == "cpu":
            return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pad_batch_sequences(
    seqs: List[torch.Tensor], device: torch.device
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Pad a list of [T_i, C] tensors into a batch [B, T_max, C] and return
    a key_padding_mask [B, T_max] with True for padded positions.
    """
    if len(seqs) == 0:
        raise ValueError("pad_batch_sequences received empty sequence list")
    lengths = [int(s.shape[0]) for s in seqs]
    C = int(seqs[0].shape[1])
    T_max = int(max(lengths))
    B = len(seqs)
    batch = torch.zeros((B, T_max, C), dtype=torch.float32, device=device)
    mask = torch.ones((B, T_max), dtype=torch.bool, device=device)  # True=padded
    for i, s in enumerate(seqs):
        t = lengths[i]
        batch[i, :t, :] = s.to(device)
        mask[i, :t] = False
    return batch, mask


def compute_concept_standardization(seqs: List[torch.Tensor | np.ndarray]):
    cat = torch.cat(
        [
            (
                s
                if isinstance(s, torch.Tensor)
                else torch.tensor(np.array(s), dtype=torch.float32)
            )
            for s in seqs
        ],
        dim=0,
    )
    mean = cat.mean(dim=0)
    std = cat.std(dim=0).clamp_min(1e-6)
    return mean, std


def apply_standardization(
    seqs: List[torch.Tensor | np.ndarray], mean: torch.Tensor, std: torch.Tensor
):
    out = []
    for s in seqs:
        s_t = (
            s
            if isinstance(s, torch.Tensor)
            else torch.tensor(np.array(s), dtype=torch.float32)
        )
        out.append((s_t - mean) / std)
    return out


def concepts_over_time_cosine(
    concepts: torch.Tensor,
    all_data_list,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    chunk_size: int | None = None,
):
    """
    Cosine-sim per frame vs concepts.
    - Normalizes in fp32 for stability, computes in fp32, then returns on CPU.
    - Optional chunked matmul to cap peak memory.
    """
    with torch.no_grad():
        # normalize concepts in fp32 on target device
        c = F.normalize(
            concepts.detach().to(device=device, dtype=torch.float32), dim=1
        )  # [K,C]
        K = c.shape[0]

        activations, embeddings = [], []

        for vid in all_data_list:
            x = vid if isinstance(vid, torch.Tensor) else torch.as_tensor(vid)
            if x.ndim == 1:
                x = x.unsqueeze(0)
            elif x.ndim > 2:
                x = x.view(-1, x.size(-1))
            x = x.detach().to(device=device, dtype=torch.float32)  # [T,C]

            if x.numel() == 0:
                sim = torch.empty((0, K), dtype=torch.float32, device=device)
            else:
                x = F.normalize(x, dim=1)
                if chunk_size is None or x.shape[0] <= chunk_size:
                    sim = x @ c.T  # [T,K]
                else:
                    # chunk over T to limit peak memory
                    outs = []
                    for s in range(0, x.shape[0], chunk_size):
                        outs.append(x[s : s + chunk_size] @ c.T)
                    sim = torch.cat(outs, dim=0)
                sim = torch.clamp(sim, min=0.0)

            # return CPU fp32
            activations.append(sim.to("cpu", dtype=dtype))
            embeddings.append(vid)  # keep original reference if needed

    return activations, embeddings


class PositionalEncoding(nn.Module):
    """
    Supports both [T, C] and [B, T, C] input tensors, automatically unsqueezing and squeezing as needed for 2D input.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(
            1
        )  # [max_len,1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)  # [max_len, C]

        # Handle even and odd indices separately to avoid dimension mismatch
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            # Even d_model: use same div_term for cosine
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # Odd d_model: need one more element for cosine
            div_term_cos = torch.exp(
                torch.arange(0, d_model - 1, 2, dtype=torch.float32)
                * (-math.log(10000.0) / d_model)
            )
            pe[:, 1::2] = torch.cos(position * div_term_cos)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        """
        Handles both 2D and 3D input, automatically unsqueezing and squeezing for [T, C] input. Positional encoding is broadcast over the batch dimension.
        """
        squeeze_back = False
        if x.dim() == 2:  # [T, C] -> [1, T, C]
            x = x.unsqueeze(0)
            squeeze_back = True
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :]  # broadcast over batch
        x = self.dropout(x)
        if squeeze_back:
            x = x.squeeze(0)
        return x


# -------------------------
# Diagonal (per-channel) Q/K/V + per-channel FFN
# -------------------------
class DiagQKVd(nn.Module):
    """Per-channel Q/K/V with width d (no cross-concept mixing)."""

    def __init__(self, C: int, d: int = 8, bias: bool = True):
        super().__init__()
        self.C, self.d = C, d
        # groups=C keeps channels isolated; each channel gets d features
        self.q = nn.Conv1d(C, C * d, 1, groups=C, bias=bias)
        self.k = nn.Conv1d(C, C * d, 1, groups=C, bias=bias)
        self.v = nn.Conv1d(C, C * d, 1, groups=C, bias=bias)

    def forward(self, x):  # x: [B,T,C]
        B, T, C = x.shape
        xc = x.transpose(1, 2)  # [B,C,T]
        Q = self.q(xc).transpose(1, 2).view(B, T, C, self.d)  # [B,T,C,d]
        K = self.k(xc).transpose(1, 2).view(B, T, C, self.d)
        V = self.v(xc).transpose(1, 2).view(B, T, C, self.d)
        return Q, K, V

class ChannelTimeNorm(nn.Module):
    def __init__(self, C, eps=1e-5, affine=True):
        super().__init__()
        self.ln = nn.LayerNorm(C, eps=eps, elementwise_affine=affine)

    def forward(self, x):  # x: [B,T,C]
        return self.ln(x)


class PerChannelFFN(nn.Module):
    """Per-channel FFN (no cross-concept mixing)."""

    def __init__(self, C: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Conv1d(
            C, C, kernel_size=1, groups=C, bias=True
        )  # group equals C to have no channel mixing!
        self.fc2 = nn.Conv1d(C, C, kernel_size=1, groups=C, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: [B, T, C]
        xc = x.transpose(1, 2)  # [B, C, T]
        y = self.fc2(self.drop(self.act(self.fc1(xc))))
        return y.transpose(1, 2)  # [B, T, C]


class PerChannelTemporalBlock(nn.Module):
    """
    Attention over time for each concept channel independently.
    Stores attn_weights: [B, C, T, T].
    """

    def __init__(self, C: int, d: int = 1, dropout: float = 0.1, T_max: int = 1024):
        super().__init__()
        self.C, self.d = C, d
        self.qkv = DiagQKVd(C, d)
        self.scale = d**-0.5
        self.logit_scale = nn.Parameter(torch.zeros(C))  # per-concept multiplier

        self.norm1 = ChannelTimeNorm(C)
        self.norm2 = ChannelTimeNorm(C)
        self.drop = nn.Dropout(dropout)

        self.ffn = PerChannelFFN(C, dropout=dropout)

        self.act = nn.GELU()

        self.attn_weights = None

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Pre-attention norm
        y = self.norm1(x)  # [B, T, C]

        # Per-channel QKV: Q/K/V are [B, T, C, d]
        Q, K, V = self.qkv(y)

        # Attention logits per channel: [B, C, T, T]
        scores = torch.einsum("btcd,bucd->bctu", Q, K) * self.scale

        # Optional masks
        if attn_mask is not None:
            # treat bool as additive -inf mask; float as-is
            if attn_mask.dtype == torch.bool:
                am = torch.zeros_like(attn_mask, dtype=scores.dtype)
                am = am.masked_fill(attn_mask, float("-inf"))
            else:
                am = attn_mask.to(dtype=scores.dtype)
            scores = scores + am.view(1, 1, T, T)

        if key_padding_mask is not None:
            kpm = key_padding_mask.view(B, 1, 1, T)  # True = masked
            scores = scores.masked_fill(kpm, float("-inf"))

        # Softmax over source time axis
        w = torch.softmax(scores, dim=-1)  # [B, C, T, T]
        self.attn_weights = w.detach()

        # Weighted sum of values, then reduce d
        out = torch.einsum("bctu,bucd->btcd", w, V).mean(dim=-1)  # [B, T, C]

        # Residual + dropout
        x = x + self.drop(out)

        # Post-attention norm + per-channel FFN (already expects [B,T,C])
        z = self.norm2(x)
        z = self.ffn(z)

        # Residual + dropout
        x = x + self.drop(z)
        return x


def _pick_num_heads(C: int, proposed: Optional[int]) -> int:
    if proposed is not None and proposed >= 1 and C % proposed == 0:
        return proposed
    for h in [8, 6, 4, 3, 2]:
        if h <= C and C % h == 0:
            return h
    return 1


class FullAttentionTemporalBlock(nn.Module):
    """
    Full multi-head self-attention over time with channel mixing (manual implementation).
    """

    def __init__(
        self,
        C: int,
        num_heads: Optional[int] = None,
        dropout: float = 0.1,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.C = C
        self.H = _pick_num_heads(C, num_heads)
        self.d = C // self.H
        assert self.H * self.d == C, "C must be divisible by num_heads"

        # Projections (mix channels)
        self.q_proj = nn.Linear(C, C, bias=True)
        self.k_proj = nn.Linear(C, C, bias=True)
        self.v_proj = nn.Linear(C, C, bias=True)
        self.o_proj = nn.Linear(C, C, bias=True)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(C, ffn_mult * C),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * C, C),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(C)
        self.norm2 = nn.LayerNorm(C)

        self.attn_weights = None  # [B, H, T, T]

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, C] -> [B, H, T, d]
        B, T, _ = x.shape
        return x.view(B, T, self.H, self.d).permute(0, 2, 1, 3)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,  # [T, T]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, T]
    ) -> torch.Tensor:
        assert x.dim() == 3, "x must be [B, T, C]"
        B, T, C = x.shape
        assert C == self.C

        # Projections
        Q = self._shape_heads(self.q_proj(x))  # [B,H,T,d]
        K = self._shape_heads(self.k_proj(x))  # [B,H,T,d]
        V = self._shape_heads(self.v_proj(x))  # [B,H,T,d]

        # Scaled dot-product attention
        scale = self.d**-0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B,H,T,T]

        # Masks
        if attn_mask is not None:
            # bool -> additive mask; float left as-is
            if attn_mask.dtype == torch.bool:
                am = torch.zeros_like(attn_mask, dtype=Q.dtype)  # 0 keep
                am = am.masked_fill(attn_mask, float("-inf"))
            else:
                am = attn_mask.to(dtype=Q.dtype)
            scores = scores + am.view(1, 1, T, T)

        if key_padding_mask is not None:
            kpm = key_padding_mask.to(torch.bool).view(
                B, 1, 1, T
            )  # broadcast on heads & queries
            scores = scores.masked_fill(kpm, float("-inf"))

        weights = F.softmax(scores, dim=-1)  # [B,H,T,T]
        weights = self.attn_drop(weights)
        self.attn_weights = weights.detach()

        out = torch.matmul(weights, V)  # [B,H,T,d]
        out = out.permute(0, 2, 1, 3).contiguous()  # [B,T,H,d]
        out = out.view(B, T, C)  # [B,T,C]
        out = self.o_proj(out)
        out = self.proj_drop(out)

        # Residual + norm
        x = self.norm1(x + out)

        # FFN + residual + norm
        ff = self.ffn(x)
        x = self.norm2(x + self.dropout(ff))
        return x


class MoTIF:
    """
    MoTIF model for video classification using concept bottleneck models.
    Assumes:
      - concepts_over_time_cosine returns signed cosine sims (no clamp).
      - self.model(window_embeddings, key_padding_mask) returns (logits, concepts, concepts_t, sharpness)
    """

    @staticmethod
    def _collate_pad(batch):
        """
        batch: list of tuples (seq:[T,C] CPU float32, y:int)
        Returns CPU pinned tensors to enable non_blocking .to(device)
        """
        B = len(batch)
        T = max(seq.shape[0] for seq, _ in batch)
        C = batch[0][0].shape[1]
        x = torch.zeros((B, T, C), dtype=torch.float32)
        mask = torch.ones((B, T), dtype=torch.bool)  # True = padded
        y = torch.empty((B,), dtype=torch.long)
        for i, (seq, yi) in enumerate(batch):
            t = seq.shape[0]
            x[i, :t].copy_(seq)  # CPU->CPU copy into pinned
            mask[i, :t] = False
            y[i] = yi
        return x, mask, y

    def __init__(self, embedder, concepts):
        self.device = get_torch_device(prefer="cuda")

        self.concepts = concepts
        self.all_data = embedder.video_embeddings  # dict: path -> [T,C]
        self.all_labels = (
            embedder.labels
        )  # list aligned with keys order (non-SSv2 case)
        self.video_paths = list(self.all_data.keys())
        self.video_spans = embedder.video_window_spans

        self.concept_bank = concepts.text_embeddings
        self.raw_activations, self.video_embeddings = concepts_over_time_cosine(
            self.concept_bank, list(self.all_data.values())
        )  # list of [T,C]

        keep_idx = [
            i
            for i, act in enumerate(self.raw_activations)
            if isinstance(act, torch.Tensor) and act.shape[0] > 0
        ]
        if len(keep_idx) != len(self.raw_activations):
            removed = len(self.raw_activations) - len(keep_idx)
            self.raw_activations = [self.raw_activations[i] for i in keep_idx]
            self.video_paths = [self.video_paths[i] for i in keep_idx]
            self.all_labels = [self.all_labels[i] for i in keep_idx]  # non-SSv2 path
            self.video_embeddings = [self.video_embeddings[i] for i in keep_idx]
            print(f"[MoTIF] Removed {removed} entries with empty activations.")

        # Stable, aligned numeric IDs (for SSv2)
        self.video_ids = [self.path_to_id(p) for p in self.video_paths]
        self.kept_ids = {vid for vid in self.video_ids if vid is not None}

        # Defer LabelEncoder to preprocess()
        self.encoder = LabelEncoder()
        self.class_weights = None

        self.mean_c, self.std_c = None, None
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.paths_train = self.paths_val = self.paths_test = None
        self.test_zero_shot = None

        # Model attached later
        self.model = None

    @staticmethod
    def path_to_id(p: str):
        base = os.path.splitext(os.path.basename(p))[0]
        m = re.search(r"(\d+)", base)
        return int(m.group(1)) if m else None

    # -------------------------
    # Zero-shot (vectorized over frames)
    # -------------------------
    @torch.inference_mode()
    def zero_shot(self, concept_embedder, wandb_run=None):
        assert (
            self.test_zero_shot is not None and self.y_test is not None
        ), "Call preprocess(...) first."

        # build text prompts and text embeddings
        class_prompts = ["a video of " + c for c in self.encoder.classes_.tolist()]
        text_embedder = copy.copy(concept_embedder)
        text_embedder.tokenizer = concept_embedder.tokenizer
        text_embedder.model = concept_embedder.model
        text_embedder.embedd_text(class_prompts)  # keep original method name

        # ensure device + dtype
        text_embeddings = text_embedder.text_embeddings.to(self.device, dtype=torch.float32)  # [K, C]
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # check model type for probability transform
        model_name = getattr(text_embedder, "model_name", "").lower()
        use_siglip = "siglip" in model_name

        if use_siglip:
            # SigLIP style scaling/bias (ensure fp32)
            scale = text_embedder.model.logit_scale.exp().to(self.device).float()
            bias = text_embedder.model.logit_bias.to(self.device).float()  # shape [K] or [1,K]

        # counters
        correct_pooled = 0
        correct_soft_avg = 0
        correct_hard_majority = 0

        for idx, frames in enumerate(self.test_zero_shot):
            # frames -> frame embeddings [T, C] on device
            frame_emb = torch.as_tensor(np.array(frames), device=self.device, dtype=torch.float32)
            frame_emb = F.normalize(frame_emb, dim=-1)  # [T, C]

            # pooled embedding (mean over time) [1, C]
            pooled_emb = F.normalize(frame_emb.mean(dim=0, keepdim=True), dim=-1)  # [1, C]

            # raw logits
            if use_siglip:
                logits_pooled = pooled_emb @ text_embeddings.T
                logits_pooled = logits_pooled * scale + bias  # [1, K]
                logits_per_frame = (frame_emb @ text_embeddings.T) * scale + bias  # [T, K]
                probs_per_frame = logits_per_frame.sigmoid()  # for soft average
            else:
                logits_pooled = pooled_emb @ text_embeddings.T  # [1, K]
                logits_per_frame = frame_emb @ text_embeddings.T  # [T, K]
                probs_per_frame = logits_per_frame.softmax(dim=-1)  # for soft average

            # predictions
            pred_pooled = logits_pooled.argmax(dim=-1).item()                       # mean-pooled embedding
            pred_soft_avg = probs_per_frame.mean(dim=0).argmax().item()             # soft voting (avg probs)

            per_frame_preds = logits_per_frame.argmax(dim=-1)                       # [T]
            counts = torch.bincount(per_frame_preds, minlength=logits_per_frame.size(1))
            pred_hard_majority = counts.argmax().item()                             # hard majority (mode)

            # ground truth
            y = int(self.y_test[idx])

            # update counters
            correct_pooled += int(pred_pooled == y)
            correct_soft_avg += int(pred_soft_avg == y)
            correct_hard_majority += int(pred_hard_majority == y)

        n = max(1, len(self.test_zero_shot))
        acc_pooled = correct_pooled / n
        acc_soft_avg = correct_soft_avg / n
        acc_hard_majority = correct_hard_majority / n

        # logging
        if wandb_run is not None:
            wandb_run.log(
                {
                    "zero_shot_acc_pooled": acc_pooled,
                    "zero_shot_acc_soft_avg": acc_soft_avg,
                    "zero_shot_acc_hard_majority": acc_hard_majority,
                }
            )

        print(
            f"[ZS] pooled={acc_pooled:.4f} | soft-avg={acc_soft_avg:.4f} | hard-majority={acc_hard_majority:.4f}"
        )

        return {
            "acc_pooled": acc_pooled,
            "acc_soft_avg": acc_soft_avg,
            "acc_hard_majority": acc_hard_majority,
        }

    # -------------------------
    # Preprocess (unchanged split logic; at end we build datasets)
    # -------------------------
    def preprocess(self, 
                   dataset: str, 
                   info: Optional[str] = None,
                   test_size: float = 0.2, 
                   random_state: int = 42,):
        binary_array = []

        def get_index(info):
            if info == "s1":
                index = 1
            elif info == "s2":
                index = 2
            elif info == "s3":
                index = 3
            else:
                index = 1
            return index

        if info:
            if dataset == "breakfast":
                RANGES = {
                    "s1": range(3, 16),
                    "s2": range(16, 29),
                    "s3": range(29, 42),
                    "s4": range(42, 54),
                }

                def split_paths_by_group(paths, group_name, ranges=RANGES):
                    if group_name not in ranges:
                        raise ValueError(
                            f"Unknown group '{group_name}'. Expected one of {list(ranges)}"
                        )
                    target = ranges[group_name]
                    for p in paths:
                        if any(re.search(rf"P{num:02}", p) for num in target):
                            binary_array.append(False)
                        else:
                            binary_array.append(True)
                    return binary_array

                binary_array = split_paths_by_group(self.video_paths, info)

            elif dataset == "ucf101":
                index = get_index(info)
                ucf_test_list = (
                    f"../Datasets/UCF101/ucfTrainTestlist/testlist0{index}.txt"
                )
                path_list = pd.read_csv(ucf_test_list, sep=" ", header=None)
                for path in self.video_paths:
                    path_rel = path.split("Video_data/")[1].replace(".mp4", ".avi")
                    binary_array.append(
                        False if path_rel in path_list[0].values else True
                    )

            elif dataset == "hmdb51":
                index = get_index(info)
                labels_path = "../Datasets/HMDB/testTrainMulti_7030_splits/"
                path_text_dirs = glob.glob(os.path.join(labels_path, "*.txt"))
                path_text_dirs_idx = [p for p in path_text_dirs if f"split{index}" in p]
                path_text_dirs_idx.sort()
                path_list_test, path_list_train, path_list_ignore = set(), set(), set()
                for txt_path in path_text_dirs_idx:
                    with open(txt_path, "r") as fh:
                        for line in fh:
                            name, flag = line.strip().split()
                            if flag == "2":
                                path_list_test.add(name)
                            elif flag == "0":
                                path_list_ignore.add(name)
                            else:
                                path_list_train.add(name)
                mask = []
                for vp in self.video_paths:
                    basename = os.path.basename(vp).replace(".mp4", ".avi")
                    if basename in path_list_test:
                        mask.append(False)
                    elif basename in path_list_train:
                        mask.append(True)
                    elif basename in path_list_ignore:
                        mask.append(None)
                    else:
                        mask.append(None)
                kept = [
                    (x, y, p, b, m)
                    for x, y, p, b, m in zip(
                        self.raw_activations,
                        self.all_labels,
                        self.video_paths,
                        self.video_embeddings,
                        mask,
                    )
                    if m is not None
                ]
                if not kept:
                    raise ValueError(
                        "HMDB split produced no usable items. Check paths and split lists."
                    )
                (
                    self.raw_activations,
                    self.all_labels,
                    self.video_paths,
                    self.video_embeddings,
                    mask_kept,
                ) = map(list, zip(*kept))
                self.video_ids = [
                    (
                        int(os.path.splitext(os.path.basename(p))[0])
                        if os.path.splitext(os.path.basename(p))[0].isdigit()
                        else None
                    )
                    for p in self.video_paths
                ]
                self.kept_ids = {vid for vid in self.video_ids if vid is not None}
                binary_array = [True if m else False for m in mask_kept]

            elif dataset == "something2":
                # ===== SSv2 handling =====
                def replace_something(text: str) -> str:
                    return re.sub(r"\[(.*?)\]", r"\1", text)

                val_json = "../Datasets/Something2/labels/validation.json"
                train_json = "../Datasets/Something2/labels/train.json"
                test_json = "../Datasets/Something2/labels/test.json"
                test_csv = "../Datasets/Something2/labels/test-answers.csv"

                df_train = pd.read_json(train_json)
                df_val = pd.read_json(val_json)
                df_test = pd.read_json(test_json)
                train_ids = [int(row[0]) for row in df_train.values.tolist()]
                val_ids = [int(row[0]) for row in df_val.values.tolist()]
                test_ids = [int(row[0]) for row in df_test.values.tolist()]
                train_labels = [replace_something(t) for t in df_train["template"]]
                val_labels = [replace_something(t) for t in df_val["template"]]
                test_tbl = pd.read_csv(
                    test_csv, sep=";", header=None, dtype={0: int, 1: str}
                )
                test_labels_map = dict(zip(test_tbl[0].tolist(), test_tbl[1].tolist()))
                test_labels = [test_labels_map[i] for i in test_ids]
                id2split = {}
                id2split.update(
                    {i: ("train", l) for i, l in zip(train_ids, train_labels)}
                )
                id2split.update({i: ("val", l) for i, l in zip(val_ids, val_labels)})
                id2split.update({i: ("test", l) for i, l in zip(test_ids, test_labels)})

                train_x, val_x, test_x = [], [], []
                train_y, val_y, test_y = [], [], []
                self.test_zero_shot = []
                self.paths_train, self.paths_val, self.paths_test = [], [], []
                self.video_ids = [self.path_to_id(p) for p in self.video_paths]
                missed = 0
                for idx, vid in enumerate(self.video_ids):
                    if vid is None:
                        missed += 1
                        continue
                    entry = id2split.get(vid)
                    if entry is None:
                        missed += 1
                        continue
                    split, lab = entry
                    if split == "train":
                        train_x.append(self.raw_activations[idx])
                        train_y.append(lab)
                        self.paths_train.append(self.video_paths[idx])
                    elif split == "val":
                        val_x.append(self.raw_activations[idx])
                        val_y.append(lab)
                        self.paths_val.append(self.video_paths[idx])
                    elif split == "test":
                        test_x.append(self.raw_activations[idx])
                        test_y.append(lab)
                        self.paths_test.append(self.video_paths[idx])
                        self.test_zero_shot.append(self.video_embeddings[idx])
                if missed:
                    print(
                        f"[SSv2] Skipped {missed} items (no parseable ID or not in official splits)."
                    )

                if len(train_x) == 0:
                    raise RuntimeError(
                        "[SSv2] No training samples matched. Check filename-to-ID parsing and dataset paths."
                    )

                self.encoder = self.encoder.fit(train_y)
                self.X_train, self.y_train = train_x, self.encoder.transform(
                    np.array(train_y, dtype=object)
                )
                self.X_val, self.y_val = val_x, (
                    self.encoder.transform(np.array(val_y, dtype=object))
                    if len(val_x)
                    else (None, None)
                )
                self.X_test, self.y_test = test_x, (
                    self.encoder.transform(np.array(test_y, dtype=object))
                    if len(test_x)
                    else (None, None)
                )

            # ===== end SSv2 =====
            if dataset != "something2":
                self.X_train = [
                    self.raw_activations[i]
                    for i in range(len(self.raw_activations))
                    if binary_array[i]
                ]
                self.X_test = [
                    self.raw_activations[i]
                    for i in range(len(self.raw_activations))
                    if not binary_array[i]
                ]
                self.y_train = [
                    self.all_labels[i]
                    for i in range(len(self.all_labels))
                    if binary_array[i]
                ]
                self.y_test = [
                    self.all_labels[i]
                    for i in range(len(self.all_labels))
                    if not binary_array[i]
                ]
                self.paths_train = [
                    self.video_paths[i]
                    for i in range(len(self.video_paths))
                    if binary_array[i]
                ]
                self.paths_test = [
                    self.video_paths[i]
                    for i in range(len(self.video_paths))
                    if not binary_array[i]
                ]
                self.encoder = self.encoder.fit(self.y_train)
                self.y_train = self.encoder.transform(self.y_train)
                self.y_test = self.encoder.transform(self.y_test)
                self.test_zero_shot = [
                    self.video_embeddings[i]
                    for i in range(len(self.video_embeddings))
                    if not binary_array[i]
                ]

        else:
            # Stratified random split (non-SSv2)
            (
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                self.paths_train,
                self.paths_test,
            ) = train_test_split(
                self.raw_activations,
                self.all_labels,
                self.video_paths,
                test_size=test_size,
                random_state=random_state,
                stratify=self.all_labels,
            )
            self.encoder = self.encoder.fit(self.y_train)
            self.y_train = self.encoder.transform(self.y_train)
            self.y_test = self.encoder.transform(self.y_test)

        # ----- Standardization -----
        self.mean_c, self.std_c = compute_concept_standardization(self.X_train)
        self.X_train = apply_standardization(self.X_train, self.mean_c, self.std_c)
        self.X_test = apply_standardization(self.X_test, self.mean_c, self.std_c)
        if self.X_val is not None:
            self.X_val = apply_standardization(self.X_val, self.mean_c, self.std_c)

        # ----- Class weights -----
        classes, counts = np.unique(self.y_train, return_counts=True)
        self.class_weights = torch.tensor(counts.max() / counts, dtype=torch.float32)
        self.num_concepts = self.X_train[0].shape[-1]
        self.num_classes = len(classes)

    def train_model(
        self,
        num_epochs: int,
        l1_lambda: float,
        lambda_sparse: float,
        batch_size: int = 8,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        enforce_nonneg: bool = True,
        class_weights: bool = True,
        wandb_run: Optional[wandb.WandbRun] = None,
        random_seed: int = 42,
        ckpt_path: Optional[str] = None,
        early_stopping_patience: int = 50,
    ):

        if wandb_run is not None:
            wandb_run.config.update(
                {
                    "num_epochs": num_epochs,
                    "l1_lambda": l1_lambda,
                    "lambda_sparse": lambda_sparse,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "enforce_nonneg": enforce_nonneg,
                    "class_weights": class_weights,
                    "transformer_layers": self.model.transformer_layers,
                    "lse_tau": self.model.lse_tau,
                    "diagonal_attention": self.model.diagonal_attention,
                    "early_stopping_patience": early_stopping_patience,
                }
            )

        # move model to device
        self.model.to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        if class_weights:
            criterion = nn.CrossEntropyLoss(
                weight=self.class_weights.to(self.device), label_smoothing=0.1
            )
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        num_train = len(self.X_train)

        best_metric = -float("inf")
        best_state = None
        best_epoch = -1
        epochs_since_improvement = 0
        use_early_stopping = (early_stopping_patience is not None) and (
            len(self.X_test) > 0
        )

        for epoch in range(num_epochs):
            self.model.train()
            correct, total = 0, 0
            last_loss, last_L_sparse = None, None
            epoch_L_sparse_sum, epoch_batches = 0.0, 0

            base_seed = int(getattr(self, "seed", random_seed))
            g = torch.Generator(device="cpu").manual_seed(base_seed + epoch)
            perm_tensor = torch.randperm(num_train, generator=g)
            perm = perm_tensor.tolist()

            for start in range(0, num_train, batch_size):
                end = min(start + batch_size, num_train)
                idx = perm[start:end]
                batch_seqs = [self.X_train[i] for i in idx]
                batch_labels = torch.tensor(
                    [int(self.y_train[i]) for i in idx],
                    dtype=torch.long,
                    device=self.device,
                )

                inputs, pad_mask = pad_batch_sequences(batch_seqs, device=self.device)
                optimizer.zero_grad()

                # updated forward: now returns sharpness
                logits, concepts_, concepts_t, sharpness = self.model(
                    inputs, key_padding_mask=pad_mask
                )

                valid = (~pad_mask).unsqueeze(-1).float()
                last_L_sparse = (concepts_t.abs() * valid).sum() / (
                    valid.sum() * concepts_t.shape[-1]
                ).clamp(min=1.0)

                ce = criterion(logits, batch_labels)
                l1 = l1_lambda * self.model.classifier.weight.abs().sum()
                loss = ce + l1 + lambda_sparse * last_L_sparse
                loss.backward()
                optimizer.step()
                last_loss = loss

                # accumulate for epoch-average L_sparse
                epoch_L_sparse_sum += float(last_L_sparse.detach().item())
                epoch_batches += 1

                if enforce_nonneg:
                    with torch.no_grad():
                        self.model.classifier.weight.clamp_(min=0.0)

                preds = logits.argmax(dim=1)
                correct += int((preds == batch_labels).sum().item())
                total += batch_labels.shape[0]

            acc = correct / max(1, total)
            epoch_L_sparse = epoch_L_sparse_sum / max(1, epoch_batches)

            # ===== evaluation =====
            def evaluate(dataset_X, dataset_y):
                self.model.eval()
                correct, total = 0, 0
                sharpness_vals = []
                with torch.no_grad():
                    for start in range(0, len(dataset_X), batch_size):
                        end = min(start + batch_size, len(dataset_X))
                        batch_seqs = [dataset_X[i] for i in range(start, end)]
                        batch_labels = torch.tensor(
                            [int(dataset_y[i]) for i in range(start, end)],
                            dtype=torch.long,
                            device=self.device,
                        )
                        inputs, pad_mask = pad_batch_sequences(
                            batch_seqs, device=self.device
                        )

                        logits, _, _, sharpness = self.model(
                            inputs, key_padding_mask=pad_mask
                        )
                        preds = logits.argmax(dim=1)
                        correct += int((preds == batch_labels).sum().item())
                        total += batch_labels.shape[0]

                        for b in range(logits.shape[0]):
                            sharpness_vals.append(
                                {
                                    "concepts_max": float(
                                        sharpness["concepts"]["max"][b]
                                        .mean()
                                        .detach()
                                        .cpu()
                                        .item()
                                    ),
                                    "concepts_entropy": float(
                                        sharpness["concepts"]["entropy"][b]
                                        .mean()
                                        .detach()
                                        .cpu()
                                        .item()
                                    ),
                                    "logits_max": float(
                                        sharpness["logits"]["max"][b]
                                        .mean()
                                        .detach()
                                        .cpu()
                                        .item()
                                    ),
                                    "logits_entropy": float(
                                        sharpness["logits"]["entropy"][b]
                                        .mean()
                                        .detach()
                                        .cpu()
                                        .item()
                                    ),
                                }
                            )

                acc = correct / max(1, total)
                if sharpness_vals:
                    mean_sharp = {
                        k: float(np.mean([s[k] for s in sharpness_vals]))
                        for k in sharpness_vals[0]
                    }
                else:
                    mean_sharp = {}
                return acc, mean_sharp

            test_acc, test_sharp = (
                (0.0, {})
                if len(self.X_test) == 0
                else evaluate(self.X_test, self.y_test)
            )
            val_acc, val_sharp = (
                (0.0, {}) if self.X_val is None else evaluate(self.X_val, self.y_val)
            )

            metric = test_acc if len(self.X_test) > 0 else acc

            # ===== checkpointing =====
            if metric > best_metric + 1e-8:
                best_metric = metric
                best_epoch = epoch
                epochs_since_improvement = 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                if ckpt_path:
                    tmp = ckpt_path + ".tmp"
                    torch.save(best_state, tmp)
                    os.replace(tmp, ckpt_path)
            else:
                epochs_since_improvement += 1

            # ===== wandb logging =====
            if wandb_run is not None:
                current_lr = (
                    optimizer.param_groups[0]["lr"] if optimizer.param_groups else None
                )
                log_data = {
                    "epoch": epoch + 1,
                    "train_loss": (
                        float(last_loss.item()) if last_loss is not None else None
                    ),
                    "train_acc": acc,
                    "test_acc": test_acc,
                    "val_acc": val_acc if self.X_val is not None else None,
                    "L_sparse": (
                        float(last_L_sparse.item())
                        if last_L_sparse is not None
                        else None
                    ),
                    "learning_rate": current_lr,
                    "best_val_acc": best_metric,
                    "epochs_since_improvement": epochs_since_improvement,
                }
                # add sharpness metrics
                for prefix, sharp in [("test_", test_sharp), ("val_", val_sharp)]:
                    for k, v in sharp.items():
                        log_data[prefix + "sharp_" + k] = v
                wandb_run.log(log_data)

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                msg_loss = (
                    float(last_loss.item()) if last_loss is not None else float("nan")
                )
                msg_sparse = (
                    float(last_L_sparse.item())
                    if last_L_sparse is not None
                    else float("nan")
                )
                print(
                    f"Epoch {epoch+1}/{num_epochs} | loss {msg_loss:.4f} | test_acc {test_acc:.4f} "
                    f"| train_acc {acc:.4f} | L_sparse {msg_sparse:.4f} "
                    f"| best_val {best_metric:.4f} | epochs_no_improve {epochs_since_improvement}"
                )

            # early stopping
            if (
                use_early_stopping
                and epochs_since_improvement >= early_stopping_patience
            ):
                print(
                    f"[MoTIF] Early stopping triggered (no improvement for {epochs_since_improvement} epochs). Stopping at epoch {epoch+1}."
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "early_stopped_epoch": epoch + 1,
                            "early_stopping_patience": early_stopping_patience,
                        }
                    )
                break

        # ===== restore best =====
        if best_state is not None:
            self.model.load_state_dict(best_state, strict=True)
            self.model.eval()
            print(
                f"[MoTIF] Restored best weights from epoch {best_epoch+1} (metric={best_metric:.4f})."
            )
        else:
            print("[MoTIF] No best_state captured (empty training?).")


# -------------------------
# PerConceptAffine + CBMTransformer using the per-channel temporal block
# -------------------------


class PerConceptAffine(nn.Module):
    def __init__(self, num_concepts: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_concepts))
        self.bias = nn.Parameter(torch.zeros(num_concepts))

    def forward(self, x: torch.Tensor):
        ## Comment out to test no scaling and bias ablation for paper
        y = F.softplus(x * self.scale + self.bias) - math.log(2.0)
        return y.clamp(min=0.0)


class CBMTransformer(nn.Module):
    def __init__(
        self,
        num_concepts: int,
        num_classes: int,
        transformer_layers: int = 1,
        dropout: float = 0.1,
        lse_tau: float = 1.0,
        nonneg_classifier: bool = False,
        diagonal_attention: bool = True,
        dimension=1,
    ):
        super().__init__()
        self.lse_tau = lse_tau
        self.diagonal_attention = diagonal_attention
        self.transformer_layers = transformer_layers

        self.posenc = PositionalEncoding(
            d_model=num_concepts, dropout=dropout, max_len=2000
        )
        if diagonal_attention:
            self.layers = nn.ModuleList(
                [
                    PerChannelTemporalBlock(
                        C=num_concepts, dropout=dropout, d=dimension
                    )
                    for _ in range(transformer_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    FullAttentionTemporalBlock(
                        C=num_concepts, num_heads=None, dropout=dropout
                    )
                    for _ in range(transformer_layers)
                ]
            )
        self.norm = nn.LayerNorm(num_concepts)
        self.concept_predictor = PerConceptAffine(num_concepts)

        if nonneg_classifier:
            self.classifier = NonNegativeLinear(num_concepts, num_classes)
        else:
            self.classifier = nn.Linear(num_concepts, num_classes)

        # for introspection
        self.last_time_importance = None  # [B,T] detached
        

    def forward(
        self,
        window_embeddings: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        channel_ids: Optional[Union[List[int], torch.Tensor]] = None,
        window_ids: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        window_embeddings: [B,T,C] or [T,C]
        key_padding_mask: [B,T] with True for padded tokens to be ignored

        Returns:
        logits:     [B,K]    pooled class logits
        concepts:   [B,C]    pooled concept activations
        concepts_t: [B,T,C]  per-time-step concepts
        sharpness:  dict with 'concepts' and 'logits' sharpness per batch
        """
        x = window_embeddings
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1,T,C]
            if key_padding_mask is not None and key_padding_mask.dim() == 1:
                key_padding_mask = key_padding_mask.unsqueeze(0)

        # --- transformer backbone ---
        x = self.posenc(x)  # [B,T,C]
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)  # [B,T,C]

        # --- concept predictions per time step ---
        concepts_t = self.concept_predictor(x)  # [B,T,C]

        # --- concept interventions ---
        if channel_ids is not None and window_ids is not None:
            concepts_t[:, window_ids, channel_ids] = 0
        elif channel_ids is not None:
            concepts_t[:, :, channel_ids] = 0
        elif window_ids is not None:
            concepts_t[:, window_ids, :] = 0

        logits_t = self.classifier(concepts_t)  # [B,T,K]

        tau = self.lse_tau

        # --- LSE pooling over time ---
        if key_padding_mask is not None:
            concepts_t_masked = concepts_t.masked_fill(
                key_padding_mask.unsqueeze(-1), float("-inf")
            )
            logits_t_masked = logits_t.masked_fill(
                key_padding_mask.unsqueeze(-1), float("-inf")
            )

            concepts = (concepts_t_masked * tau).logsumexp(dim=1) / tau  # [B,C]
            logits = (logits_t_masked * tau).logsumexp(dim=1) / tau  # [B,K]
        else:
            concepts = (concepts_t * tau).logsumexp(dim=1) / tau
            logits = (logits_t * tau).logsumexp(dim=1) / tau

        # --- temporal importance for explanation ---
        with torch.no_grad():
            pred = logits.argmax(dim=1)  # [B]
            sel = torch.gather(logits_t, dim=2, index=pred[:, None, None]).squeeze(
                -1
            )  # [B,T]
            if key_padding_mask is not None:
                sel = sel.masked_fill(key_padding_mask, float("-inf"))
            self.last_time_importance = torch.softmax(
                sel / tau, dim=1
            ).detach()  # softmax importance

        # --- compute sharpness of LSE pooled distributions ---
        def compute_sharpness(x_t, mask=None):
            """Compute max / entropy as sharpness metric for batch"""
            if mask is not None:
                x_t = x_t.masked_fill(mask.unsqueeze(-1), float("-inf"))
            probs = torch.softmax(tau * x_t, dim=1)
            probs = probs.clamp(min=1e-8)  # avoids log(0)
            max_prob = probs.max(dim=1).values  # [B]
            entropy = -(probs * probs.log()).sum(dim=1)
            return {"max": max_prob, "entropy": entropy}

        sharpness = {
            "concepts": compute_sharpness(concepts_t, key_padding_mask),
            "logits": compute_sharpness(logits_t, key_padding_mask),
        }

        return logits, concepts, concepts_t, sharpness

    def get_attention_maps(self):
        # list of [B, C, T, T] (detached)
        return [
            layer.attn_weights.cpu() if layer.attn_weights is not None else None
            for layer in self.layers
        ]


def mean_cbm(model, wandb_run=None):
    X_train, X_test = model.X_train.copy(), model.X_test.copy()
    y_train, y_test = model.y_train.copy(), model.y_test.copy()
    num_classes = model.num_classes
    num_concepts = model.num_concepts
    batch_size = 1

    device = getattr(model, "device", get_torch_device())

    random = False # was for testing 
    if random:

        def get_random_image(x):
            idx = np.random.randint(0, len(x))
            return x[idx]

        # Replace each video with a random frame (as np array)
        X_train_random = [get_random_image(x) for x in X_train]
        X_test_random = [get_random_image(x) for x in X_test]

        X_train_mean = X_train_random
        X_test_mean = X_test_random

    else:
        # take mean
        X_train_mean = [torch.mean(x, axis=0) for x in X_train]  # [T,C] -> [C]
        X_test_mean = [torch.mean(x, axis=0) for x in X_test]  # [T,C] -> [C]

    # Stack into arrays before converting to torch tensors
    X_train_arr = np.stack(
        [
            t.cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)
            for t in X_train_mean
        ]
    )
    X_test_arr = np.stack(
        [
            t.cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)
            for t in X_test_mean
        ]
    )

    tensor_train = torch.tensor(X_train_arr, dtype=torch.float32, device=device)
    tensor_test = torch.tensor(X_test_arr, dtype=torch.float32, device=device)

    # train a linear model on the random/mean frames

    linear_model = nn.Linear(num_concepts, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(linear_model.parameters(), lr=0.001)
    num_epochs = 200
    for epoch in range(num_epochs):
        linear_model.train()
        optimizer.zero_grad()
        outputs = linear_model(tensor_train)
        loss = criterion(
            outputs, torch.tensor(y_train, dtype=torch.long, device=device)
        )
        loss.backward()
        optimizer.step()
        if wandb_run is not None:
            with torch.no_grad():
                preds = outputs.argmax(dim=1)
                acc = (preds.detach().cpu().numpy() == y_train).mean()
                current_lr = (
                    optimizer.param_groups[0]["lr"] if optimizer.param_groups else None
                )
                wandb_run.log(
                    {
                        "mean_train_loss": loss.item(),
                        "mean_train_acc": acc,
                        "mean_learning_rate": current_lr,
                    }
                )
    linear_model.eval()
    with torch.no_grad():
        outputs = linear_model(tensor_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted.detach().cpu().numpy() == y_test).mean()
    print(f"CBM accuracy test: {accuracy:.4f}")
    if wandb_run is not None:
        wandb_run.log({"mean_test_acc": accuracy})


class NonNegativeLinear:
    def __init__(self, in_features, out_features, bias=True):
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        self.linear.weight.data.clamp_(min=0.0)
        return self.linear(x)
