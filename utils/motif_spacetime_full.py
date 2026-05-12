from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, List


class PerChannelFFN(nn.Module):
    """Per-channel FFN (no cross-concept mixing)."""

    def __init__(self, C: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Conv1d(C, C, kernel_size=1, groups=C, bias=True)
        self.fc2 = nn.Conv1d(C, C, kernel_size=1, groups=C, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        xc = x.transpose(1, 2)
        y = self.fc2(self.drop(self.act(self.fc1(xc))))
        return y.transpose(1, 2)


class PerTimeSpatialBlock(nn.Module):
    """
    Attention across concepts/channels (C dimension) at each time step independently.
    This is the "spatial" component of space-time attention.
    Input: [B, T, C] -> attention over C for each T -> [B, T, C, C] attention maps.
    
    Uses identity-preserving gating to maintain concept interpretability while allowing
    controlled spatial interaction.
    """

    def __init__(self, C: int, d: int = 1, dropout: float = 0.1, spatial_gate: float = 0.01, identity_bias: float = 5.0):
        super().__init__()
        self.C, self.d = C, d
        self.spatial_gate = spatial_gate
        self.identity_bias = identity_bias
        self.q = nn.Linear(C, C * d, bias=True)
        self.k = nn.Linear(C, C * d, bias=True)
        self.v = nn.Linear(C, C * d, bias=True)
        self.scale = d ** -0.5

        self.norm1 = nn.LayerNorm(C)
        self.norm2 = nn.LayerNorm(C)
        self.drop = nn.Dropout(dropout)
        self.ffn = PerChannelFFN(C, dropout=dropout)
        self.attn_weights = None

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B, T, C]
        attn_mask: Optional [C, C] mask for concept-to-concept attention
        key_padding_mask: Optional [B, T] mask for padded time steps
        """
        B, T, C = x.shape
        y = self.norm1(x)

        Q = self.q(y).view(B, T, C, self.d)
        K = self.k(y).view(B, T, C, self.d)
        V = self.v(y).view(B, T, C, self.d)

        scores = torch.einsum('btid,btjd->btij', Q, K) * self.scale
        identity_mask = torch.eye(C, device=scores.device, dtype=scores.dtype)
        scores = scores + self.identity_bias * identity_mask.view(1, 1, C, C)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                am = torch.zeros_like(attn_mask, dtype=scores.dtype)
                am = am.masked_fill(attn_mask, float('-inf'))
            else:
                am = attn_mask.to(dtype=scores.dtype)
            scores = scores + am.view(1, 1, C, C)

        if key_padding_mask is not None:
            kpm = key_padding_mask.view(B, T, 1, 1).to(scores.dtype)
            scores = scores.masked_fill(kpm.bool(), float('-inf'))

        scores = scores.clamp(min=-1e4, max=1e4)
        w = torch.softmax(scores, dim=-1)
        
        if torch.isnan(w).any():
            nan_mask = torch.isnan(w)
            w = torch.where(nan_mask, torch.ones_like(w) / C, w)
        
        self.attn_weights = w.detach()

        out = torch.einsum('btij,btjd->btid', w, V)
        if self.d > 1:
            out = out.mean(dim=-1)
        else:
            out = out.squeeze(-1)
        
        if key_padding_mask is not None:
            out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        if self.spatial_gate > 0:
            x = x + self.spatial_gate * self.drop(out)

        z = self.norm2(x)
        z = self.ffn(z)
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
    Full multi-head self-attention over time with channel mixing.
    This is the "temporal" component of space-time attention with full (non-diagonal) attention.
    Stores attn_weights: [B, H, T, T].
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

        self.attn_weights = None

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
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

        y = self.norm1(x)

        Q = self._shape_heads(self.q_proj(y))
        K = self._shape_heads(self.k_proj(y))
        V = self._shape_heads(self.v_proj(y))

        scale = self.d**-0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                am = torch.zeros_like(attn_mask, dtype=Q.dtype)
                am = am.masked_fill(attn_mask, float("-inf"))
            else:
                am = attn_mask.to(dtype=Q.dtype)
            scores = scores + am.view(1, 1, T, T)

        if key_padding_mask is not None:
            kpm = key_padding_mask.to(torch.bool).view(B, 1, 1, T)
            scores = scores.masked_fill(kpm, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)
        self.attn_weights = weights.detach()

        out = torch.matmul(weights, V)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(B, T, C)
        out = self.o_proj(out)
        out = self.proj_drop(out)

        x = x + out

        z = self.norm2(x)
        ff = self.ffn(z)
        x = x + self.dropout(ff)
        return x


class SpaceTimeBlock(nn.Module):
    """
    Factorized space-time attention block with full temporal attention.
    First applies spatial attention (across concepts at each time step),
    then applies full temporal attention (over time with channel mixing).
    """

    def __init__(
        self,
        C: int,
        spatial_d: int = 1,
        num_heads: Optional[int] = None,
        dropout: float = 0.1,
        ffn_mult: int = 4,
        spatial_gate: float = 0.01,
        identity_bias: float = 5.0,
    ):
        super().__init__()
        self.spatial_attn = PerTimeSpatialBlock(
            C, d=spatial_d, dropout=dropout, spatial_gate=spatial_gate, identity_bias=identity_bias
        )
        self.temporal_attn = FullAttentionTemporalBlock(
            C, num_heads=num_heads, dropout=dropout, ffn_mult=ffn_mult
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B, T, C]
        attn_mask: Optional [T, T] mask for temporal attention
        key_padding_mask: Optional [B, T] mask for padded time steps
        """
        x = self.spatial_attn(x, attn_mask=None, key_padding_mask=key_padding_mask)
        x = self.temporal_attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        return x

    def get_attention_maps(self):
        """Returns both spatial and temporal attention maps."""
        return {
            "spatial": self.spatial_attn.attn_weights,
            "temporal": self.temporal_attn.attn_weights,
        }


class PositionalEncoding(nn.Module):
    """
    Supports both [T, C] and [B, T, C] input tensors, automatically unsqueezing and squeezing as needed for 2D input.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
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
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_back = True
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :]
        x = self.dropout(x)
        if squeeze_back:
            x = x.squeeze(0)
        return x


class PerConceptAffine(nn.Module):
    def __init__(self, num_concepts: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_concepts))
        self.bias = nn.Parameter(torch.zeros(num_concepts))

    def forward(self, x: torch.Tensor):
        y = F.softplus(x * self.scale + self.bias) - math.log(2.0)
        return y.clamp(min=0.0)


class CBMTransformerSTFull(nn.Module):
    """
    Space-Time CBM Transformer with factorized attention and full temporal attention.
    
    Architecture:
    1. Positional encoding over time
    2. Stack of Space-Time blocks:
       - Spatial attention: concepts interact at each time step (with identity-preserving mechanisms)
       - Temporal attention: full multi-head attention over time with channel mixing
    3. Concept predictor and classifier
    
    The key difference from CBMTransformerST is that we use full (non-diagonal) 
    temporal attention, allowing concepts to mix during temporal processing.
    
    Identity-preserving mechanisms in spatial attention:
    - Identity bias (default 5.0): encourages self-attention
    - Spatial gating (default 0.01): weakens spatial mixing to preserve concept identity
    - Per-channel FFN: prevents cross-concept mixing in feed-forward network
    """

    def __init__(
        self,
        num_concepts: int,
        num_classes: int,
        transformer_layers: int = 1,
        dropout: float = 0.1,
        lse_tau: float = 1.0,
        nonneg_classifier: bool = False,
        spatial_d: int = 1,
        num_heads: Optional[int] = None,
        ffn_mult: int = 4,
        spatial_gate: float = 0.01,
        identity_bias: float = 5.0,
    ):
        super().__init__()
        self.lse_tau = lse_tau
        self.transformer_layers = transformer_layers
        self.diagonal_attention = False

        self.posenc = PositionalEncoding(
            d_model=num_concepts, dropout=dropout, max_len=2000
        )

        self.layers = nn.ModuleList(
            [
                SpaceTimeBlock(
                    C=num_concepts,
                    spatial_d=spatial_d,
                    num_heads=num_heads,
                    dropout=dropout,
                    ffn_mult=ffn_mult,
                    spatial_gate=spatial_gate,
                    identity_bias=identity_bias,
                )
                for _ in range(transformer_layers)
            ]
        )

        self.norm = nn.LayerNorm(num_concepts)
        self.concept_predictor = PerConceptAffine(num_concepts)

        if nonneg_classifier:
            self.classifier = nn.Linear(num_concepts, num_classes)
            self.nonneg_classifier = True
        else:
            self.classifier = nn.Linear(num_concepts, num_classes)
            self.nonneg_classifier = False

        self.last_time_importance = None

    def forward(
        self,
        window_embeddings: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        channel_ids: Optional[Union[List[int], torch.Tensor]] = None,
        window_ids: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
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

        x = self.posenc(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)

        concepts_t = self.concept_predictor(x)

        if channel_ids is not None and window_ids is not None:
            concepts_t[:, window_ids, channel_ids] = 0
        elif channel_ids is not None:
            concepts_t[:, :, channel_ids] = 0
        elif window_ids is not None:
            concepts_t[:, window_ids, :] = 0

        if self.nonneg_classifier:
            with torch.no_grad():
                self.classifier.weight.data.clamp_(min=0.0)

        logits_t = self.classifier(concepts_t)

        tau = self.lse_tau

        if key_padding_mask is not None:
            concepts_t_masked = concepts_t.masked_fill(
                key_padding_mask.unsqueeze(-1), float("-inf")
            )
            logits_t_masked = logits_t.masked_fill(
                key_padding_mask.unsqueeze(-1), float("-inf")
            )

            concepts = (concepts_t_masked * tau).logsumexp(dim=1) / tau
            logits = (logits_t_masked * tau).logsumexp(dim=1) / tau
        else:
            concepts = (concepts_t * tau).logsumexp(dim=1) / tau
            logits = (logits_t * tau).logsumexp(dim=1) / tau

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            sel = torch.gather(logits_t, dim=2, index=pred[:, None, None]).squeeze(-1)
            if key_padding_mask is not None:
                sel = sel.masked_fill(key_padding_mask, float("-inf"))
            self.last_time_importance = torch.softmax(sel / tau, dim=1).detach()

        def compute_sharpness(x_t, mask=None):
            """Compute max / entropy as sharpness metric for batch"""
            if mask is not None:
                x_t = x_t.masked_fill(mask.unsqueeze(-1), float("-inf"))
            probs = torch.softmax(tau * x_t, dim=1)
            probs = probs.clamp(min=1e-8)
            max_prob = probs.max(dim=1).values
            entropy = -(probs * probs.log()).sum(dim=1)
            return {"max": max_prob, "entropy": entropy}

        sharpness = {
            "concepts": compute_sharpness(concepts_t, key_padding_mask),
            "logits": compute_sharpness(logits_t, key_padding_mask),
        }

        return logits, concepts, concepts_t, sharpness

    def get_attention_maps(self):
        """
        Returns attention maps from all layers.
        Each element is a dict with 'spatial' and 'temporal' keys.
        """
        return [layer.get_attention_maps() for layer in self.layers]

