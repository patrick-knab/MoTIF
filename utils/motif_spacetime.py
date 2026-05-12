from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, List


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


class DiagQKVd(nn.Module):
    """Per-channel Q/K/V with width d (no cross-concept mixing)."""

    def __init__(self, C: int, d: int = 1, bias: bool = True):
        super().__init__()
        self.C, self.d = C, d
        self.q = nn.Conv1d(C, C * d, 1, groups=C, bias=bias)
        self.k = nn.Conv1d(C, C * d, 1, groups=C, bias=bias)
        self.v = nn.Conv1d(C, C * d, 1, groups=C, bias=bias)

    def forward(self, x):
        B, T, C = x.shape
        xc = x.transpose(1, 2)
        Q = self.q(xc).transpose(1, 2).view(B, T, C, self.d)
        K = self.k(xc).transpose(1, 2).view(B, T, C, self.d)
        V = self.v(xc).transpose(1, 2).view(B, T, C, self.d)
        return Q, K, V


class ChannelTimeNorm(nn.Module):
    def __init__(self, C, eps=1e-5, affine=True):
        super().__init__()
        self.ln = nn.LayerNorm(C, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        return self.ln(x)


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


class PerChannelTemporalBlock(nn.Module):
    """
    Attention over time for each concept channel independently (diagonal attention).
    This is the "temporal" component of space-time attention.
    Stores attn_weights: [B, C, T, T].
    """

    def __init__(self, C: int, d: int = 1, dropout: float = 0.1, T_max: int = 1024):
        super().__init__()
        self.C, self.d = C, d
        self.qkv = DiagQKVd(C, d)
        self.scale = d**-0.5
        self.logit_scale = nn.Parameter(torch.zeros(C))

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
        y = self.norm1(x)

        Q, K, V = self.qkv(y)
        scores = torch.einsum("btcd,bucd->bctu", Q, K) * self.scale

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                am = torch.zeros_like(attn_mask, dtype=scores.dtype)
                am = am.masked_fill(attn_mask, float("-inf"))
            else:
                am = attn_mask.to(dtype=scores.dtype)
            scores = scores + am.view(1, 1, T, T)

        if key_padding_mask is not None:
            kpm = key_padding_mask.view(B, 1, 1, T)
            scores = scores.masked_fill(kpm, float("-inf"))

        w = torch.softmax(scores, dim=-1)
        self.attn_weights = w.detach()

        out = torch.einsum("bctu,bucd->btcd", w, V).mean(dim=-1)
        x = x + self.drop(out)

        z = self.norm2(x)
        z = self.ffn(z)
        x = x + self.drop(z)
        return x


class SpaceTimeBlock(nn.Module):
    """
    Factorized space-time attention block.
    First applies spatial attention (across concepts at each time step),
    then applies temporal attention (diagonal, per-channel over time).
    """

    def __init__(
        self,
        C: int,
        spatial_d: int = 1,
        temporal_d: int = 1,
        dropout: float = 0.1,
        T_max: int = 1024,
        spatial_gate: float = 0.1,
        identity_bias: float = 1.0,
    ):
        super().__init__()
        self.spatial_attn = PerTimeSpatialBlock(C, d=spatial_d, dropout=dropout, spatial_gate=spatial_gate, identity_bias=identity_bias)
        self.temporal_attn = PerChannelTemporalBlock(
            C, d=temporal_d, dropout=dropout, T_max=T_max
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B, T, C]
        attn_mask: Optional [T, T] or [C, C] mask (for temporal or spatial attention)
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


class CBMTransformerST(nn.Module):
    """
    Space-Time CBM Transformer with factorized attention.
    
    Architecture:
    1. Positional encoding over time
    2. Stack of Space-Time blocks:
       - Spatial attention: concepts interact at each time step (with identity-preserving mechanisms)
       - Temporal attention: diagonal attention over time for each concept
    3. Concept predictor and classifier
    
    The key difference from CBMTransformer is that we factorize attention
    into spatial (concept-to-concept) and temporal (time-to-time) components,
    while maintaining diagonal attention in the temporal dimension.
    
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
        temporal_d: int = 1,
        dimension: int = 1,
        spatial_gate: float = 0.01,
        identity_bias: float = 5.0,
    ):
        super().__init__()
        self.lse_tau = lse_tau
        self.transformer_layers = transformer_layers
        self.diagonal_attention = True

        if dimension != 1:
            temporal_d = dimension

        self.posenc = PositionalEncoding(
            d_model=num_concepts, dropout=dropout, max_len=2000
        )

        self.layers = nn.ModuleList(
            [
                SpaceTimeBlock(
                    C=num_concepts,
                    spatial_d=spatial_d,
                    temporal_d=temporal_d,
                    dropout=dropout,
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
            x = x.unsqueeze(0)
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

