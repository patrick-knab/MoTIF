"""Minimal, dependency-free LoRA (Low-Rank Adaptation) for the vision tower.

LoRA re-parameterises a frozen weight matrix ``W`` as ``W + (alpha / r) * B @ A`` with
``A`` (r x in) and ``B`` (out x r); only ``A`` and ``B`` are trained. This module applies
that re-parameterisation with ``torch.nn.utils.parametrize`` so it works for both ordinary
``nn.Linear`` layers and the fused ``in_proj_weight`` of the Perception Encoder attention
blocks without changing their forward code.

Typical use::

    config = LoRAConfig(rank=8, alpha=16, targets=("attn",), last_blocks=0)
    names = apply_lora(model.visual, config)          # adds trainable lora_A / lora_B
    ...train...
    torch.save({"lora_config": config.to_dict(), "adapter": lora_state_dict(model)}, "adapter.pt")

    # later: rebuild the same adapter and merge it into the frozen weights
    load_lora_adapter(model, "adapter.pt")
    merge_lora(model)                                 # plain module again, pickle-safe
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

# Which 2-D weight tensors of a Perception Encoder / CLIP residual block each target covers.
# Names are relative to one ``ResidualAttentionBlock``.
TARGET_WEIGHTS = {
    "attn": ("attn.in_proj_weight", "attn.out_proj.weight"),
    "mlp": ("mlp.c_fc.weight", "mlp.c_proj.weight"),
}


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    targets: Tuple[str, ...] = ("attn",)
    last_blocks: int = 0  # 0 = adapt every residual block

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        unknown = [t for t in self.targets if t not in TARGET_WEIGHTS]
        if unknown:
            raise ValueError(f"Unknown LoRA targets {unknown}; choose from {sorted(TARGET_WEIGHTS)}")
        if self.last_blocks < 0:
            raise ValueError("last_blocks cannot be negative")
        self.targets = tuple(self.targets)

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["targets"] = list(self.targets)  # JSON/torch.save friendly
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "LoRAConfig":
        return cls(
            rank=int(data["rank"]),
            alpha=float(data["alpha"]),
            targets=tuple(data["targets"]),
            last_blocks=int(data.get("last_blocks", 0)),
        )


class LoRAParametrization(nn.Module):
    """``weight -> weight + scale * B @ A`` with ``B`` zero-initialised (identity at start)."""

    def __init__(self, out_features: int, in_features: int, rank: int, alpha: float):
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = float(alpha) / float(rank)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        delta = (self.lora_B @ self.lora_A) * self.scale
        return weight + delta.to(dtype=weight.dtype)


def _split_param_name(module: nn.Module, dotted: str) -> Tuple[nn.Module, str]:
    parts = dotted.split(".")
    owner = module
    for part in parts[:-1]:
        owner = getattr(owner, part)
    return owner, parts[-1]


def _target_blocks(visual: nn.Module, last_blocks: int) -> Iterable[Tuple[int, nn.Module]]:
    blocks = list(visual.transformer.resblocks)
    if last_blocks > 0:
        blocks = blocks[-last_blocks:]
    offset = len(visual.transformer.resblocks) - len(blocks)
    for index, block in enumerate(blocks, start=offset):
        yield index, block


def apply_lora(visual: nn.Module, config: LoRAConfig) -> List[str]:
    """Attach LoRA parametrizations to ``visual.transformer.resblocks``; returns adapted names."""
    adapted: List[str] = []
    for index, block in _target_blocks(visual, config.last_blocks):
        for target in config.targets:
            for dotted in TARGET_WEIGHTS[target]:
                try:
                    owner, name = _split_param_name(block, dotted)
                    weight = getattr(owner, name)
                except AttributeError:
                    continue  # e.g. nn.MultiheadAttention blocks without these attributes
                if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
                    continue
                if parametrize.is_parametrized(owner, name):
                    continue
                weight.requires_grad_(False)
                lora = LoRAParametrization(weight.shape[0], weight.shape[1], config.rank, config.alpha)
                lora.to(device=weight.device)
                parametrize.register_parametrization(owner, name, lora, unsafe=True)
                adapted.append(f"transformer.resblocks.{index}.{dotted}")
    if not adapted:
        raise RuntimeError("apply_lora adapted no weights; check --lora-targets and the backbone")
    return adapted


def lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for n, p in model.named_parameters() if n.endswith(("lora_A", "lora_B"))]


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Adapter-only weights (``lora_A`` / ``lora_B`` tensors) keyed by their parameter names."""
    return {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if n.endswith(("lora_A", "lora_B"))
    }


def load_lora_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    missing = [k for k in state if k not in params]
    if missing:
        raise KeyError(f"Adapter contains weights the model does not have: {missing[:5]} ...")
    with torch.no_grad():
        for key, value in state.items():
            params[key].copy_(value.to(params[key].device, params[key].dtype))


def merge_lora(model: nn.Module) -> int:
    """Fold every LoRA delta into its base weight and remove the parametrizations."""
    merged = 0
    for module in list(model.modules()):
        if not parametrize.is_parametrized(module):
            continue
        for name in list(module.parametrizations.keys()):
            parametrize.remove_parametrizations(module, name, leave_parametrized=True)
            merged += 1
    return merged


def save_lora_adapter(model: nn.Module, config: LoRAConfig, path, **extra) -> None:
    payload = {"lora_config": config.to_dict(), "adapter": lora_state_dict(model)}
    payload.update(extra)
    torch.save(payload, path)


def load_lora_adapter(model: nn.Module, path, visual_attr: str = "visual") -> LoRAConfig:
    """Re-create the adapter described in ``path`` on ``model.<visual_attr>`` and load its weights."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = LoRAConfig.from_dict(payload["lora_config"])
    visual = getattr(model, visual_attr)
    apply_lora(visual, config)
    state = {
        (k if k.startswith(visual_attr + ".") else f"{visual_attr}.{k}"): v
        for k, v in payload["adapter"].items()
    }
    load_lora_state_dict(model, state)
    return config


__all__: Sequence[str] = (
    "LoRAConfig",
    "LoRAParametrization",
    "TARGET_WEIGHTS",
    "apply_lora",
    "load_lora_adapter",
    "load_lora_state_dict",
    "lora_parameters",
    "lora_state_dict",
    "merge_lora",
    "save_lora_adapter",
)
