from __future__ import annotations

import hashlib
from typing import Dict, Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class DCOFAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult), nn.GELU(), nn.Linear(dim * ff_mult, dim)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attended, weights = self.attn(x, x, x, need_weights=True)
        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x, weights


class LoRAAdapter(nn.Module):
    def __init__(self, hidden_dim: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, hidden_dim))
        self.B = nn.Parameter(torch.zeros(hidden_dim, rank))
        nn.init.normal_(self.A, mean=0.0, std=0.02)
        self.scale = float(alpha) / max(1, int(rank))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return ((hidden @ self.A.t()) @ self.B.t()) * self.scale


class TinyTFTBackbone(nn.Module):
    """Feature-token TinyTFT used by the historical SG-Share runs."""

    def __init__(self, input_dim: int, cfg: ModelConfig) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.feat_w = nn.Parameter(torch.empty(input_dim, dim))
        self.feat_b = nn.Parameter(torch.zeros(input_dim, dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.empty(1, input_dim + 1, dim))
        nn.init.normal_(self.feat_w, mean=0.0, std=0.02)
        nn.init.normal_(self.pos, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList([
            DCOFAttentionBlock(dim, cfg.num_heads, cfg.ff_mult, cfg.dropout)
            for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.route_w = nn.Parameter(torch.zeros(dim, dim))
        self.route_enabled = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        tokens = x.unsqueeze(-1) * self.feat_w.unsqueeze(0) + self.feat_b.unsqueeze(0)
        h = torch.cat([self.cls.expand(x.shape[0], -1, -1), tokens], dim=1) + self.pos
        attention = []
        for block in self.blocks:
            h, weights = block(h)
            attention.append(weights)
        z = self.norm(h[:, 0, :])
        if self.route_enabled:
            z = z + z @ self.route_w.t()
        return z, attention


class SGModel(nn.Module):
    def __init__(self, input_dim: int, cfg: ModelConfig, device: str = "cpu") -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = TinyTFTBackbone(input_dim, cfg)
        self.head = nn.Linear(cfg.hidden_dim, 2)
        self.adapters = nn.ModuleDict()
        self._external_to_internal: Dict[str, str] = {}
        self.device_name = device
        self.to(torch.device(device))

    @staticmethod
    def _internal_key(external_key: str) -> str:
        digest = hashlib.sha1(external_key.encode("utf-8")).hexdigest()[:20]
        return f"adapter_{digest}"

    def ensure_adapter(self, key: str) -> LoRAAdapter:
        if key not in self._external_to_internal:
            internal = self._internal_key(key)
            self._external_to_internal[key] = internal
            self.adapters[internal] = LoRAAdapter(
                self.cfg.hidden_dim, self.cfg.lora_rank, self.cfg.lora_alpha
            ).to(torch.device(self.device_name))
        return self.adapters[self._external_to_internal[key]]

    def adapter(self, key: str) -> LoRAAdapter:
        return self.ensure_adapter(key)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)[0]

    def base_logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden)

    def adapter_logits_from_hidden(self, hidden: torch.Tensor, adapter_key: str) -> torch.Tensor:
        delta = self.ensure_adapter(adapter_key)(hidden)
        return F.linear(delta, self.head.weight, bias=None)

    def adapter_logits_from_hidden_mean(
        self, hidden: torch.Tensor, adapter_keys: Iterable[str],
    ) -> torch.Tensor:
        adapters = [self.ensure_adapter(key) for key in adapter_keys]
        if not adapters:
            raise ValueError("adapter_keys must not be empty")
        mean_a = torch.stack([adapter.A for adapter in adapters]).mean(dim=0)
        mean_b = torch.stack([adapter.B for adapter in adapters]).mean(dim=0)
        delta = ((hidden @ mean_a.t()) @ mean_b.t()) * adapters[0].scale
        return F.linear(delta, self.head.weight, bias=None)

    def logits_from_hidden(self, hidden: torch.Tensor, adapter_key: str) -> torch.Tensor:
        return self.base_logits_from_hidden(hidden) + self.adapter_logits_from_hidden(hidden, adapter_key)

    def logits_from_hidden_mean(
        self, hidden: torch.Tensor, adapter_keys: Iterable[str],
    ) -> torch.Tensor:
        return self.base_logits_from_hidden(hidden) + self.adapter_logits_from_hidden_mean(
            hidden, adapter_keys,
        )

    def detached_backbone_logits_from_hidden(
        self, hidden: torch.Tensor, adapter_key: str,
    ) -> torch.Tensor:
        """Return logits whose gradient is restricted to one adapter."""
        detached_hidden = hidden.detach()
        base = self.base_logits_from_hidden(detached_hidden).detach()
        delta = self.ensure_adapter(adapter_key)(detached_hidden)
        adapter_logits = F.linear(delta, self.head.weight.detach(), bias=None)
        return base + adapter_logits

    def forward(self, x: torch.Tensor, adapter_key: str) -> torch.Tensor:
        return self.logits_from_hidden(self.hidden(x), adapter_key)

    def forward_mean(self, x: torch.Tensor, adapter_keys: Iterable[str]) -> torch.Tensor:
        return self.logits_from_hidden_mean(self.hidden(x), adapter_keys)

    def clone_adapter_state(self, key: str) -> Dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in self.ensure_adapter(key).state_dict().items()}

    def set_adapter_state(self, key: str, state: Mapping[str, torch.Tensor]) -> None:
        self.ensure_adapter(key).load_state_dict(dict(state), strict=True)

    def set_adapter_mean(self, destination: str, source_keys: Iterable[str]) -> None:
        sources = list(source_keys)
        if not sources:
            self.ensure_adapter(destination)
            return
        states = [self.clone_adapter_state(key) for key in sources]
        mean_state = {
            name: torch.stack([state[name] for state in states]).mean(dim=0)
            for name in states[0]
        }
        self.set_adapter_state(destination, mean_state)

    def adapter_mean_state(self, source_keys: Iterable[str]) -> Dict[str, torch.Tensor]:
        sources = list(source_keys)
        if not sources:
            raise ValueError("source_keys must not be empty")
        states = [self.clone_adapter_state(key) for key in sources]
        return {
            name: torch.stack([state[name] for state in states]).mean(dim=0)
            for name in states[0]
        }

    def blend_adapter_with_mean(
        self, destination: str, source_keys: Iterable[str], alpha: float,
    ) -> None:
        dst = self.clone_adapter_state(destination)
        mean_state = self.adapter_mean_state(source_keys)
        weight = float(alpha)
        self.set_adapter_state(destination, {
            name: (1.0 - weight) * dst[name] + weight * mean_state[name]
            for name in dst
        })

    def blend_adapter(self, destination: str, source: str, alpha: float) -> None:
        dst = self.clone_adapter_state(destination)
        src = self.clone_adapter_state(source)
        self.set_adapter_state(destination, {
            name: (1.0 - alpha) * dst[name] + alpha * src[name] for name in dst
        })

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        return list(self.backbone.parameters()) + list(self.head.parameters())


def weighted_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    positive_class_weight: float = 5.0,
) -> torch.Tensor:
    target_float = target.float().reshape(-1)
    positive_logit = logits[:, 1] - logits[:, 0]
    bce = F.binary_cross_entropy_with_logits(positive_logit, target_float, reduction="none")
    probability = torch.sigmoid(positive_logit)
    p_true = target_float * probability + (1.0 - target_float) * (1.0 - probability)
    alpha_t = alpha * target_float + (1.0 - alpha) * (1.0 - target_float)
    class_weight = positive_class_weight * target_float + (1.0 - target_float)
    return (alpha_t * class_weight * (1.0 - p_true).pow(gamma) * bce).mean()


def cross_entropy_per_sample(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target.long(), reduction="none")
