from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

from .config import ExperimentConfig, apply_variant


@dataclass(frozen=True)
class NamedPreset:
    dataset: str
    name: str
    route_grad_ema_alpha: float
    eligibility_lambda: int
    k_min: int
    normalize_route_gradient: bool = False
    strict_initial_eligibility: bool = False
    verified_split: bool = False
    mature_min_observations: int = 20
    mature_loss_margin: float = 0.01

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


PRESETS: Dict[tuple[str, str], NamedPreset] = {
    ("ces", "classification_tuned"): NamedPreset(
        "ces", "classification_tuned", 0.20, 20, 5,
    ),
    ("ces", "cold_safe"): NamedPreset(
        "ces", "cold_safe", 0.05, 20, 3,
    ),
    ("globem", "classification_tuned"): NamedPreset(
        "globem", "classification_tuned", 0.10, 2, 5,
    ),
    ("globem", "cold_safe"): NamedPreset(
        "globem", "cold_safe", 0.12, 2, 4,
    ),
}


def get_preset(dataset: str, name: str) -> NamedPreset:
    key = (dataset.strip().lower(), name.strip().lower())
    if key not in PRESETS:
        available = ", ".join(f"{d}/{n}" for d, n in sorted(PRESETS))
        raise ValueError(f"unknown preset {dataset}/{name}; available: {available}")
    return PRESETS[key]


def apply_named_preset(
    config: ExperimentConfig, name: str,
) -> tuple[ExperimentConfig, NamedPreset]:
    preset = get_preset(config.dataset, name)
    resolved = apply_variant(config, "full_final")
    resolved.training.route_grad_ema_decay = 1.0 - preset.route_grad_ema_alpha
    resolved.training.route_grad_normalize_before_ema = (
        preset.normalize_route_gradient
    )
    resolved.grouping.min_observations = preset.eligibility_lambda
    resolved.grouping.k_min = preset.k_min
    resolved.refinements.verified_split = preset.verified_split
    resolved.refinements.mature_min_observations = (
        preset.mature_min_observations
    )
    resolved.refinements.mature_loss_margin = preset.mature_loss_margin
    if preset.strict_initial_eligibility:
        resolved.grouping.warmup_initial_min_samples = preset.eligibility_lambda
        resolved.grouping.warmup_initial_min_samples_fallback = (
            preset.eligibility_lambda
        )
    return resolved, preset
