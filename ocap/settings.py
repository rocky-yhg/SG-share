from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

from .config import ExperimentConfig, apply_variant


@dataclass(frozen=True)
class MainConfiguration:
    dataset: str
    name: str
    gradient_history_alpha: float
    grouping_min_observations: int
    minimum_group_count: int
    normalize_gradient_direction: bool = False
    strict_initial_grouping: bool = False
    verified_split: bool = False
    reassignment_min_observations: int = 20
    reassignment_loss_margin: float = 0.01

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


MAIN_CONFIGURATIONS: Dict[tuple[str, str], MainConfiguration] = {
    ("ces", "main"): MainConfiguration(
        "ces", "main", 0.20, 20, 5,
    ),
    ("globem", "main"): MainConfiguration(
        "globem", "main", 0.10, 2, 5,
    ),
}


def get_main_configuration(dataset: str, name: str) -> MainConfiguration:
    key = (dataset.strip().lower(), name.strip().lower())
    if key not in MAIN_CONFIGURATIONS:
        available = ", ".join(f"{d}/{n}" for d, n in sorted(MAIN_CONFIGURATIONS))
        raise ValueError(f"unknown main_configuration {dataset}/{name}; available: {available}")
    return MAIN_CONFIGURATIONS[key]


def apply_main_configuration(
    config: ExperimentConfig, name: str,
) -> tuple[ExperimentConfig, MainConfiguration]:
    main_configuration = get_main_configuration(config.dataset, name)
    resolved = apply_variant(config, "ocap")
    resolved.training.gradient_history_decay = 1.0 - main_configuration.gradient_history_alpha
    resolved.training.normalize_gradient_direction = (
        main_configuration.normalize_gradient_direction
    )
    resolved.grouping.min_observations = main_configuration.grouping_min_observations
    resolved.grouping.minimum_group_count = main_configuration.minimum_group_count
    resolved.refinements.verified_split = main_configuration.verified_split
    resolved.refinements.reassignment_min_observations = (
        main_configuration.reassignment_min_observations
    )
    resolved.refinements.reassignment_loss_margin = main_configuration.reassignment_loss_margin
    if main_configuration.strict_initial_grouping:
        resolved.grouping.warmup_initial_min_samples = main_configuration.grouping_min_observations
        resolved.grouping.warmup_initial_min_samples_fallback = (
            main_configuration.grouping_min_observations
        )
    return resolved, main_configuration
