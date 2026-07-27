from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class ModelConfig:
    hidden_dim: int = 32
    depth: int = 2
    num_heads: int = 4
    ff_mult: int = 4
    lora_rank: int = 4
    lora_alpha: float = 8.0
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    backbone_lr: float = 2e-4
    adapter_lr: float = 7e-3
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    positive_class_weight: float = 5.0
    weight_decay: float = 1e-4
    gradient_history_decay: float = 0.95
    normalize_gradient_direction: bool = False
    backbone_lr_scale: float = 0.3
    gradient_direction_clip_norm: float = 1.0
    smoothing_window: int = 5
    threshold_mode: str = "in_window"
    threshold_window: int = 1000
    threshold_grid_size: int = 101
    threshold_min_history: int = 200
    threshold_refresh_every: int = 100
    threshold_default: float = 0.5
    user_threshold_posrate_bias: bool = False
    user_threshold_posrate_bias_alpha: float = 0.0
    user_threshold_min_samples: int = 3


@dataclass
class GroupingConfig:
    backbone_warmup_steps: int = 300
    individual_warmup_steps: int = 20
    min_observations: int = 20
    group_update_interval: int = 100
    periodic_group_update_enabled: bool = True
    minimum_group_count: int = 4
    min_pair_cosine: float = 0.0
    cfl_coherence_threshold: float = 0.55
    cfl_disagreement_cosine_threshold: float = 0.20
    group_merge_alpha: float = 0.3
    warmup_gradient_history_enabled: bool = True
    warmup_initial_grouping_enabled: bool = True
    warmup_initial_min_samples: int = 5
    warmup_initial_min_samples_fallback: int = 1
    warmup_initial_min_grouping_users_fallback: int = 2
    warmup_gradient_history_count: int = 10
    warmup_min_grouping_users: int = 10
    warmup_fallback_gradient_history_count: int = 1
    warmup_fallback_min_grouping_users: int = 2
    warmup_initial_force_accept: bool = True
    warmup_initial_target_groups: int = 0
    warmup_initial_min_groups: int = 2
    warmup_initial_max_groups: int = 12
    warmup_gradient_history_keep_after_initial: bool = False
    personal_adapter_group_init_enabled: bool = True
    personal_adapter_group_init_min_records: int = 3
    personal_adapter_group_init_weight: float = 0.35
    low_evidence_adapter_mixing_enabled: bool = True
    low_evidence_mixing_max_records: int = 20
    low_evidence_mixing_eta: float = 0.5
    low_evidence_prior_global: float = 0.30
    low_evidence_prior_personal: float = 0.50
    low_evidence_prior_group: float = 0.20
    shadow_group_refresh_alpha: float = 0.10
    shadow_group_refresh_min_updates: int = 1
    shadow_group_refresh_changed_only: bool = True
    group_update_mode: str = "full"
    incremental_split_min_users: int = 4
    incremental_objective_margin: float = 0.0
    incremental_max_split_merge_swaps: int = 1


@dataclass
class RefinementConfig:
    per_user_bias: bool = True
    per_user_bias_lr: float = 0.005
    per_user_bias_l2: float = 1e-3
    per_user_bias_clip: float = 2.0
    per_user_bias_init_mode: str = "group_mean"
    per_user_bias_init_group_weight: float = 0.5
    verified_split: bool = True
    split_loss_margin: float = 0.005
    split_min_users: int = 10
    split_min_events: int = 50
    split_holdout_min_users: int = 5
    split_max_groups: int = 8
    split_pred_posrate_delta_tol: float = 0.10
    split_cohesion_tolerance: float = 0.05
    split_holdout_fallback_to_recent: bool = False
    reassignment_enabled: bool = True
    reassignment_min_observations: int = 20
    reassignment_min_buffer: int = 5
    reassignment_loss_margin: float = 0.01
    reassignment_every: int = 10


@dataclass
class EvaluationConfig:
    first_k: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    require_complete_first_k: bool = True


@dataclass
class ExperimentConfig:
    dataset: str = "synthetic"
    seed: int = 42
    device: str = "cpu"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    refinements: RefinementConfig = field(default_factory=RefinementConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _construct(data: Dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        dataset=str(data.get("dataset", "synthetic")),
        seed=int(data.get("seed", 42)),
        device=str(data.get("device", "cpu")),
        model=ModelConfig(**data.get("model", {})),
        training=TrainingConfig(**data.get("training", {})),
        grouping=GroupingConfig(**data.get("grouping", {})),
        refinements=RefinementConfig(**data.get("refinements", {})),
        evaluation=EvaluationConfig(**data.get("evaluation", {})),
    )


def load_config(path: str | Path | None = None, dataset: str = "synthetic") -> ExperimentConfig:
    if path is None:
        if dataset == "synthetic":
            cfg = ExperimentConfig(dataset="synthetic")
            cfg.model.hidden_dim = 16
            cfg.model.depth = 1
            cfg.model.num_heads = 2
            cfg.model.ff_mult = 2
            cfg.grouping.backbone_warmup_steps = 20
            cfg.grouping.individual_warmup_steps = 2
            cfg.grouping.min_observations = 4
            cfg.grouping.group_update_interval = 20
            cfg.grouping.warmup_gradient_history_count = 2
            cfg.grouping.warmup_min_grouping_users = 4
            cfg.grouping.warmup_initial_max_groups = 4
            cfg.refinements.split_min_users = 2
            cfg.refinements.split_min_events = 4
            cfg.refinements.split_holdout_min_users = 2
            cfg.evaluation.first_k = [5, 10]
            return cfg
        root = Path(__file__).resolve().parents[1]
        path = root / "configs" / f"{dataset}.yaml"
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _construct(data)


def apply_variant(cfg: ExperimentConfig, variant: str) -> ExperimentConfig:
    out = _construct(cfg.to_dict())
    variant = variant.lower()
    if variant in {"full", "ocap"}:
        return out
    if variant in {"clean", "clean_base"}:
        out.refinements.per_user_bias = False
        out.refinements.verified_split = False
        out.refinements.reassignment_enabled = False
        return out
    if variant == "clean_plus_bias":
        out.refinements.verified_split = False
        out.refinements.reassignment_enabled = False
        return out
    if variant == "clean_plus_bias_plus_split":
        out.refinements.reassignment_enabled = False
        return out
    if variant == "minus_bias":
        out.refinements.per_user_bias = False
    elif variant == "minus_split":
        out.refinements.verified_split = False
    elif variant == "without_reassignment":
        out.refinements.reassignment_enabled = False
    return out
