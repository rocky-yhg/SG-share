from __future__ import annotations

import copy
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set

import numpy as np
import torch
from torch.nn import functional as F

from .config import ExperimentConfig
from .data import StreamEvent
from .grouping import agglomerative_groups, binary_split, cosine, match_clusters, normalize, random_groups
from .metrics import binary_metrics, first_k_metrics
from .model import SGModel, weighted_focal_loss


@dataclass
class BufferedObservation:
    features: np.ndarray
    label: int


class SGShareLearner:
    """Prequential reconstruction of p0_bias_best_cflsplit_a_m005_u10_e50."""

    def __init__(
        self,
        input_dim: int,
        config: ExperimentConfig,
        method: str = "full_final",
        grouping_signal: str = "gradient",
        group_adapter_mode: str = "standalone",
    ) -> None:
        self.cfg = copy.deepcopy(config)
        self.method = method
        self.grouping_signal = grouping_signal
        if group_adapter_mode not in {"standalone", "user_mean"}:
            raise ValueError(f"unknown group_adapter_mode: {group_adapter_mode}")
        self.group_adapter_mode = group_adapter_mode
        self.device = torch.device(self.cfg.device)
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        self.model = SGModel(input_dim, self.cfg.model, self.cfg.device)
        self.backbone_optimizer = torch.optim.AdamW(
            self.model.backbone_parameters(), lr=self.cfg.training.backbone_lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        self.adapter_optimizers: Dict[str, torch.optim.Adam] = {}
        self.seen: MutableMapping[str, int] = defaultdict(int)
        self.positive_seen: MutableMapping[str, int] = defaultdict(int)
        self.user_bias: Dict[str, float] = {}
        self.route_signature: Dict[str, np.ndarray] = {}
        self.signature_count: MutableMapping[str, int] = defaultdict(int)
        self.window_route_sum: Dict[str, np.ndarray] = {}
        self.window_route_count: MutableMapping[str, int] = defaultdict(int)
        self.feature_signature: Dict[str, np.ndarray] = {}
        self.recent: Dict[str, Deque[BufferedObservation]] = defaultdict(lambda: deque(maxlen=64))
        self.window_buffer: Dict[str, List[BufferedObservation]] = defaultdict(list)
        self.holdout_buffer: Dict[str, List[BufferedObservation]] = defaultdict(list)
        self.assignment: Dict[str, str] = {}
        self.groups: Dict[str, Set[str]] = {}
        self.next_group_index = 0
        self.regroup_count = 0
        self.events: List[Dict[str, object]] = []
        self.group_trace: List[Dict[str, object]] = []
        self._last_split_diagnostics: Dict[str, int] = {}
        self._last_mature_diagnostics: Dict[str, int] = {}
        self.route_group_buffer: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.smooth_probabilities: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=max(1, self.cfg.training.smoothing_window))
        )
        self.threshold_history: Deque[tuple[float, int]] = deque(
            maxlen=max(1, self.cfg.training.threshold_window)
        )
        self.cached_threshold = float(self.cfg.training.threshold_default)
        self.threshold_updates = 0
        self.mix_weights: Dict[str, Dict[str, float]] = {}
        self.prototype_initialized: Set[str] = set()
        self._ensure_adapter("global")

    def _ensure_adapter(self, key: str) -> None:
        adapter = self.model.ensure_adapter(key)
        if key not in self.adapter_optimizers:
            self.adapter_optimizers[key] = torch.optim.Adam(
                adapter.parameters(), lr=self.cfg.training.adapter_lr,
            )

    def _reset_adapter_optimizer(self, key: str) -> None:
        self._ensure_adapter(key)
        self.adapter_optimizers[key] = torch.optim.Adam(
            self.model.adapter(key).parameters(), lr=self.cfg.training.adapter_lr,
        )

    def _temp_key(self, uid: str) -> str:
        key = f"temp::{uid}"
        self._ensure_adapter(key)
        return key

    def _active_adapter(self, uid: str) -> str:
        if self.method == "global_shared":
            return "global"
        if self.method == "per_user_adapter":
            return self._temp_key(uid)
        if uid in self.assignment:
            key = f"group::{self.assignment[uid]}"
            self._ensure_adapter(key)
            return key
        return self._temp_key(uid)

    def _adapter_keys_for_user(self, uid: str) -> List[str]:
        if (
            self.group_adapter_mode == "user_mean"
            and self.method not in {"global_shared", "per_user_adapter"}
            and uid in self.assignment
        ):
            gid = self.assignment[uid]
            keys = [self._temp_key(member) for member in sorted(self.groups.get(gid, set()))]
            if keys:
                return keys
        return [self._active_adapter(uid)]

    def _forward_with_adapters(
        self, features: np.ndarray, adapter_keys: Sequence[str],
    ) -> torch.Tensor:
        x = self._tensor(features)
        if len(adapter_keys) == 1:
            return self.model(x, adapter_keys[0])
        return self.model.forward_mean(x, adapter_keys)

    def _tensor(self, features: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(features, dtype=torch.float32, device=self.device).reshape(1, -1)

    def _task_loss(self, logits: torch.Tensor, label: int) -> torch.Tensor:
        target = torch.tensor([label], dtype=torch.long, device=self.device)
        train = self.cfg.training
        return weighted_focal_loss(
            logits, target, train.focal_alpha, train.focal_gamma, train.positive_class_weight
        )

    def _group_mean_user_bias(self, uid: str) -> float:
        gid = self.assignment.get(uid)
        if gid is None:
            return 0.0
        values = [
            float(self.user_bias[member])
            for member in self.groups.get(gid, set())
            if member != uid and member in self.user_bias
        ]
        return float(np.mean(values)) if values else 0.0

    def _ensure_user_bias(self, uid: str) -> float:
        if uid in self.user_bias:
            return float(self.user_bias[uid])
        ref = self.cfg.refinements
        value = 0.0
        if ref.per_user_bias_init_mode.strip().lower() in {"group_mean", "global_group_mix"}:
            value = ref.per_user_bias_init_group_weight * self._group_mean_user_bias(uid)
        value = float(np.clip(value, -ref.per_user_bias_clip, ref.per_user_bias_clip))
        self.user_bias[uid] = value
        return value

    def _apply_bias(
        self, logits: torch.Tensor, uid: str, *, warmup: bool = False,
    ) -> torch.Tensor:
        if not self.cfg.refinements.per_user_bias or warmup:
            return logits
        offset = torch.zeros_like(logits)
        offset[:, 1] = self._ensure_user_bias(uid)
        return logits + offset

    def _group_prototypes(self) -> Dict[str, np.ndarray]:
        output = {}
        for gid, members in self.groups.items():
            rows = [self.route_signature[u] for u in members if u in self.route_signature]
            if rows:
                output[gid] = normalize(np.mean(np.stack(rows), axis=0))
        return output

    def _nearest_group(self, uid: str) -> str | None:
        if uid in self.assignment:
            return self.assignment[uid]
        if uid not in self.route_signature:
            return None
        prototypes = self._group_prototypes()
        if not prototypes:
            return None
        return max(prototypes, key=lambda gid: cosine(self.route_signature[uid], prototypes[gid]))

    def _expert_logits(self, features: np.ndarray, uid: str) -> Dict[str, torch.Tensor]:
        x = self._tensor(features)
        hidden = self.model.hidden(x)
        base = self.model.base_logits_from_hidden(hidden)
        experts: Dict[str, torch.Tensor] = {"global": base}
        personal_key = self._temp_key(uid)
        experts["personal"] = base + self.model.adapter_logits_from_hidden(hidden, personal_key)
        gid = self._nearest_group(uid)
        if gid is not None:
            group_key = f"group::{gid}"
            if self.group_adapter_mode == "user_mean":
                member_keys = [
                    self._temp_key(member)
                    for member in sorted(self.groups.get(gid, set()))
                ]
                if member_keys:
                    experts[group_key] = base + self.model.adapter_logits_from_hidden_mean(
                        hidden, member_keys,
                    )
            else:
                self._ensure_adapter(group_key)
                experts[group_key] = base + self.model.adapter_logits_from_hidden(
                    hidden, group_key,
                )
        return experts

    def _mix_logits(self, uid: str, experts: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, float]]:
        grouping = self.cfg.grouping
        priors = {
            "global": grouping.cold_start_prior_global,
            "personal": grouping.cold_start_prior_personal,
        }
        for key in experts:
            if key.startswith("group::"):
                priors[key] = grouping.cold_start_prior_group
        previous = self.mix_weights.get(uid, {})
        weights = {key: float(previous.get(key, priors.get(key, 0.0))) for key in experts}
        total = sum(max(0.0, value) for value in weights.values())
        if total <= 0.0:
            weights = {key: 1.0 / len(experts) for key in experts}
        else:
            weights = {key: max(0.0, value) / total for key, value in weights.items()}
        self.mix_weights[uid] = weights
        mixed = sum(float(weights[key]) * value for key, value in experts.items())
        return mixed, weights

    def _prediction_logits(
        self, event: StreamEvent,
    ) -> tuple[torch.Tensor, str, Dict[str, torch.Tensor], Dict[str, float]]:
        uid = event.user_id
        active = self._active_adapter(uid)
        grouping = self.cfg.grouping
        can_mix = (
            self.method not in {"global_shared", "per_user_adapter"}
            and grouping.cold_start_safe_mixing
            and event.index >= grouping.global_warmup_steps
            and self.seen[uid] < grouping.cold_start_mixing_until
        )
        if can_mix:
            experts = self._expert_logits(event.features, uid)
            mixed, weights = self._mix_logits(uid, experts)
            return self._apply_bias(mixed, uid, warmup=False), "safe_mix", experts, weights
        adapter_keys = self._adapter_keys_for_user(uid)
        logits = self._forward_with_adapters(event.features, adapter_keys)
        if self.group_adapter_mode == "user_mean" and uid in self.assignment:
            active = f"user_mean::{self.assignment[uid]}"
        return self._apply_bias(
            logits, uid, warmup=event.index < grouping.global_warmup_steps,
        ), active, {}, {}

    @staticmethod
    def _positive_probability(logits: torch.Tensor) -> float:
        return float(torch.sigmoid(logits[:, 1] - logits[:, 0])[0].item())

    @staticmethod
    def _f1_at_threshold(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> float:
        predictions = probabilities >= threshold
        tp = int(np.sum(predictions & (labels == 1)))
        fp = int(np.sum(predictions & (labels == 0)))
        fn = int(np.sum((~predictions) & (labels == 1)))
        return 2.0 * tp / (2.0 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

    def _current_threshold(self) -> float:
        train = self.cfg.training
        if train.threshold_mode != "in_window" or len(self.threshold_history) < train.threshold_min_history:
            return float(train.threshold_default)
        if self.threshold_updates % max(1, train.threshold_refresh_every) == 0:
            probabilities = np.asarray([row[0] for row in self.threshold_history], dtype=np.float64)
            labels = np.asarray([row[1] for row in self.threshold_history], dtype=np.int64)
            best_score = -1.0
            best_threshold = float(train.threshold_default)
            for threshold in np.linspace(0.0, 1.0, max(2, train.threshold_grid_size)):
                score = self._f1_at_threshold(probabilities, labels, float(threshold))
                if score > best_score:
                    best_score = score
                    best_threshold = float(threshold)
            self.cached_threshold = best_threshold
        return self.cached_threshold

    def _biased_threshold(self, uid: str, base: float) -> float:
        train = self.cfg.training
        if not train.user_threshold_posrate_bias or self.seen[uid] < train.user_threshold_min_samples:
            return base
        user_rate = self.positive_seen[uid] / max(1, self.seen[uid])
        gid = self.assignment.get(uid)
        members = self.groups.get(gid, {uid}) if gid is not None else {uid}
        positives = sum(self.positive_seen[u] for u in members)
        observations = sum(self.seen[u] for u in members)
        group_rate = positives / observations if observations else user_rate
        adjusted = base - train.user_threshold_posrate_bias_alpha * (user_rate - group_rate)
        return float(np.clip(adjusted, 0.05, 0.95))

    def _predict(self, event: StreamEvent) -> tuple[int, float, float, str, Dict[str, torch.Tensor], Dict[str, float]]:
        self.model.eval()
        with torch.no_grad():
            logits, source, experts, weights = self._prediction_logits(event)
            raw_probability = self._positive_probability(logits)
        history = self.smooth_probabilities[event.user_id]
        history.append(raw_probability)
        probability = float(np.mean(history))
        threshold = self._biased_threshold(event.user_id, self._current_threshold())
        return int(probability >= threshold), probability, raw_probability, source, experts, weights

    def _update_mix_weights(
        self, uid: str, experts: Mapping[str, torch.Tensor], weights: Mapping[str, float], label: int,
    ) -> None:
        if not experts:
            return
        eta = self.cfg.grouping.cold_start_mixing_eta
        updated = {}
        with torch.no_grad():
            for key, logits in experts.items():
                loss = float(self._task_loss(self._apply_bias(logits, uid), label).item())
                updated[key] = float(weights[key]) * math.exp(-eta * loss)
        total = sum(updated.values())
        if total > 0.0:
            self.mix_weights[uid] = {key: value / total for key, value in updated.items()}

    def _update_signature(self, uid: str, features: np.ndarray) -> np.ndarray | None:
        grad = self.model.backbone.route_w.grad
        route_vector = None
        if grad is not None:
            route_vector = grad.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=True)
            if np.all(np.isfinite(route_vector)) and np.linalg.norm(route_vector) > 1e-12:
                if self.cfg.training.route_grad_normalize_before_ema:
                    route_vector = normalize(route_vector)
                decay = self.cfg.training.route_grad_ema_decay
                old = self.route_signature.get(uid)
                updated = route_vector if old is None else decay * old + (1.0 - decay) * route_vector
                self.route_signature[uid] = updated.copy()
                self.signature_count[uid] += 1
                if uid not in self.window_route_sum:
                    self.window_route_sum[uid] = route_vector.copy()
                else:
                    self.window_route_sum[uid] = (
                        decay * self.window_route_sum[uid]
                        + (1.0 - decay) * route_vector
                    )
                # The historical implementation stores a persistent EMA, not a
                # sample sum. The count is therefore a presence marker.
                self.window_route_count[uid] = 1
        feature = normalize(features)
        old_feature = self.feature_signature.get(uid)
        decay = self.cfg.training.route_grad_ema_decay
        updated_feature = feature if old_feature is None else decay * old_feature + (1.0 - decay) * feature
        self.feature_signature[uid] = normalize(updated_feature)
        return route_vector

    def _train_event(self, event: StreamEvent, raw_probability_before_update: float) -> None:
        uid = event.user_id
        adapter_keys = self._adapter_keys_for_user(uid)
        for adapter_key in adapter_keys:
            self._ensure_adapter(adapter_key)
        self.model.train()
        self.backbone_optimizer.zero_grad(set_to_none=True)
        for adapter_key in adapter_keys:
            self.adapter_optimizers[adapter_key].zero_grad(set_to_none=True)
        is_warmup = event.index < self.cfg.grouping.global_warmup_steps
        logits = self._apply_bias(
            self._forward_with_adapters(event.features, adapter_keys), uid, warmup=is_warmup,
        )
        loss = self._task_loss(logits, event.label)
        loss.backward()
        route_vector = self._update_signature(uid, event.features)
        if is_warmup:
            self.backbone_optimizer.step()
        for adapter_key in adapter_keys:
            self.adapter_optimizers[adapter_key].step()
        if not is_warmup and route_vector is not None and uid in self.assignment:
            self.route_group_buffer[self.assignment[uid]].append(route_vector)
        self.backbone_optimizer.zero_grad(set_to_none=True)
        for adapter_key in adapter_keys:
            self.adapter_optimizers[adapter_key].zero_grad(set_to_none=True)
        if self.cfg.refinements.per_user_bias and not is_warmup:
            ref = self.cfg.refinements
            current = self._ensure_user_bias(uid)
            gradient = raw_probability_before_update - float(event.label)
            updated = (1.0 - ref.per_user_bias_lr * ref.per_user_bias_l2) * current
            updated -= ref.per_user_bias_lr * gradient
            self.user_bias[uid] = float(np.clip(updated, -ref.per_user_bias_clip, ref.per_user_bias_clip))

    def _apply_group_route_update(self) -> int:
        group_means = [np.mean(np.stack(rows), axis=0) for rows in self.route_group_buffer.values() if rows]
        self.route_group_buffer.clear()
        if not group_means:
            return 0
        update = np.mean(np.stack(group_means), axis=0)
        norm = float(np.linalg.norm(update))
        clip = self.cfg.training.route_grad_clip_norm
        if norm > clip > 0.0:
            update = update * (clip / norm)
        tensor = torch.as_tensor(
            update.reshape(self.model.backbone.route_w.shape),
            dtype=self.model.backbone.route_w.dtype,
            device=self.device,
        )
        with torch.no_grad():
            self.model.backbone.route_w.add_(
                tensor, alpha=-self.cfg.training.backbone_lr * self.cfg.training.backbone_lr_scale
            )
        return len(group_means)

    def _eligible_users(self, initial: bool = False) -> List[str]:
        grouping = self.cfg.grouping
        minimum = grouping.warmup_initial_min_samples if initial else grouping.min_observations
        users = [
            uid for uid in sorted(self.window_route_sum)
            if self.seen[uid] >= minimum and self.window_route_count[uid] > 0
        ]
        if not initial:
            return users
        if len(users) >= grouping.warmup_min_eligible_users:
            return users
        fallback = [
            uid for uid in sorted(self.window_route_sum)
            if self.seen[uid] >= grouping.warmup_initial_min_samples_fallback
            and self.window_route_count[uid] > 0
        ]
        return (
            fallback
            if len(fallback) >= grouping.warmup_initial_min_eligible_users_fallback
            else []
        )

    def _source_key(self, uid: str) -> str:
        # New dynamic groups are initialized from each member's personal source,
        # even when that user is currently routed through an existing group.
        return self._temp_key(uid)

    @staticmethod
    def _project_regroup_signatures(
        users: Sequence[str], signatures: Mapping[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        ordered = [uid for uid in sorted(users) if uid in signatures]
        if not ordered:
            return {}
        rows = np.stack([
            np.asarray(signatures[uid], dtype=np.float32).reshape(-1)
            for uid in ordered
        ])
        matrix = torch.as_tensor(rows, dtype=torch.float32)
        standardized = (matrix - matrix.mean(dim=0)) / matrix.std(
            dim=0, unbiased=False,
        ).clamp_min(1e-6)
        rank = min(8, max(1, len(ordered) - 1), int(standardized.shape[1]))
        try:
            # X @ V_k from the historical SVD equals U_k @ Sigma_k. Computing
            # the latter from the user-by-user Gram matrix preserves all row
            # inner products in the selected principal subspace and avoids a
            # repeated SVD over the much wider route-parameter dimension.
            eigenvalues, eigenvectors = torch.linalg.eigh(standardized @ standardized.T)
            order = torch.argsort(eigenvalues, descending=True)[:rank]
            scales = torch.sqrt(torch.clamp(eigenvalues[order], min=0.0))
            coordinates = eigenvectors[:, order] * scales.unsqueeze(0)
        except RuntimeError:
            coordinates = standardized[:, :rank]
        values = coordinates.detach().cpu().numpy().astype(np.float64, copy=False)
        return {uid: values[index].copy() for index, uid in enumerate(ordered)}

    def _buffer_for(self, uid: str, holdout: bool) -> Sequence[BufferedObservation]:
        return self.holdout_buffer.get(uid, []) if holdout else list(self.recent[uid])

    def _adapter_loss(
        self, members: Iterable[str], adapter_key: str, holdout: bool,
    ) -> tuple[float, float, int, int]:
        losses: List[float] = []
        positive_predictions = 0
        users_with_data = 0
        self.model.eval()
        with torch.no_grad():
            for uid in sorted(members):
                observations = self._buffer_for(uid, holdout)
                if observations:
                    users_with_data += 1
                for observation in observations:
                    logits = self.model(self._tensor(observation.features), adapter_key)
                    losses.append(float(self._task_loss(logits, observation.label).item()))
                    positive_predictions += int(self._positive_probability(logits) >= 0.5)
        total = len(losses)
        return (
            float(np.mean(losses)) if losses else float("inf"),
            positive_predictions / total if total else float("nan"),
            total,
            users_with_data,
        )

    @staticmethod
    def _cluster_signature_stats(
        members: Iterable[str], signatures: Mapping[str, np.ndarray],
    ) -> tuple[float, float]:
        rows = [
            np.asarray(signatures[uid], dtype=np.float64).reshape(-1)
            for uid in sorted(members) if uid in signatures
        ]
        if not rows:
            return float("nan"), float("nan")
        matrix = np.stack(rows)
        norms = np.linalg.norm(matrix, axis=1)
        finite = np.isfinite(norms) & (norms > 1e-12)
        if not np.any(finite):
            return 0.0, -1.0
        matrix = matrix[finite]
        norms = norms[finite]
        coherence = float(
            np.linalg.norm(np.mean(matrix, axis=0))
            / max(float(np.max(norms)), 1e-12)
        )
        normalized = matrix / norms[:, None]
        if len(normalized) < 2:
            min_cosine = 1.0
        else:
            similarities = normalized @ normalized.T
            min_cosine = float(
                np.min(similarities[np.triu_indices(len(normalized), k=1)])
            )
        return coherence, min_cosine

    def _verified_splits(
        self,
        clusters: Sequence[Set[str]],
        signatures: Mapping[str, np.ndarray],
        source_keys: Mapping[str, str],
        step: int,
    ) -> tuple[List[Set[str]], int]:
        ref = self.cfg.refinements
        diagnostics = {
            "disabled": 0,
            "clusters": len(clusters),
            "skipped_group_limit": 0,
            "skipped_user_support": 0,
            "skipped_geometry": 0,
            "skipped_proposal": 0,
            "skipped_child_users": 0,
            "skipped_child_events": 0,
            "rejected_validation": 0,
            "accepted": 0,
        }
        if not ref.verified_split:
            diagnostics["disabled"] = 1
            self._last_split_diagnostics = diagnostics
            return list(clusters), 0
        output: List[Set[str]] = []
        accepted = 0
        current_groups = len(clusters)
        for index, cluster in enumerate(clusters):
            if current_groups >= ref.split_max_groups:
                diagnostics["skipped_group_limit"] += 1
                output.append(set(cluster))
                continue
            if len(cluster) < 2 * ref.split_min_users:
                diagnostics["skipped_user_support"] += 1
                output.append(set(cluster))
                continue
            parent_cohesion, parent_min_cosine = self._cluster_signature_stats(
                cluster, signatures,
            )
            if (
                not np.isfinite(parent_cohesion)
                or not np.isfinite(parent_min_cosine)
                or parent_cohesion > self.cfg.grouping.cfl_coherence_threshold
                or parent_min_cosine > self.cfg.grouping.cfl_disagreement_cosine_threshold
            ):
                diagnostics["skipped_geometry"] += 1
                output.append(set(cluster))
                continue
            proposal = binary_split(set(cluster), signatures)
            if proposal is None:
                diagnostics["skipped_proposal"] += 1
                output.append(set(cluster))
                continue
            left, right = proposal
            if min(len(left), len(right)) < ref.split_min_users:
                diagnostics["skipped_child_users"] += 1
                output.append(set(cluster))
                continue
            left_events = sum(len(self.holdout_buffer.get(uid, [])) for uid in left)
            right_events = sum(len(self.holdout_buffer.get(uid, [])) for uid in right)
            if min(left_events, right_events) < ref.split_min_events:
                diagnostics["skipped_child_events"] += 1
                output.append(set(cluster))
                continue
            keys = [f"diagnostic::{step}::{index}::{name}" for name in ("parent", "left", "right")]
            self.model.set_adapter_mean(keys[0], [source_keys[u] for u in sorted(cluster)])
            self.model.set_adapter_mean(keys[1], [source_keys[u] for u in sorted(left)])
            self.model.set_adapter_mean(keys[2], [source_keys[u] for u in sorted(right)])
            parent_loss, parent_rate, parent_n, parent_users = self._adapter_loss(cluster, keys[0], True)
            left_loss, left_rate, left_n, _ = self._adapter_loss(left, keys[1], True)
            right_loss, right_rate, right_n, _ = self._adapter_loss(right, keys[2], True)
            child_n = left_n + right_n
            child_loss = (left_loss * left_n + right_loss * right_n) / child_n if child_n else float("inf")
            child_rate = (left_rate * left_n + right_rate * right_n) / child_n if child_n else float("nan")
            left_cohesion, _ = self._cluster_signature_stats(left, signatures)
            right_cohesion, _ = self._cluster_signature_stats(right, signatures)
            child_cohesion = (
                left_cohesion * len(left) + right_cohesion * len(right)
            ) / len(cluster)
            delta = parent_loss - child_loss
            rate_ok = (
                np.isfinite(parent_rate) and np.isfinite(child_rate)
                and child_rate <= parent_rate + ref.split_pred_posrate_delta_tol
            )
            cohesion_ok = (
                np.isfinite(child_cohesion)
                and child_cohesion >= parent_cohesion - ref.split_cohesion_tolerance
            )
            if (
                parent_users >= ref.split_holdout_min_users and parent_n and child_n
                and delta > ref.split_loss_margin and rate_ok and cohesion_ok
            ):
                output.extend([left, right])
                accepted += 1
                current_groups += 1
            else:
                diagnostics["rejected_validation"] += 1
                output.append(set(cluster))
        diagnostics["accepted"] = accepted
        self._last_split_diagnostics = diagnostics
        return output, accepted

    def _recent_loss(self, uid: str, adapter_key: str) -> float:
        observations = list(self.recent[uid])
        if len(observations) < self.cfg.refinements.mature_min_buffer:
            return float("inf")
        losses = []
        self.model.eval()
        with torch.no_grad():
            for observation in observations:
                logits = self.model(self._tensor(observation.features), adapter_key)
                losses.append(float(self._task_loss(logits, observation.label).item()))
        return float(np.mean(losses))

    def _mature_refine_clusters(
        self,
        clusters: Sequence[Set[str]],
        eligible: Sequence[str],
        source_keys: Mapping[str, str],
        step: int,
    ) -> tuple[List[Set[str]], int]:
        ref = self.cfg.refinements
        result = [set(cluster) for cluster in clusters]
        boundary_index = self.regroup_count
        diagnostics = {
            "disabled": int(not ref.mature_refine),
            "skipped_cadence": 0,
            "skipped_group_count": 0,
            "eligible_users": len(eligible),
            "skipped_observations": 0,
            "skipped_buffer": 0,
            "skipped_unassigned": 0,
            "skipped_singleton": 0,
            "evaluated_users": 0,
            "moves": 0,
        }
        if not ref.mature_refine:
            self._last_mature_diagnostics = diagnostics
            return result, 0
        if boundary_index % max(1, ref.mature_every) != 0:
            diagnostics["skipped_cadence"] = 1
            self._last_mature_diagnostics = diagnostics
            return result, 0
        if len(result) < 2:
            diagnostics["skipped_group_count"] = 1
            self._last_mature_diagnostics = diagnostics
            return result, 0
        temporary_keys: Dict[int, str] = {}
        for index, cluster in enumerate(result):
            key = f"diagnostic::mature::{step}::{index}"
            self.model.set_adapter_mean(key, [source_keys[uid] for uid in sorted(cluster)])
            temporary_keys[index] = key
        cluster_by_user = {
            uid: index for index, cluster in enumerate(result) for uid in cluster
        }
        sizes = {index: len(cluster) for index, cluster in enumerate(result)}
        moves = 0
        for uid in sorted(eligible):
            if self.seen[uid] < ref.mature_min_observations:
                diagnostics["skipped_observations"] += 1
                continue
            if len(self.recent[uid]) < ref.mature_min_buffer:
                diagnostics["skipped_buffer"] += 1
                continue
            if uid not in cluster_by_user:
                diagnostics["skipped_unassigned"] += 1
                continue
            current = cluster_by_user[uid]
            if sizes[current] <= 1:
                diagnostics["skipped_singleton"] += 1
                continue
            diagnostics["evaluated_users"] += 1
            losses = {
                index: self._recent_loss(uid, key)
                for index, key in temporary_keys.items()
            }
            best = min(losses, key=losses.get)
            if (
                best != current
                and np.isfinite(losses[current])
                and np.isfinite(losses[best])
                and losses[current] - losses[best] > ref.mature_loss_margin
            ):
                result[current].discard(uid)
                result[best].add(uid)
                sizes[current] -= 1
                sizes[best] += 1
                cluster_by_user[uid] = best
                moves += 1
        diagnostics["moves"] = moves
        self._last_mature_diagnostics = diagnostics
        return [cluster for cluster in result if cluster], moves

    def _initialize_group_adapter(
        self, gid: str, members: Set[str], source_keys: Mapping[str, str], created: bool,
    ) -> None:
        key = f"group::{gid}"
        sources = [source_keys[uid] for uid in sorted(members)]
        if self.group_adapter_mode == "user_mean":
            self.model.set_adapter_mean(key, sources)
            self._reset_adapter_optimizer(key)
            return
        if created or key not in self.model._external_to_internal:
            self.model.set_adapter_mean(key, sources)
            self._reset_adapter_optimizer(key)
        else:
            self._ensure_adapter(key)

    def _initialize_user_from_group(self, uid: str, gid: str) -> None:
        if uid in self.prototype_initialized:
            return
        grouping = self.cfg.grouping
        if not grouping.prototype_init_enabled or self.seen[uid] < grouping.prototype_init_min_samples:
            return
        self.model.blend_adapter(
            self._temp_key(uid), f"group::{gid}", grouping.prototype_init_alpha
        )
        self._reset_adapter_optimizer(self._temp_key(uid))
        self.prototype_initialized.add(uid)

    def _regroup(self, step: int, initial: bool = False) -> None:
        eligible = self._eligible_users(initial)
        if not eligible:
            return
        sources = {uid: self._source_key(uid) for uid in eligible}
        raw_signatures = (
            self.feature_signature
            if self.grouping_signal == "feature"
            else self.window_route_sum
        )
        eligible = [uid for uid in eligible if uid in raw_signatures]
        if not eligible:
            return
        projected_signatures = self._project_regroup_signatures(eligible, raw_signatures)
        eligible = [uid for uid in eligible if uid in projected_signatures]
        if not eligible:
            return
        if self.method == "one_group_adapter":
            clusters = [set(eligible)]
            stop_reason = "forced_one_group"
            merges = max(0, len(eligible) - 1)
            last_similarity = float("nan")
        elif self.grouping_signal == "random":
            clusters = random_groups(eligible, self.cfg.grouping.k_min, self.cfg.seed + self.regroup_count)
            stop_reason = "random_k"
            merges = max(0, len(eligible) - len(clusters))
            last_similarity = float("nan")
        elif initial:
            target = int(self.cfg.grouping.warmup_initial_target_groups)
            if target <= 0:
                target = int(math.ceil(math.sqrt(float(len(eligible)))))
            target = max(
                self.cfg.grouping.warmup_initial_min_groups,
                min(len(eligible), self.cfg.grouping.warmup_initial_max_groups, target),
            )
            initial_floor = (
                -1.0
                if self.cfg.grouping.warmup_initial_force_accept
                else self.cfg.grouping.min_pair_cosine
            )
            result = agglomerative_groups(
                eligible, projected_signatures, target, initial_floor,
            )
            clusters = result.clusters
            stop_reason = "warmup_forced_target"
            merges = result.merges
            last_similarity = result.last_similarity
        else:
            result = agglomerative_groups(
                eligible, projected_signatures,
                self.cfg.grouping.k_min, self.cfg.grouping.min_pair_cosine,
            )
            clusters = result.clusters
            stop_reason = result.stop_reason
            merges = result.merges
            last_similarity = result.last_similarity
        clusters, split_count = self._verified_splits(
            clusters, raw_signatures, sources, step,
        )
        clusters, mature_moves = self._mature_refine_clusters(
            clusters, eligible, sources, step,
        )
        previous = {gid: set(users) for gid, users in self.groups.items()}
        new_groups, self.next_group_index, created = match_clusters(previous, clusters, self.next_group_index)
        for gid, members in new_groups.items():
            self._initialize_group_adapter(gid, members, sources, gid in created)
        self.groups = new_groups
        self.assignment = {uid: gid for gid, users in self.groups.items() for uid in users}
        for uid, gid in self.assignment.items():
            self._initialize_user_from_group(uid, gid)
        self.regroup_count += 1
        old_assignment = {uid: gid for gid, users in previous.items() for uid in users}
        common = set(old_assignment) & set(self.assignment)
        churn = float(np.mean([old_assignment[u] != self.assignment[u] for u in common])) if common else 0.0
        self.group_trace.append({
            "step": step,
            "initial": initial,
            "eligible_users": len(eligible),
            "n_groups": len(self.groups),
            "n_merges": merges,
            "n_verified_splits": split_count,
            "n_mature_moves": mature_moves,
            "churn": churn,
            "stop_reason": stop_reason,
            "last_similarity": last_similarity,
            **{
                f"split_{key}": value
                for key, value in self._last_split_diagnostics.items()
            },
            **{
                f"reassignment_{key}": value
                for key, value in self._last_mature_diagnostics.items()
            },
        })

    def _boundary(self, step: int, initial: bool) -> None:
        route_groups = self._apply_group_route_update()
        if self.method not in {"global_shared", "per_user_adapter"}:
            self._regroup(step, initial=initial)
            if self.group_trace and self.group_trace[-1]["step"] == step:
                self.group_trace[-1]["route_update_groups"] = route_groups
        self.holdout_buffer = {uid: list(rows) for uid, rows in self.window_buffer.items()}
        self.window_buffer = defaultdict(list)
        if initial and not self.cfg.grouping.warmup_signature_keep_after_initial:
            self.window_route_sum = {}
            self.window_route_count = defaultdict(int)
        else:
            self.window_route_count = defaultdict(
                int, {uid: 1 for uid in self.window_route_sum}
            )

    def process(self, event: StreamEvent) -> Dict[str, object]:
        uid = event.user_id
        self._temp_key(uid)
        prediction, probability, raw_probability, source, experts, weights = self._predict(event)
        is_warmup = event.index < self.cfg.grouping.global_warmup_steps
        row: Dict[str, object] = {
            "event_index": event.index,
            "timestamp": event.timestamp,
            "user_id": uid,
            "user_event_index": self.seen[uid] + 1,
            "label": event.label,
            "prediction": prediction,
            "probability": probability,
            "raw_probability": raw_probability,
            "threshold": self._biased_threshold(uid, self._current_threshold()),
            "adapter_key": source,
            "group_id": self.assignment.get(uid, ""),
            "phase": "warmup" if is_warmup else "post_warmup",
            "stage": "warmup" if is_warmup else ("group" if uid in self.assignment else "temporary"),
            "was_warmup": is_warmup,
        }
        self.events.append(row)
        self._update_mix_weights(uid, experts, weights, event.label)
        self._train_event(event, raw_probability)
        observation = BufferedObservation(event.features.copy(), event.label)
        self.recent[uid].append(observation)
        self.window_buffer[uid].append(observation)
        self.threshold_history.append((probability, event.label))
        self.threshold_updates += 1
        self.seen[uid] += 1
        self.positive_seen[uid] += int(event.label == 1)
        step = event.index + 1
        warmup = self.cfg.grouping.global_warmup_steps
        interval = self.cfg.grouping.regroup_interval
        if step == warmup:
            self._boundary(step, initial=self.cfg.grouping.warmup_initial_grouping_enabled)
        elif step > warmup and (step - warmup) % max(1, interval) == 0:
            self._boundary(step, initial=False)
        return row

    def run(self, stream: Sequence[StreamEvent]) -> Dict[str, object]:
        for event in stream:
            self.process(event)
        y = [int(row["label"]) for row in self.events]
        predictions = [int(row["prediction"]) for row in self.events]
        post_rows = [row for row in self.events if not bool(row["was_warmup"])]
        post_y = [int(row["label"]) for row in post_rows]
        post_predictions = [int(row["prediction"]) for row in post_rows]
        cold = first_k_metrics(
            self.events, self.cfg.evaluation.first_k, self.cfg.evaluation.require_complete_first_k
        )
        return {
            "overall": binary_metrics(y, predictions),
            "post_warmup": binary_metrics(post_y, post_predictions),
            "cold_start": cold,
            "mean_first_k_f1": float(np.mean([row["user_macro_f1"] for row in cold])) if cold else float("nan"),
            "n_groups_final": len(self.groups),
            "mean_groups": float(np.mean([row["n_groups"] for row in self.group_trace])) if self.group_trace else 0.0,
            "n_events": len(self.events),
            "n_users": len(self.seen),
            "target_variant": "p0_bias_best_cflsplit_a_m005_u10_e50",
        }
