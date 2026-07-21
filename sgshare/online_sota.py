from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Dict, List, MutableMapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import ExperimentConfig
from .data import StreamEvent
from .metrics import binary_metrics, first_k_metrics


SUPPORTED_ONLINE_SOTA = {"oli2ds", "obal", "hbp", "odl", "koil", "olifl", "olfl"}


@dataclass(frozen=True)
class OnlineSOTAConfig:
    """Protocol-aligned parameters; paper defaults are retained where public."""

    oli2ds_c: float = 0.01
    oli2ds_theta: float = 8.0
    hbp_hidden_dim: int = 100
    hbp_layers: int = 19
    hbp_learning_rate: float = 0.01
    hbp_beta: float = 0.99
    hbp_min_expert_weight: float = 0.01
    koil_eta: float = 0.01
    koil_c: float = 5.0
    koil_sigma: float = 2.0
    koil_budget_positive: int = 64
    koil_budget_negative: int = 64
    koil_neighbours: int = 8
    obal_learning_rate: float = 0.02
    obal_source_limit: int = 16
    obal_reliability_decay: float = 0.98
    olifl_c: float = 0.1
    olifl_gamma: float = 1.0
    olfl_prototype_decay: float = 0.02


def _sigmoid(value: float) -> float:
    value = float(np.clip(value, -40.0, 40.0))
    return 1.0 / (1.0 + math.exp(-value))


def _safe_unit(vector: np.ndarray) -> np.ndarray:
    vector = np.nan_to_num(np.asarray(vector, dtype=np.float64))
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


class _OnlineModel:
    provenance = "independent_mechanism_reproduction"

    def predict_proba(self, x: np.ndarray, uid: str) -> float:
        raise NotImplementedError

    def learn(self, x: np.ndarray, y: int, uid: str) -> None:
        raise NotImplementedError

    def diagnostics(self) -> Dict[str, float]:
        return {}


class _OLI2DSModel(_OnlineModel):
    """Complete-feature specialization of the authors' public OLI2DS code."""

    provenance = "author_code_core_complete_feature_specialization"

    def __init__(self, input_dim: int, cfg: OnlineSOTAConfig) -> None:
        self.weights = np.zeros(input_dim, dtype=np.float64)
        self.c = float(cfg.oli2ds_c)
        self.theta = float(cfg.oli2ds_theta)
        self.label_counts = np.ones(2, dtype=np.float64)
        self.update_count = 0

    def predict_proba(self, x: np.ndarray, uid: str) -> float:
        return _sigmoid(float(self.weights @ x))

    def learn(self, x: np.ndarray, y: int, uid: str) -> None:
        signed = 1 if int(y) == 1 else -1
        score = float(self.weights @ x)
        loss = max(0.0, 1.0 - signed * score)
        pos, neg = float(self.label_counts[1]), float(self.label_counts[0])
        if signed > 0:
            imbalance_weight = 1.0 / max(pos / neg + 1.0, 1e-12)
        else:
            imbalance_weight = 1.0 / max(neg / pos + 1.0, 1e-12)
        tau = min(self.c, loss / max(float(x @ x), 1e-12))
        self.weights += tau * signed * self.theta * imbalance_weight * x
        self.label_counts[int(y)] += 1.0
        self.update_count += int(tau > 0.0)

    def diagnostics(self) -> Dict[str, float]:
        return {"updates": float(self.update_count), "weight_norm": float(np.linalg.norm(self.weights))}


class _HBPNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        self.hidden = nn.ModuleList()
        self.heads = nn.ModuleList()
        for index in range(depth):
            self.hidden.append(nn.Linear(input_dim if index == 0 else hidden_dim, hidden_dim))
            self.heads.append(nn.Linear(hidden_dim, 2))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        hidden = x
        for layer, head in zip(self.hidden, self.heads):
            hidden = F.relu(layer(hidden))
            outputs.append(head(hidden))
        return outputs


class _HBPModel(_OnlineModel):
    """Hedge Backpropagation following the public IJCAI implementation."""

    provenance = "public_code_equation_reproduction"

    def __init__(self, input_dim: int, cfg: OnlineSOTAConfig, seed: int) -> None:
        torch.manual_seed(int(seed))
        self.network = _HBPNetwork(input_dim, cfg.hbp_hidden_dim, cfg.hbp_layers)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=cfg.hbp_learning_rate)
        self.expert_weights = np.full(cfg.hbp_layers, 1.0 / cfg.hbp_layers, dtype=np.float64)
        self.beta = float(cfg.hbp_beta)
        self.minimum = float(cfg.hbp_min_expert_weight)
        self.updates = 0

    def _tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32).reshape(1, -1)

    def predict_proba(self, x: np.ndarray, uid: str) -> float:
        self.network.eval()
        with torch.no_grad():
            outputs = self.network(self._tensor(x))
            values = np.asarray([F.softmax(logits, dim=1)[0, 1].item() for logits in outputs])
        return float(np.dot(self.expert_weights, values))

    def learn(self, x: np.ndarray, y: int, uid: str) -> None:
        self.network.train()
        tensor = self._tensor(x)
        label = torch.as_tensor([int(y)], dtype=torch.long)
        outputs = self.network(tensor)
        losses = torch.stack([F.cross_entropy(logits, label) for logits in outputs])
        old_weights = self.expert_weights.copy()
        with torch.no_grad():
            loss_values = losses.detach().cpu().numpy()
        self.expert_weights *= np.power(self.beta, loss_values)
        self.expert_weights = np.maximum(self.expert_weights, self.minimum)
        self.expert_weights /= self.expert_weights.sum()
        self.optimizer.zero_grad(set_to_none=True)
        total = sum(float(weight) * loss for weight, loss in zip(old_weights, losses))
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
        self.optimizer.step()
        self.updates += 1

    def diagnostics(self) -> Dict[str, float]:
        entropy = -float(np.sum(self.expert_weights * np.log(self.expert_weights + 1e-12)))
        return {"updates": float(self.updates), "expert_entropy": entropy}


@dataclass
class _SupportVector:
    x: np.ndarray
    alpha: float


class _KOILModel(_OnlineModel):
    """Python port of the authors' fixed-budget FIFO-compensated KOIL core."""

    provenance = "author_code_python_port_fifo_compensation"

    def __init__(self, input_dim: int, cfg: OnlineSOTAConfig) -> None:
        self.eta = float(cfg.koil_eta)
        self.c = float(cfg.koil_c)
        self.sigma = float(cfg.koil_sigma)
        self.budgets = {1: int(cfg.koil_budget_positive), -1: int(cfg.koil_budget_negative)}
        self.k = int(cfg.koil_neighbours)
        self.buffers: Dict[int, List[_SupportVector]] = {1: [], -1: []}
        self.bias = 0.0
        self.replacements = 0
        self.update_count = 0

    def _kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        distance = float(np.sum((x - y) ** 2))
        return math.exp(-distance / max(2.0 * self.sigma * self.sigma, 1e-12))

    def _score(self, x: np.ndarray) -> float:
        return float(self._score_batch(np.asarray(x, dtype=np.float64).reshape(1, -1))[0])

    def _support_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        entries = [item for values in self.buffers.values() for item in values]
        if not entries:
            return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=np.float64)
        return np.stack([item.x for item in entries]), np.asarray([item.alpha for item in entries])

    def _score_batch(self, values: np.ndarray) -> np.ndarray:
        support, alpha = self._support_arrays()
        values = np.asarray(values, dtype=np.float64)
        if alpha.size == 0:
            return np.zeros(values.shape[0], dtype=np.float64)
        squared = np.sum((values[:, None, :] - support[None, :, :]) ** 2, axis=2)
        kernels = np.exp(-squared / max(2.0 * self.sigma * self.sigma, 1e-12))
        return kernels @ alpha

    def predict_proba(self, x: np.ndarray, uid: str) -> float:
        return _sigmoid(self._score(x) - self.bias)

    def _update_bias(self) -> None:
        positive = self.buffers[1]
        negative = self.buffers[-1]
        if not positive or not negative:
            self.bias = 0.0
            return
        all_values = np.stack([item.x for item in positive + negative])
        all_scores = self._score_batch(all_values)
        p_scores = all_scores[: len(positive)]
        n_scores = all_scores[len(positive):]
        candidates = np.unique(np.concatenate([p_scores, n_scores]))
        if candidates.size > 128:
            candidates = np.quantile(candidates, np.linspace(0.0, 1.0, 128))
        best_accuracy = -1.0
        best = self.bias
        for threshold in candidates:
            accuracy = float((np.sum(p_scores >= threshold) + np.sum(n_scores < threshold)) / (len(p_scores) + len(n_scores)))
            if accuracy > best_accuracy:
                best_accuracy, best = accuracy, float(threshold)
        self.bias = best

    def _insert(self, signed: int, x: np.ndarray, alpha: float) -> None:
        if abs(alpha) <= 1e-15:
            return
        target = self.buffers[signed]
        item = _SupportVector(np.asarray(x, dtype=np.float64).copy(), float(alpha))
        if len(target) < self.budgets[signed]:
            target.append(item)
            return
        removed = target.pop(0)
        target.append(item)
        nearest = max(target, key=lambda candidate: self._kernel(removed.x, candidate.x))
        nearest.alpha += removed.alpha
        self.replacements += 1

    def learn(self, x: np.ndarray, y: int, uid: str) -> None:
        signed = 1 if int(y) == 1 else -1
        opposite = self.buffers[-signed]
        same = self.buffers[signed]
        score_x = self._score(x)
        if opposite:
            nearest = sorted(opposite, key=lambda item: -self._kernel(x, item.x))[: self.k]
            opposite_scores = self._score_batch(np.stack([item.x for item in nearest]))
            valid = [item for item, score in zip(nearest, opposite_scores)
                     if 1.0 - signed * (score_x - float(score)) > 0.0]
        else:
            valid = []
        for values in self.buffers.values():
            for item in values:
                item.alpha *= 1.0 - self.eta
        for item in valid:
            item.alpha -= signed * self.eta * self.c
        if not opposite or (not same and not valid):
            count = 1
        else:
            count = len(valid)
        alpha = self.eta * self.c * signed * min(count, self.k)
        self._insert(signed, x, alpha)
        for values in self.buffers.values():
            for item in values:
                item.alpha = float(np.clip(item.alpha, -100.0, 100.0))
        self.update_count += 1
        if self.update_count % 10 == 0 or self.bias == 0.0:
            self._update_bias()

    def diagnostics(self) -> Dict[str, float]:
        return {
            "positive_support_vectors": float(len(self.buffers[1])),
            "negative_support_vectors": float(len(self.buffers[-1])),
            "buffer_replacements": float(self.replacements),
        }


class _OnlineLogistic:
    def __init__(self, input_dim: int, learning_rate: float) -> None:
        self.weights = np.zeros(input_dim + 1, dtype=np.float64)
        self.learning_rate = float(learning_rate)
        self.count = 0
        self.mean = np.zeros(input_dim, dtype=np.float64)
        self.m2 = np.zeros(input_dim, dtype=np.float64)

    def probability(self, x: np.ndarray) -> float:
        return _sigmoid(float(self.weights[:-1] @ x + self.weights[-1]))

    def likelihood(self, x: np.ndarray) -> float:
        if self.count < 3:
            return 0.5
        variance = self.m2 / max(1, self.count - 1)
        z2 = (x - self.mean) ** 2 / np.maximum(variance, 0.25)
        return float(math.exp(-0.5 * min(float(np.mean(z2)), 20.0)))

    def learn(self, x: np.ndarray, y: int, weight: float = 1.0) -> None:
        error = self.probability(x) - int(y)
        self.weights[:-1] -= self.learning_rate * weight * error * x
        self.weights[-1] -= self.learning_rate * weight * error
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)


class _OBALModel(_OnlineModel):
    """Protocol adaptation of OBAL's multistream adaptive source ensemble."""

    provenance = "paper_mechanism_multistream_adaptation_no_public_code"

    def __init__(self, input_dim: int, cfg: OnlineSOTAConfig) -> None:
        self.input_dim = input_dim
        self.cfg = cfg
        self.global_expert = _OnlineLogistic(input_dim, cfg.obal_learning_rate)
        self.sources: Dict[str, _OnlineLogistic] = {}
        self.reliability: MutableMapping[str, float] = defaultdict(lambda: 0.5)
        self.last_candidates: Dict[str, List[tuple[str, _OnlineLogistic, float]]] = {}

    def _source(self, uid: str) -> _OnlineLogistic:
        if uid not in self.sources:
            self.sources[uid] = _OnlineLogistic(self.input_dim, self.cfg.obal_learning_rate)
        return self.sources[uid]

    def _candidates(self, x: np.ndarray, uid: str) -> List[tuple[str, _OnlineLogistic, float]]:
        own = self._source(uid)
        ranked = sorted(
            ((name, model, model.likelihood(x)) for name, model in self.sources.items()),
            key=lambda row: row[2] * (0.1 + self.reliability[row[0]]),
            reverse=True,
        )[: self.cfg.obal_source_limit]
        if all(name != uid for name, _, _ in ranked):
            ranked.append((uid, own, own.likelihood(x)))
        ranked.append(("__global__", self.global_expert, self.global_expert.likelihood(x)))
        return ranked

    def predict_proba(self, x: np.ndarray, uid: str) -> float:
        candidates = self._candidates(x, uid)
        values, weights = [], []
        for name, model, likelihood in candidates:
            bonus = 2.0 if name == uid else 1.0
            weights.append(max(1e-4, bonus * likelihood * (0.1 + self.reliability[name])))
            values.append(model.probability(x))
        normalized = np.asarray(weights) / np.sum(weights)
        self.last_candidates[uid] = candidates
        return float(np.dot(normalized, values))

    def learn(self, x: np.ndarray, y: int, uid: str) -> None:
        decay = float(self.cfg.obal_reliability_decay)
        candidates = self.last_candidates.pop(uid, self._candidates(x, uid))
        for name, model, _ in candidates:
            correct = float((model.probability(x) >= 0.5) == bool(y))
            self.reliability[name] = decay * self.reliability[name] + (1.0 - decay) * correct
        self._source(uid).learn(x, y)
        self.global_expert.learn(x, y, weight=0.5)

    def diagnostics(self) -> Dict[str, float]:
        return {
            "source_stream_models": float(len(self.sources)),
            "mean_source_reliability": float(np.mean(list(self.reliability.values()))) if self.reliability else 0.0,
        }


class _OLIFLModel(_OnlineModel):
    """Complete-feature/complete-label specialization of OLIFL's informative update."""

    provenance = "paper_mechanism_complete_feature_label_specialization_no_public_code"

    def __init__(self, input_dim: int, cfg: OnlineSOTAConfig) -> None:
        self.weights = np.zeros(input_dim, dtype=np.float64)
        self.count = 0
        self.mean = np.zeros(input_dim, dtype=np.float64)
        self.m2 = np.zeros(input_dim, dtype=np.float64)
        self.c = float(cfg.olifl_c)
        self.gamma = float(cfg.olifl_gamma)

    def _informativeness(self) -> np.ndarray:
        if self.count < 2:
            return np.full_like(self.weights, 1.0 / max(1, self.weights.size))
        variance = self.m2 / max(1, self.count - 1)
        total = float(np.sum(variance))
        return variance / total if total > 1e-12 else np.full_like(variance, 1.0 / variance.size)

    def predict_proba(self, x: np.ndarray, uid: str) -> float:
        return _sigmoid(float(self.weights @ x))

    def learn(self, x: np.ndarray, y: int, uid: str) -> None:
        signed = 1 if int(y) == 1 else -1
        info = self._informativeness()
        effective = x * np.sqrt(np.maximum(info * len(info), 1e-6))
        loss = max(0.0, 1.0 - signed * float(self.weights @ x))
        tau = min(self.c, loss / max(float(effective @ effective) + 1.0 / (2.0 * self.gamma), 1e-12))
        self.weights += tau * signed * effective
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)


class _OLFLModel(_OnlineModel):
    """Labeled protocol wrapper around OLFL-style evolving feature prototypes.

    OLFL itself assumes a label-free stream and cannot identify the positive-risk
    class in this benchmark without an external semantic anchor. Here each label,
    observed strictly after prediction, only anchors the evolving class prototype.
    """

    provenance = "protocol_aligned_labeled_surrogate_original_task_unlabeled"

    def __init__(self, input_dim: int, cfg: OnlineSOTAConfig) -> None:
        self.centroids = {0: np.zeros(input_dim, dtype=np.float64), 1: np.zeros(input_dim, dtype=np.float64)}
        self.counts = {0: 0, 1: 0}
        self.decay = float(cfg.olfl_prototype_decay)

    def predict_proba(self, x: np.ndarray, uid: str) -> float:
        if self.counts[0] == 0 and self.counts[1] == 0:
            return 0.5
        if self.counts[0] == 0:
            return 1.0
        if self.counts[1] == 0:
            return 0.0
        d0 = float(np.sum((x - self.centroids[0]) ** 2))
        d1 = float(np.sum((x - self.centroids[1]) ** 2))
        scale = max(math.sqrt(d0 + d1), 1e-6)
        return _sigmoid((d0 - d1) / scale)

    def learn(self, x: np.ndarray, y: int, uid: str) -> None:
        label = int(y)
        self.counts[label] += 1
        rate = max(self.decay, 1.0 / self.counts[label])
        self.centroids[label] = (1.0 - rate) * self.centroids[label] + rate * x


class OnlineSOTALearner:
    """Common strict-prequential wrapper for global and independent-user scopes."""

    def __init__(
        self,
        input_dim: int,
        config: ExperimentConfig,
        method: str,
        scope: str,
        sota_config: OnlineSOTAConfig | None = None,
    ) -> None:
        self.input_dim = int(input_dim)
        self.cfg = config
        self.method = method.lower()
        self.scope = scope.lower()
        if self.method not in SUPPORTED_ONLINE_SOTA:
            raise ValueError(f"Unsupported online SOTA method: {method}")
        if self.scope not in {"global", "per_user"}:
            raise ValueError("scope must be global or per_user")
        self.sota_cfg = sota_config or OnlineSOTAConfig()
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        self.states: Dict[str, _OnlineModel] = {}
        self.user_seen: MutableMapping[str, int] = defaultdict(int)
        self.events: List[Dict[str, object]] = []
        self.threshold_history: MutableMapping[str, deque[tuple[float, int]]] = defaultdict(
            lambda: deque(maxlen=max(1, self.cfg.training.threshold_window))
        )
        self.threshold_updates: MutableMapping[str, int] = defaultdict(int)
        self.cached_threshold: MutableMapping[str, float] = defaultdict(
            lambda: float(self.cfg.training.threshold_default)
        )
        self.smooth_probabilities: MutableMapping[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=max(1, self.cfg.training.smoothing_window))
        )

    @staticmethod
    def _f1_at_threshold(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> float:
        predictions = probabilities >= threshold
        tp = int(np.sum(predictions & (labels == 1)))
        fp = int(np.sum(predictions & (labels == 0)))
        fn = int(np.sum((~predictions) & (labels == 1)))
        return 2.0 * tp / (2.0 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

    def _threshold_key(self, uid: str) -> str:
        return "__global__" if self.scope == "global" else uid

    def _current_threshold(self, uid: str) -> float:
        train = self.cfg.training
        key = self._threshold_key(uid)
        history = self.threshold_history[key]
        if train.threshold_mode != "in_window" or len(history) < train.threshold_min_history:
            return float(train.threshold_default)
        if self.threshold_updates[key] % max(1, train.threshold_refresh_every) == 0:
            probabilities = np.asarray([row[0] for row in history], dtype=np.float64)
            labels = np.asarray([row[1] for row in history], dtype=np.int64)
            best_score = -1.0
            best_threshold = float(train.threshold_default)
            for threshold in np.linspace(0.0, 1.0, max(2, train.threshold_grid_size)):
                score = self._f1_at_threshold(probabilities, labels, float(threshold))
                if score > best_score:
                    best_score = score
                    best_threshold = float(threshold)
            self.cached_threshold[key] = best_threshold
        return self.cached_threshold[key]

    def _key(self, uid: str) -> str:
        return "__global__" if self.scope == "global" else uid

    def _seed(self, key: str) -> int:
        token = sum((index + 1) * byte for index, byte in enumerate(key.encode("utf-8")))
        return int((self.cfg.seed * 1_000_003 + token) % (2**31 - 1))

    def _new_model(self, key: str) -> _OnlineModel:
        if self.method == "oli2ds":
            return _OLI2DSModel(self.input_dim, self.sota_cfg)
        if self.method in {"hbp", "odl"}:
            return _HBPModel(self.input_dim, self.sota_cfg, self._seed(key))
        if self.method == "koil":
            return _KOILModel(self.input_dim, self.sota_cfg)
        if self.method == "obal":
            return _OBALModel(self.input_dim, self.sota_cfg)
        if self.method == "olifl":
            return _OLIFLModel(self.input_dim, self.sota_cfg)
        return _OLFLModel(self.input_dim, self.sota_cfg)

    def _state(self, uid: str) -> _OnlineModel:
        key = self._key(uid)
        if key not in self.states:
            self.states[key] = self._new_model(key)
        return self.states[key]

    def process(self, event: StreamEvent) -> Dict[str, object]:
        state = self._state(event.user_id)
        x = np.nan_to_num(np.asarray(event.features, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        raw_probability = float(np.clip(state.predict_proba(x, event.user_id), 0.0, 1.0))
        history = self.smooth_probabilities[event.user_id]
        history.append(raw_probability)
        probability = float(np.mean(history))
        threshold = self._current_threshold(event.user_id)
        row: Dict[str, object] = {
            "event_index": event.index,
            "timestamp": event.timestamp,
            "user_id": event.user_id,
            "user_event_index": self.user_seen[event.user_id] + 1,
            "label": int(event.label),
            "prediction": int(probability >= threshold),
            "probability": probability,
            "raw_probability": raw_probability,
            "threshold": threshold,
            "method": self.method,
            "scope": self.scope,
            "method_provenance": state.provenance,
        }
        self.events.append(row)
        state.learn(x, int(event.label), event.user_id)
        threshold_key = self._threshold_key(event.user_id)
        self.threshold_history[threshold_key].append((probability, int(event.label)))
        self.threshold_updates[threshold_key] += 1
        self.user_seen[event.user_id] += 1
        return row

    def run(self, stream: Sequence[StreamEvent]) -> Dict[str, object]:
        for event in stream:
            self.process(event)
        y = [int(row["label"]) for row in self.events]
        p = [int(row["prediction"]) for row in self.events]
        offset = min(len(self.events), self.cfg.grouping.global_warmup_steps)
        cold = first_k_metrics(
            self.events,
            self.cfg.evaluation.first_k,
            self.cfg.evaluation.require_complete_first_k,
        )
        state_diagnostics = [state.diagnostics() for state in self.states.values()]
        diagnostic_keys = sorted({key for row in state_diagnostics for key in row})
        diagnostics = {
            key: float(np.mean([row[key] for row in state_diagnostics if key in row]))
            for key in diagnostic_keys
        }
        provenance = sorted({state.provenance for state in self.states.values()})
        return {
            "overall": binary_metrics(y, p),
            "post_warmup": binary_metrics(y[offset:], p[offset:]),
            "cold_start": cold,
            "mean_first_k_f1": float(np.mean([row["user_macro_f1"] for row in cold])) if cold else float("nan"),
            "n_events": len(self.events),
            "n_users": len(self.user_seen),
            "n_models": len(self.states),
            "provenance": provenance[0] if len(provenance) == 1 else provenance,
            "baseline_protocol": {
                "prediction_order": "predict_then_observe_then_update",
                "first_event_included": True,
                "scope": self.scope,
                "prediction_threshold": self.cfg.training.threshold_mode,
                "threshold_scope": self.scope,
                "shared_causal_preprocessing": True,
                "method_parameters": asdict(self.sota_cfg),
            },
            "baseline_diagnostics": diagnostics,
        }
