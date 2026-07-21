from __future__ import annotations

import copy
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, MutableMapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F

from .config import ExperimentConfig
from .data import StreamEvent
from .metrics import binary_metrics, first_k_metrics
from .model import weighted_focal_loss


@dataclass(frozen=True)
class PaperBaselineConfig:
    """Shared budget plus paper-specific defaults for the tabular adaptation."""

    hidden_dim: int = 64
    memory_size: int = 128
    replay_batch_size: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    pdfk_kd_lambda: float = 5.5
    pdfk_kd_temperature: float = 3.0
    pdfk_ema_alpha: float = 0.01
    pdfk_perturb_gamma: float = 5e-4
    pdfk_perturb_lambda: float = 5e-3
    budgeted_fisher_ema: float = 0.01
    budgeted_similarity_ema: float = 0.01
    budgeted_frequency_scale: float = 4.0
    budgeted_temperature: float = 0.125
    budgeted_fisher_refresh: int = 4
    supermask_alpha: float = 0.01
    supermask_kd_lambda: float = 5.5
    supermask_fisher_ema: float = 0.01
    supermask_view_noise: float = 0.01


class TabularOCLNet(nn.Module):
    """Common two-hidden-layer model used by all per-user SOTA adaptations."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 2),
        ])
        for layer in self.layers:
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5.0))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.layers[0](x))
        return F.gelu(self.layers[1](x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers[2](self.features(x))


class ReplayBuffer:
    """Fixed-size replay with reservoir or greedy class-balanced insertion."""

    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self.capacity = int(capacity)
        self.rng = rng
        self.features: List[np.ndarray] = []
        self.labels: List[int] = []
        self.frequency: List[float] = []
        self.n_seen = 0

    def __len__(self) -> int:
        return len(self.labels)

    def _append(self, x: np.ndarray, y: int) -> None:
        self.features.append(np.asarray(x, dtype=np.float32).copy())
        self.labels.append(int(y))
        self.frequency.append(0.0)

    def _replace(self, index: int, x: np.ndarray, y: int) -> None:
        self.features[index] = np.asarray(x, dtype=np.float32).copy()
        self.labels[index] = int(y)
        self.frequency[index] = 0.0

    def reservoir_update(self, x: np.ndarray, y: int) -> None:
        self.n_seen += 1
        if len(self) < self.capacity:
            self._append(x, y)
            return
        index = int(self.rng.integers(0, self.n_seen))
        if index < self.capacity:
            self._replace(index, x, y)

    def balanced_update(self, x: np.ndarray, y: int) -> None:
        self.n_seen += 1
        if len(self) < self.capacity:
            self._append(x, y)
            return
        counts = np.bincount(np.asarray(self.labels, dtype=np.int64), minlength=2)
        majority = int(np.argmax(counts))
        incoming = int(y)
        if incoming == majority:
            return
        candidates = np.flatnonzero(np.asarray(self.labels) == majority)
        self._replace(int(self.rng.choice(candidates)), x, incoming)

    def random_indices(self, count: int) -> np.ndarray:
        count = min(int(count), len(self))
        if count <= 0:
            return np.empty(0, dtype=np.int64)
        return np.asarray(
            self.rng.choice(len(self), size=count, replace=False), dtype=np.int64
        )

    def tensors(
        self, indices: Sequence[int], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = np.stack([self.features[int(index)] for index in indices])
        y = np.asarray([self.labels[int(index)] for index in indices], dtype=np.int64)
        return (
            torch.as_tensor(x, dtype=torch.float32, device=device),
            torch.as_tensor(y, dtype=torch.long, device=device),
        )

    def similarity_aware_indices(
        self,
        count: int,
        class_similarity: np.ndarray,
        frequency_scale: float,
        temperature: float,
    ) -> np.ndarray:
        count = min(int(count), len(self))
        if count <= 0:
            return np.empty(0, dtype=np.int64)
        labels = np.asarray(self.labels, dtype=np.int64)
        frequencies = np.asarray(self.frequency, dtype=np.float64)
        class_frequency = np.asarray(
            [frequencies[labels == label].sum() for label in range(2)]
        )
        effective = frequencies.copy()
        for index, label in enumerate(labels):
            effective[index] += float(np.dot(class_similarity[:, label], class_frequency))
        logits = -effective / max(float(temperature), 1e-6)
        logits -= logits.max()
        probabilities = np.exp(np.clip(logits, -50.0, 50.0))
        probabilities /= probabilities.sum()
        selected = np.asarray(
            self.rng.choice(len(self), size=count, replace=False, p=probabilities),
            dtype=np.int64,
        )
        decay = count / max(float(frequency_scale) * len(self), 1.0)
        self.frequency = list(frequencies * (1.0 - decay))
        for index in selected:
            self.frequency[int(index)] += 1.0
        return selected


@dataclass
class PersonalState:
    model: TabularOCLNet
    optimizer: torch.optim.Optimizer
    replay: ReplayBuffer
    teacher: TabularOCLNet | None = None
    fisher: Dict[str, torch.Tensor] = field(default_factory=dict)
    layer_fisher: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    class_grad_ema: Dict[int, np.ndarray] = field(default_factory=dict)
    class_similarity: np.ndarray = field(
        default_factory=lambda: np.eye(2, dtype=np.float64)
    )
    update_count: int = 0
    frozen_layer_total: int = 0
    perturb_count: int = 0
    feature_grad_norm_ema: float = 0.0


class PortedPerUserBaseline:
    """Paper-faithful cross-domain adaptations of PDFK, aL-SAR, and Supermask."""

    def __init__(self, input_dim: int, config: ExperimentConfig, method: str) -> None:
        self.cfg = copy.deepcopy(config)
        self.input_dim = int(input_dim)
        self.method = method.lower()
        if self.method not in {"pdfk", "budgeted", "supermask"}:
            raise ValueError(f"Unsupported paper baseline: {method}")
        self.paper_cfg = PaperBaselineConfig()
        self.device = torch.device(self.cfg.device)
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        template = TabularOCLNet(self.input_dim, self.paper_cfg.hidden_dim).to(self.device)
        self.template_state = copy.deepcopy(template.state_dict())
        self.states: Dict[str, PersonalState] = {}
        self.user_seen: MutableMapping[str, int] = defaultdict(int)
        self.events: List[Dict[str, object]] = []

    def _seed_for_user(self, uid: str) -> int:
        token = sum((i + 1) * byte for i, byte in enumerate(uid.encode("utf-8")))
        return int((self.cfg.seed * 1_000_003 + token) % (2**32 - 1))

    def _new_model(self) -> TabularOCLNet:
        model = TabularOCLNet(self.input_dim, self.paper_cfg.hidden_dim).to(self.device)
        model.load_state_dict(copy.deepcopy(self.template_state), strict=True)
        return model

    def _state(self, uid: str) -> PersonalState:
        if uid in self.states:
            return self.states[uid]
        model = self._new_model()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.paper_cfg.learning_rate,
            weight_decay=self.paper_cfg.weight_decay,
        )
        replay = ReplayBuffer(self.paper_cfg.memory_size, np.random.default_rng(
            self._seed_for_user(uid)
        ))
        teacher = copy.deepcopy(model).eval() if self.method == "pdfk" else None
        fisher = {
            name: torch.zeros_like(parameter)
            for name, parameter in model.named_parameters()
        }
        self.states[uid] = PersonalState(
            model=model,
            optimizer=optimizer,
            replay=replay,
            teacher=teacher,
            fisher=fisher,
        )
        return self.states[uid]

    def _tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            x, dtype=torch.float32, device=self.device
        ).reshape(1, -1)

    def _task_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return weighted_focal_loss(
            logits,
            labels,
            self.cfg.training.focal_alpha,
            self.cfg.training.focal_gamma,
            self.cfg.training.positive_class_weight,
        )

    def _training_batch(
        self,
        state: PersonalState,
        x: np.ndarray,
        y: int,
        similarity_aware: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if similarity_aware:
            indices = state.replay.similarity_aware_indices(
                self.paper_cfg.replay_batch_size,
                state.class_similarity,
                self.paper_cfg.budgeted_frequency_scale,
                self.paper_cfg.budgeted_temperature,
            )
        else:
            indices = state.replay.random_indices(self.paper_cfg.replay_batch_size)
        current_x = self._tensor(x)
        current_y = torch.tensor([int(y)], dtype=torch.long, device=self.device)
        if not len(indices):
            return current_x, current_y
        replay_x, replay_y = state.replay.tensors(indices, self.device)
        return torch.cat([current_x, replay_x]), torch.cat([current_y, replay_y])

    @staticmethod
    def _weighted_kd(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        teacher = F.softmax(teacher_logits / temperature, dim=1)
        student = F.log_softmax(student_logits / temperature, dim=1)
        return F.kl_div(student, teacher, reduction="batchmean") * temperature**2

    def _pdfk_perturb_backward(self, state: PersonalState) -> None:
        indices = state.replay.random_indices(self.paper_cfg.replay_batch_size)
        if len(indices) < 2:
            return
        replay_x, replay_y = state.replay.tensors(indices, self.device)
        model = state.model
        model.eval()
        with torch.no_grad():
            original_probabilities = F.softmax(model(replay_x), dim=1)
            correct = original_probabilities.argmax(dim=1).eq(replay_y)
        if int(correct.sum().item()) < 2:
            return

        proxy = copy.deepcopy(model).train()
        with torch.no_grad():
            for parameter in proxy.parameters():
                if parameter.ndim > 1:
                    parameter.add_(
                        torch.randn_like(parameter),
                        alpha=1e-6 * (float(parameter.norm().item()) + 1e-12),
                    )
        proxy.zero_grad(set_to_none=True)
        divergence = F.kl_div(
            F.log_softmax(proxy(replay_x), dim=1),
            original_probabilities,
            reduction="none",
        ).sum(dim=1)[correct].mean()
        (-divergence).backward()
        with torch.no_grad():
            for proxy_parameter, reference in zip(proxy.parameters(), model.parameters()):
                if proxy_parameter.grad is None or proxy_parameter.ndim <= 1:
                    continue
                gradient = proxy_parameter.grad
                gradient.mul_(reference.norm() / (gradient.norm() + 1e-20))
                proxy_parameter.add_(
                    gradient, alpha=-self.paper_cfg.pdfk_perturb_gamma
                )
            differences = [
                proxy_parameter.detach() - parameter.detach()
                for proxy_parameter, parameter in zip(proxy.parameters(), model.parameters())
            ]
            for parameter, difference in zip(model.parameters(), differences):
                if parameter.ndim > 1:
                    parameter.add_(difference)

        consistency = self.paper_cfg.pdfk_perturb_lambda * F.kl_div(
            F.log_softmax(model(replay_x), dim=1),
            original_probabilities,
            reduction="none",
        ).sum(dim=1)[correct].mean()
        consistency.backward()
        with torch.no_grad():
            for parameter, difference in zip(model.parameters(), differences):
                if parameter.ndim > 1:
                    parameter.sub_(difference)
        state.perturb_count += 1

    def _pdfk_step(self, state: PersonalState, x: np.ndarray, label: int) -> None:
        teacher = state.teacher
        assert teacher is not None
        batch_x, batch_y = self._training_batch(state, x, label)
        state.model.train()
        state.optimizer.zero_grad(set_to_none=True)
        self._pdfk_perturb_backward(state)
        student_logits = state.model(batch_x)
        with torch.no_grad():
            teacher_logits = teacher(batch_x)
        loss = self._task_loss(student_logits, batch_y)
        loss += self.paper_cfg.pdfk_kd_lambda * self._weighted_kd(
            teacher_logits,
            student_logits,
            self.paper_cfg.pdfk_kd_temperature,
        )
        loss.backward()
        state.optimizer.step()
        alpha = self.paper_cfg.pdfk_ema_alpha
        with torch.no_grad():
            for teacher_parameter, student_parameter in zip(
                teacher.parameters(), state.model.parameters()
            ):
                teacher_parameter.mul_(1.0 - alpha).add_(
                    student_parameter, alpha=alpha
                )

    @staticmethod
    def _gradient_vector(model: TabularOCLNet) -> np.ndarray:
        values = [
            parameter.grad.detach().reshape(-1).cpu().numpy()
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        return np.concatenate(values) if values else np.zeros(1, dtype=np.float32)

    def _update_budgeted_similarity(
        self, state: PersonalState, label: int, gradient: np.ndarray
    ) -> None:
        stride = 2000
        subset = gradient[::stride]
        if subset.size < 4:
            subset = gradient[: min(32, gradient.size)]
        alpha = self.paper_cfg.budgeted_similarity_ema
        old = state.class_grad_ema.get(int(label))
        state.class_grad_ema[int(label)] = (
            subset.copy() if old is None else (1.0 - alpha) * old + alpha * subset
        )
        for first in range(2):
            for second in range(2):
                a = state.class_grad_ema.get(first)
                b = state.class_grad_ema.get(second)
                if a is None or b is None:
                    continue
                state.class_similarity[first, second] = float(
                    np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
                )

    def _budgeted_freeze_count(
        self, state: PersonalState, feature_grad_norm: float
    ) -> int:
        if state.update_count % self.paper_cfg.budgeted_fisher_refresh == 0:
            return 0
        layers = state.model.layers
        forward_cost = float(sum(layer.weight.numel() for layer in layers))
        backward_cost = np.asarray([
            float(sum(parameter.numel() for parameter in layer.parameters()))
            for layer in layers
        ])
        information_per_cost = []
        for frozen in range(len(layers) + 1):
            information = state.layer_fisher[frozen:].sum()
            cost = forward_cost + backward_cost[frozen:].sum()
            information_per_cost.append(float(information / max(cost, 1.0)))
        best_future = max(information_per_cost)
        scores = [0.0]
        information_ratio = (
            feature_grad_norm / max(state.feature_grad_norm_ema, 1e-12)
            if state.feature_grad_norm_ema > 0.0
            else 1.0
        )
        for frozen in range(1, len(layers) + 1):
            scores.append(float(
                best_future * backward_cost[:frozen].sum()
                - information_ratio * state.layer_fisher[:frozen].sum()
            ))
        best = int(np.argmax(scores))
        return best if scores[best] > 0.0 else 0

    def _budgeted_step(self, state: PersonalState, x: np.ndarray, label: int) -> None:
        batch_x, batch_y = self._training_batch(
            state, x, label, similarity_aware=True
        )
        state.model.train()
        state.optimizer.zero_grad(set_to_none=True)
        final_features = state.model.features(batch_x)
        final_features.retain_grad()
        logits = state.model.layers[2](final_features)
        self._task_loss(logits, batch_y).backward()
        gradient = self._gradient_vector(state.model)
        feature_grad_norm = float(final_features.grad.detach().pow(2).sum().item())
        batch_fisher = np.asarray([
            sum(
                float(parameter.grad.detach().pow(2).sum().item())
                for parameter in layer.parameters()
                if parameter.grad is not None
            )
            for layer in state.model.layers
        ])
        freeze_count = self._budgeted_freeze_count(state, feature_grad_norm)
        for layer in state.model.layers[:freeze_count]:
            for parameter in layer.parameters():
                parameter.grad = None
        state.optimizer.step()
        alpha = self.paper_cfg.budgeted_fisher_ema
        if state.update_count % self.paper_cfg.budgeted_fisher_refresh == 0:
            state.layer_fisher = (
                (1.0 - alpha) * state.layer_fisher + alpha * batch_fisher
            )
        state.feature_grad_norm_ema = (
            (1.0 - alpha) * state.feature_grad_norm_ema
            + alpha * feature_grad_norm
        )
        self._update_budgeted_similarity(state, label, gradient)
        state.frozen_layer_total += freeze_count

    def _supermask_step(self, state: PersonalState, x: np.ndarray, label: int) -> None:
        batch_x, batch_y = self._training_batch(state, x, label)
        model = state.model
        model.train()
        old_parameters = [
            parameter.detach().clone() for parameter in model.parameters()
        ]
        noise = torch.randn_like(batch_x) * self.paper_cfg.supermask_view_noise
        with torch.no_grad():
            old_logits = model(batch_x)
            old_augmented_logits = model(batch_x + noise)

        state.optimizer.zero_grad(set_to_none=True)
        new_logits = model(batch_x)
        new_augmented_logits = model(batch_x + noise)
        distillation = 0.5 * (
            F.kl_div(
                F.log_softmax(new_augmented_logits, dim=1),
                F.softmax(old_logits, dim=1),
                reduction="batchmean",
            )
            + F.kl_div(
                F.log_softmax(new_augmented_logits, dim=1),
                F.softmax(old_augmented_logits, dim=1),
                reduction="batchmean",
            )
        )
        loss = self._task_loss(new_logits, batch_y)
        loss += self.paper_cfg.supermask_kd_lambda * distillation
        loss.backward()
        first_gradients = [
            torch.zeros_like(parameter)
            if parameter.grad is None
            else parameter.grad.detach().clone()
            for parameter in model.parameters()
        ]
        fisher_alpha = self.paper_cfg.supermask_fisher_ema
        for (name, _), gradient in zip(model.named_parameters(), first_gradients):
            state.fisher[name].mul_(1.0 - fisher_alpha).add_(
                gradient.pow(2), alpha=fisher_alpha
            )
        state.optimizer.step()

        alpha = self.paper_cfg.supermask_alpha
        with torch.no_grad():
            for (name, parameter), old, first in zip(
                model.named_parameters(), old_parameters, first_gradients
            ):
                new = parameter.detach().clone()
                first_mask = torch.sigmoid(first / (old.abs().sum() + 1e-12))
                second_mask = torch.sigmoid(
                    state.fisher[name] / (old.norm() + 1e-12)
                )
                merge = alpha * (
                    first_mask + second_mask - alpha * first_mask * second_mask
                )
                parameter.copy_(old + merge * (new - old))

    def process(self, event: StreamEvent) -> Dict[str, object]:
        state = self._state(event.user_id)
        state.model.eval()
        with torch.no_grad():
            tensor = self._tensor(event.features)
            if self.method == "pdfk" and state.teacher is not None:
                averaged = {
                    name: 0.5 * (parameter + teacher_parameter)
                    for (name, parameter), (_, teacher_parameter) in zip(
                        state.model.named_parameters(),
                        state.teacher.named_parameters(),
                    )
                }
                logits = functional_call(state.model, averaged, (tensor,))
            else:
                logits = state.model(tensor)
            probability = float(
                torch.sigmoid(logits[:, 1] - logits[:, 0])[0].item()
            )
        row: Dict[str, object] = {
            "event_index": event.index,
            "timestamp": event.timestamp,
            "user_id": event.user_id,
            "user_event_index": self.user_seen[event.user_id] + 1,
            "label": event.label,
            "prediction": int(probability >= 0.5),
            "probability": probability,
            "threshold": 0.5,
            "method_provenance": "paper_faithful_cross_domain_port",
        }
        self.events.append(row)
        if self.method == "pdfk":
            self._pdfk_step(state, event.features, event.label)
            state.replay.reservoir_update(event.features, event.label)
        elif self.method == "budgeted":
            self._budgeted_step(state, event.features, event.label)
            state.replay.balanced_update(event.features, event.label)
        else:
            self._supermask_step(state, event.features, event.label)
            state.replay.reservoir_update(event.features, event.label)
        state.update_count += 1
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
        diagnostics = {
            "mean_pdfk_perturbations_per_user": float(np.mean([
                state.perturb_count for state in self.states.values()
            ])) if self.states else 0.0,
            "mean_budgeted_frozen_layers": float(
                sum(state.frozen_layer_total for state in self.states.values())
                / max(1, sum(state.update_count for state in self.states.values()))
            ),
        }
        return {
            "overall": binary_metrics(y, p),
            "post_warmup": binary_metrics(y[offset:], p[offset:]),
            "cold_start": cold,
            "mean_first_k_f1": float(np.mean([
                row["user_macro_f1"] for row in cold
            ])) if cold else float("nan"),
            "n_events": len(self.events),
            "n_users": len(self.states),
            "provenance": "paper_faithful_cross_domain_port",
            "baseline_protocol": {
                "prediction_order": "predict_then_observe_then_update",
                "first_event_included": True,
                "task_loss": "common_weighted_focal",
                "model": "two_hidden_layer_mlp",
                "per_user_independent_models": True,
                "paper_hyperparameters": asdict(self.paper_cfg),
            },
            "baseline_diagnostics": diagnostics,
        }
