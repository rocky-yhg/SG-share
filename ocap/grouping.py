from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np


@dataclass
class AgglomerationResult:
    clusters: List[Set[str]]
    merges: int
    stop_reason: str
    last_similarity: float


@dataclass
class IncrementalGroupingResult:
    clusters: List[Set[str]]
    merges: int
    split_attempts: int
    split_accepts: int
    stop_reason: str
    last_similarity: float
    objective_before: float
    objective_after: float
    reused_users: int
    new_users: int
    similarity_evaluations: int


def normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        return np.zeros_like(value)
    return value / norm


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = normalize(left)
    b = normalize(right)
    if not np.any(a) or not np.any(b):
        return -1.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def centroid(members: Iterable[str], representations: Mapping[str, np.ndarray]) -> np.ndarray:
    rows = [
        np.asarray(representations[user], dtype=np.float64).reshape(-1)
        for user in sorted(members) if user in representations
    ]
    if not rows:
        raise ValueError("Cannot calculate a centroid without representations")
    return np.mean(np.stack(rows), axis=0)


def agglomerative_groups(
    users: Sequence[str],
    representations: Mapping[str, np.ndarray],
    minimum_group_count: int,
    min_pair_cosine: float = 0.0,
) -> AgglomerationResult:
    ordered = [user for user in sorted(users) if user in representations]
    if not ordered:
        return AgglomerationResult([], 0, "no_representations", float("nan"))
    clusters: Dict[int, Set[str]] = {
        index: {user} for index, user in enumerate(ordered)
    }
    centroids: Dict[int, np.ndarray] = {
        index: np.asarray(representations[user], dtype=np.float64).reshape(-1).copy()
        for index, user in enumerate(ordered)
    }
    sizes = {index: 1 for index in clusters}
    heap: List[Tuple[float, int, int, int]] = []
    sequence = 0
    initial_matrix = np.stack([centroids[index] for index in range(len(ordered))])
    initial_norms = np.linalg.norm(initial_matrix, axis=1)
    valid_initial = np.isfinite(initial_norms) & (initial_norms > 1e-12)
    normalized_initial = np.zeros_like(initial_matrix)
    normalized_initial[valid_initial] = (
        initial_matrix[valid_initial] / initial_norms[valid_initial, None]
    )
    initial_similarities = np.clip(
        normalized_initial @ normalized_initial.T, -1.0, 1.0,
    )
    for left in range(len(ordered)):
        for right in range(left + 1, len(ordered)):
            score = (
                float(initial_similarities[left, right])
                if valid_initial[left] and valid_initial[right]
                else -1.0
            )
            heapq.heappush(heap, (-score, sequence, left, right))
            sequence += 1
    merges = 0
    last_similarity = float("nan")
    next_cluster_id = len(ordered)
    target = max(1, int(minimum_group_count))
    while len(clusters) > target:
        while heap:
            neg_score, _, left, right = heapq.heappop(heap)
            if left in clusters and right in clusters:
                break
        else:
            return AgglomerationResult(
                list(clusters.values()), merges, "no_pair", last_similarity,
            )
        score = -neg_score
        last_similarity = score
        if score < min_pair_cosine:
            return AgglomerationResult(
                sorted(clusters.values(), key=lambda group: tuple(sorted(group))),
                merges, "negative_best_gain", score,
            )
        left_size = sizes[left]
        right_size = sizes[right]
        merged_size = left_size + right_size
        merged_centroid = (
            left_size * centroids[left] + right_size * centroids[right]
        ) / merged_size
        merged_members = clusters[left] | clusters[right]
        del clusters[left], clusters[right]
        del centroids[left], centroids[right]
        del sizes[left], sizes[right]
        merged_id = next_cluster_id
        next_cluster_id += 1
        clusters[merged_id] = merged_members
        centroids[merged_id] = merged_centroid
        sizes[merged_id] = merged_size
        other_ids = sorted(cluster_id for cluster_id in clusters if cluster_id != merged_id)
        other_matrix = np.stack([centroids[other] for other in other_ids])
        other_norms = np.linalg.norm(other_matrix, axis=1)
        merged_norm = float(np.linalg.norm(merged_centroid))
        valid_merged = np.isfinite(merged_norm) and merged_norm > 1e-12
        if valid_merged:
            scores = np.clip(
                (other_matrix @ merged_centroid)
                / np.maximum(other_norms * merged_norm, 1e-12),
                -1.0,
                1.0,
            )
        else:
            scores = np.full(len(other_ids), -1.0, dtype=np.float64)
        for position, other in enumerate(other_ids):
            score = (
                float(scores[position])
                if np.isfinite(other_norms[position]) and other_norms[position] > 1e-12
                else -1.0
            )
            left_id, right_id = sorted((other, merged_id))
            heapq.heappush(heap, (-score, sequence, left_id, right_id))
            sequence += 1
        merges += 1
    return AgglomerationResult(
        sorted(clusters.values(), key=lambda group: tuple(sorted(group))),
        merges, "minimum_group_count", last_similarity,
    )


def random_groups(users: Sequence[str], k: int, seed: int) -> List[Set[str]]:
    rng = np.random.default_rng(seed)
    shuffled = list(sorted(users))
    rng.shuffle(shuffled)
    groups: List[Set[str]] = [set() for _ in range(max(1, min(k, len(shuffled))))]
    for index, user in enumerate(shuffled):
        groups[index % len(groups)].add(user)
    return [group for group in groups if group]


def binary_split(members: Set[str], representations: Mapping[str, np.ndarray]) -> tuple[Set[str], Set[str]] | None:
    users = sorted(user for user in members if user in representations)
    if len(users) < 2:
        return None
    farthest = None
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            distance = 1.0 - cosine(representations[users[i]], representations[users[j]])
            candidate = (distance, users[i], users[j])
            if farthest is None or candidate > farthest:
                farthest = candidate
    if farthest is None:
        return None
    _, seed_a, seed_b = farthest
    left: Set[str] = set()
    right: Set[str] = set()
    for user in users:
        sim_a = cosine(representations[user], representations[seed_a])
        sim_b = cosine(representations[user], representations[seed_b])
        (left if sim_a >= sim_b else right).add(user)
    if not left or not right:
        return None
    return left, right


def partition_cohesion(
    clusters: Sequence[Set[str]], representations: Mapping[str, np.ndarray],
) -> float:
    values: List[float] = []
    for cluster in clusters:
        center = centroid(cluster, representations)
        values.extend(cosine(representations[user], center) for user in sorted(cluster))
    return float(np.mean(values)) if values else float("-inf")


def incremental_split_merge_groups(
    users: Sequence[str],
    representations: Mapping[str, np.ndarray],
    previous: Sequence[Set[str]],
    minimum_group_count: int,
    min_pair_cosine: float = 0.0,
    split_min_users: int = 4,
    objective_margin: float = 0.0,
    max_split_merge_swaps: int = 1,
) -> IncrementalGroupingResult:
    """Update a partition with group-level merges and local split-merge swaps.

    Existing groups are retained as the starting point. Newly users_for_grouping users enter
    as singletons, after which only group centroids are compared. A split is
    accepted only when pairing it with a non-sibling merge improves mean
    user-to-centroid cosine cohesion.
    """
    ordered = [user for user in sorted(users) if user in representations]
    allowed = set(ordered)
    clusters: List[Set[str]] = []
    reused: Set[str] = set()
    for old in previous:
        kept = set(old) & allowed
        if kept:
            clusters.append(kept)
            reused.update(kept)
    new_users = [user for user in ordered if user not in reused]
    clusters.extend({user} for user in new_users)
    if not clusters:
        return IncrementalGroupingResult(
            [], 0, 0, 0, "no_representations", float("nan"), float("nan"),
            float("nan"), 0, 0, 0,
        )

    target = max(1, int(minimum_group_count))
    similarity_evaluations = 0
    merges = 0
    last_similarity = float("nan")

    def best_pair(groups: Sequence[Set[str]], excluded: Set[frozenset[str]] | None = None):
        nonlocal similarity_evaluations
        centers = [centroid(group, representations) for group in groups]
        best = None
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                if excluded and frozenset(groups[left] | groups[right]) in excluded:
                    continue
                score = cosine(centers[left], centers[right])
                similarity_evaluations += 1
                candidate = (score, -left, -right, left, right)
                if best is None or candidate > best:
                    best = candidate
        return None if best is None else (best[0], best[3], best[4])

    while len(clusters) > target:
        pair = best_pair(clusters)
        if pair is None:
            break
        score, left, right = pair
        last_similarity = score
        if score < min_pair_cosine:
            break
        merged = clusters[left] | clusters[right]
        clusters = [
            group for index, group in enumerate(clusters)
            if index not in {left, right}
        ] + [merged]
        merges += 1

    objective_before = partition_cohesion(clusters, representations)
    current_objective = objective_before
    split_attempts = 0
    split_accepts = 0
    for _ in range(max(0, int(max_split_merge_swaps))):
        candidates = [
            (partition_cohesion([group], representations), tuple(sorted(group)), index)
            for index, group in enumerate(clusters)
            if len(group) >= max(2, int(split_min_users))
        ]
        if not candidates or len(clusters) < 2:
            break
        _, _, parent_index = min(candidates)
        children = binary_split(clusters[parent_index], representations)
        split_attempts += 1
        if children is None:
            break
        child_left, child_right = children
        proposed = [
            group for index, group in enumerate(clusters) if index != parent_index
        ] + [child_left, child_right]
        sibling_union = frozenset(child_left | child_right)
        pair = best_pair(proposed, {sibling_union})
        if pair is None:
            break
        score, left, right = pair
        last_similarity = score
        if score < min_pair_cosine:
            break
        merged = proposed[left] | proposed[right]
        proposed = [
            group for index, group in enumerate(proposed)
            if index not in {left, right}
        ] + [merged]
        proposed_objective = partition_cohesion(proposed, representations)
        if proposed_objective <= current_objective + float(objective_margin):
            break
        clusters = proposed
        current_objective = proposed_objective
        merges += 1
        split_accepts += 1

    stop_reason = "incremental_split_merge" if split_accepts else "incremental_reuse"
    return IncrementalGroupingResult(
        sorted(clusters, key=lambda group: tuple(sorted(group))),
        merges,
        split_attempts,
        split_accepts,
        stop_reason,
        last_similarity,
        objective_before,
        current_objective,
        len(reused),
        len(new_users),
        similarity_evaluations,
    )


def match_clusters(
    previous: Mapping[str, Set[str]], proposed: Sequence[Set[str]], next_group_index: int,
) -> tuple[Dict[str, Set[str]], int, Set[str]]:
    remaining_old = set(previous)
    assignments: Dict[str, Set[str]] = {}
    created: Set[str] = set()
    for cluster in sorted(proposed, key=lambda group: (-len(group), tuple(sorted(group)))):
        best_gid = None
        best_overlap = 0.0
        for gid in sorted(remaining_old):
            old = previous[gid]
            union = len(cluster | old)
            overlap = len(cluster & old) / union if union else 0.0
            if overlap > best_overlap:
                best_overlap = overlap
                best_gid = gid
        if best_gid is not None and best_overlap >= 0.5:
            gid = best_gid
            remaining_old.remove(gid)
        else:
            gid = f"g{next_group_index:05d}"
            next_group_index += 1
            created.add(gid)
        assignments[gid] = set(cluster)
    return assignments, next_group_index, created
