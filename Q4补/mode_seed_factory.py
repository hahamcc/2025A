from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Sequence

import numpy as np

from Q4 import q4_main as base


MODE_ORDER = ("relay", "overlap_relay", "spatial_exploration", "redundant")
MODE_LABELS = {
    "relay": "全程接力型",
    "overlap_relay": "交叠接力型",
    "spatial_exploration": "空间分工探索型",
    "redundant": "冗余集中型",
}


@dataclass(frozen=True)
class PopulationRecord:
    seed: int
    index: int
    source: str
    requested_mode: str
    vector: np.ndarray
    strategy: base.Strategy
    classification: str
    fallback: bool


@dataclass(frozen=True)
class PopulationBundle:
    population: np.ndarray
    records: tuple[PopulationRecord, ...]
    requested_counts: dict[str, int]


def _mode_counts(population_size: int) -> dict[str, int]:
    """Keep the agreed 72-member composition, proportionally for quick runs."""
    if population_size == 72:
        return {
            "relay": 12,
            "overlap_relay": 12,
            "spatial_exploration": 12,
            "redundant": 6,
            "single_library": 8,
            "global_lhs": 22,
        }
    weights = {
        "relay": 12,
        "overlap_relay": 12,
        "spatial_exploration": 12,
        "redundant": 6,
        "single_library": 8,
    }
    counts = {key: int(round(population_size * value / 72.0)) for key, value in weights.items()}
    counts["global_lhs"] = population_size - sum(counts.values())
    if counts["global_lhs"] < 1:
        counts["single_library"] = max(0, counts["single_library"] + counts["global_lhs"] - 1)
        counts["global_lhs"] = 1
    return counts


def _angular_distance(left: float, right: float) -> float:
    return abs((left - right + math.pi) % (2.0 * math.pi) - math.pi)


def _pair_data(strategy: base.Strategy) -> list[tuple[int, int, float, float, float]]:
    data = []
    for left, right in itertools.combinations(range(3), 2):
        first, second = strategy.plans[left], strategy.plans[right]
        time_gap = abs(first.burst_time - second.burst_time)
        heading_gap = math.degrees(_angular_distance(first.heading, second.heading))
        point_gap = float(np.linalg.norm(first.burst_point[:2] - second.burst_point[:2]))
        data.append((left, right, time_gap, heading_gap, point_gap))
    return data


def classify(strategy: base.Strategy) -> str:
    times = sorted(plan.burst_time for plan in strategy.plans)
    pairs = _pair_data(strategy)
    if min(right - left for left, right in zip(times[:-1], times[1:])) >= 2.0:
        return "relay"
    for left, right, gap, _, _ in pairs:
        if gap <= 1.5:
            third = next(index for index in range(3) if index not in (left, right))
            third_time = strategy.plans[third].burst_time
            if min(abs(third_time - strategy.plans[left].burst_time), abs(third_time - strategy.plans[right].burst_time)) >= 2.0:
                return "overlap_relay"
    if max(times) - min(times) <= 2.0:
        return "redundant"
    return "other"


def matches_mode(strategy: base.Strategy, mode: str) -> bool:
    times = [plan.burst_time for plan in strategy.plans]
    pairs = _pair_data(strategy)
    if mode == "relay":
        ordered = sorted(times)
        return min(right - left for left, right in zip(ordered[:-1], ordered[1:])) >= 2.0
    if mode == "overlap_relay":
        for left, right, gap, _, _ in pairs:
            if gap <= 1.5:
                third = next(index for index in range(3) if index not in (left, right))
                third_time = times[third]
                if min(abs(third_time - times[left]), abs(third_time - times[right])) >= 2.0:
                    return True
        return False
    if mode == "spatial_exploration":
        return any(gap <= 1.5 and heading >= 30.0 and distance >= 50.0 for _, _, gap, heading, distance in pairs)
    if mode == "redundant":
        return max(times) - min(times) <= 2.0
    raise ValueError(f"Unknown seed mode: {mode}")


def _vector_key(vector: np.ndarray) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(vector, dtype=float), 8))


def _feasible_strategy(vector: np.ndarray) -> base.Strategy | None:
    strategy = base.decode_strategy(vector)
    return strategy if base.validate_strategy(strategy) is None else None


def _single_anchor_pools(seed: int) -> tuple[np.ndarray, ...]:
    """Geometric anchors plus a small LHS reserve for all three aircraft."""
    pools = []
    for index in range(3):
        geometric = list(base.geometric_single_anchors(index))
        known = list(base.known_single_anchors(index))
        random_vectors = base.lhs_population(base.variable_bounds((index,)), 480, seed + 101 * index)
        pool = np.vstack([*geometric, *known, random_vectors]).astype(float)
        pools.append(pool)
    return tuple(pools)


def _draw_mode_vector(
    pools: Sequence[np.ndarray], mode: str, rng: np.random.Generator, attempts: int = 40000
) -> np.ndarray | None:
    for _ in range(attempts):
        vector = np.concatenate([pool[int(rng.integers(len(pool)))] for pool in pools]).astype(float)
        strategy = _feasible_strategy(vector)
        if strategy is not None and matches_mode(strategy, mode):
            return vector
    return None


def _jitter_mode_vector(
    references: Sequence[np.ndarray], mode: str, rng: np.random.Generator, attempts: int = 4000
) -> np.ndarray | None:
    if not references:
        return None
    bounds = np.asarray(base.JOINT_BOUNDS, dtype=float)
    for _ in range(attempts):
        vector = np.asarray(references[int(rng.integers(len(references)))], dtype=float).copy()
        for uav_index in range(3):
            start = 4 * uav_index
            vector[start] = (vector[start] + rng.normal(0.0, 0.035)) % (2.0 * math.pi)
            vector[start + 1] = np.clip(vector[start + 1] + rng.normal(0.0, 0.025), 0.0, 1.0)
            vector[start + 2] = np.clip(
                vector[start + 2] + rng.normal(0.0, 0.015 * base.BURST_TIME_LIMITS[uav_index]),
                bounds[start + 2, 0],
                bounds[start + 2, 1],
            )
            vector[start + 3] = np.clip(vector[start + 3] + rng.normal(0.0, 0.025), 0.0, 1.0)
        strategy = _feasible_strategy(vector)
        if strategy is not None and matches_mode(strategy, mode):
            return vector
    return None


def _library_vectors(libraries: Sequence[Sequence[base.SingleCandidate]], count: int, seed: int) -> list[np.ndarray]:
    combinations = list(itertools.product(*libraries))
    rng = np.random.default_rng(seed + 20_000)
    rng.shuffle(combinations)
    return [np.concatenate([candidate.encoded for candidate in combo]).astype(float) for combo in combinations[:count]]


def _make_records(
    seed: int,
    vectors: Sequence[tuple[np.ndarray, str, str, bool]],
) -> tuple[PopulationRecord, ...]:
    records = []
    for index, (vector, source, requested_mode, fallback) in enumerate(vectors):
        strategy = _feasible_strategy(vector)
        if strategy is None:
            raise AssertionError("Initial population contains an infeasible individual.")
        records.append(
            PopulationRecord(
                seed=seed,
                index=index,
                source=source,
                requested_mode=requested_mode,
                vector=np.asarray(vector, dtype=float),
                strategy=strategy,
                classification=classify(strategy),
                fallback=fallback,
            )
        )
    return tuple(records)


def make_mode_hybrid_population(
    profile: base.SearchProfile,
    libraries: Sequence[Sequence[base.SingleCandidate]],
    seed: int,
) -> PopulationBundle:
    counts = _mode_counts(profile.population_size)
    pools = _single_anchor_pools(seed)
    rng = np.random.default_rng(seed + 30_000)
    vectors: list[tuple[np.ndarray, str, str, bool]] = []
    seen: set[tuple[float, ...]] = set()
    accepted_by_mode: dict[str, list[np.ndarray]] = {mode: [] for mode in MODE_ORDER}

    for mode in MODE_ORDER:
        for _ in range(counts[mode]):
            vector = _draw_mode_vector(pools, mode, rng)
            source = f"mode_{mode}"
            fallback = False
            if vector is None or _vector_key(vector) in seen:
                vector = _jitter_mode_vector(accepted_by_mode[mode], mode, rng)
                source = f"mode_{mode}_jitter"
            if vector is None or _vector_key(vector) in seen:
                # A global feasible sample is the documented final fallback.  Its true
                # class is retained separately in the CSV rather than being relabelled.
                for candidate in base.lhs_population(base.JOINT_BOUNDS, 2000, int(rng.integers(1, 2**31 - 1))):
                    strategy = _feasible_strategy(candidate)
                    if strategy is not None and _vector_key(candidate) not in seen:
                        vector = candidate
                        source = f"lhs_backfill_{mode}"
                        fallback = True
                        break
            if vector is None:
                raise RuntimeError(f"Unable to create a feasible {mode} seed.")
            seen.add(_vector_key(vector))
            if matches_mode(_feasible_strategy(vector), mode):
                accepted_by_mode[mode].append(vector)
            vectors.append((vector, source, mode, fallback))

    for vector in _library_vectors(libraries, counts["single_library"], seed):
        if _vector_key(vector) not in seen:
            seen.add(_vector_key(vector))
            vectors.append((vector, "single_library", "single_library", False))
    while sum(item[2] == "single_library" for item in vectors) < counts["single_library"]:
        vector = base.lhs_population(base.JOINT_BOUNDS, 1, int(rng.integers(1, 2**31 - 1)))[0]
        if _vector_key(vector) not in seen:
            seen.add(_vector_key(vector))
            vectors.append((vector, "lhs_backfill_single_library", "single_library", True))

    global_vectors = base.lhs_population(base.JOINT_BOUNDS, counts["global_lhs"], seed + 10_000)
    for vector in global_vectors:
        if _vector_key(vector) in seen:
            continue
        seen.add(_vector_key(vector))
        vectors.append((vector, "global_lhs", "global_lhs", False))
    while sum(item[2] == "global_lhs" for item in vectors) < counts["global_lhs"]:
        vector = base.lhs_population(base.JOINT_BOUNDS, 1, int(rng.integers(1, 2**31 - 1)))[0]
        if _vector_key(vector) not in seen:
            seen.add(_vector_key(vector))
            vectors.append((vector, "global_lhs_backfill", "global_lhs", True))

    if len(vectors) != profile.population_size:
        raise AssertionError(f"Expected {profile.population_size} initial members, got {len(vectors)}.")
    records = _make_records(seed, vectors)
    return PopulationBundle(np.vstack([record.vector for record in records]), records, counts)


def make_baseline_population(
    profile: base.SearchProfile,
    libraries: Sequence[Sequence[base.SingleCandidate]],
    seed: int,
) -> PopulationBundle:
    """Reproduce the old Q4 initial-population rule, with source labels."""
    population = base.lhs_population(base.JOINT_BOUNDS, profile.population_size, seed + 10_000)
    seeded_count = int(round(profile.population_size * (1.0 - profile.global_fraction)))
    combinations = list(itertools.product(*libraries))
    rng = np.random.default_rng(seed + 20_000)
    rng.shuffle(combinations)
    vectors: list[tuple[np.ndarray, str, str, bool]] = []
    for row in range(seeded_count):
        combo = combinations[row % len(combinations)]
        vector = np.concatenate([item.encoded for item in combo]).astype(float)
        if row >= len(combinations):
            raise AssertionError("The standard library is expected to have enough combinations.")
        population[row] = vector
        vectors.append((vector, "single_library", "single_library", False))
    for vector in population[seeded_count:]:
        vectors.append((vector, "global_lhs", "global_lhs", False))
    counts = {"single_library": seeded_count, "global_lhs": profile.population_size - seeded_count}
    return PopulationBundle(population, _make_records(seed, vectors), counts)
