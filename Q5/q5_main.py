from __future__ import annotations

import argparse
import csv
import itertools
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
Q5_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = PROJECT_ROOT / "Resources" / "result3.xlsx"

GRAVITY = 9.8
MISSILE_SPEED = 300.0
CLOUD_RADIUS = 10.0
CLOUD_LIFETIME = 20.0
CLOUD_SINK_SPEED = 3.0
UAV_SPEED_MIN = 70.0
UAV_SPEED_MAX = 140.0
MIN_RELEASE_GAP = 1.0

MISSILE_NAMES = ("M1", "M2", "M3")
MISSILE_INITIALS = np.array(
    ((20000.0, 0.0, 2000.0), (19000.0, 600.0, 2100.0), (18000.0, -600.0, 1900.0)),
    dtype=float,
)
MISSILE_DIRECTIONS = -MISSILE_INITIALS / np.linalg.norm(MISSILE_INITIALS, axis=1)[:, None]
MISSILE_VELOCITIES = MISSILE_SPEED * MISSILE_DIRECTIONS
MISSILE_IMPACT_TIMES = np.linalg.norm(MISSILE_INITIALS, axis=1) / MISSILE_SPEED
GLOBAL_HORIZON = float(np.max(MISSILE_IMPACT_TIMES))

UAV_NAMES = ("FY1", "FY2", "FY3", "FY4", "FY5")
UAV_INITIALS = np.array(
    (
        (17800.0, 0.0, 1800.0),
        (12000.0, 1400.0, 1400.0),
        (6000.0, -3000.0, 700.0),
        (11000.0, 2000.0, 1800.0),
        (13000.0, -2000.0, 1300.0),
    ),
    dtype=float,
)
FUSE_DELAY_LIMITS = np.sqrt(2.0 * UAV_INITIALS[:, 2] / GRAVITY)

TARGET_BOTTOM_CENTER = np.array((0.0, 200.0, 0.0), dtype=float)
TARGET_CENTER = TARGET_BOTTOM_CENTER + np.array((0.0, 0.0, 5.0))
TARGET_RADIUS = 7.0
TARGET_HEIGHT = 10.0
TARGET_BOUND_RADIUS = float(np.hypot(TARGET_RADIUS, TARGET_HEIGHT / 2.0))

BOMBS_PER_UAV = 3
GROUP_DIMENSION = 2 + 2 * BOMBS_PER_UAV
DIMENSION = len(UAV_NAMES) * GROUP_DIMENSION
JOINT_BOUNDS = tuple(
    bound
    for _ in UAV_NAMES
    for bound in ((0.0, 2.0 * np.pi), (0.0, 1.0), *((0.0, 1.0),) * 6)
)


@dataclass(frozen=True)
class SearchProfile:
    name: str
    seeds: tuple[int, ...]
    single_population: int
    single_maxiter: int
    single_pool_count: int
    single_library_size: int
    routes_per_pair: int
    local_population: int
    local_maxiter: int
    local_library_size: int
    beam_width: int
    combination_review_count: int
    population_size: int
    point_maxiter: int
    surface_maxiter: int
    point_grid: tuple[int, int, int, float]
    search_grid: tuple[int, int, int, float]
    rerank_grid: tuple[int, int, int, float]
    final_grid: tuple[int, int, int, float]
    verification_grid: tuple[int, int, int, float]
    top_per_seed: int
    rerank_count: int
    final_count: int
    global_fraction: float = 0.15


PROFILES = {
    "quick": SearchProfile(
        name="quick",
        seeds=(51,),
        single_population=24,
        single_maxiter=18,
        single_pool_count=12,
        single_library_size=2,
        routes_per_pair=1,
        local_population=28,
        local_maxiter=20,
        local_library_size=1,
        beam_width=24,
        combination_review_count=12,
        population_size=48,
        point_maxiter=20,
        surface_maxiter=12,
        point_grid=(1, 1, 1, 0.20),
        search_grid=(12, 3, 2, 0.30),
        rerank_grid=(36, 5, 4, 0.08),
        final_grid=(90, 7, 6, 0.03),
        verification_grid=(180, 9, 9, 0.01),
        top_per_seed=8,
        rerank_count=8,
        final_count=3,
    ),
    "standard": SearchProfile(
        name="standard",
        seeds=(20250951, 20250952, 20250953),
        single_population=48,
        single_maxiter=70,
        single_pool_count=30,
        single_library_size=4,
        routes_per_pair=2,
        local_population=48,
        local_maxiter=70,
        local_library_size=2,
        beam_width=96,
        combination_review_count=48,
        population_size=112,
        point_maxiter=140,
        surface_maxiter=90,
        point_grid=(1, 1, 1, 0.08),
        search_grid=(18, 3, 3, 0.20),
        rerank_grid=(60, 5, 5, 0.05),
        final_grid=(180, 9, 9, 0.02),
        verification_grid=(360, 13, 11, 0.01),
        top_per_seed=16,
        rerank_count=24,
        final_count=5,
    ),
    "extensive": SearchProfile(
        name="extensive",
        seeds=(20250951, 20250952, 20250953, 20250954, 20250955),
        single_population=64,
        single_maxiter=120,
        single_pool_count=48,
        single_library_size=6,
        routes_per_pair=3,
        local_population=64,
        local_maxiter=120,
        local_library_size=3,
        beam_width=160,
        combination_review_count=80,
        population_size=160,
        point_maxiter=260,
        surface_maxiter=180,
        point_grid=(1, 1, 1, 0.05),
        search_grid=(24, 4, 4, 0.12),
        rerank_grid=(90, 7, 7, 0.03),
        final_grid=(240, 11, 11, 0.01),
        verification_grid=(480, 17, 15, 0.005),
        top_per_seed=24,
        rerank_count=40,
        final_count=8,
    ),
}


@dataclass(frozen=True)
class BombPlan:
    release_time: float
    fuse_delay: float

    @property
    def burst_time(self) -> float:
        return self.release_time + self.fuse_delay


@dataclass(frozen=True)
class UavPlan:
    name: str
    initial: tuple[float, float, float]
    heading: float
    speed: float
    bombs: tuple[BombPlan, ...]

    @property
    def direction(self) -> np.ndarray:
        return np.array((math.cos(self.heading), math.sin(self.heading), 0.0), dtype=float)

    def release_point(self, bomb: BombPlan) -> np.ndarray:
        return np.asarray(self.initial, dtype=float) + self.speed * bomb.release_time * self.direction

    def burst_point(self, bomb: BombPlan) -> np.ndarray:
        point = np.asarray(self.initial, dtype=float) + self.speed * bomb.burst_time * self.direction
        point = point.copy()
        point[2] -= 0.5 * GRAVITY * bomb.fuse_delay**2
        return point


@dataclass(frozen=True)
class Strategy:
    uavs: tuple[UavPlan, ...]


@dataclass(frozen=True)
class Evaluation:
    strategy: Strategy
    feasible: bool
    reason: str
    missile_intervals: tuple[tuple[tuple[float, float], ...], ...]
    missile_durations: tuple[float, ...]
    total_duration: float
    minimum_duration: float
    positive_count: int
    score: float
    grid: tuple[int, int, int, float]


@dataclass(frozen=True)
class SingleCandidate:
    uav_index: int
    missile_index: int
    heading: float
    speed: float
    bomb: BombPlan
    encoded: np.ndarray
    point_duration: float
    surface_duration: float


@dataclass(frozen=True)
class LocalCandidate:
    uav_index: int
    missile_index: int
    plan: UavPlan
    point_duration: float
    surface_duration: float


@dataclass(frozen=True)
class JointCandidate:
    encoded: np.ndarray
    result: Evaluation
    source: str
    seed: int


@dataclass(frozen=True)
class SearchRun:
    seed: int
    elapsed_seconds: float
    point_iterations: int
    surface_iterations: int
    evaluations: int
    best: JointCandidate
    candidates: tuple[JointCandidate, ...]
    point_history: tuple[float, ...]
    surface_history: tuple[float, ...]


def surface_points(angle_count: int, height_count: int, radial_count: int) -> np.ndarray:
    if (angle_count, height_count, radial_count) == (1, 1, 1):
        return TARGET_CENTER[None, :].copy()
    angles = np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False)
    cosines, sines = np.cos(angles), np.sin(angles)
    x0, y0, z0 = TARGET_BOTTOM_CENTER
    heights = np.linspace(z0, z0 + TARGET_HEIGHT, height_count)
    side = np.vstack(
        [
            np.column_stack(
                (
                    x0 + TARGET_RADIUS * cosines,
                    y0 + TARGET_RADIUS * sines,
                    np.full_like(angles, height),
                )
            )
            for height in heights
        ]
    )
    caps = []
    for height in (z0, z0 + TARGET_HEIGHT):
        for radius in np.linspace(0.0, TARGET_RADIUS, radial_count):
            caps.append(
                np.column_stack(
                    (
                        x0 + radius * cosines,
                        y0 + radius * sines,
                        np.full_like(angles, height),
                    )
                )
            )
    return np.vstack((side, *caps))


def decode_releases(raw: Sequence[float], horizon: float) -> np.ndarray:
    usable = max(0.0, horizon - MIN_RELEASE_GAP * (BOMBS_PER_UAV - 1))
    return np.sort(np.clip(np.asarray(raw, dtype=float), 0.0, 1.0)) * usable + np.arange(BOMBS_PER_UAV)


def decode_group(values: Sequence[float], uav_index: int, horizon: float = GLOBAL_HORIZON) -> UavPlan:
    values = np.asarray(values, dtype=float)
    if values.shape != (GROUP_DIMENSION,):
        raise ValueError("每架无人机必须包含8个编码变量。")
    heading = float(values[0] % (2.0 * np.pi))
    speed = UAV_SPEED_MIN + (UAV_SPEED_MAX - UAV_SPEED_MIN) * float(np.clip(values[1], 0.0, 1.0))
    releases = decode_releases(values[2:5], horizon)
    bombs = []
    for release, delay_unit in zip(releases, values[5:8]):
        ceiling = min(float(FUSE_DELAY_LIMITS[uav_index]), max(0.0, horizon - float(release)))
        bombs.append(BombPlan(float(release), ceiling * float(np.clip(delay_unit, 0.0, 1.0))))
    return UavPlan(
        UAV_NAMES[uav_index],
        tuple(float(item) for item in UAV_INITIALS[uav_index]),
        heading,
        speed,
        tuple(bombs),
    )


def decode_strategy(vector: Iterable[float]) -> Strategy:
    values = np.asarray(tuple(float(item) for item in vector), dtype=float)
    if values.shape != (DIMENSION,):
        raise ValueError(f"问题五编码向量必须包含{DIMENSION}个变量。")
    return Strategy(
        tuple(
            decode_group(values[GROUP_DIMENSION * i : GROUP_DIMENSION * (i + 1)], i)
            for i in range(len(UAV_NAMES))
        )
    )


def encode_plan(plan: UavPlan, uav_index: int, horizon: float = GLOBAL_HORIZON) -> np.ndarray:
    speed_unit = (plan.speed - UAV_SPEED_MIN) / (UAV_SPEED_MAX - UAV_SPEED_MIN)
    usable = max(1.0e-12, horizon - MIN_RELEASE_GAP * (BOMBS_PER_UAV - 1))
    releases = np.array([bomb.release_time for bomb in plan.bombs], dtype=float)
    raw = np.clip((releases - np.arange(BOMBS_PER_UAV)) / usable, 0.0, 1.0)
    delays = []
    for bomb in plan.bombs:
        ceiling = min(float(FUSE_DELAY_LIMITS[uav_index]), max(0.0, horizon - bomb.release_time))
        delays.append(0.0 if ceiling <= 1.0e-12 else bomb.fuse_delay / ceiling)
    return np.array((plan.heading, speed_unit, *raw, *delays), dtype=float)


def validate_strategy(strategy: Strategy) -> str | None:
    if not 1 <= len(strategy.uavs) <= len(UAV_NAMES):
        return "策略包含的无人机数量不正确。"
    seen: set[str] = set()
    for plan in strategy.uavs:
        if plan.name in seen:
            return f"无人机{plan.name}重复出现。"
        seen.add(plan.name)
        if plan.name not in UAV_NAMES:
            return f"未知无人机：{plan.name}。"
        uav_index = UAV_NAMES.index(plan.name)
        if not np.isfinite((plan.heading, plan.speed)).all():
            return f"{plan.name}航向或速度不是有限数。"
        if not 0.0 <= plan.heading < 2.0 * np.pi + 1.0e-12:
            return f"{plan.name}航向越界。"
        if not UAV_SPEED_MIN - 1.0e-10 <= plan.speed <= UAV_SPEED_MAX + 1.0e-10:
            return f"{plan.name}速度越界。"
        if not 1 <= len(plan.bombs) <= BOMBS_PER_UAV:
            return f"{plan.name}投弹数量越界。"
        releases = [bomb.release_time for bomb in plan.bombs]
        if any(not np.isfinite((bomb.release_time, bomb.fuse_delay)).all() for bomb in plan.bombs):
            return f"{plan.name}投弹时序不是有限数。"
        if any(value < -1.0e-10 for value in releases):
            return f"{plan.name}存在负投放时刻。"
        if any(right - left < MIN_RELEASE_GAP - 1.0e-9 for left, right in zip(releases[:-1], releases[1:])):
            return f"{plan.name}相邻投弹间隔小于1秒。"
        for bomb in plan.bombs:
            if bomb.fuse_delay < -1.0e-10 or bomb.burst_time > GLOBAL_HORIZON + 1.0e-9:
                return f"{plan.name}存在非法起爆时序。"
            if plan.burst_point(bomb)[2] < -1.0e-8:
                return f"{plan.name}存在地面以下起爆。"
            if bomb.fuse_delay > FUSE_DELAY_LIMITS[uav_index] + 1.0e-9:
                return f"{plan.name}引信延迟超过高度上界。"
    return None


class MultiMissileEvaluator:
    def __init__(
        self,
        angle_count: int,
        height_count: int,
        radial_count: int,
        scan_step: float,
        *,
        root_tolerance: float | None = None,
        target_points: np.ndarray | None = None,
        use_geometric_skip: bool = True,
    ) -> None:
        self.angle_count = angle_count
        self.height_count = height_count
        self.radial_count = radial_count
        self.scan_step = float(scan_step)
        self.root_tolerance = float(root_tolerance or max(1.0e-7, scan_step * 1.0e-4))
        self.points = surface_points(angle_count, height_count, radial_count) if target_points is None else np.asarray(target_points, dtype=float)
        self.use_geometric_skip = use_geometric_skip

    @property
    def grid(self) -> tuple[int, int, int, float]:
        return self.angle_count, self.height_count, self.radial_count, self.scan_step

    @staticmethod
    def flatten(strategy: Strategy) -> tuple[tuple[UavPlan, BombPlan, np.ndarray], ...]:
        return tuple(
            (plan, bomb, plan.burst_point(bomb))
            for plan in strategy.uavs
            for bomb in plan.bombs
        )

    @staticmethod
    def missile_positions(missile_index: int, times: np.ndarray) -> np.ndarray:
        return MISSILE_INITIALS[missile_index][None, :] + times[:, None] * MISSILE_VELOCITIES[missile_index][None, :]

    @staticmethod
    def active_indices(flat: Sequence[tuple[UavPlan, BombPlan, np.ndarray]], missile_index: int, time_value: float) -> tuple[int, ...]:
        impact = float(MISSILE_IMPACT_TIMES[missile_index])
        return tuple(
            index
            for index, (_, bomb, _) in enumerate(flat)
            if bomb.burst_time - 1.0e-12 <= time_value <= min(bomb.burst_time + CLOUD_LIFETIME, impact) + 1.0e-12
        )

    def _potential_mask(self, centers: np.ndarray, missiles: np.ndarray) -> np.ndarray:
        if not self.use_geometric_skip:
            return np.ones(len(centers), dtype=bool)
        direction = TARGET_CENTER[None, :] - missiles
        denominator = np.einsum("ij,ij->i", direction, direction)
        projection = np.einsum("ij,ij->i", centers - missiles, direction) / denominator
        projection = np.clip(projection, 0.0, 1.0)
        closest = missiles + projection[:, None] * direction
        distance = np.linalg.norm(centers - closest, axis=1)
        return distance <= CLOUD_RADIUS + TARGET_BOUND_RADIUS + 1.0e-10

    def status_batch(
        self,
        flat: Sequence[tuple[UavPlan, BombPlan, np.ndarray]],
        missile_index: int,
        times: np.ndarray,
        active: tuple[int, ...],
    ) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        if not active or len(times) == 0:
            return np.zeros(len(times), dtype=bool)
        missiles = self.missile_positions(missile_index, times)
        directions = self.points[None, :, :] - missiles[:, None, :]
        denominators = np.einsum("tpc,tpc->tp", directions, directions)
        minimum = np.full((len(times), len(self.points)), np.inf, dtype=float)
        for index in active:
            _, bomb, burst_point = flat[index]
            centers = np.repeat(burst_point[None, :], len(times), axis=0)
            centers[:, 2] -= CLOUD_SINK_SPEED * (times - bomb.burst_time)
            useful = self._potential_mask(centers, missiles)
            if not np.any(useful):
                continue
            selected = np.flatnonzero(useful)
            selected_directions = directions[selected]
            selected_missiles = missiles[selected]
            selected_centers = centers[selected]
            projection = np.einsum(
                "tc,tpc->tp", selected_centers - selected_missiles, selected_directions
            ) / denominators[selected]
            projection = np.clip(projection, 0.0, 1.0)
            closest = selected_missiles[:, None, :] + projection[:, :, None] * selected_directions
            distances = np.linalg.norm(closest - selected_centers[:, None, :], axis=2)
            minimum[selected] = np.minimum(minimum[selected], distances)
        return np.max(minimum, axis=1) <= CLOUD_RADIUS + 1.0e-12

    def status(self, flat, missile_index: int, time_value: float, active: tuple[int, ...]) -> bool:
        return bool(self.status_batch(flat, missile_index, np.array((time_value,)), active)[0])

    def event_segments(self, flat, missile_index: int):
        impact = float(MISSILE_IMPACT_TIMES[missile_index])
        events = {0.0, impact}
        for _, bomb, _ in flat:
            events.add(float(np.clip(bomb.burst_time, 0.0, impact)))
            events.add(float(np.clip(bomb.burst_time + CLOUD_LIFETIME, 0.0, impact)))
        ordered = sorted(events)
        segments = []
        for left, right in zip(ordered[:-1], ordered[1:]):
            if right - left <= 1.0e-12:
                continue
            active = self.active_indices(flat, missile_index, 0.5 * (left + right))
            segments.append((left, right, active))
        return segments

    def segment_times(self, left: float, right: float) -> np.ndarray:
        times = np.arange(left, right, self.scan_step, dtype=float)
        if times.size == 0 or abs(times[-1] - right) > 1.0e-12:
            times = np.append(times, right)
        return times

    def bisect_boundary(self, flat, missile_index: int, left: float, right: float, active: tuple[int, ...], left_valid: bool) -> float:
        while right - left > self.root_tolerance:
            middle = 0.5 * (left + right)
            valid = self.status(flat, missile_index, middle, active)
            if valid == left_valid:
                left = middle
            else:
                right = middle
        return 0.5 * (left + right)

    @staticmethod
    def merge_intervals(intervals: Iterable[tuple[float, float]], tolerance: float = 1.0e-8) -> tuple[tuple[float, float], ...]:
        ordered = sorted(intervals)
        if not ordered:
            return ()
        merged = [list(ordered[0])]
        for start, end in ordered[1:]:
            if start <= merged[-1][1] + tolerance:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return tuple((float(start), float(end)) for start, end in merged)

    def missile_intervals(self, flat, missile_index: int) -> tuple[tuple[float, float], ...]:
        intervals: list[tuple[float, float]] = []
        for left, right, active in self.event_segments(flat, missile_index):
            if not active:
                continue
            times = self.segment_times(left, right)
            valid = self.status_batch(flat, missile_index, times, active)
            current_start = float(times[0]) if bool(valid[0]) else None
            for position in range(1, len(times)):
                if bool(valid[position]) == bool(valid[position - 1]):
                    continue
                boundary = self.bisect_boundary(
                    flat,
                    missile_index,
                    float(times[position - 1]),
                    float(times[position]),
                    active,
                    bool(valid[position - 1]),
                )
                if bool(valid[position]):
                    current_start = boundary
                elif current_start is not None:
                    intervals.append((current_start, boundary))
                    current_start = None
            if current_start is not None:
                intervals.append((current_start, right))
        return self.merge_intervals(intervals)

    def evaluate(self, strategy: Strategy) -> Evaluation:
        reason = validate_strategy(strategy)
        if reason is not None:
            return Evaluation(strategy, False, reason, ((), (), ()), (0.0, 0.0, 0.0), 0.0, 0.0, 0, -1.0e6, self.grid)
        flat = self.flatten(strategy)
        intervals = tuple(self.missile_intervals(flat, index) for index in range(len(MISSILE_NAMES)))
        durations = tuple(float(sum(end - start for start, end in values)) for values in intervals)
        total = float(sum(durations))
        minimum = float(min(durations))
        positive = sum(value > 1.0e-8 for value in durations)
        # 总导弹秒为主目标；最差导弹时长和已覆盖导弹数只用于弱化并列与搜索平坦区。
        score = total + 0.05 * minimum + 0.01 * positive
        return Evaluation(strategy, True, "", intervals, durations, total, minimum, positive, score, self.grid)


class JointObjective:
    def __init__(self, evaluator: MultiMissileEvaluator) -> None:
        self.evaluator = evaluator
        self.cache: dict[tuple[float, ...], Evaluation] = {}

    @staticmethod
    def key(vector: Iterable[float]) -> tuple[float, ...]:
        return tuple(round(float(item), 9) for item in vector)

    def result_for(self, vector: Iterable[float]) -> Evaluation:
        key = self.key(vector)
        if key not in self.cache:
            self.cache[key] = self.evaluator.evaluate(decode_strategy(key))
        return self.cache[key]

    def __call__(self, vector: np.ndarray) -> float:
        result = self.result_for(vector)
        return -result.score if result.feasible else 1.0e6


def lhs_population(bounds: Sequence[tuple[float, float]], count: int, seed: int) -> np.ndarray:
    unit = qmc.LatinHypercube(d=len(bounds), seed=seed).random(count)
    lower = np.array([item[0] for item in bounds], dtype=float)
    upper = np.array([item[1] for item in bounds], dtype=float)
    return lower + unit * (upper - lower)


def decode_single(vector: Sequence[float], uav_index: int, missile_index: int) -> tuple[float, float, BombPlan]:
    heading, speed_unit, burst_time, delay_unit = map(float, vector)
    heading %= 2.0 * np.pi
    speed = UAV_SPEED_MIN + (UAV_SPEED_MAX - UAV_SPEED_MIN) * float(np.clip(speed_unit, 0.0, 1.0))
    burst_time = float(np.clip(burst_time, 0.0, MISSILE_IMPACT_TIMES[missile_index]))
    ceiling = min(float(FUSE_DELAY_LIMITS[uav_index]), burst_time)
    delay = ceiling * float(np.clip(delay_unit, 0.0, 1.0))
    return heading, speed, BombPlan(burst_time - delay, delay)


def single_strategy(vector: Sequence[float], uav_index: int, missile_index: int) -> Strategy:
    heading, speed, bomb = decode_single(vector, uav_index, missile_index)
    return Strategy((UavPlan(UAV_NAMES[uav_index], tuple(UAV_INITIALS[uav_index]), heading, speed, (bomb,)),))


def single_duration(evaluator: MultiMissileEvaluator, vector: Sequence[float], uav_index: int, missile_index: int) -> float:
    result = evaluator.evaluate(single_strategy(vector, uav_index, missile_index))
    return result.missile_durations[missile_index] if result.feasible else 0.0


def geometric_single_anchors(uav_index: int, missile_index: int) -> list[np.ndarray]:
    initial = UAV_INITIALS[uav_index]
    impact = float(MISSILE_IMPACT_TIMES[missile_index])
    anchors: list[np.ndarray] = []
    for burst_time in np.linspace(max(0.25, 0.01 * impact), impact, 64):
        missile = MISSILE_INITIALS[missile_index] + burst_time * MISSILE_VELOCITIES[missile_index]
        for sight in np.linspace(0.04, 0.98, 64):
            desired = missile + sight * (TARGET_CENTER - missile)
            drop = initial[2] - desired[2]
            if drop < -1.0e-10:
                continue
            delay = math.sqrt(max(0.0, 2.0 * drop / GRAVITY))
            if delay > min(burst_time, FUSE_DELAY_LIMITS[uav_index]) + 1.0e-10:
                continue
            displacement = desired[:2] - initial[:2]
            distance = float(np.linalg.norm(displacement))
            speed = distance / burst_time
            if not UAV_SPEED_MIN <= speed <= UAV_SPEED_MAX:
                continue
            heading = math.atan2(displacement[1], displacement[0]) % (2.0 * np.pi)
            speed_unit = (speed - UAV_SPEED_MIN) / (UAV_SPEED_MAX - UAV_SPEED_MIN)
            ceiling = min(float(FUSE_DELAY_LIMITS[uav_index]), burst_time)
            anchors.append(np.array((heading, speed_unit, burst_time, delay / ceiling if ceiling else 0.0)))
    return anchors


def angular_distance(left: float, right: float) -> float:
    return abs((left - right + np.pi) % (2.0 * np.pi) - np.pi)


def build_single_library(profile: SearchProfile, uav_index: int, missile_index: int, seed: int) -> tuple[SingleCandidate, ...]:
    point = MultiMissileEvaluator(*profile.point_grid)
    low_surface = MultiMissileEvaluator(24 if profile.name != "quick" else 12, 3, 3, 0.10)
    impact = float(MISSILE_IMPACT_TIMES[missile_index])
    bounds = ((0.0, 2.0 * np.pi), (0.0, 1.0), (0.0, impact), (0.0, 1.0))
    cache: dict[tuple[float, ...], float] = {}

    def objective(vector: np.ndarray) -> float:
        key = tuple(round(float(item), 9) for item in vector)
        if key not in cache:
            cache[key] = single_duration(point, key, uav_index, missile_index)
        return -cache[key]

    initial = lhs_population(bounds, profile.single_population, seed)
    anchors = geometric_single_anchors(uav_index, missile_index)
    anchor_records = sorted(((single_duration(point, item, uav_index, missile_index), item) for item in anchors), key=lambda item: -item[0])
    anchor_limit = min(len(anchor_records), max(2, profile.single_population // 2))
    if anchor_limit:
        initial[:anchor_limit] = np.vstack([item[1] for item in anchor_records[:anchor_limit]])
    result = differential_evolution(
        objective,
        bounds=bounds,
        init=initial,
        maxiter=profile.single_maxiter,
        mutation=(0.5, 1.0),
        recombination=0.9,
        seed=seed,
        polish=False,
        tol=1.0e-6,
        atol=0.0,
        updating="immediate",
        workers=1,
    )
    population = np.asarray(result.population, dtype=float)
    vectors = [population[index].copy() for index in np.argsort(result.population_energies)]
    vectors.extend(item[1] for item in anchor_records[: profile.single_pool_count])
    vectors.sort(key=objective)
    candidates = []
    for vector in vectors[: profile.single_pool_count]:
        heading, speed, bomb = decode_single(vector, uav_index, missile_index)
        plan = UavPlan(UAV_NAMES[uav_index], tuple(UAV_INITIALS[uav_index]), heading, speed, (bomb,))
        surface = low_surface.evaluate(Strategy((plan,)))
        candidates.append(
            SingleCandidate(
                uav_index,
                missile_index,
                heading,
                speed,
                bomb,
                np.asarray(vector).copy(),
                single_duration(point, vector, uav_index, missile_index),
                surface.missile_durations[missile_index],
            )
        )
    candidates.sort(key=lambda item: (-item.surface_duration, -item.point_duration))
    selected: list[SingleCandidate] = []
    for candidate in candidates:
        if not selected or all(
            math.hypot(
                angular_distance(candidate.heading, other.heading) / np.pi,
                (candidate.speed - other.speed) / (UAV_SPEED_MAX - UAV_SPEED_MIN),
            ) >= 0.015
            for other in selected
        ):
            selected.append(candidate)
        if len(selected) >= profile.single_library_size:
            break
    if not selected:
        raise RuntimeError(f"{UAV_NAMES[uav_index]}-{MISSILE_NAMES[missile_index]}未生成单弹候选。")
    return tuple(selected)


def timing_plan(vector: Sequence[float], uav_index: int, missile_index: int, heading: float, speed: float) -> UavPlan:
    horizon = float(MISSILE_IMPACT_TIMES[missile_index])
    releases = decode_releases(vector[:3], horizon)
    bombs = []
    for release, delay_unit in zip(releases, vector[3:6]):
        ceiling = min(float(FUSE_DELAY_LIMITS[uav_index]), max(0.0, horizon - release))
        bombs.append(BombPlan(float(release), ceiling * float(np.clip(delay_unit, 0.0, 1.0))))
    return UavPlan(UAV_NAMES[uav_index], tuple(UAV_INITIALS[uav_index]), heading, speed, tuple(bombs))


def seed_timing_from_single(candidate: SingleCandidate) -> np.ndarray:
    horizon = float(MISSILE_IMPACT_TIMES[candidate.missile_index])
    center = candidate.bomb.release_time
    releases = np.array((max(0.0, center - 1.0), center, min(horizon, center + 1.0)))
    releases.sort()
    releases[1] = max(releases[1], releases[0] + 1.0)
    releases[2] = max(releases[2], releases[1] + 1.0)
    if releases[2] > horizon:
        releases -= releases[2] - horizon
    releases = np.maximum(releases, np.arange(3, dtype=float))
    usable = max(1.0e-12, horizon - 2.0)
    raw = np.clip((releases - np.arange(3)) / usable, 0.0, 1.0)
    delays = []
    for release in releases:
        ceiling = min(float(FUSE_DELAY_LIMITS[candidate.uav_index]), max(0.0, horizon - release))
        delays.append(0.0 if ceiling <= 1.0e-12 else min(candidate.bomb.fuse_delay, ceiling) / ceiling)
    return np.array((*raw, *delays), dtype=float)


def build_local_candidates(
    profile: SearchProfile,
    singles: Sequence[SingleCandidate],
    uav_index: int,
    missile_index: int,
    seed: int,
) -> tuple[LocalCandidate, ...]:
    point = MultiMissileEvaluator(*profile.point_grid)
    low_surface = MultiMissileEvaluator(18 if profile.name != "quick" else 12, 3, 3, 0.12)
    produced: list[LocalCandidate] = []
    for route_rank, route in enumerate(singles[: profile.routes_per_pair]):
        cache: dict[tuple[float, ...], float] = {}

        def objective(vector: np.ndarray) -> float:
            key = tuple(round(float(item), 9) for item in vector)
            if key not in cache:
                plan = timing_plan(key, uav_index, missile_index, route.heading, route.speed)
                cache[key] = point.evaluate(Strategy((plan,))).missile_durations[missile_index]
            return -cache[key]

        bounds = ((0.0, 1.0),) * 6
        initial = lhs_population(bounds, profile.local_population, seed + route_rank)
        initial[0] = seed_timing_from_single(route)
        result = differential_evolution(
            objective,
            bounds=bounds,
            init=initial,
            maxiter=profile.local_maxiter,
            mutation=(0.5, 1.0),
            recombination=0.9,
            seed=seed + route_rank,
            polish=False,
            tol=1.0e-6,
            atol=0.0,
            updating="immediate",
            workers=1,
        )
        population = np.asarray(result.population, dtype=float)
        for index in np.argsort(result.population_energies)[: profile.local_library_size]:
            vector = population[index]
            plan = timing_plan(vector, uav_index, missile_index, route.heading, route.speed)
            point_duration = point.evaluate(Strategy((plan,))).missile_durations[missile_index]
            surface_duration = low_surface.evaluate(Strategy((plan,))).missile_durations[missile_index]
            produced.append(LocalCandidate(uav_index, missile_index, plan, point_duration, surface_duration))
    produced.sort(key=lambda item: (-item.surface_duration, -item.point_duration))
    return tuple(produced[: max(profile.local_library_size, profile.routes_per_pair)])


def schedule_combinations(profile: SearchProfile, local_by_uav: Sequence[Sequence[LocalCandidate]]) -> list[JointCandidate]:
    point = MultiMissileEvaluator(*profile.point_grid)
    beams: list[tuple[Strategy, tuple[int, ...], Evaluation]] = [(Strategy(()), (), point.evaluate(Strategy((local_by_uav[0][0].plan,))))]
    # 上面的占位评价不会用于首轮排序；首轮会完全替换它。
    for uav_index, options in enumerate(local_by_uav):
        expanded = []
        for partial, labels, _ in beams:
            for option in options:
                strategy = Strategy((*partial.uavs, option.plan))
                result = point.evaluate(strategy)
                expanded.append((strategy, (*labels, option.missile_index), result))
        expanded.sort(key=lambda item: (-item[2].positive_count, -item[2].score))
        diverse = []
        per_mask: dict[int, int] = {}
        quota = max(2, profile.beam_width // 8)
        for item in expanded:
            mask = sum(1 << label for label in set(item[1]))
            if per_mask.get(mask, 0) >= quota and len(diverse) >= profile.beam_width // 2:
                continue
            diverse.append(item)
            per_mask[mask] = per_mask.get(mask, 0) + 1
            if len(diverse) >= profile.beam_width:
                break
        beams = diverse
        if not beams:
            raise RuntimeError(f"组合调度在加入{UAV_NAMES[uav_index]}后没有候选。")
    coarse = MultiMissileEvaluator(*profile.search_grid)
    candidates = []
    for rank, (strategy, _, _) in enumerate(beams[: profile.combination_review_count]):
        encoded = np.concatenate([encode_plan(plan, index) for index, plan in enumerate(strategy.uavs)])
        result = coarse.evaluate(strategy)
        candidates.append(JointCandidate(encoded, result, "beam_schedule", -rank - 1))
    return sorted(candidates, key=lambda item: (-item.result.positive_count, -item.result.score))


def build_joint_initial_population(profile: SearchProfile, scheduled: Sequence[JointCandidate], seed: int) -> np.ndarray:
    initial = lhs_population(JOINT_BOUNDS, profile.population_size, seed + 10_000)
    seeded_count = int(round(profile.population_size * (1.0 - profile.global_fraction)))
    rng = np.random.default_rng(seed + 20_000)
    for row in range(seeded_count):
        vector = scheduled[row % len(scheduled)].encoded.copy()
        if row >= len(scheduled):
            for uav_index in range(len(UAV_NAMES)):
                start = GROUP_DIMENSION * uav_index
                vector[start] = (vector[start] + rng.normal(0.0, 0.05)) % (2.0 * np.pi)
                stop = start + GROUP_DIMENSION
                vector[start + 1 : stop] = np.clip(
                    vector[start + 1 : stop]
                    + rng.normal(0.0, 0.035, GROUP_DIMENSION - 1),
                    0.0,
                    1.0,
                )
        initial[row] = vector
    return initial


def run_joint_seed(profile: SearchProfile, scheduled: Sequence[JointCandidate], seed: int, workers: int) -> SearchRun:
    initial = build_joint_initial_population(profile, scheduled, seed)
    point_evaluator = MultiMissileEvaluator(*profile.point_grid)
    point_objective = JointObjective(point_evaluator)
    point_history: list[float] = []

    def point_callback(vector: np.ndarray, convergence: float) -> bool:
        del convergence
        score = point_objective.result_for(vector).score
        point_history.append(max(point_history[-1], score) if point_history else score)
        return False

    started = time.perf_counter()
    point_result = differential_evolution(
        point_objective,
        bounds=JOINT_BOUNDS,
        init=initial,
        maxiter=profile.point_maxiter,
        mutation=(0.45, 1.0),
        recombination=0.9,
        seed=seed,
        polish=False,
        tol=1.0e-6,
        atol=0.0,
        callback=point_callback,
        updating="immediate" if workers == 1 else "deferred",
        workers=workers,
    )
    point_population = np.asarray(point_result.population, dtype=float)
    point_order = np.argsort(point_result.population_energies)
    surface_initial = point_population[point_order].copy()
    for row, candidate in enumerate(scheduled[: min(len(scheduled), len(surface_initial) // 3)]):
        surface_initial[-row - 1] = candidate.encoded

    surface_evaluator = MultiMissileEvaluator(*profile.search_grid)
    surface_objective = JointObjective(surface_evaluator)
    surface_history: list[float] = []

    def surface_callback(vector: np.ndarray, convergence: float) -> bool:
        del convergence
        score = surface_objective.result_for(vector).score
        surface_history.append(max(surface_history[-1], score) if surface_history else score)
        return False

    surface_result = differential_evolution(
        surface_objective,
        bounds=JOINT_BOUNDS,
        init=surface_initial,
        maxiter=profile.surface_maxiter,
        mutation=(0.45, 1.0),
        recombination=0.9,
        seed=seed + 1_000,
        polish=False,
        tol=1.0e-6,
        atol=0.0,
        callback=surface_callback,
        updating="immediate" if workers == 1 else "deferred",
        workers=workers,
    )
    elapsed = time.perf_counter() - started
    population = np.asarray(surface_result.population, dtype=float)
    order = np.argsort(surface_result.population_energies)[: profile.top_per_seed]
    candidates = tuple(
        JointCandidate(
            population[index].copy(),
            surface_objective.result_for(population[index]),
            "surface_de_population",
            seed,
        )
        for index in order
    )
    best = JointCandidate(np.asarray(surface_result.x).copy(), surface_objective.result_for(surface_result.x), "surface_de_best", seed)
    return SearchRun(
        seed,
        elapsed,
        int(point_result.nit),
        int(surface_result.nit),
        int(point_result.nfev + surface_result.nfev),
        best,
        candidates,
        tuple(point_history),
        tuple(surface_history),
    )


def candidate_key(candidate: JointCandidate) -> tuple[float, ...]:
    return tuple(round(float(item), 7) for item in candidate.encoded)


def deduplicate(candidates: Iterable[JointCandidate]) -> list[JointCandidate]:
    unique: dict[tuple[float, ...], JointCandidate] = {}
    for candidate in candidates:
        key = candidate_key(candidate)
        current = unique.get(key)
        if current is None or candidate.result.score > current.result.score:
            unique[key] = candidate
    return sorted(unique.values(), key=lambda item: (-item.result.positive_count, -item.result.score))


def rerank(candidates: Sequence[JointCandidate], grid: tuple[int, int, int, float], count: int, source: str) -> list[JointCandidate]:
    evaluator = MultiMissileEvaluator(*grid)
    reviewed = [
        JointCandidate(item.encoded, evaluator.evaluate(decode_strategy(item.encoded)), source, item.seed)
        for item in candidates[:count]
    ]
    return sorted(reviewed, key=lambda item: (-item.result.positive_count, -item.result.score))


def format_intervals(intervals: Iterable[tuple[float, float]]) -> str:
    return "; ".join(f"[{start:.10f}, {end:.10f}]" for start, end in intervals)


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"没有可写入{path.name}的记录。")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def individual_bomb_durations(strategy: Strategy, evaluator: MultiMissileEvaluator) -> list[tuple[int, tuple[float, ...]]]:
    rows = []
    for uav_index, plan in enumerate(strategy.uavs):
        for bomb in plan.bombs:
            single = Strategy((UavPlan(plan.name, plan.initial, plan.heading, plan.speed, (bomb,)),))
            result = evaluator.evaluate(single)
            rows.append((uav_index, result.missile_durations))
    return rows


def save_outputs(
    profile: SearchProfile,
    singles: Sequence[Sequence[Sequence[SingleCandidate]]],
    locals_: Sequence[Sequence[LocalCandidate]],
    scheduled: Sequence[JointCandidate],
    runs: Sequence[SearchRun],
    finalists: Sequence[JointCandidate],
    verification: Evaluation,
) -> tuple[Path, ...]:
    Q5_DIR.mkdir(parents=True, exist_ok=True)
    single_rows = []
    for uav_index in range(len(UAV_NAMES)):
        for missile_index in range(len(MISSILE_NAMES)):
            for rank, item in enumerate(singles[uav_index][missile_index], start=1):
                single_rows.append({
                    "profile": profile.name, "uav": UAV_NAMES[uav_index], "missile": MISSILE_NAMES[missile_index], "rank": rank,
                    "heading_deg": np.degrees(item.heading), "speed_mps": item.speed,
                    "release_time_s": item.bomb.release_time, "fuse_delay_s": item.bomb.fuse_delay,
                    "burst_time_s": item.bomb.burst_time, "point_duration_s": item.point_duration,
                    "surface_duration_s": item.surface_duration,
                })
    local_rows = []
    for uav_index, items in enumerate(locals_):
        for rank, item in enumerate(items, start=1):
            local_rows.append({
                "profile": profile.name, "uav": UAV_NAMES[uav_index], "planned_missile": MISSILE_NAMES[item.missile_index], "rank": rank,
                "heading_deg": np.degrees(item.plan.heading), "speed_mps": item.plan.speed,
                "release_times_s": "; ".join(f"{bomb.release_time:.10f}" for bomb in item.plan.bombs),
                "fuse_delays_s": "; ".join(f"{bomb.fuse_delay:.10f}" for bomb in item.plan.bombs),
                "burst_times_s": "; ".join(f"{bomb.burst_time:.10f}" for bomb in item.plan.bombs),
                "point_duration_s": item.point_duration, "surface_duration_s": item.surface_duration,
            })
    schedule_rows = [
        {
            "rank": rank, "positive_missiles": item.result.positive_count, "score": item.result.score,
            "total_duration_s": item.result.total_duration, "minimum_duration_s": item.result.minimum_duration,
            **{f"{MISSILE_NAMES[m]}_duration_s": item.result.missile_durations[m] for m in range(3)},
        }
        for rank, item in enumerate(scheduled, start=1)
    ]
    search_rows = [
        {
            "profile": profile.name, "seed": run.seed, "elapsed_seconds": run.elapsed_seconds,
            "point_iterations": run.point_iterations, "surface_iterations": run.surface_iterations,
            "function_evaluations": run.evaluations, "positive_missiles": run.best.result.positive_count,
            "score": run.best.result.score, "total_duration_s": run.best.result.total_duration,
            "minimum_duration_s": run.best.result.minimum_duration,
            **{f"{MISSILE_NAMES[m]}_duration_s": run.best.result.missile_durations[m] for m in range(3)},
        }
        for run in runs
    ]
    history_rows = []
    for run in runs:
        history_rows.extend(
            {"profile": profile.name, "seed": run.seed, "stage": "point", "generation": index, "best_score": value}
            for index, value in enumerate(run.point_history, start=1)
        )
        history_rows.extend(
            {"profile": profile.name, "seed": run.seed, "stage": "surface", "generation": index, "best_score": value}
            for index, value in enumerate(run.surface_history, start=1)
        )
    review_evaluator = MultiMissileEvaluator(*profile.verification_grid)
    bomb_stats = individual_bomb_durations(verification.strategy, review_evaluator)
    best_rows = []
    flat_index = 0
    for uav_index, plan in enumerate(verification.strategy.uavs):
        for bomb_index, bomb in enumerate(plan.bombs, start=1):
            release = plan.release_point(bomb)
            burst = plan.burst_point(bomb)
            durations = bomb_stats[flat_index][1]
            planned = int(np.argmax(durations))
            best_rows.append({
                "profile": profile.name, "uav": plan.name, "bomb": bomb_index,
                "heading_rad": plan.heading, "heading_deg": np.degrees(plan.heading), "speed_mps": plan.speed,
                "release_time_s": bomb.release_time, "fuse_delay_s": bomb.fuse_delay, "burst_time_s": bomb.burst_time,
                "release_x_m": release[0], "release_y_m": release[1], "release_z_m": release[2],
                "burst_x_m": burst[0], "burst_y_m": burst[1], "burst_z_m": burst[2],
                "planned_missile": MISSILE_NAMES[planned], "individual_duration_s": durations[planned],
                "M1_individual_s": durations[0], "M2_individual_s": durations[1], "M3_individual_s": durations[2],
                "joint_M1_s": verification.missile_durations[0], "joint_M2_s": verification.missile_durations[1],
                "joint_M3_s": verification.missile_durations[2], "joint_total_s": verification.total_duration,
                "minimum_missile_s": verification.minimum_duration,
                "verification_grid": f"{verification.grid[0]}x{verification.grid[1]}x{verification.grid[2]}@{verification.grid[3]}",
            })
            flat_index += 1
    interval_rows = [
        {"missile": MISSILE_NAMES[m], "duration_s": verification.missile_durations[m], "intervals_s": format_intervals(verification.missile_intervals[m])}
        for m in range(3)
    ]
    finalist_rows = [
        {"rank": rank, "source": item.source, "seed": item.seed, "score": item.result.score,
         "total_duration_s": item.result.total_duration, "minimum_duration_s": item.result.minimum_duration,
         **{f"{MISSILE_NAMES[m]}_duration_s": item.result.missile_durations[m] for m in range(3)}}
        for rank, item in enumerate(finalists, start=1)
    ]
    paths = (
        Q5_DIR / "q5_single_library.csv",
        Q5_DIR / "q5_local_library.csv",
        Q5_DIR / "q5_schedule_candidates.csv",
        Q5_DIR / "q5_search_runs.csv",
        Q5_DIR / "q5_de_history.csv",
        Q5_DIR / "q5_finalists.csv",
        Q5_DIR / "q5_best_solution.csv",
        Q5_DIR / "q5_missile_intervals.csv",
    )
    for path, rows in zip(paths, (single_rows, local_rows, schedule_rows, search_rows, history_rows, finalist_rows, best_rows, interval_rows)):
        write_csv(path, rows)
    return paths


def run_regression_checks() -> None:
    expected = np.array((66.9991708075, 63.7503812629, 60.3664734008))
    if not np.allclose(MISSILE_IMPACT_TIMES, expected, atol=2.0e-8):
        raise AssertionError(f"导弹命中时刻回归失败：{MISSILE_IMPACT_TIMES}")
    rng = np.random.default_rng(20250950)
    for vector in rng.random((100, DIMENSION)):
        for uav_index in range(len(UAV_NAMES)):
            vector[GROUP_DIMENSION * uav_index] *= 2.0 * np.pi
        strategy = decode_strategy(vector)
        reason = validate_strategy(strategy)
        if reason is not None:
            raise AssertionError(f"可行化解码失败：{reason}")
    q1 = Strategy((UavPlan("FY1", tuple(UAV_INITIALS[0]), math.pi, 120.0, (BombPlan(1.5, 3.6),)),))
    result = MultiMissileEvaluator(180, 9, 9, 0.02).evaluate(q1)
    if abs(result.missile_durations[0] - 1.3916426693) > 2.0e-3:
        raise AssertionError(f"Q1对M1回归失败：{result.missile_durations[0]:.10f}s")


def solve(profile: SearchProfile, workers: int):
    print("正在执行问题一和物理约束回归……", flush=True)
    run_regression_checks()
    singles: list[list[tuple[SingleCandidate, ...]]] = []
    for uav_index in range(len(UAV_NAMES)):
        row = []
        for missile_index in range(len(MISSILE_NAMES)):
            print(f"正在生成{UAV_NAMES[uav_index]}-{MISSILE_NAMES[missile_index]}单弹候选库……", flush=True)
            library = build_single_library(profile, uav_index, missile_index, 20251000 + 10 * uav_index + missile_index)
            row.append(library)
            print(f"  保留{len(library)}个候选，最佳低密度完整表面时长{library[0].surface_duration:.6f}s。", flush=True)
        singles.append(row)
    local_by_uav: list[tuple[LocalCandidate, ...]] = []
    for uav_index in range(len(UAV_NAMES)):
        local_items = []
        for missile_index in range(len(MISSILE_NAMES)):
            print(f"正在优化{UAV_NAMES[uav_index]}面向{MISSILE_NAMES[missile_index]}的三弹共享轨迹策略……", flush=True)
            local_items.extend(build_local_candidates(profile, singles[uav_index][missile_index], uav_index, missile_index, 20252000 + 10 * uav_index + missile_index))
        local_items.sort(key=lambda item: (-item.surface_duration, -item.point_duration))
        local_by_uav.append(tuple(local_items))
    print("正在执行五架无人机策略库的束搜索组合调度……", flush=True)
    scheduled = schedule_combinations(profile, local_by_uav)
    print(f"  保留{len(scheduled)}个组合；最佳粗网格总时长{scheduled[0].result.total_duration:.6f}s，分项{scheduled[0].result.missile_durations}。", flush=True)
    runs = []
    for seed in profile.seeds:
        print(f"正在执行40维点目标长代数定位与完整表面联合精修：seed={seed}……", flush=True)
        run = run_joint_seed(profile, scheduled, seed, workers)
        runs.append(run)
        print(f"  完成：{run.elapsed_seconds:.1f}s，粗网格总时长{run.best.result.total_duration:.6f}s，分项{run.best.result.missile_durations}。", flush=True)
    pool = deduplicate(itertools.chain(scheduled, *(itertools.chain((run.best,), run.candidates) for run in runs)))
    print("正在执行中密度候选重排……", flush=True)
    middle = rerank(pool, profile.rerank_grid, profile.rerank_count, "middle_rerank")
    print("正在执行高密度完整表面复核……", flush=True)
    finalists = rerank(middle, profile.final_grid, profile.final_count, "final_review")
    if not finalists:
        raise RuntimeError("没有候选通过高密度复核。")
    print("正在执行独立加密交叉复核……", flush=True)
    verification = MultiMissileEvaluator(*profile.verification_grid).evaluate(decode_strategy(finalists[0].encoded))
    return tuple(tuple(row) for row in singles), tuple(local_by_uav), scheduled, runs, finalists, verification


def print_result(result: Evaluation) -> None:
    print("\n问题五推荐方案（完整圆柱、三导弹联合评价）")
    for plan in result.strategy.uavs:
        print(f"  {plan.name}: 航向{np.degrees(plan.heading):.8f}°，速度{plan.speed:.8f}m/s")
        for index, bomb in enumerate(plan.bombs, start=1):
            print(f"    弹{index}: 投放{bomb.release_time:.8f}s，延迟{bomb.fuse_delay:.8f}s，起爆{bomb.burst_time:.8f}s")
    for missile_index, name in enumerate(MISSILE_NAMES):
        print(f"  {name}: {format_intervals(result.missile_intervals[missile_index])}，合计{result.missile_durations[missile_index]:.10f}s")
    print(f"  三枚导弹遮蔽总时长：{result.total_duration:.10f}导弹·秒")
    print(f"  最差导弹遮蔽时长：{result.minimum_duration:.10f}s")
    print(f"  复核网格：{result.grid[0]}×{result.grid[1]}×{result.grid[2]}，时间步长{result.grid[3]}s")


def parse_seeds(text: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("随机种子必须为逗号分隔的整数。") from error
    if not seeds:
        raise argparse.ArgumentTypeError("至少需要一个随机种子。")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="问题五分层候选库、组合调度与40维联合差分进化")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="standard")
    parser.add_argument("--seeds", type=parse_seeds, help="覆盖配置随机种子")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-workbook", action="store_true", help="仅输出CSV，不调用结果工作簿写入器")
    arguments = parser.parse_args()
    if arguments.workers == 0 or arguments.workers < -1:
        parser.error("workers必须为-1或正整数。")
    profile = PROFILES[arguments.profile]
    if arguments.seeds is not None:
        profile = replace(profile, seeds=arguments.seeds)
    started = time.perf_counter()
    singles, locals_, scheduled, runs, finalists, verification = solve(profile, arguments.workers)
    print_result(verification)
    paths = save_outputs(profile, singles, locals_, scheduled, runs, finalists, verification)
    for path in paths:
        print(f"结果已保存至：{path}")
    if not arguments.skip_workbook:
        contribution_path = Q5_DIR / "q5_bomb_contributions.csv"
        subprocess.run(
            [
                sys.executable,
                str(Q5_DIR / "q5_review_existing.py"),
                "--input",
                str(Q5_DIR / "q5_best_solution.csv"),
                "--output",
                str(contribution_path),
                "--angles",
                str(verification.grid[0]),
                "--heights",
                str(verification.grid[1]),
                "--radials",
                str(verification.grid[2]),
                "--step",
                str(verification.grid[3]),
            ],
            check=True,
        )
        writer = Q5_DIR / "q5_write_result.mjs"
        if writer.exists():
            node_executable = shutil.which("node")
            if node_executable is None:
                bundled_node = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "node" / "bin" / "node.exe"
                if bundled_node.exists():
                    node_executable = str(bundled_node)
            if node_executable is None:
                raise FileNotFoundError("未找到Node.js，无法调用q5_write_result.mjs生成result3.xlsx。")
            subprocess.run(
                [
                    node_executable,
                    str(writer),
                    str(TEMPLATE_PATH),
                    str(Q5_DIR / "q5_best_solution.csv"),
                    str(contribution_path),
                    str(Q5_DIR / "result3.xlsx"),
                    str(Q5_DIR / "q5_result3_preview.png"),
                ],
                check=True,
            )
        else:
            print("未找到q5_write_result.mjs，已保留CSV结果。", flush=True)
    print(f"总运行时间：{time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
