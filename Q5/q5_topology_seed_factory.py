from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .q5_main import (
        BOMBS_PER_UAV,
        FUSE_DELAY_LIMITS,
        GLOBAL_HORIZON,
        MIN_RELEASE_GAP,
        Strategy,
        UAV_INITIALS,
        UAV_NAMES,
        UAV_SPEED_MAX,
        UAV_SPEED_MIN,
        BombPlan,
        UavPlan,
        validate_strategy,
    )
except ImportError:
    from q5_main import (
        BOMBS_PER_UAV,
        FUSE_DELAY_LIMITS,
        GLOBAL_HORIZON,
        MIN_RELEASE_GAP,
        Strategy,
        UAV_INITIALS,
        UAV_NAMES,
        UAV_SPEED_MAX,
        UAV_SPEED_MIN,
        BombPlan,
        UavPlan,
        validate_strategy,
    )


@dataclass(frozen=True)
class SeededPlan:
    uav_index: int
    plan: UavPlan
    source: str


# These pairs describe where a useful initial direction is likely to be found.
# They are deliberately not task assignments or constraints in the final model.
PRIMARY_CHANNELS = (
    ("M1", "center"),
    ("M2", "positive"),
    ("M3", "negative"),
    ("M2", "positive"),
    ("M3", "negative"),
)
SECONDARY_CHANNELS = (
    ("M2", "cross"),
    ("M1", "cross"),
    ("M1", "cross"),
    ("M1", "cross"),
    ("M1", "cross"),
)


def _load_single_anchors(path: Path) -> dict[tuple[str, str], list[UavPlan]]:
    grouped: dict[tuple[str, str], list[tuple[int, UavPlan]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            uav = row["uav"]
            missile = row["missile"]
            uav_index = UAV_NAMES.index(uav)
            plan = UavPlan(
                uav,
                tuple(float(value) for value in UAV_INITIALS[uav_index]),
                math.radians(float(row["heading_deg"])),
                float(row["speed_mps"]),
                (BombPlan(float(row["release_time_s"]), float(row["fuse_delay_s"])),),
            )
            grouped.setdefault((uav, missile), []).append((int(row["rank"]), plan))
    return {key: [plan for _, plan in sorted(values)] for key, values in grouped.items()}


def _ladder_plan(anchor: UavPlan, uav_index: int, shift: float) -> UavPlan:
    """Make three 5-second-spaced releases around a good one-bomb anchor."""

    reference = anchor.bombs[0]
    delay = min(reference.fuse_delay, float(FUSE_DELAY_LIMITS[uav_index]))
    latest_start = max(0.0, GLOBAL_HORIZON - delay - 2.0 * MIN_RELEASE_GAP - 10.0)
    start = float(np.clip(reference.release_time + shift, 0.0, latest_start))
    releases = start + np.array((0.0, 5.0, 10.0))
    bombs = tuple(BombPlan(float(release), delay) for release in releases)
    return UavPlan(anchor.name, anchor.initial, anchor.heading, anchor.speed, bombs)


def _lhs_plan(uav_index: int, seed: int) -> UavPlan:
    rng = np.random.default_rng(seed)
    heading = float(rng.uniform(0.0, 2.0 * math.pi))
    speed = float(rng.uniform(UAV_SPEED_MIN, UAV_SPEED_MAX))
    delays = rng.uniform(0.15, 0.85, BOMBS_PER_UAV) * float(FUSE_DELAY_LIMITS[uav_index])
    latest_start = max(0.0, GLOBAL_HORIZON - float(np.max(delays)) - 2.0 * MIN_RELEASE_GAP)
    start = float(rng.uniform(0.0, latest_start))
    releases = start + MIN_RELEASE_GAP * np.arange(BOMBS_PER_UAV, dtype=float)
    bombs = tuple(BombPlan(float(release), float(delay)) for release, delay in zip(releases, delays))
    return UavPlan(UAV_NAMES[uav_index], tuple(float(value) for value in UAV_INITIALS[uav_index]), heading, speed, bombs)


def _plan_key(plan: UavPlan) -> tuple[float, ...]:
    values = [plan.heading, plan.speed]
    for bomb in plan.bombs:
        values.extend((bomb.release_time, bomb.fuse_delay))
    return tuple(round(float(value), 7) for value in values)


def build_topology_seed_plans(single_path: Path) -> tuple[tuple[SeededPlan, ...], ...]:
    """Return 9 physical-prior/LHS plans per UAV, all already feasible."""

    anchors = _load_single_anchors(single_path)
    result: list[tuple[SeededPlan, ...]] = []
    for uav_index, uav_name in enumerate(UAV_NAMES):
        produced: list[SeededPlan] = []
        primary_missile, primary_channel = PRIMARY_CHANNELS[uav_index]
        secondary_missile, secondary_channel = SECONDARY_CHANNELS[uav_index]

        # Six channel plans: two anchors with early/mid/late ladders.  If the
        # existing one-bomb library has only one anchor, use additional time
        # ladders around it rather than silently shrinking that UAV's coverage.
        primary_anchors = anchors.get((uav_name, primary_missile), [])[:2]
        for rank, anchor in enumerate(primary_anchors, start=1):
            ladders = (("early", -6.0), ("middle", 0.0), ("late", 6.0))
            if len(primary_anchors) == 1:
                ladders += (("early_dense", -9.0), ("middle_dense", -3.0), ("late_dense", 3.0))
            for label, shift in ladders:
                produced.append(
                    SeededPlan(
                        uav_index,
                        _ladder_plan(anchor, uav_index, shift),
                        f"topology_{primary_channel}_{label}_r{rank}",
                    )
                )

        # One deliberately cross-channel direction.  It can still cover any missile later.
        secondary = anchors.get((uav_name, secondary_missile), [])
        if secondary:
            produced.append(
                SeededPlan(
                    uav_index,
                    _ladder_plan(secondary[0], uav_index, 0.0),
                    f"topology_{secondary_channel}_{secondary_missile}",
                )
            )

        # Two global samples prevent the spatial prior from becoming the only source.
        for number in range(2):
            produced.append(SeededPlan(uav_index, _lhs_plan(uav_index, 20257000 + 100 * uav_index + number), f"topology_global_lhs_{number + 1}"))

        unique: dict[tuple[float, ...], SeededPlan] = {}
        for item in produced:
            if validate_strategy(Strategy((item.plan,))) is None:
                unique.setdefault(_plan_key(item.plan), item)
        fallback = 3
        while len(unique) < 9:
            item = SeededPlan(
                uav_index,
                _lhs_plan(uav_index, 20258000 + 100 * uav_index + fallback),
                f"topology_global_lhs_fallback_{fallback}",
            )
            unique.setdefault(_plan_key(item.plan), item)
            fallback += 1
        result.append(tuple(unique.values()))
    return tuple(result)
