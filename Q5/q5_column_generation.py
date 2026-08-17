from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, differential_evolution, linprog, milp
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack
from scipy.special import expit
from scipy.stats import qmc

try:
    from .q5_main import (
        BOMBS_PER_UAV,
        CLOUD_LIFETIME,
        CLOUD_RADIUS,
        CLOUD_SINK_SPEED,
        FUSE_DELAY_LIMITS,
        GLOBAL_HORIZON,
        GRAVITY,
        MIN_RELEASE_GAP,
        MISSILE_IMPACT_TIMES,
        MISSILE_INITIALS,
        MISSILE_NAMES,
        MISSILE_VELOCITIES,
        MultiMissileEvaluator,
        Strategy,
        TARGET_BOTTOM_CENTER,
        TARGET_HEIGHT,
        TARGET_RADIUS,
        UAV_INITIALS,
        UAV_NAMES,
        UAV_SPEED_MAX,
        UAV_SPEED_MIN,
        BombPlan,
        UavPlan,
        format_intervals,
        write_csv,
    )
    from .q5_topology_seed_factory import build_topology_seed_plans
except ImportError:
    from q5_main import (
    BOMBS_PER_UAV,
    CLOUD_LIFETIME,
    CLOUD_RADIUS,
    CLOUD_SINK_SPEED,
    FUSE_DELAY_LIMITS,
    GLOBAL_HORIZON,
    GRAVITY,
    MIN_RELEASE_GAP,
    MISSILE_IMPACT_TIMES,
    MISSILE_INITIALS,
    MISSILE_NAMES,
    MISSILE_VELOCITIES,
    MultiMissileEvaluator,
    Strategy,
    TARGET_BOTTOM_CENTER,
    TARGET_HEIGHT,
    TARGET_RADIUS,
    UAV_INITIALS,
    UAV_NAMES,
    UAV_SPEED_MAX,
    UAV_SPEED_MIN,
    BombPlan,
    UavPlan,
    format_intervals,
        write_csv,
    )
    from q5_topology_seed_factory import build_topology_seed_plans


Q5_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    time_step: float
    angle_cells: int
    height_cells: int
    radial_cells: int
    column_iterations: int
    pricing_population: int
    pricing_maxiter: int
    pricing_sample_count: int
    pricing_new_per_uav: int
    master_solution_count: int
    mip_time_limit: float
    mip_gap: float
    refinement_candidates: int
    refinement_population: int
    refinement_maxiter: int
    refinement_sweeps: int


PROFILES = {
    "quick": ColumnProfile(
        "quick", 0.40, 12, 3, 3, 1, 24, 15, 800, 2, 4, 60.0, 1.0e-6, 1, 24, 10, 1
    ),
    "standard": ColumnProfile(
        "standard", 0.20, 24, 4, 4, 3, 48, 50, 3000, 4, 8, 300.0, 1.0e-8, 2, 36, 25, 1
    ),
    "extensive": ColumnProfile(
        "extensive", 0.10, 30, 5, 5, 5, 72, 90, 5000, 6, 12, 900.0, 1.0e-9, 3, 48, 45, 2
    ),
}


@dataclass(frozen=True)
class PatchGrid:
    centers: np.ndarray
    radii: np.ndarray
    labels: tuple[str, ...]


@dataclass(frozen=True)
class MasterGrid:
    missile_indices: np.ndarray
    times: np.ndarray
    weights: np.ndarray
    patches: PatchGrid
    missiles: np.ndarray
    directions: np.ndarray
    denominators: np.ndarray

    @property
    def pair_count(self) -> int:
        return len(self.times)

    @property
    def patch_count(self) -> int:
        return len(self.patches.centers)

    @property
    def constraint_count(self) -> int:
        return self.pair_count * self.patch_count


@dataclass(frozen=True)
class PlanColumn:
    uav_index: int
    plan: UavPlan
    source: str
    coverage_center: np.ndarray
    coverage_lower: np.ndarray
    coverage_upper: np.ndarray

    @property
    def key(self) -> tuple[float, ...]:
        values = [self.plan.heading, self.plan.speed]
        for bomb in self.plan.bombs:
            values.extend((bomb.release_time, bomb.fuse_delay))
        return (float(len(self.plan.bombs)), *(round(float(value), 7) for value in values))


@dataclass(frozen=True)
class MasterMatrices:
    columns: tuple[PlanColumn, ...]
    column_groups: tuple[tuple[int, ...], ...]
    a_ub: csr_matrix
    b_ub: np.ndarray
    a_eq: csr_matrix
    b_eq: np.ndarray
    objective: np.ndarray
    bounds: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class MasterSolution:
    mode: str
    sampled_duration: float
    selected_global_indices: tuple[int, ...]
    selected_local_indices: tuple[int, ...]
    mip_gap: float
    mip_nodes: int
    status: int
    message: str
    sampled_minimum: float = math.nan


@dataclass(frozen=True)
class PricingRecord:
    iteration: int
    uav: str
    lp_duration: float
    dual_constraints: int
    existing_score: float
    best_score: float
    added_columns: int


def build_patch_grid(angle_cells: int, height_cells: int, radial_cells: int) -> PatchGrid:
    x0, y0, z0 = TARGET_BOTTOM_CENTER
    centers: list[np.ndarray] = []
    radii: list[float] = []
    labels: list[str] = []
    delta_phi = 2.0 * np.pi / angle_cells
    phis = (np.arange(angle_cells) + 0.5) * delta_phi
    side_z_edges = np.linspace(z0, z0 + TARGET_HEIGHT, height_cells + 1)
    for angle_index, phi in enumerate(phis):
        for height_index in range(height_cells):
            z_left, z_right = side_z_edges[height_index : height_index + 2]
            centers.append(
                np.array(
                    (
                        x0 + TARGET_RADIUS * math.cos(phi),
                        y0 + TARGET_RADIUS * math.sin(phi),
                        0.5 * (z_left + z_right),
                    ),
                    dtype=float,
                )
            )
            horizontal = 2.0 * TARGET_RADIUS * math.sin(0.25 * delta_phi)
            radii.append(math.hypot(horizontal, 0.5 * (z_right - z_left)))
            labels.append(f"side-a{angle_index}-z{height_index}")
    radial_edges = np.linspace(0.0, TARGET_RADIUS, radial_cells + 1)
    for cap_name, cap_z in (("bottom", z0), ("top", z0 + TARGET_HEIGHT)):
        for angle_index, phi in enumerate(phis):
            for radial_index in range(radial_cells):
                r_left, r_right = radial_edges[radial_index : radial_index + 2]
                r_mid = 0.5 * (r_left + r_right)
                centers.append(
                    np.array(
                        (x0 + r_mid * math.cos(phi), y0 + r_mid * math.sin(phi), cap_z),
                        dtype=float,
                    )
                )
                corner_distances = [
                    math.sqrt(max(0.0, radius**2 + r_mid**2 - 2.0 * radius * r_mid * math.cos(0.5 * delta_phi)))
                    for radius in (r_left, r_right)
                ]
                radii.append(max(corner_distances))
                labels.append(f"{cap_name}-a{angle_index}-r{radial_index}")
    return PatchGrid(np.vstack(centers), np.asarray(radii, dtype=float), tuple(labels))


def build_master_grid(profile: ColumnProfile) -> MasterGrid:
    missiles = []
    times = []
    weights = []
    for missile_index, impact in enumerate(MISSILE_IMPACT_TIMES):
        edges = np.arange(0.0, impact, profile.time_step, dtype=float)
        for left in edges:
            right = min(float(impact), float(left + profile.time_step))
            missiles.append(missile_index)
            times.append(0.5 * (left + right))
            weights.append(right - left)
    missile_indices = np.asarray(missiles, dtype=int)
    time_values = np.asarray(times, dtype=float)
    patch_grid = build_patch_grid(profile.angle_cells, profile.height_cells, profile.radial_cells)
    missile_positions = MISSILE_INITIALS[missile_indices] + time_values[:, None] * MISSILE_VELOCITIES[missile_indices]
    directions = patch_grid.centers[None, :, :] - missile_positions[:, None, :]
    denominators = np.einsum("pqc,pqc->pq", directions, directions)
    return MasterGrid(
        missile_indices,
        time_values,
        np.asarray(weights, dtype=float),
        patch_grid,
        missile_positions,
        directions,
        denominators,
    )


def plan_coverage(plan: UavPlan, grid: MasterGrid, batch_size: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair_count, patch_count = grid.pair_count, grid.patch_count
    minimum = np.full((pair_count, patch_count), np.inf, dtype=np.float32)
    for bomb in plan.bombs:
        burst = plan.burst_point(bomb)
        active = (grid.times >= bomb.burst_time - 1.0e-12) & (
            grid.times <= bomb.burst_time + CLOUD_LIFETIME + 1.0e-12
        )
        if not np.any(active):
            continue
        active_indices = np.flatnonzero(active)
        for start in range(0, len(active_indices), batch_size):
            selected = active_indices[start : start + batch_size]
            selected_times = grid.times[selected]
            centers = np.repeat(burst[None, :], len(selected), axis=0)
            centers[:, 2] -= CLOUD_SINK_SPEED * (selected_times - bomb.burst_time)
            projection = np.einsum(
                "pc,pqc->pq", centers - grid.missiles[selected], grid.directions[selected]
            ) / grid.denominators[selected]
            projection = np.clip(projection, 0.0, 1.0)
            closest = grid.missiles[selected, None, :] + projection[:, :, None] * grid.directions[selected]
            distances = np.linalg.norm(closest - centers[:, None, :], axis=2)
            minimum[selected] = np.minimum(minimum[selected], distances.astype(np.float32))
    center = minimum <= CLOUD_RADIUS + 1.0e-7
    lower = minimum + grid.patches.radii[None, :] <= CLOUD_RADIUS + 1.0e-7
    upper = minimum - grid.patches.radii[None, :] <= CLOUD_RADIUS + 1.0e-7
    return center, lower, upper


def make_column(uav_index: int, plan: UavPlan, source: str, grid: MasterGrid) -> PlanColumn:
    center, lower, upper = plan_coverage(plan, grid)
    return PlanColumn(uav_index, plan, source, center, lower, upper)


def parse_semicolon_values(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(";") if item.strip())


def load_best_plans(path: Path) -> tuple[UavPlan, ...]:
    grouped: dict[str, list[dict[str, str]]] = {name: [] for name in UAV_NAMES}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["uav"]].append(row)
    plans = []
    for uav_index, name in enumerate(UAV_NAMES):
        rows = sorted(grouped[name], key=lambda row: int(row["bomb"]))
        bombs = tuple(BombPlan(float(row["release_time_s"]), float(row["fuse_delay_s"])) for row in rows)
        plans.append(
            UavPlan(
                name,
                tuple(float(value) for value in UAV_INITIALS[uav_index]),
                float(rows[0]["heading_rad"]),
                float(rows[0]["speed_mps"]),
                bombs,
            )
        )
    return tuple(plans)


def load_pruned_plans(best_path: Path, contribution_path: Path) -> tuple[UavPlan, ...]:
    full = load_best_plans(best_path)
    used: set[tuple[str, int]] = set()
    with contribution_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["used"].lower() == "true":
                used.add((row["uav"], int(row["bomb"])))
    return tuple(
        UavPlan(
            plan.name,
            plan.initial,
            plan.heading,
            plan.speed,
            tuple(bomb for bomb_index, bomb in enumerate(plan.bombs, start=1) if (plan.name, bomb_index) in used),
        )
        for plan in full
    )


def load_local_plans(path: Path) -> tuple[tuple[UavPlan, ...], ...]:
    grouped: list[list[UavPlan]] = [[] for _ in UAV_NAMES]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            uav_index = UAV_NAMES.index(row["uav"])
            releases = parse_semicolon_values(row["release_times_s"])
            delays = parse_semicolon_values(row["fuse_delays_s"])
            grouped[uav_index].append(
                UavPlan(
                    row["uav"],
                    tuple(float(value) for value in UAV_INITIALS[uav_index]),
                    math.radians(float(row["heading_deg"])),
                    float(row["speed_mps"]),
                    tuple(BombPlan(release, delay) for release, delay in zip(releases, delays)),
                )
            )
    return tuple(tuple(items) for items in grouped)


def load_single_plans(path: Path) -> tuple[tuple[UavPlan, ...], ...]:
    grouped: list[list[UavPlan]] = [[] for _ in UAV_NAMES]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            uav_index = UAV_NAMES.index(row["uav"])
            grouped[uav_index].append(
                UavPlan(
                    row["uav"],
                    tuple(float(value) for value in UAV_INITIALS[uav_index]),
                    math.radians(float(row["heading_deg"])),
                    float(row["speed_mps"]),
                    (BombPlan(float(row["release_time_s"]), float(row["fuse_delay_s"])),),
                )
            )
    return tuple(tuple(items) for items in grouped)


def initial_columns(grid: MasterGrid) -> list[list[PlanColumn]]:
    best_path = Q5_DIR / "q5_best_solution.csv"
    contribution_path = Q5_DIR / "q5_bomb_contributions.csv"
    full = load_best_plans(best_path)
    pruned = load_pruned_plans(best_path, contribution_path)
    local = load_local_plans(Q5_DIR / "q5_local_library.csv")
    single = load_single_plans(Q5_DIR / "q5_single_library.csv")
    topology = build_topology_seed_plans(Q5_DIR / "q5_single_library.csv")
    columns: list[list[PlanColumn]] = [[] for _ in UAV_NAMES]
    for uav_index in range(len(UAV_NAMES)):
        candidates: list[tuple[UavPlan, str]] = [(full[uav_index], "previous_best_full")]
        if pruned[uav_index].bombs:
            candidates.append((pruned[uav_index], "previous_best_pruned"))
        candidates.extend((plan, "local_library") for plan in local[uav_index])
        candidates.extend((plan, "single_library") for plan in single[uav_index])
        candidates.extend((item.plan, item.source) for item in topology[uav_index])
        seen: set[tuple[float, ...]] = set()
        for plan, source in candidates:
            key_values = [len(plan.bombs), plan.heading, plan.speed]
            for bomb in plan.bombs:
                key_values.extend((bomb.release_time, bomb.fuse_delay))
            key = tuple(round(float(value), 7) for value in key_values)
            if key in seen:
                continue
            seen.add(key)
            columns[uav_index].append(make_column(uav_index, plan, source, grid))
    return columns


def coverage_for_mode(column: PlanColumn, mode: str) -> np.ndarray:
    if mode == "center":
        return column.coverage_center
    if mode == "lower":
        return column.coverage_lower
    if mode == "upper":
        return column.coverage_upper
    raise ValueError(f"未知覆盖模式：{mode}")


def build_master_matrices(columns_by_uav: Sequence[Sequence[PlanColumn]], grid: MasterGrid, mode: str = "center") -> MasterMatrices:
    columns = tuple(column for group in columns_by_uav for column in group)
    groups: list[tuple[int, ...]] = []
    offset = 0
    for group in columns_by_uav:
        groups.append(tuple(range(offset, offset + len(group))))
        offset += len(group)
    column_count = len(columns)
    pair_count = grid.pair_count
    constraint_count = grid.constraint_count
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    for column_index, column in enumerate(columns):
        covered = np.flatnonzero(coverage_for_mode(column, mode).reshape(-1))
        if len(covered):
            row_parts.append(covered.astype(np.int32))
            col_parts.append(np.full(len(covered), column_index, dtype=np.int32))
            data_parts.append(np.full(len(covered), -1.0, dtype=float))
    rows = np.arange(constraint_count, dtype=np.int32)
    row_parts.append(rows)
    col_parts.append(column_count + rows // grid.patch_count)
    data_parts.append(np.ones(constraint_count, dtype=float))
    a_ub = coo_matrix(
        (np.concatenate(data_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
        shape=(constraint_count, column_count + pair_count),
    ).tocsr()
    eq_rows = []
    eq_cols = []
    for uav_index, group in enumerate(groups):
        eq_rows.extend([uav_index] * len(group))
        eq_cols.extend(group)
    a_eq = coo_matrix(
        (np.ones(len(eq_rows)), (eq_rows, eq_cols)),
        shape=(len(UAV_NAMES), column_count + pair_count),
    ).tocsr()
    objective = np.zeros(column_count + pair_count, dtype=float)
    objective[column_count:] = -grid.weights
    return MasterMatrices(
        columns,
        tuple(groups),
        a_ub,
        np.zeros(constraint_count),
        a_eq,
        np.ones(len(UAV_NAMES)),
        objective,
        tuple((0.0, 1.0) for _ in range(column_count + pair_count)),
    )


def solve_lp_master(matrices: MasterMatrices):
    result = linprog(
        matrices.objective,
        A_ub=matrices.a_ub,
        b_ub=matrices.b_ub,
        A_eq=matrices.a_eq,
        b_eq=matrices.b_eq,
        bounds=matrices.bounds,
        method="highs",
        options={"presolve": True},
    )
    if not result.success:
        raise RuntimeError(f"列生成LP主问题失败：{result.message}")
    return result


def selected_from_vector(vector: np.ndarray, matrices: MasterMatrices) -> tuple[tuple[int, ...], tuple[int, ...]]:
    global_indices = []
    local_indices = []
    for group in matrices.column_groups:
        values = vector[np.asarray(group, dtype=int)]
        local = int(np.argmax(values))
        global_indices.append(group[local])
        local_indices.append(local)
    return tuple(global_indices), tuple(local_indices)


def solve_integer_master(
    matrices: MasterMatrices,
    mode: str,
    time_limit: float,
    mip_gap: float,
    exclusions: Sequence[tuple[int, ...]] = (),
) -> MasterSolution:
    variable_count = len(matrices.objective)
    constraints: list[LinearConstraint] = [
        LinearConstraint(matrices.a_ub, -np.inf, matrices.b_ub),
        LinearConstraint(matrices.a_eq, matrices.b_eq, matrices.b_eq),
    ]
    if exclusions:
        rows = []
        cols = []
        for row_index, selection in enumerate(exclusions):
            rows.extend([row_index] * len(selection))
            cols.extend(selection)
        exclusion_matrix = coo_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(len(exclusions), variable_count),
        ).tocsr()
        constraints.append(LinearConstraint(exclusion_matrix, -np.inf, len(UAV_NAMES) - 1))
    result = milp(
        matrices.objective,
        integrality=np.ones(variable_count, dtype=int),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=constraints,
        options={"time_limit": time_limit, "mip_rel_gap": mip_gap, "presolve": True},
    )
    if result.x is None:
        raise RuntimeError(f"0-1主问题没有可行解：{result.message}")
    global_indices, local_indices = selected_from_vector(result.x, matrices)
    return MasterSolution(
        mode=mode,
        sampled_duration=float(-result.fun),
        selected_global_indices=global_indices,
        selected_local_indices=local_indices,
        mip_gap=float(getattr(result, "mip_gap", math.nan)),
        mip_nodes=int(getattr(result, "mip_node_count", -1)),
        status=int(result.status),
        message=str(result.message),
    )


def solve_balanced_master(
    matrices: MasterMatrices,
    grid: MasterGrid,
    primary_optimum: float,
    primary_tolerance: float,
    time_limit: float,
    mip_gap: float,
) -> MasterSolution:
    """在总时长距离散最优不超过一个时间单元时最大化最差导弹时长。"""

    base_variables = len(matrices.objective)
    column_count = len(matrices.columns)
    variable_count = base_variables + 1
    zero_column_ub = csr_matrix((matrices.a_ub.shape[0], 1))
    zero_column_eq = csr_matrix((matrices.a_eq.shape[0], 1))
    a_coverage = hstack((matrices.a_ub, zero_column_ub), format="csr")
    a_select = hstack((matrices.a_eq, zero_column_eq), format="csr")
    constraints: list[LinearConstraint] = [
        LinearConstraint(a_coverage, -np.inf, matrices.b_ub),
        LinearConstraint(a_select, matrices.b_eq, matrices.b_eq),
    ]
    primary_row = np.zeros(variable_count, dtype=float)
    primary_row[column_count:base_variables] = -grid.weights
    constraints.append(
        LinearConstraint(
            csr_matrix(primary_row[None, :]),
            -np.inf,
            -(primary_optimum - primary_tolerance),
        )
    )
    fairness_rows = []
    for missile_index in range(len(MISSILE_NAMES)):
        row = np.zeros(variable_count, dtype=float)
        pair_mask = grid.missile_indices == missile_index
        pair_positions = np.flatnonzero(pair_mask)
        row[column_count + pair_positions] = -grid.weights[pair_positions]
        row[-1] = 1.0
        fairness_rows.append(row)
    constraints.append(
        LinearConstraint(csr_matrix(np.vstack(fairness_rows)), -np.inf, np.zeros(len(MISSILE_NAMES)))
    )
    objective = np.zeros(variable_count, dtype=float)
    objective[-1] = -1.0
    lower = np.zeros(variable_count, dtype=float)
    upper = np.ones(variable_count, dtype=float)
    upper[-1] = float(np.max(MISSILE_IMPACT_TIMES))
    integrality = np.ones(variable_count, dtype=int)
    integrality[-1] = 0
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"time_limit": time_limit, "mip_rel_gap": mip_gap, "presolve": True},
    )
    if result.x is None:
        raise RuntimeError(f"均衡第二阶段主问题没有可行解：{result.message}")
    global_indices, local_indices = selected_from_vector(result.x, matrices)
    y_values = result.x[column_count:base_variables]
    missile_durations = tuple(
        float(np.dot(grid.weights[grid.missile_indices == missile_index], y_values[grid.missile_indices == missile_index]))
        for missile_index in range(len(MISSILE_NAMES))
    )
    return MasterSolution(
        mode="balanced",
        sampled_duration=float(sum(missile_durations)),
        selected_global_indices=global_indices,
        selected_local_indices=local_indices,
        mip_gap=float(getattr(result, "mip_gap", math.nan)),
        mip_nodes=int(getattr(result, "mip_node_count", -1)),
        status=int(result.status),
        message=str(result.message),
        sampled_minimum=float(min(missile_durations)),
    )


def softmax_gap_decode(vector: Sequence[float], uav_index: int) -> UavPlan:
    values = np.asarray(vector, dtype=float)
    if values.shape != (9,):
        raise ValueError("定价向量必须包含9个变量。")
    heading = float(values[0] % (2.0 * np.pi))
    speed = UAV_SPEED_MIN + (UAV_SPEED_MAX - UAV_SPEED_MIN) * float(np.clip(values[1], 0.0, 1.0))
    logits = np.clip(values[2:6], -8.0, 8.0)
    weights = np.exp(logits - np.max(logits))
    gaps = (GLOBAL_HORIZON - 2.0 * MIN_RELEASE_GAP) * weights / np.sum(weights)
    releases = np.array(
        (
            gaps[0],
            gaps[0] + MIN_RELEASE_GAP + gaps[1],
            gaps[0] + MIN_RELEASE_GAP + gaps[1] + MIN_RELEASE_GAP + gaps[2],
        )
    )
    bombs = []
    for release, delay_unit in zip(releases, values[6:9]):
        ceiling = min(float(FUSE_DELAY_LIMITS[uav_index]), max(0.0, GLOBAL_HORIZON - release))
        bombs.append(BombPlan(float(release), ceiling * float(np.clip(delay_unit, 0.0, 1.0))))
    return UavPlan(
        UAV_NAMES[uav_index],
        tuple(float(value) for value in UAV_INITIALS[uav_index]),
        heading,
        speed,
        tuple(bombs),
    )


def encode_softmax_plan(plan: UavPlan, uav_index: int) -> np.ndarray | None:
    if len(plan.bombs) != BOMBS_PER_UAV:
        return None
    bombs = tuple(sorted(plan.bombs, key=lambda bomb: bomb.release_time))
    releases = np.array([bomb.release_time for bomb in bombs])
    gaps = np.array(
        (
            releases[0],
            releases[1] - releases[0] - MIN_RELEASE_GAP,
            releases[2] - releases[1] - MIN_RELEASE_GAP,
            GLOBAL_HORIZON - releases[2],
        ),
        dtype=float,
    )
    if np.min(gaps) < -1.0e-7:
        return None
    gaps = np.maximum(gaps, 1.0e-8)
    logits = np.log(gaps)
    logits -= np.mean(logits)
    logits = np.clip(logits, -8.0, 8.0)
    delays = []
    for bomb in bombs:
        ceiling = min(float(FUSE_DELAY_LIMITS[uav_index]), max(0.0, GLOBAL_HORIZON - bomb.release_time))
        delays.append(0.0 if ceiling <= 1.0e-12 else bomb.fuse_delay / ceiling)
    speed_unit = (plan.speed - UAV_SPEED_MIN) / (UAV_SPEED_MAX - UAV_SPEED_MIN)
    return np.array((plan.heading, speed_unit, *logits, *delays), dtype=float)


class PricingObjective:
    def __init__(self, uav_index: int, grid: MasterGrid, constraint_indices: np.ndarray, weights: np.ndarray, epsilon: float = 0.8) -> None:
        self.uav_index = uav_index
        self.grid = grid
        self.constraint_indices = constraint_indices
        self.weights = weights / max(float(np.sum(weights)), 1.0e-12)
        self.epsilon = epsilon
        self.pair_indices = constraint_indices // grid.patch_count
        self.patch_indices = constraint_indices % grid.patch_count
        self.times = grid.times[self.pair_indices]
        self.missiles = grid.missiles[self.pair_indices]
        self.directions = grid.directions[self.pair_indices, self.patch_indices]
        self.denominators = grid.denominators[self.pair_indices, self.patch_indices]

    def score_plan(self, plan: UavPlan) -> float:
        minimum = np.full(len(self.constraint_indices), np.inf, dtype=float)
        for bomb in plan.bombs:
            active = (self.times >= bomb.burst_time - 1.0e-12) & (
                self.times <= bomb.burst_time + CLOUD_LIFETIME + 1.0e-12
            )
            if not np.any(active):
                continue
            selected = np.flatnonzero(active)
            center = plan.burst_point(bomb)[None, :].repeat(len(selected), axis=0)
            center[:, 2] -= CLOUD_SINK_SPEED * (self.times[selected] - bomb.burst_time)
            projection = np.einsum(
                "pc,pc->p", center - self.missiles[selected], self.directions[selected]
            ) / self.denominators[selected]
            projection = np.clip(projection, 0.0, 1.0)
            closest = self.missiles[selected] + projection[:, None] * self.directions[selected]
            distance = np.linalg.norm(closest - center, axis=1)
            minimum[selected] = np.minimum(minimum[selected], distance)
        margin = CLOUD_RADIUS - minimum
        smooth = expit(np.clip(margin / self.epsilon, -50.0, 50.0))
        hard = margin >= 0.0
        return float(np.dot(self.weights, 0.9 * smooth + 0.1 * hard))

    def __call__(self, vector: np.ndarray) -> float:
        return -self.score_plan(softmax_gap_decode(vector, self.uav_index))


def latin_population(bounds: Sequence[tuple[float, float]], count: int, seed: int) -> np.ndarray:
    unit = qmc.LatinHypercube(d=len(bounds), seed=seed).random(count)
    lower = np.array([value[0] for value in bounds])
    upper = np.array([value[1] for value in bounds])
    return lower + unit * (upper - lower)


def pricing_columns(
    iteration: int,
    uav_index: int,
    columns: Sequence[PlanColumn],
    grid: MasterGrid,
    dual_weights: np.ndarray,
    profile: ColumnProfile,
) -> tuple[list[UavPlan], PricingRecord]:
    positive = np.flatnonzero(dual_weights > 1.0e-12)
    if not len(positive):
        positive = np.arange(len(dual_weights))
    order = positive[np.argsort(dual_weights[positive])[::-1]]
    selected = order[: profile.pricing_sample_count]
    weights = dual_weights[selected]
    objective = PricingObjective(uav_index, grid, selected, weights)
    bounds = ((0.0, 2.0 * np.pi), (0.0, 1.0), *((-8.0, 8.0),) * 4, *((0.0, 1.0),) * 3)
    initial = latin_population(bounds, profile.pricing_population, 20253000 + 100 * iteration + uav_index)
    anchors = [encoded for encoded in (encode_softmax_plan(column.plan, uav_index) for column in columns) if encoded is not None]
    anchor_scores = sorted(((objective.score_plan(softmax_gap_decode(anchor, uav_index)), anchor) for anchor in anchors), key=lambda item: -item[0])
    for row, (_, anchor) in enumerate(anchor_scores[: min(len(anchor_scores), len(initial) // 2)]):
        initial[row] = anchor
    existing_score = max((objective.score_plan(column.plan) for column in columns), default=0.0)
    result = differential_evolution(
        objective,
        bounds=bounds,
        init=initial,
        maxiter=profile.pricing_maxiter,
        mutation=(0.45, 1.0),
        recombination=0.9,
        seed=20254000 + 100 * iteration + uav_index,
        polish=False,
        tol=1.0e-7,
        atol=0.0,
        updating="immediate",
        workers=1,
    )
    population = np.asarray(result.population)
    order = np.argsort(result.population_energies)
    plans: list[UavPlan] = []
    seen = {column.key for column in columns}
    best_score = existing_score
    for index in order:
        plan = softmax_gap_decode(population[index], uav_index)
        values = [len(plan.bombs), plan.heading, plan.speed]
        for bomb in plan.bombs:
            values.extend((bomb.release_time, bomb.fuse_delay))
        key = (float(len(plan.bombs)), *(round(float(value), 7) for value in values[1:]))
        score = objective.score_plan(plan)
        best_score = max(best_score, score)
        if key in seen:
            continue
        seen.add(key)
        plans.append(plan)
        if len(plans) >= profile.pricing_new_per_uav:
            break
    record = PricingRecord(iteration, UAV_NAMES[uav_index], math.nan, len(selected), existing_score, best_score, len(plans))
    return plans, record


def column_generation(profile: ColumnProfile, grid: MasterGrid, columns_by_uav: list[list[PlanColumn]]):
    records: list[PricingRecord] = []
    previous_lp = -math.inf
    for iteration in range(1, profile.column_iterations + 1):
        print(f"列生成第{iteration}轮：构建并求解LP主问题……", flush=True)
        matrices = build_master_matrices(columns_by_uav, grid, "center")
        lp = solve_lp_master(matrices)
        lp_duration = float(-lp.fun)
        print(f"  LP上界（当前策略列库、中心网格）：{lp_duration:.6f}s，列数{len(matrices.columns)}。", flush=True)
        dual_weights = np.maximum(0.0, -np.asarray(lp.ineqlin.marginals, dtype=float))
        added = 0
        iteration_records = []
        for uav_index in range(len(UAV_NAMES)):
            print(f"  正在求解{UAV_NAMES[uav_index]}的9维平滑定价子问题……", flush=True)
            new_plans, record = pricing_columns(iteration, uav_index, columns_by_uav[uav_index], grid, dual_weights, profile)
            for rank, plan in enumerate(new_plans, start=1):
                column = make_column(uav_index, plan, f"pricing_iter{iteration}_rank{rank}", grid)
                if column.key not in {existing.key for existing in columns_by_uav[uav_index]}:
                    columns_by_uav[uav_index].append(column)
                    added += 1
            iteration_records.append(
                PricingRecord(
                    record.iteration,
                    record.uav,
                    lp_duration,
                    record.dual_constraints,
                    record.existing_score,
                    record.best_score,
                    len(new_plans),
                )
            )
        records.extend(iteration_records)
        print(f"  本轮新增{added}个策略列。", flush=True)
        if added == 0:
            print("  没有产生新列，停止列生成。", flush=True)
            break
        previous_lp = lp_duration
    return columns_by_uav, records


def strategy_from_master(solution: MasterSolution, matrices: MasterMatrices) -> Strategy:
    plans = tuple(matrices.columns[index].plan for index in solution.selected_global_indices)
    return Strategy(plans)


def solve_top_master_solutions(
    matrices: MasterMatrices,
    profile: ColumnProfile,
    mode: str,
    count: int,
) -> list[MasterSolution]:
    solutions = []
    exclusions: list[tuple[int, ...]] = []
    for rank in range(count):
        solution = solve_integer_master(
            matrices,
            mode,
            profile.mip_time_limit,
            profile.mip_gap,
            exclusions,
        )
        solutions.append(solution)
        exclusions.append(solution.selected_global_indices)
        print(
            f"  {mode}主问题第{rank + 1}解：{solution.sampled_duration:.6f}s，"
            f"MIP gap={solution.mip_gap:.3e}。",
            flush=True,
        )
    return solutions


def pad_to_three_bombs(plan: UavPlan, uav_index: int) -> UavPlan:
    if len(plan.bombs) == 3:
        return plan
    bombs = list(sorted(plan.bombs, key=lambda bomb: bomb.release_time))
    if not bombs:
        bombs.append(BombPlan(0.0, 0.0))
    while len(bombs) < 3:
        previous = bombs[-1]
        release = min(GLOBAL_HORIZON, previous.release_time + MIN_RELEASE_GAP)
        if release >= GLOBAL_HORIZON and bombs[0].release_time >= 2.0:
            bombs = [BombPlan(bomb.release_time - 1.0, bomb.fuse_delay) for bomb in bombs]
            release = bombs[-1].release_time + MIN_RELEASE_GAP
        ceiling = min(float(FUSE_DELAY_LIMITS[uav_index]), max(0.0, GLOBAL_HORIZON - release))
        bombs.append(BombPlan(release, min(previous.fuse_delay, ceiling)))
    return UavPlan(plan.name, plan.initial, plan.heading, plan.speed, tuple(bombs[:3]))


def refine_strategy(strategy: Strategy, profile: ColumnProfile, seed: int) -> tuple[Strategy, list[dict[str, object]]]:
    plans = [pad_to_three_bombs(plan, UAV_NAMES.index(plan.name)) for plan in strategy.uavs]
    evaluator = MultiMissileEvaluator(18, 3, 3, 0.15)
    current = Strategy(tuple(plans))
    current_result = evaluator.evaluate(current)
    history: list[dict[str, object]] = []
    bounds = ((0.0, 2.0 * np.pi), (0.0, 1.0), *((-8.0, 8.0),) * 4, *((0.0, 1.0),) * 3)
    for sweep in range(1, profile.refinement_sweeps + 1):
        for uav_index in range(len(UAV_NAMES)):
            anchor = encode_softmax_plan(current.uavs[uav_index], uav_index)
            if anchor is None:
                continue
            initial = latin_population(bounds, profile.refinement_population, seed + 1000 * sweep + uav_index)
            initial[0] = anchor
            rng = np.random.default_rng(seed + 2000 * sweep + uav_index)
            for row in range(1, min(len(initial), len(initial) // 2)):
                perturbed = anchor.copy()
                perturbed[0] = (perturbed[0] + rng.normal(0.0, 0.05)) % (2.0 * np.pi)
                perturbed[1] = np.clip(perturbed[1] + rng.normal(0.0, 0.04), 0.0, 1.0)
                perturbed[2:6] = np.clip(perturbed[2:6] + rng.normal(0.0, 0.35, 4), -8.0, 8.0)
                perturbed[6:9] = np.clip(perturbed[6:9] + rng.normal(0.0, 0.04, 3), 0.0, 1.0)
                initial[row] = perturbed

            def objective(vector: np.ndarray) -> float:
                candidate_plans = list(current.uavs)
                candidate_plans[uav_index] = softmax_gap_decode(vector, uav_index)
                result = evaluator.evaluate(Strategy(tuple(candidate_plans)))
                if not result.feasible or result.positive_count < len(MISSILE_NAMES):
                    return 1.0e4 - result.total_duration
                return -(result.total_duration + 1.0e-4 * result.minimum_duration)

            result = differential_evolution(
                objective,
                bounds=bounds,
                init=initial,
                maxiter=profile.refinement_maxiter,
                mutation=(0.45, 1.0),
                recombination=0.9,
                seed=seed + 3000 * sweep + uav_index,
                polish=False,
                tol=1.0e-7,
                atol=0.0,
                updating="immediate",
                workers=1,
            )
            candidate_plans = list(current.uavs)
            candidate_plans[uav_index] = softmax_gap_decode(result.x, uav_index)
            candidate = Strategy(tuple(candidate_plans))
            candidate_result = evaluator.evaluate(candidate)
            accepted = candidate_result.total_duration > current_result.total_duration + 1.0e-7
            if accepted:
                current = candidate
                current_result = candidate_result
            history.append(
                {
                    "sweep": sweep,
                    "uav": UAV_NAMES[uav_index],
                    "accepted": accepted,
                    "candidate_total_s": candidate_result.total_duration,
                    "current_total_s": current_result.total_duration,
                    "M1_s": current_result.missile_durations[0],
                    "M2_s": current_result.missile_durations[1],
                    "M3_s": current_result.missile_durations[2],
                }
            )
    return current, history


def exact_rank(strategies: Sequence[tuple[str, Strategy]], grid_spec: tuple[int, int, int, float]):
    evaluator = MultiMissileEvaluator(*grid_spec)
    reviewed = []
    for source, strategy in strategies:
        result = evaluator.evaluate(strategy)
        reviewed.append((source, strategy, result))
    return sorted(reviewed, key=lambda item: (-item[2].total_duration, -item[2].minimum_duration))


def individual_durations(strategy: Strategy, evaluator: MultiMissileEvaluator):
    values = []
    for plan in strategy.uavs:
        for bomb in plan.bombs:
            single = Strategy((UavPlan(plan.name, plan.initial, plan.heading, plan.speed, (bomb,)),))
            values.append(evaluator.evaluate(single).missile_durations)
    return values


def save_best_solution(path: Path, result, strategy: Strategy) -> None:
    evaluator = MultiMissileEvaluator(*result.grid)
    individual = individual_durations(strategy, evaluator)
    rows = []
    flat_index = 0
    for plan in strategy.uavs:
        for bomb_index, bomb in enumerate(plan.bombs, start=1):
            release = plan.release_point(bomb)
            burst = plan.burst_point(bomb)
            durations = individual[flat_index]
            target = int(np.argmax(durations))
            rows.append(
                {
                    "profile": "column_generation",
                    "uav": plan.name,
                    "bomb": bomb_index,
                    "heading_rad": plan.heading,
                    "heading_deg": np.degrees(plan.heading),
                    "speed_mps": plan.speed,
                    "release_time_s": bomb.release_time,
                    "fuse_delay_s": bomb.fuse_delay,
                    "burst_time_s": bomb.burst_time,
                    "release_x_m": release[0],
                    "release_y_m": release[1],
                    "release_z_m": release[2],
                    "burst_x_m": burst[0],
                    "burst_y_m": burst[1],
                    "burst_z_m": burst[2],
                    "planned_missile": MISSILE_NAMES[target],
                    "individual_duration_s": durations[target],
                    "M1_individual_s": durations[0],
                    "M2_individual_s": durations[1],
                    "M3_individual_s": durations[2],
                    "joint_M1_s": result.missile_durations[0],
                    "joint_M2_s": result.missile_durations[1],
                    "joint_M3_s": result.missile_durations[2],
                    "joint_total_s": result.total_duration,
                    "minimum_missile_s": result.minimum_duration,
                    "verification_grid": f"{result.grid[0]}x{result.grid[1]}x{result.grid[2]}@{result.grid[3]}",
                }
            )
            flat_index += 1
    write_csv(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="问题五策略列生成、0-1主问题与连续精修")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="standard")
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    started = time.perf_counter()
    print("正在建立导弹×时间单元×圆柱面片主问题网格……", flush=True)
    grid = build_master_grid(profile)
    print(
        f"  时间单元{grid.pair_count}个，圆柱面片{grid.patch_count}个，"
        f"覆盖约束{grid.constraint_count}条。",
        flush=True,
    )
    print("正在载入既有最优解、单弹库和共享轨迹局部库作为初始策略列……", flush=True)
    columns_by_uav = initial_columns(grid)
    print("  初始列数：" + ", ".join(f"{UAV_NAMES[i]}={len(group)}" for i, group in enumerate(columns_by_uav)), flush=True)
    columns_by_uav, pricing_records = column_generation(profile, grid, columns_by_uav)
    print("列生成完成，正在求解中心网格0-1主问题及前若干离散组合……", flush=True)
    center_matrices = build_master_matrices(columns_by_uav, grid, "center")
    center_solutions = solve_top_master_solutions(center_matrices, profile, "center", profile.master_solution_count)
    print("正在求解总时长近优约束下的最差导弹最大化第二阶段主问题……", flush=True)
    balanced_solution = solve_balanced_master(
        center_matrices,
        grid,
        center_solutions[0].sampled_duration,
        profile.time_step,
        profile.mip_time_limit,
        profile.mip_gap,
    )
    print(
        f"  均衡解采样总时长{balanced_solution.sampled_duration:.6f}s，"
        f"采样最差导弹{balanced_solution.sampled_minimum:.6f}s，MIP gap={balanced_solution.mip_gap:.3e}。",
        flush=True,
    )
    print("正在求解保守面片与乐观面片0-1主问题，形成空间离散区间……", flush=True)
    lower_matrices = build_master_matrices(columns_by_uav, grid, "lower")
    upper_matrices = build_master_matrices(columns_by_uav, grid, "upper")
    lower_solution = solve_integer_master(lower_matrices, "lower", profile.mip_time_limit, profile.mip_gap)
    upper_solution = solve_integer_master(upper_matrices, "upper", profile.mip_time_limit, profile.mip_gap)
    print(
        f"  保守/中心/乐观采样主问题：{lower_solution.sampled_duration:.6f} / "
        f"{center_solutions[0].sampled_duration:.6f} / {upper_solution.sampled_duration:.6f}s。",
        flush=True,
    )
    candidates: list[tuple[str, Strategy]] = [
        (f"master_center_rank{rank}", strategy_from_master(solution, center_matrices))
        for rank, solution in enumerate(center_solutions, start=1)
    ]
    candidates.append(("master_balanced", strategy_from_master(balanced_solution, center_matrices)))
    candidates.append(("master_lower", strategy_from_master(lower_solution, lower_matrices)))
    candidates.append(("previous_best", Strategy(load_best_plans(Q5_DIR / "q5_best_solution.csv"))))
    print("正在以中密度完整圆柱评价器重排主问题组合……", flush=True)
    medium = exact_rank(candidates, (90, 7, 7, 0.03))
    refinement_history = []
    refined_candidates = []
    for rank, (source, strategy, result) in enumerate(medium[: profile.refinement_candidates], start=1):
        print(
            f"正在对第{rank}个主问题组合进行逐无人机9维连续精修；"
            f"精修前{result.total_duration:.6f}s……",
            flush=True,
        )
        refined, history = refine_strategy(strategy, profile, 20255000 + rank)
        refinement_history.extend({"candidate": rank, "source": source, **row} for row in history)
        refined_candidates.append((f"refined_{source}", refined))
    all_candidates = [(source, strategy) for source, strategy, _ in medium] + refined_candidates
    print("正在执行180×9×9@0.02候选重排……", flush=True)
    dense = exact_rank(all_candidates, (180, 9, 9, 0.02))
    print("正在执行360×13×11@0.01高密度复核……", flush=True)
    high = exact_rank([(source, strategy) for source, strategy, _ in dense[:5]], (360, 13, 11, 0.01))
    print("正在执行720×21×15@0.005独立加密复核……", flush=True)
    ultra = exact_rank([(source, strategy) for source, strategy, _ in high[:2]], (720, 21, 15, 0.005))
    best_source, best_strategy, best_result = ultra[0]
    pricing_rows = [record.__dict__ for record in pricing_records]
    master_rows = [
        {
            "rank": rank,
            "mode": solution.mode,
            "sampled_duration_s": solution.sampled_duration,
            "mip_gap": solution.mip_gap,
            "mip_nodes": solution.mip_nodes,
            "selected_local_columns": ";".join(str(value) for value in solution.selected_local_indices),
            "message": solution.message,
        }
        for rank, solution in enumerate((*center_solutions, balanced_solution, lower_solution, upper_solution), start=1)
    ]
    convergence_rows = []
    for grid_name, reviewed in (("medium", medium), ("dense", dense), ("high", high), ("ultra", ultra)):
        for rank, (source, _, result) in enumerate(reviewed, start=1):
            convergence_rows.append(
                {
                    "grid": grid_name,
                    "rank": rank,
                    "source": source,
                    "M1_s": result.missile_durations[0],
                    "M2_s": result.missile_durations[1],
                    "M3_s": result.missile_durations[2],
                    "total_s": result.total_duration,
                    "minimum_s": result.minimum_duration,
                    "grid_spec": f"{result.grid[0]}x{result.grid[1]}x{result.grid[2]}@{result.grid[3]}",
                }
            )
    write_csv(Q5_DIR / "q5_cg_pricing.csv", pricing_rows)
    write_csv(Q5_DIR / "q5_cg_master_solutions.csv", master_rows)
    write_csv(Q5_DIR / "q5_cg_refinement.csv", refinement_history)
    write_csv(Q5_DIR / "q5_cg_convergence.csv", convergence_rows)
    best_path = Q5_DIR / "q5_cg_best_solution.csv"
    save_best_solution(best_path, best_result, best_strategy)
    interval_rows = [
        {
            "missile": MISSILE_NAMES[index],
            "duration_s": best_result.missile_durations[index],
            "intervals_s": format_intervals(best_result.missile_intervals[index]),
        }
        for index in range(3)
    ]
    write_csv(Q5_DIR / "q5_cg_missile_intervals.csv", interval_rows)
    print("\n策略列生成与0-1主问题推荐方案")
    print(f"  来源：{best_source}")
    for index, name in enumerate(MISSILE_NAMES):
        print(f"  {name}: {best_result.missile_durations[index]:.10f}s，区间{format_intervals(best_result.missile_intervals[index])}")
    print(f"  总时长：{best_result.total_duration:.10f}导弹·秒")
    print(f"  最差导弹：{best_result.minimum_duration:.10f}s")
    print(f"  最终网格：{best_result.grid[0]}×{best_result.grid[1]}×{best_result.grid[2]}@{best_result.grid[3]}")
    print(f"  结果保存至：{best_path}")
    print(f"  总运行时间：{time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
