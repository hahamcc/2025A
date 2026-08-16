"""多枚烟幕弹的联合完整遮蔽评价内核。

该模块用于问题三。与单弹 ``smoke_evaluator`` 并列存在，不改变问题一、二
已经回归通过的单弹逻辑。联合遮蔽要求目标表面每一点均至少被一枚当前有效
烟幕遮住；空间上使用 Lipschitz 上界驱动的自适应面片认证。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Iterable, Literal

import numpy as np

from .smoke_evaluator import ScenarioParameters

PatchKind = Literal["side", "cap"]
PatchStatus = Literal["pass", "fail", "uncertain"]
TimeStatus = Literal["valid", "invalid", "uncertain"]


@dataclass(frozen=True)
class ThreeDeployment:
    """同一无人机投放的一至三枚烟幕弹的物理策略。"""

    heading: float
    speed: float
    release_times: tuple[float, ...]
    fuse_delays: tuple[float, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.release_times) <= 3:
            raise ValueError("联合评价器支持一至三枚烟幕弹。")
        if len(self.release_times) != len(self.fuse_delays):
            raise ValueError("投放时刻与引爆延迟数量必须一致。")

    @property
    def burst_times(self) -> tuple[float, ...]:
        return tuple(
            release + delay
            for release, delay in zip(self.release_times, self.fuse_delays)
        )


@dataclass(frozen=True)
class AdaptiveSurfaceConfig:
    """完整圆柱表面自适应认证和时间扫描配置。"""

    initial_angle_count: int = 8
    initial_height_count: int = 2
    initial_radial_count: int = 2
    rho_min: float = 0.5
    max_depth: int = 8
    max_patches: int = 20_000
    scan_step: float = 0.10
    root_tolerance: float = 1.0e-6
    merge_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        if min(
            self.initial_angle_count,
            self.initial_height_count,
            self.initial_radial_count,
        ) <= 0:
            raise ValueError("初始面片划分数必须为正。")
        if self.rho_min <= 0.0 or self.scan_step <= 0.0:
            raise ValueError("rho_min 和扫描步长必须为正。")
        if self.max_depth < 0 or self.max_patches <= 0:
            raise ValueError("最大递归深度和最大面片数必须有效。")


@dataclass(frozen=True)
class SurfacePatch:
    """侧面矩形面片或端面扇环面片。"""

    kind: PatchKind
    phi0: float
    phi1: float
    lower: float
    upper: float
    cap_height: float | None = None

    def center(self, parameters: ScenarioParameters) -> np.ndarray:
        phi = 0.5 * (self.phi0 + self.phi1)
        x0, y0, z0 = parameters.target_bottom_center
        if self.kind == "side":
            return np.array(
                (
                    x0 + parameters.target_radius * np.cos(phi),
                    y0 + parameters.target_radius * np.sin(phi),
                    0.5 * (self.lower + self.upper),
                ),
                dtype=float,
            )
        radius = 0.5 * (self.lower + self.upper)
        return np.array(
            (
                x0 + radius * np.cos(phi),
                y0 + radius * np.sin(phi),
                z0 if self.cap_height is None else self.cap_height,
            ),
            dtype=float,
        )

    def rho_bound(self, parameters: ScenarioParameters) -> float:
        delta_phi = self.phi1 - self.phi0
        if self.kind == "side":
            angle_distance = 2.0 * parameters.target_radius * np.sin(
                delta_phi / 4.0
            )
            return float(
                np.hypot(angle_distance, 0.5 * (self.upper - self.lower))
            )
        return float(
            0.5 * (self.upper - self.lower)
            + 2.0 * self.upper * np.sin(delta_phi / 4.0)
        )

    def children(self) -> tuple["SurfacePatch", ...]:
        mid_phi = 0.5 * (self.phi0 + self.phi1)
        mid_value = 0.5 * (self.lower + self.upper)
        return tuple(
            SurfacePatch(
                self.kind,
                phi0,
                phi1,
                lower,
                upper,
                self.cap_height,
            )
            for phi0, phi1 in ((self.phi0, mid_phi), (mid_phi, self.phi1))
            for lower, upper in ((self.lower, mid_value), (mid_value, self.upper))
        )


@dataclass(frozen=True)
class TimeDiagnostic:
    time: float
    status: TimeStatus
    active_indices: tuple[int, ...]
    checked_patches: int
    passed_patches: int
    uncertain_patches: int
    witness_point: np.ndarray | None
    witness_distance: float | None


@dataclass(frozen=True)
class UniformReview:
    intervals: tuple[tuple[float, float], ...]
    duration: float
    angle_count: int
    height_count: int
    radial_count: int


@dataclass(frozen=True)
class JointEvaluationResult:
    deployment: ThreeDeployment
    feasible: bool
    reason: str
    release_points: tuple[np.ndarray, ...]
    burst_points: tuple[np.ndarray, ...]
    intervals: tuple[tuple[float, float], ...]
    duration: float
    diagnostics: tuple[TimeDiagnostic, ...] = field(default_factory=tuple)

    @property
    def uncertain_checks(self) -> int:
        return sum(item.status == "uncertain" for item in self.diagnostics)

    @property
    def checked_patches(self) -> int:
        return sum(item.checked_patches for item in self.diagnostics)


class MultiSmokeEvaluator:
    """场景常量、面片划分和三烟幕仿真的只读工厂。"""

    def __init__(
        self,
        parameters: ScenarioParameters | None = None,
        config: AdaptiveSurfaceConfig | None = None,
    ) -> None:
        self.parameters = parameters or ScenarioParameters(max_burst_time=13.9423)
        self.config = config or AdaptiveSurfaceConfig()
        self.missile_initial = np.asarray(self.parameters.missile_initial, dtype=float)
        self.uav_initial = np.asarray(self.parameters.uav_initial, dtype=float)
        self.target_bottom_center = np.asarray(
            self.parameters.target_bottom_center, dtype=float
        )
        self.missile_direction = -self.missile_initial / np.linalg.norm(
            self.missile_initial
        )
        self.missile_velocity = self.parameters.missile_speed * self.missile_direction
        self.missile_impact_time = (
            np.linalg.norm(self.missile_initial) / self.parameters.missile_speed
        )
        self.initial_patches = self._build_initial_patches()

    def _build_initial_patches(self) -> tuple[SurfacePatch, ...]:
        angle_edges = np.linspace(
            0.0, 2.0 * np.pi, self.config.initial_angle_count + 1
        )
        z0 = self.parameters.target_bottom_center[2]
        height_edges = np.linspace(
            z0,
            z0 + self.parameters.target_height,
            self.config.initial_height_count + 1,
        )
        radial_edges = np.linspace(
            0.0, self.parameters.target_radius, self.config.initial_radial_count + 1
        )
        patches: list[SurfacePatch] = []
        for phi0, phi1 in zip(angle_edges[:-1], angle_edges[1:]):
            for lower, upper in zip(height_edges[:-1], height_edges[1:]):
                patches.append(SurfacePatch("side", phi0, phi1, lower, upper))
            for cap_height in (z0, z0 + self.parameters.target_height):
                for lower, upper in zip(radial_edges[:-1], radial_edges[1:]):
                    patches.append(
                        SurfacePatch("cap", phi0, phi1, lower, upper, cap_height)
                    )
        return tuple(patches)

    def validate(self, deployment: ThreeDeployment) -> str | None:
        values = (
            deployment.heading,
            deployment.speed,
            *deployment.release_times,
            *deployment.fuse_delays,
        )
        if not all(isfinite(value) for value in values):
            return "决策变量必须为有限实数。"
        if not 0.0 <= deployment.heading <= 2.0 * np.pi:
            return "航向角必须位于 [0, 2π]。"
        if not self.parameters.uav_speed_min <= deployment.speed <= self.parameters.uav_speed_max:
            return "无人机速度超出题设范围。"
        if any(value < 0.0 for value in deployment.release_times):
            return "投放时刻必须非负。"
        if any(value < 0.0 for value in deployment.fuse_delays):
            return "引爆延迟必须非负。"
        if any(
            burst > self.parameters.max_burst_time + 1.0e-12
            for burst in deployment.burst_times
        ):
            return "起爆时刻超过允许上界。"
        for delay in deployment.fuse_delays:
            height = self.uav_initial[2] - 0.5 * self.parameters.gravity * delay**2
            if height < -1.0e-10:
                return "干扰弹将在地面以下起爆。"
        return None

    def simulation(self, deployment: ThreeDeployment) -> "ThreeSmokeSimulation | None":
        if self.validate(deployment) is not None:
            return None
        return ThreeSmokeSimulation(self, deployment)

    def evaluate(
        self, deployment: ThreeDeployment, *, collect_diagnostics: bool = False
    ) -> JointEvaluationResult:
        reason = self.validate(deployment)
        if reason is not None:
            return JointEvaluationResult(
                deployment=deployment,
                feasible=False,
                reason=reason,
                release_points=(),
                burst_points=(),
                intervals=(),
                duration=0.0,
            )
        simulation = ThreeSmokeSimulation(self, deployment)
        intervals, diagnostics = simulation.find_joint_intervals(collect_diagnostics)
        return JointEvaluationResult(
            deployment=deployment,
            feasible=True,
            reason="",
            release_points=tuple(point.copy() for point in simulation.release_points),
            burst_points=tuple(point.copy() for point in simulation.burst_points),
            intervals=tuple(intervals),
            duration=simulation.total_duration(intervals),
            diagnostics=tuple(diagnostics),
        )


class ThreeSmokeSimulation:
    def __init__(self, evaluator: MultiSmokeEvaluator, deployment: ThreeDeployment) -> None:
        self.evaluator = evaluator
        self.parameters = evaluator.parameters
        self.config = evaluator.config
        self.deployment = deployment
        self.missile_initial = evaluator.missile_initial
        self.missile_velocity = evaluator.missile_velocity
        self.missile_impact_time = evaluator.missile_impact_time
        self.uav_initial = evaluator.uav_initial
        self.uav_velocity = deployment.speed * np.array(
            [np.cos(deployment.heading), np.sin(deployment.heading), 0.0], dtype=float
        )
        self.release_points = tuple(
            self.uav_position(time) for time in deployment.release_times
        )
        self.burst_points = tuple(
            self._bomb_position(index, time)
            for index, time in enumerate(deployment.burst_times)
        )

    def missile_position(self, time: float) -> np.ndarray:
        return self.missile_initial + self.missile_velocity * time

    def uav_position(self, time: float) -> np.ndarray:
        return self.uav_initial + self.uav_velocity * time

    def _bomb_position(self, index: int, time: float) -> np.ndarray:
        elapsed = time - self.deployment.release_times[index]
        if elapsed < -1.0e-12:
            raise ValueError("烟幕弹尚未投放。")
        position = self.release_points[index] + self.uav_velocity * elapsed
        position = position.copy()
        position[2] -= 0.5 * self.parameters.gravity * elapsed**2
        return position

    def cloud_center(self, index: int, time: float) -> np.ndarray:
        elapsed = time - self.deployment.burst_times[index]
        if elapsed < -1.0e-12:
            raise ValueError("烟幕弹尚未起爆。")
        center = self.burst_points[index].copy()
        center[2] -= self.parameters.cloud_sink_speed * elapsed
        return center

    def active_indices(self, time: float) -> tuple[int, ...]:
        return tuple(
            index
            for index, burst in enumerate(self.deployment.burst_times)
            if burst <= time <= min(
                burst + self.parameters.cloud_lifetime, self.missile_impact_time
            )
        )

    def _minimum_distances(
        self, time: float, target_points: np.ndarray, active_indices: tuple[int, ...]
    ) -> np.ndarray:
        if not active_indices:
            return np.full(len(target_points), np.inf, dtype=float)
        missile = self.missile_position(time)
        directions = target_points - missile
        denominators = np.einsum("ij,ij->i", directions, directions)
        centers = np.vstack(
            [self.cloud_center(index, time) for index in active_indices]
        )
        projection = (centers - missile) @ directions.T / denominators[None, :]
        projection = np.clip(projection, 0.0, 1.0)
        closest = missile + projection[:, :, None] * directions[None, :, :]
        distances = np.linalg.norm(closest - centers[:, None, :], axis=2)
        return np.min(distances, axis=0)

    def _check_patch(
        self,
        patch: SurfacePatch,
        time: float,
        active_indices: tuple[int, ...],
        depth: int,
        counters: dict[str, object],
    ) -> PatchStatus:
        counters["checked"] = int(counters["checked"]) + 1
        if int(counters["checked"]) > self.config.max_patches:
            counters["uncertain"] = int(counters["uncertain"]) + 1
            return "uncertain"
        point = patch.center(self.parameters)
        value = float(
            self._minimum_distances(time, point[None, :], active_indices)[0]
        )
        rho = patch.rho_bound(self.parameters)
        if value > self.parameters.cloud_radius:
            counters["witness_point"] = point
            counters["witness_distance"] = value
            return "fail"
        if value + rho <= self.parameters.cloud_radius:
            counters["passed"] = int(counters["passed"]) + 1
            return "pass"
        if rho <= self.config.rho_min or depth >= self.config.max_depth:
            counters["uncertain"] = int(counters["uncertain"]) + 1
            return "uncertain"
        statuses: list[PatchStatus] = []
        for child in patch.children():
            child_status = self._check_patch(
                child, time, active_indices, depth + 1, counters
            )
            statuses.append(child_status)
            if child_status == "fail":
                return "fail"
        if "uncertain" in statuses:
            return "uncertain"
        return "pass"

    def check_time(
        self,
        time: float,
        active_indices: tuple[int, ...] | None = None,
    ) -> TimeDiagnostic:
        active = self.active_indices(time) if active_indices is None else active_indices
        if not active:
            return TimeDiagnostic(
                time=float(time),
                status="invalid",
                active_indices=(),
                checked_patches=0,
                passed_patches=0,
                uncertain_patches=0,
                witness_point=None,
                witness_distance=None,
            )
        # 同一层所有面片共用同一组导弹、烟幕球心。按层批量计算距离，避免
        # 递归时对每个面片都单独构造一次“点—有限视线段”距离，才能使 DE
        # 阶段的完整表面认证保持可承受的运行时间。
        checked = 0
        passed = 0
        uncertain = 0
        witness_point: np.ndarray | None = None
        witness_distance: float | None = None
        frontier: list[tuple[SurfacePatch, int]] = [
            (patch, 0) for patch in self.evaluator.initial_patches
        ]
        status: TimeStatus = "valid"
        while frontier:
            if checked + len(frontier) > self.config.max_patches:
                uncertain += 1
                status = "uncertain"
                break
            patches = [item[0] for item in frontier]
            depths = np.fromiter((item[1] for item in frontier), dtype=int)
            points = np.vstack([patch.center(self.parameters) for patch in patches])
            distances = self._minimum_distances(time, points, active)
            rhos = np.fromiter(
                (patch.rho_bound(self.parameters) for patch in patches), dtype=float
            )
            checked += len(patches)
            failed = distances > self.parameters.cloud_radius
            if np.any(failed):
                index = int(np.flatnonzero(failed)[0])
                witness_point = points[index]
                witness_distance = float(distances[index])
                status = "invalid"
                break
            certified = distances + rhos <= self.parameters.cloud_radius
            passed += int(np.sum(certified))
            unresolved = np.flatnonzero(~certified)
            if not len(unresolved):
                break
            cannot_refine = (rhos[unresolved] <= self.config.rho_min) | (
                depths[unresolved] >= self.config.max_depth
            )
            if np.any(cannot_refine):
                uncertain += int(np.sum(cannot_refine))
                status = "uncertain"
                break
            next_frontier: list[tuple[SurfacePatch, int]] = []
            for index in unresolved:
                next_frontier.extend(
                    (child, int(depths[index]) + 1)
                    for child in patches[int(index)].children()
                )
            frontier = next_frontier
        return TimeDiagnostic(
            time=float(time),
            status=status,
            active_indices=active,
            checked_patches=checked,
            passed_patches=passed,
            uncertain_patches=uncertain,
            witness_point=witness_point,
            witness_distance=witness_distance,
        )

    def _event_segments(self) -> tuple[tuple[float, float, tuple[int, ...]], ...]:
        events = {0.0, self.missile_impact_time}
        for burst in self.deployment.burst_times:
            if 0.0 <= burst <= self.missile_impact_time:
                events.add(burst)
            end = min(burst + self.parameters.cloud_lifetime, self.missile_impact_time)
            if 0.0 <= end <= self.missile_impact_time:
                events.add(end)
        ordered = sorted(events)
        segments: list[tuple[float, float, tuple[int, ...]]] = []
        for left, right in zip(ordered[:-1], ordered[1:]):
            if right - left <= 1.0e-12:
                continue
            midpoint = 0.5 * (left + right)
            segments.append((left, right, self.active_indices(midpoint)))
        return tuple(segments)

    def _segment_times(self, left: float, right: float) -> np.ndarray:
        times = np.arange(left, right, self.config.scan_step, dtype=float)
        if times.size == 0 or abs(times[-1] - right) > 1.0e-12:
            times = np.append(times, right)
        return times

    def _bisect_boundary(
        self,
        left: float,
        right: float,
        active_indices: tuple[int, ...],
        diagnostics: list[TimeDiagnostic] | None,
    ) -> float:
        left_state = self.check_time(left, active_indices)
        right_state = self.check_time(right, active_indices)
        if diagnostics is not None:
            diagnostics.extend((left_state, right_state))
        left_valid = left_state.status == "valid"
        right_valid = right_state.status == "valid"
        if left_valid == right_valid:
            raise ValueError("二分端点没有联合遮蔽状态变化。")
        while right - left > self.config.root_tolerance:
            middle = 0.5 * (left + right)
            middle_state = self.check_time(middle, active_indices)
            if diagnostics is not None:
                diagnostics.append(middle_state)
            middle_valid = middle_state.status == "valid"
            if middle_valid == left_valid:
                left = middle
                left_valid = middle_valid
            else:
                right = middle
                right_valid = middle_valid
        return 0.5 * (left + right)

    def find_joint_intervals(
        self, collect_diagnostics: bool = False
    ) -> tuple[list[tuple[float, float]], list[TimeDiagnostic]]:
        intervals: list[tuple[float, float]] = []
        diagnostics: list[TimeDiagnostic] = []
        for left, right, active in self._event_segments():
            if not active:
                continue
            times = self._segment_times(left, right)
            statuses = [self.check_time(float(time), active) for time in times]
            if collect_diagnostics:
                diagnostics.extend(statuses)
            valid = [state.status == "valid" for state in statuses]
            current_start = float(times[0]) if valid[0] else None
            for index in range(1, len(times)):
                if valid[index] == valid[index - 1]:
                    continue
                boundary = self._bisect_boundary(
                    float(times[index - 1]),
                    float(times[index]),
                    active,
                    diagnostics if collect_diagnostics else None,
                )
                if valid[index]:
                    current_start = boundary
                elif current_start is not None:
                    intervals.append((current_start, boundary))
                    current_start = None
            if current_start is not None:
                intervals.append((current_start, right))
        return self.merge_intervals(intervals, self.config.merge_tolerance), diagnostics

    @staticmethod
    def merge_intervals(
        intervals: Iterable[tuple[float, float]], tolerance: float = 1.0e-8
    ) -> list[tuple[float, float]]:
        ordered = sorted(intervals)
        if not ordered:
            return []
        merged = [list(ordered[0])]
        for start, end in ordered[1:]:
            if start <= merged[-1][1] + tolerance:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(float(start), float(end)) for start, end in merged]

    @staticmethod
    def total_duration(intervals: Iterable[tuple[float, float]]) -> float:
        return float(sum(end - start for start, end in intervals))

    @staticmethod
    def uniform_surface_points(
        parameters: ScenarioParameters,
        angle_count: int,
        height_count: int,
        radial_count: int,
    ) -> np.ndarray:
        angles = np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False)
        cosines, sines = np.cos(angles), np.sin(angles)
        x0, y0, z0 = parameters.target_bottom_center
        radius = parameters.target_radius
        heights = np.linspace(z0, z0 + parameters.target_height, height_count)
        side = np.vstack(
            [
                np.column_stack(
                    (
                        x0 + radius * cosines,
                        y0 + radius * sines,
                        np.full_like(angles, height),
                    )
                )
                for height in heights
            ]
        )
        caps: list[np.ndarray] = []
        for height in (z0, z0 + parameters.target_height):
            for current_radius in np.linspace(0.0, radius, radial_count):
                caps.append(
                    np.column_stack(
                        (
                            x0 + current_radius * cosines,
                            y0 + current_radius * sines,
                            np.full_like(angles, height),
                        )
                    )
                )
        return np.vstack((side, *caps))

    def uniform_review(
        self, angle_count: int, height_count: int, radial_count: int
    ) -> UniformReview:
        points = self.uniform_surface_points(
            self.parameters, angle_count, height_count, radial_count
        )
        return self.uniform_review_points(
            points,
            angle_count=angle_count,
            height_count=height_count,
            radial_count=radial_count,
        )

    def uniform_review_points(
        self,
        points: np.ndarray,
        *,
        angle_count: int,
        height_count: int,
        radial_count: int,
    ) -> UniformReview:
        """在预生成的完整表面点集上评价联合遮蔽。

        搜索阶段会评价大量候选。将固定的圆柱表面点集放到评价器中缓存，
        可以避免每次适应度调用都重新生成三角函数和坐标数组。这里仍使用
        与严格模型相同的 ``max_Q min_j d_j`` 判据，只改变空间离散精度。
        """

        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError("完整表面采样点必须是非空的 N×3 数组。")
        intervals: list[tuple[float, float]] = []
        for left, right, active in self._event_segments():
            if not active:
                continue
            times = self._segment_times(left, right)
            valid = [
                bool(
                    np.max(self._minimum_distances(float(time), points, active))
                    <= self.parameters.cloud_radius
                )
                for time in times
            ]
            current_start = float(times[0]) if valid[0] else None
            for index in range(1, len(times)):
                if valid[index] == valid[index - 1]:
                    continue
                left_time, right_time = float(times[index - 1]), float(times[index])
                left_valid = valid[index - 1]
                while right_time - left_time > self.config.root_tolerance:
                    middle = 0.5 * (left_time + right_time)
                    middle_valid = bool(
                        np.max(self._minimum_distances(middle, points, active))
                        <= self.parameters.cloud_radius
                    )
                    if middle_valid == left_valid:
                        left_time = middle
                    else:
                        right_time = middle
                boundary = 0.5 * (left_time + right_time)
                if valid[index]:
                    current_start = boundary
                elif current_start is not None:
                    intervals.append((current_start, boundary))
                    current_start = None
            if current_start is not None:
                intervals.append((current_start, right))
        merged = self.merge_intervals(intervals, self.config.merge_tolerance)
        return UniformReview(
            intervals=tuple(merged),
            duration=self.total_duration(merged),
            angle_count=angle_count,
            height_count=height_count,
            radial_count=radial_count,
        )
