"""单枚烟幕弹的共用运动学与遮蔽评价内核。

该模块不关心“问题一的固定策略”或“问题二的优化算法”。它只接收一组
投放方案，返回在给定目标采样口径下的有效遮蔽区间与总时长。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Literal

import numpy as np

SamplingMode = Literal["point", "rim", "surface"]


@dataclass(frozen=True)
class ScenarioParameters:
    """题面给定的物理与几何常量。"""

    gravity: float = 9.8
    missile_speed: float = 300.0
    cloud_sink_speed: float = 3.0
    cloud_radius: float = 10.0
    cloud_lifetime: float = 20.0

    missile_initial: tuple[float, float, float] = (20000.0, 0.0, 2000.0)
    uav_initial: tuple[float, float, float] = (17800.0, 0.0, 1800.0)
    target_bottom_center: tuple[float, float, float] = (0.0, 200.0, 0.0)
    target_radius: float = 7.0
    target_height: float = 10.0

    uav_speed_min: float = 70.0
    uav_speed_max: float = 140.0
    max_burst_time: float = 13.94


@dataclass(frozen=True)
class SamplingConfig:
    """目标离散与时间求解精度。"""

    angle_count: int = 1440
    height_count: int = 41
    radial_count: int = 31
    scan_step: float = 0.05
    root_tolerance: float = 1.0e-10


@dataclass(frozen=True)
class Deployment:
    """一枚烟幕弹的题目原始决策变量。"""

    heading: float
    speed: float
    release_time: float
    fuse_delay: float

    @property
    def burst_time(self) -> float:
        return self.release_time + self.fuse_delay


@dataclass(frozen=True)
class EvaluationResult:
    """单次评价的结构化结果。"""

    deployment: Deployment
    mode: SamplingMode
    feasible: bool
    reason: str
    release_point: np.ndarray | None
    burst_point: np.ndarray | None
    valid_start: float | None
    valid_end: float | None
    intervals: tuple[tuple[float, float], ...]
    duration: float


@dataclass(frozen=True)
class MarginDiagnostic:
    """某一时刻、某一目标采样口径下的最不利视线诊断结果。"""

    time: float
    mode: SamplingMode
    cloud_center: np.ndarray
    missile_position: np.ndarray
    target_point: np.ndarray
    closest_point: np.ndarray
    closest_projection: float
    max_distance: float
    margin: float


class SmokeEvaluator:
    """共享且只读的评价内核；目标采样点只在构造时生成一次。"""

    def __init__(
        self,
        parameters: ScenarioParameters | None = None,
        sampling: SamplingConfig | None = None,
    ) -> None:
        self.parameters = parameters or ScenarioParameters()
        self.sampling = sampling or SamplingConfig()

        self.missile_initial = np.asarray(self.parameters.missile_initial, dtype=float)
        self.uav_initial = np.asarray(self.parameters.uav_initial, dtype=float)
        self.target_point = np.asarray(
            self.parameters.target_bottom_center,
            dtype=float,
        )
        self.missile_direction = -self.missile_initial / np.linalg.norm(
            self.missile_initial
        )
        self.missile_velocity = (
            self.parameters.missile_speed * self.missile_direction
        )
        self.missile_impact_time = (
            np.linalg.norm(self.missile_initial) / self.parameters.missile_speed
        )

        self.cylinder_surface_points = self._build_cylinder_surface_points()
        self.cylinder_rim_points = self._build_cylinder_rim_points()

    def validate(self, deployment: Deployment) -> str | None:
        """返回不可行原因；返回 None 表示可行。"""

        values = (
            deployment.heading,
            deployment.speed,
            deployment.release_time,
            deployment.fuse_delay,
        )
        if not all(isfinite(value) for value in values):
            return "决策变量必须为有限实数。"
        if not 0.0 <= deployment.heading <= 2.0 * np.pi:
            return "航向角必须位于 [0, 2π]。"
        if not self.parameters.uav_speed_min <= deployment.speed <= self.parameters.uav_speed_max:
            return "无人机速度超出题设范围。"
        if deployment.release_time < 0.0 or deployment.fuse_delay < 0.0:
            return "投放时刻和引爆延时必须非负。"
        if deployment.burst_time > self.parameters.max_burst_time + 1.0e-12:
            return "投放时刻与引爆延时之和超过 13.94 s。"

        burst_height = (
            self.uav_initial[2]
            - 0.5 * self.parameters.gravity * deployment.fuse_delay**2
        )
        if burst_height < -1.0e-10:
            return "干扰弹将在地面以下起爆。"
        return None

    def simulation(self, deployment: Deployment) -> SmokeSimulation | None:
        """为一组可行方案建立轻量仿真实例；采样点与本内核共享。"""

        if self.validate(deployment) is not None:
            return None
        return SmokeSimulation(self, deployment)

    def evaluate(
        self,
        deployment: Deployment,
        mode: SamplingMode = "rim",
    ) -> EvaluationResult:
        reason = self.validate(deployment)
        if reason is not None:
            return EvaluationResult(
                deployment=deployment,
                mode=mode,
                feasible=False,
                reason=reason,
                release_point=None,
                burst_point=None,
                valid_start=None,
                valid_end=None,
                intervals=(),
                duration=0.0,
            )

        simulation = SmokeSimulation(self, deployment)
        intervals = tuple(simulation.find_effective_intervals_for_mode(mode))
        return EvaluationResult(
            deployment=deployment,
            mode=mode,
            feasible=True,
            reason="",
            release_point=simulation.release_point.copy(),
            burst_point=simulation.burst_point.copy(),
            valid_start=simulation.valid_start,
            valid_end=simulation.valid_end,
            intervals=intervals,
            duration=simulation.total_duration(intervals),
        )

    def _build_cylinder_surface_points(self) -> np.ndarray:
        angles = np.linspace(
            0.0,
            2.0 * np.pi,
            self.sampling.angle_count,
            endpoint=False,
        )
        cos_values = np.cos(angles)
        sin_values = np.sin(angles)
        x0, y0, z0 = self.target_point
        radius = self.parameters.target_radius

        heights = np.linspace(
            z0,
            z0 + self.parameters.target_height,
            self.sampling.height_count,
        )
        side_points = np.vstack(
            [
                np.column_stack(
                    (
                        x0 + radius * cos_values,
                        y0 + radius * sin_values,
                        np.full_like(angles, height),
                    )
                )
                for height in heights
            ]
        )

        radii = np.linspace(0.0, radius, self.sampling.radial_count)
        cap_points: list[np.ndarray] = []
        for cap_height in (z0, z0 + self.parameters.target_height):
            for current_radius in radii:
                cap_points.append(
                    np.column_stack(
                        (
                            x0 + current_radius * cos_values,
                            y0 + current_radius * sin_values,
                            np.full_like(angles, cap_height),
                        )
                    )
                )
        return np.vstack([side_points, *cap_points])

    def _build_cylinder_rim_points(self) -> np.ndarray:
        angles = np.linspace(
            0.0,
            2.0 * np.pi,
            self.sampling.angle_count,
            endpoint=False,
        )
        x0, y0, z0 = self.target_point
        radius = self.parameters.target_radius
        return np.vstack(
            [
                np.column_stack(
                    (
                        x0 + radius * np.cos(angles),
                        y0 + radius * np.sin(angles),
                        np.full_like(angles, height),
                    )
                )
                for height in (z0, z0 + self.parameters.target_height)
            ]
        )


class SmokeSimulation:
    """共享评价内核在给定单枚弹方案下的一次仿真。"""

    def __init__(self, evaluator: SmokeEvaluator, deployment: Deployment) -> None:
        self.evaluator = evaluator
        self.parameters = evaluator.parameters
        self.sampling = evaluator.sampling
        self.deployment = deployment

        self.missile_initial = evaluator.missile_initial
        self.uav_initial = evaluator.uav_initial
        self.target_point = evaluator.target_point
        self.missile_direction = evaluator.missile_direction
        self.missile_velocity = evaluator.missile_velocity
        self.missile_impact_time = evaluator.missile_impact_time
        self.cylinder_surface_points = evaluator.cylinder_surface_points
        self.cylinder_rim_points = evaluator.cylinder_rim_points

        self.uav_direction = np.array(
            [np.cos(deployment.heading), np.sin(deployment.heading), 0.0],
            dtype=float,
        )
        self.uav_velocity = deployment.speed * self.uav_direction
        self.burst_time = deployment.burst_time
        self.release_point = self.uav_position(deployment.release_time)
        self.burst_point = self.bomb_position(self.burst_time)
        self.valid_start = self.burst_time
        self.valid_end = min(
            self.burst_time + self.parameters.cloud_lifetime,
            self.missile_impact_time,
        )

    def missile_position(self, time: float) -> np.ndarray:
        return self.missile_initial + self.missile_velocity * time

    def uav_position(self, time: float) -> np.ndarray:
        return self.uav_initial + self.uav_velocity * time

    def bomb_position(self, time: float) -> np.ndarray:
        elapsed = time - self.deployment.release_time
        if elapsed < 0.0:
            raise ValueError("烟幕弹尚未投放。")
        horizontal_position = self.release_point + self.uav_velocity * elapsed
        position = horizontal_position.copy()
        position[2] = (
            self.release_point[2]
            - 0.5 * self.parameters.gravity * elapsed**2
        )
        return position

    def cloud_center(self, time: float) -> np.ndarray:
        elapsed = time - self.burst_time
        if elapsed < 0.0:
            raise ValueError("烟幕弹尚未起爆。")
        center = self.burst_point.copy()
        center[2] -= self.parameters.cloud_sink_speed * elapsed
        return center

    @staticmethod
    def distances_to_sight_segments(
        cloud_center: np.ndarray,
        missile_position: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        directions = target_points - missile_position
        denominators = np.einsum("ij,ij->i", directions, directions)
        projection = directions @ (cloud_center - missile_position) / denominators
        projection = np.clip(projection, 0.0, 1.0)
        closest_points = missile_position + projection[:, None] * directions
        return np.linalg.norm(closest_points - cloud_center, axis=1)

    def _margin_for_points(self, time: float, target_points: np.ndarray) -> float:
        distances = self.distances_to_sight_segments(
            self.cloud_center(time),
            self.missile_position(time),
            target_points,
        )
        return self.parameters.cloud_radius - float(np.max(distances))

    def margin_diagnostic(
        self,
        time: float,
        mode: SamplingMode = "surface",
    ) -> MarginDiagnostic:
        """返回最不利目标点及其对应有限视线段的完整几何信息。

        该接口服务于结果复核与绘图；计算口径与 ``margin_function`` 完全
        一致。调用时刻必须位于烟幕起爆后。
        """

        if mode == "point":
            target_points = self.target_point[None, :]
        elif mode == "rim":
            target_points = self.cylinder_rim_points
        elif mode == "surface":
            target_points = self.cylinder_surface_points
        else:
            raise ValueError(f"未知采样模式：{mode}")

        cloud_center = self.cloud_center(time)
        missile_position = self.missile_position(time)
        directions = target_points - missile_position
        denominators = np.einsum("ij,ij->i", directions, directions)
        projection = directions @ (cloud_center - missile_position) / denominators
        projection = np.clip(projection, 0.0, 1.0)
        closest_points = missile_position + projection[:, None] * directions
        distances = np.linalg.norm(closest_points - cloud_center, axis=1)
        worst_index = int(np.argmax(distances))
        max_distance = float(distances[worst_index])
        return MarginDiagnostic(
            time=float(time),
            mode=mode,
            cloud_center=cloud_center.copy(),
            missile_position=missile_position.copy(),
            target_point=target_points[worst_index].copy(),
            closest_point=closest_points[worst_index].copy(),
            closest_projection=float(projection[worst_index]),
            max_distance=max_distance,
            margin=self.parameters.cloud_radius - max_distance,
        )

    def point_target_margin(self, time: float) -> float:
        return self._margin_for_points(time, self.target_point[None, :])

    def cylinder_target_margin(self, time: float) -> float:
        return self._margin_for_points(time, self.cylinder_surface_points)

    def cylinder_rim_margin(self, time: float) -> float:
        return self._margin_for_points(time, self.cylinder_rim_points)

    def margin_function(self, mode: SamplingMode) -> Callable[[float], float]:
        if mode == "point":
            return self.point_target_margin
        if mode == "rim":
            return self.cylinder_rim_margin
        if mode == "surface":
            return self.cylinder_target_margin
        raise ValueError(f"未知采样模式：{mode}")

    def _bisect_boundary(
        self,
        margin_function: Callable[[float], float],
        left: float,
        right: float,
    ) -> float:
        left_value = margin_function(left)
        right_value = margin_function(right)
        if left_value == 0.0:
            return left
        if right_value == 0.0:
            return right
        if left_value * right_value > 0.0:
            raise ValueError("二分区间两端没有发生遮蔽状态变化。")

        while right - left > self.sampling.root_tolerance:
            middle = 0.5 * (left + right)
            middle_value = margin_function(middle)
            if left_value * middle_value <= 0.0:
                right = middle
                right_value = middle_value
            else:
                left = middle
                left_value = middle_value
        return 0.5 * (left + right)

    def find_effective_intervals(
        self,
        margin_function: Callable[[float], float],
    ) -> list[tuple[float, float]]:
        if self.valid_start >= self.valid_end:
            return []

        times = np.arange(
            self.valid_start,
            self.valid_end,
            self.sampling.scan_step,
            dtype=float,
        )
        if times.size == 0 or times[-1] < self.valid_end:
            times = np.append(times, self.valid_end)
        margins = np.array([margin_function(float(time)) for time in times])
        inside = margins >= 0.0

        intervals: list[tuple[float, float]] = []
        current_start: float | None = self.valid_start if inside[0] else None
        for index in range(1, len(times)):
            if inside[index] == inside[index - 1]:
                continue
            boundary = self._bisect_boundary(
                margin_function,
                float(times[index - 1]),
                float(times[index]),
            )
            if inside[index]:
                current_start = boundary
            else:
                if current_start is None:
                    raise RuntimeError("发现遮蔽结束边界，但没有对应的开始边界。")
                intervals.append((current_start, boundary))
                current_start = None
        if current_start is not None:
            intervals.append((current_start, self.valid_end))
        return intervals

    def find_effective_intervals_for_mode(
        self,
        mode: SamplingMode,
    ) -> list[tuple[float, float]]:
        return self.find_effective_intervals(self.margin_function(mode))

    @staticmethod
    def total_duration(intervals: tuple[tuple[float, float], ...] | list[tuple[float, float]]) -> float:
        return sum(end - start for start, end in intervals)
