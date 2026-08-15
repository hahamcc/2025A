from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class CriticalState:
    """完整圆柱遮蔽区间端点处的几何诊断结果。"""

    boundary_name: str
    time: float
    worst_point: np.ndarray
    closest_point: np.ndarray
    maximum_distance: float
    projection_parameter: float
    missile_cloud_distance: float
    mechanism: str
    worst_point_note: str


@dataclass(frozen=True)
class Q1Parameters:
    """第一问的题面参数与数值计算参数。"""

    gravity: float = 9.8
    missile_speed: float = 300.0
    uav_speed: float = 120.0
    release_time: float = 1.5
    fuse_delay: float = 3.6
    cloud_sink_speed: float = 3.0
    cloud_radius: float = 10.0
    cloud_lifetime: float = 20.0

    missile_initial: tuple[float, float, float] = (20000.0, 0.0, 2000.0)
    uav_initial: tuple[float, float, float] = (17800.0, 0.0, 1800.0)
    target_bottom_center: tuple[float, float, float] = (0.0, 200.0, 0.0)
    target_radius: float = 7.0
    target_height: float = 10.0

    # 圆柱表面采样：侧面 + 上下端面。
    angle_count: int = 1440
    height_count: int = 41
    radial_count: int = 31

    # 粗扫描只定位边界，最终边界由二分法精修。
    scan_step: float = 0.05
    root_tolerance: float = 1.0e-10


class Q1Model:
    """按主体法组织第一问的运动、几何判定与时间求解。"""

    def __init__(self, parameters: Q1Parameters) -> None:
        self.p = parameters
        self.missile_initial = np.asarray(parameters.missile_initial, dtype=float)
        self.uav_initial = np.asarray(parameters.uav_initial, dtype=float)
        self.target_point = np.asarray(parameters.target_bottom_center, dtype=float)

        # M1 以恒定速率直指假目标原点。
        self.missile_direction = -self.missile_initial / np.linalg.norm(
            self.missile_initial
        )
        self.missile_velocity = parameters.missile_speed * self.missile_direction

        # FY1 等高度朝假目标的水平投影方向飞行，即沿 x 轴负方向。
        self.uav_direction = np.array([-1.0, 0.0, 0.0])
        self.uav_velocity = parameters.uav_speed * self.uav_direction

        self.burst_time = parameters.release_time + parameters.fuse_delay
        self.release_point = self.uav_position(parameters.release_time)
        self.burst_point = self.bomb_position(self.burst_time)

        self.missile_impact_time = (
            np.linalg.norm(self.missile_initial) / parameters.missile_speed
        )
        self.valid_start = self.burst_time
        self.valid_end = min(
            self.burst_time + parameters.cloud_lifetime,
            self.missile_impact_time,
        )

        self.cylinder_surface_points = self._build_cylinder_surface_points()
        self.cylinder_rim_points = self._build_cylinder_rim_points()

    # ------------------------------------------------------------------
    # 主体一：来袭导弹 M1
    # ------------------------------------------------------------------
    def missile_position(self, time: float) -> np.ndarray:
        return self.missile_initial + self.missile_velocity * time

    # ------------------------------------------------------------------
    # 主体二：无人机 FY1
    # ------------------------------------------------------------------
    def uav_position(self, time: float) -> np.ndarray:
        return self.uav_initial + self.uav_velocity * time

    # ------------------------------------------------------------------
    # 主体三：烟幕干扰弹
    # ------------------------------------------------------------------
    def bomb_position(self, time: float) -> np.ndarray:
        """返回投放后、起爆前烟幕弹的位置。"""

        elapsed = time - self.p.release_time
        if elapsed < 0.0:
            raise ValueError("烟幕弹尚未投放。")

        horizontal_position = self.release_point + self.uav_velocity * elapsed
        position = horizontal_position.copy()
        position[2] = self.release_point[2] - 0.5 * self.p.gravity * elapsed**2
        return position

    # ------------------------------------------------------------------
    # 主体四：烟幕云团
    # ------------------------------------------------------------------
    def cloud_center(self, time: float) -> np.ndarray:
        """返回起爆后烟幕云团的球心位置。"""

        elapsed = time - self.burst_time
        if elapsed < 0.0:
            raise ValueError("烟幕弹尚未起爆。")

        center = self.burst_point.copy()
        center[2] -= self.p.cloud_sink_speed * elapsed
        return center

    # ------------------------------------------------------------------
    # 主体五：真实圆柱目标
    # ------------------------------------------------------------------
    def _build_cylinder_surface_points(self) -> np.ndarray:
        """生成圆柱侧面和上下端面的规则采样点。"""

        angles = np.linspace(
            0.0,
            2.0 * np.pi,
            self.p.angle_count,
            endpoint=False,
        )
        cos_values = np.cos(angles)
        sin_values = np.sin(angles)

        x0, y0, z0 = self.target_point
        radius = self.p.target_radius

        # 圆柱侧面。
        heights = np.linspace(z0, z0 + self.p.target_height, self.p.height_count)
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

        # 上、下端面。中心点会随角度重复，但不影响最大距离判定。
        radii = np.linspace(0.0, radius, self.p.radial_count)
        cap_points = []
        for cap_height in (z0, z0 + self.p.target_height):
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
        """生成圆柱上、下边缘圆周的规则采样点。

        在导弹位于烟幕球外且遮挡关系可表示为实心凸锥时，
        两条边缘圆周均被遮挡蕴含整个圆柱被遮挡。本点集用于
        验证这一降维判据，并为后续问题提供快速评价模式。
        """

        angles = np.linspace(
            0.0,
            2.0 * np.pi,
            self.p.angle_count,
            endpoint=False,
        )
        x0, y0, z0 = self.target_point
        radius = self.p.target_radius

        return np.vstack(
            [
                np.column_stack(
                    (
                        x0 + radius * np.cos(angles),
                        y0 + radius * np.sin(angles),
                        np.full_like(angles, height),
                    )
                )
                for height in (z0, z0 + self.p.target_height)
            ]
        )

    # ------------------------------------------------------------------
    # 主体间关系：导弹视线段与烟幕球的相交判定
    # ------------------------------------------------------------------
    @staticmethod
    def _distances_to_sight_segments(
        cloud_center: np.ndarray,
        missile_position: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        """计算球心到多条“导弹—目标点”视线段的最短距离。"""

        directions = target_points - missile_position
        denominators = np.einsum("ij,ij->i", directions, directions)
        projection = directions @ (cloud_center - missile_position) / denominators

        # 截断到 [0, 1]，保证计算的是线段而不是无限直线。
        projection = np.clip(projection, 0.0, 1.0)
        closest_points = missile_position + projection[:, None] * directions
        return np.linalg.norm(closest_points - cloud_center, axis=1)

    def point_target_margin(self, time: float) -> float:
        """点目标模型的遮蔽裕量；非负表示有效遮蔽。"""

        distances = self._distances_to_sight_segments(
            self.cloud_center(time),
            self.missile_position(time),
            self.target_point[None, :],
        )
        return self.p.cloud_radius - float(distances[0])

    def cylinder_target_margin(self, time: float) -> float:
        """完整圆柱模型的遮蔽裕量；非负表示整个目标均被遮蔽。"""

        distances = self._distances_to_sight_segments(
            self.cloud_center(time),
            self.missile_position(time),
            self.cylinder_surface_points,
        )
        worst_distance = float(np.max(distances))
        return self.p.cloud_radius - worst_distance

    def cylinder_rim_margin(self, time: float) -> float:
        """双圆周降维模型的遮蔽裕量；非负表示两条边缘圆周均被遮挡。"""

        distances = self._distances_to_sight_segments(
            self.cloud_center(time),
            self.missile_position(time),
            self.cylinder_rim_points,
        )
        worst_distance = float(np.max(distances))
        return self.p.cloud_radius - worst_distance

    def diagnose_cylinder_boundary(
        self,
        time: float,
        boundary_name: str,
    ) -> CriticalState:
        """提取区间端点的最不利视线，并识别临界状态的形成原因。"""

        missile = self.missile_position(time)
        cloud = self.cloud_center(time)
        target_points = self.cylinder_surface_points

        directions = target_points - missile
        denominators = np.einsum("ij,ij->i", directions, directions)
        raw_projection = directions @ (cloud - missile) / denominators
        segment_projection = np.clip(raw_projection, 0.0, 1.0)
        closest_points = missile + segment_projection[:, None] * directions
        distances = np.linalg.norm(closest_points - cloud, axis=1)

        worst_index = int(np.argmax(distances))
        worst_distance = float(distances[worst_index])
        worst_projection = float(raw_projection[worst_index])
        missile_cloud_distance = float(np.linalg.norm(cloud - missile))

        distance_tolerance = 1.0e-5
        if (
            0.0 < worst_projection < 1.0
            and abs(worst_distance - self.p.cloud_radius) <= distance_tolerance
        ):
            mechanism = "最不利目标视线与烟幕球相切"
            worst_point_note = "圆柱底部边缘的临界采样点"
        elif (
            worst_projection <= 0.0
            and abs(missile_cloud_distance - self.p.cloud_radius)
            <= distance_tolerance
        ):
            mechanism = "导弹穿出烟幕球，烟幕转至导弹后方"
            worst_point_note = "非唯一；此时线段最近点退化为导弹端点"
        else:
            mechanism = "其他线段—球体临界状态"
            worst_point_note = "数值诊断所对应的代表性采样点"

        return CriticalState(
            boundary_name=boundary_name,
            time=time,
            worst_point=target_points[worst_index].copy(),
            closest_point=closest_points[worst_index].copy(),
            maximum_distance=worst_distance,
            projection_parameter=worst_projection,
            missile_cloud_distance=missile_cloud_distance,
            mechanism=mechanism,
            worst_point_note=worst_point_note,
        )

    # ------------------------------------------------------------------
    # 时间算法：粗扫描定位 + 二分法精修
    # ------------------------------------------------------------------
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

        while right - left > self.p.root_tolerance:
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
        """求烟幕有效时间窗内所有遮蔽区间。"""

        times = np.arange(
            self.valid_start,
            self.valid_end,
            self.p.scan_step,
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

    @staticmethod
    def total_duration(intervals: list[tuple[float, float]]) -> float:
        return sum(end - start for start, end in intervals)


def format_vector(vector: np.ndarray) -> str:
    cleaned = np.where(np.abs(vector) < 5.0e-13, 0.0, vector)
    return "(" + ", ".join(f"{value:.6f}" for value in cleaned) + ")"


def print_intervals(
    title: str,
    intervals: list[tuple[float, float]],
) -> None:
    print(f"\n{title}")
    if not intervals:
        print("  未出现有效遮蔽区间。")
        return

    for number, (start, end) in enumerate(intervals, start=1):
        print(
            f"  区间 {number}: [{start:.10f}, {end:.10f}] s, "
            f"时长 {end - start:.10f} s"
        )
    print(f"  总有效遮蔽时长: {Q1Model.total_duration(intervals):.10f} s")


def save_results_csv(
    model: Q1Model,
    point_intervals: list[tuple[float, float]],
    cylinder_intervals: list[tuple[float, float]],
    rim_intervals: list[tuple[float, float]],
) -> Path:
    """将论文需要的关键结果保存为 Excel 可直接打开的 CSV 表。"""

    output_path = Path(__file__).resolve().parent / "q1_results.csv"
    fieldnames = [
        "模型口径",
        "区间编号",
        "投放时刻_s",
        "投放点x_m",
        "投放点y_m",
        "投放点z_m",
        "起爆时刻_s",
        "起爆点x_m",
        "起爆点y_m",
        "起爆点z_m",
        "遮蔽开始时刻_s",
        "遮蔽结束时刻_s",
        "本区间有效时长_s",
        "总有效遮蔽时长_s",
        "是否作为最终结果",
        "结果用途",
    ]

    rows: list[dict[str, str | int | float]] = []
    model_results = [
        (
            "下底面圆心特征点",
            point_intervals,
            "否",
            "基础简化模型与程序校验",
        ),
        (
            "完整圆柱全部遮蔽",
            cylinder_intervals,
            "是",
            "第一问最终严格结果",
        ),
        (
            "上下边缘双圆周遮蔽",
            rim_intervals,
            "否",
            "凸锥降维判据与后续快速评价器验证",
        ),
    ]

    for model_name, intervals, is_final, purpose in model_results:
        total = model.total_duration(intervals)
        for interval_number, (start, end) in enumerate(intervals, start=1):
            rows.append(
                {
                    "模型口径": model_name,
                    "区间编号": interval_number,
                    "投放时刻_s": f"{model.p.release_time:.4f}",
                    "投放点x_m": f"{model.release_point[0]:.6f}",
                    "投放点y_m": f"{model.release_point[1]:.6f}",
                    "投放点z_m": f"{model.release_point[2]:.6f}",
                    "起爆时刻_s": f"{model.burst_time:.4f}",
                    "起爆点x_m": f"{model.burst_point[0]:.6f}",
                    "起爆点y_m": f"{model.burst_point[1]:.6f}",
                    "起爆点z_m": f"{model.burst_point[2]:.6f}",
                    "遮蔽开始时刻_s": f"{start:.10f}",
                    "遮蔽结束时刻_s": f"{end:.10f}",
                    "本区间有效时长_s": f"{end - start:.10f}",
                    "总有效遮蔽时长_s": f"{total:.10f}",
                    "是否作为最终结果": is_final,
                    "结果用途": purpose,
                }
            )

    # utf-8-sig 可使 Windows Excel 直接正确识别中文表头。
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def save_critical_states_csv(states: list[CriticalState]) -> Path:
    """保存两个临界时刻的几何诊断量，供论文制表引用。"""

    output_path = Path(__file__).resolve().parent / "q1_critical_states.csv"
    fieldnames = [
        "边界",
        "临界时刻_s",
        "最不利点x_m",
        "最不利点y_m",
        "最不利点z_m",
        "最大视线距离_m",
        "原始投影参数s",
        "导弹至烟幕球心距离_m",
        "临界机制",
        "最不利点说明",
    ]

    rows = []
    for state in states:
        rows.append(
            {
                "边界": state.boundary_name,
                "临界时刻_s": f"{state.time:.10f}",
                "最不利点x_m": f"{state.worst_point[0]:.6f}",
                "最不利点y_m": f"{state.worst_point[1]:.6f}",
                "最不利点z_m": f"{state.worst_point[2]:.6f}",
                "最大视线距离_m": f"{state.maximum_distance:.10f}",
                "原始投影参数s": f"{state.projection_parameter:.10f}",
                "导弹至烟幕球心距离_m": (
                    f"{state.missile_cloud_distance:.10f}"
                ),
                "临界机制": state.mechanism,
                "最不利点说明": state.worst_point_note,
            }
        )

    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def plot_critical_state_figure(
    model: Q1Model,
    interval: tuple[float, float],
    states: list[CriticalState],
) -> Path:
    """绘制紧凑的裕度曲线和两个三维关系的二维法截面示意图。"""

    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Circle, Polygon, Rectangle

    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if font_name in installed_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 13
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12

    start, end = interval
    start_state, end_state = states

    figure = plt.figure(figsize=(11.4, 6.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(0.62, 1.38))
    margin_axis = figure.add_subplot(grid[0, :])
    tangent_axis = figure.add_subplot(grid[1, 0])
    exit_axis = figure.add_subplot(grid[1, 1])
    figure.suptitle("有效遮蔽区间及两端临界状态的二维法截面解释", fontsize=17)

    # 上部：用唯一的主模型裕度函数直接标出有效区间。
    time_left = max(model.valid_start, start - 0.25)
    time_right = min(model.valid_end, end + 0.08)
    times = np.linspace(time_left, time_right, 181)
    margins = np.array(
        [model.cylinder_target_margin(float(time)) for time in times]
    )

    margin_axis.plot(times, margins, color="#245b8a", linewidth=2.3)
    margin_axis.axhline(0.0, color="#505050", linewidth=0.9)
    margin_axis.axvspan(start, end, color="#4c9f70", alpha=0.14)
    margin_axis.scatter(
        [start, end],
        [0.0, 0.0],
        s=50,
        color="#c84b31",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
    upper_limit = max(2.0, float(np.max(margins)) + 1.4)
    lower_limit = -max(3.0, 0.78 * upper_limit)

    margin_axis.annotate(
        f"$t_1={start:.4f}$ s\n视线相切",
        xy=(start, 0.0),
        xytext=(start + 0.06, 0.62 * lower_limit),
        arrowprops={"arrowstyle": "->", "color": "#c84b31", "lw": 1.0},
        ha="left",
        va="center",
        color="#8c2f22",
    )
    margin_axis.annotate(
        f"$t_2={end:.4f}$ s\n导弹穿出云团",
        xy=(end, 0.0),
        xytext=(end - 0.06, 0.62 * lower_limit),
        arrowprops={"arrowstyle": "->", "color": "#c84b31", "lw": 1.0},
        ha="right",
        va="center",
        color="#8c2f22",
    )
    margin_axis.text(
        0.5,
        0.86,
        f"完整遮蔽：$H(t)=10-D_{{\\max}}(t)\\geq0$，时长 {end - start:.4f} s",
        transform=margin_axis.transAxes,
        ha="center",
        va="center",
        color="#286442",
        bbox={"facecolor": "#eef7f1", "edgecolor": "none", "alpha": 0.94, "pad": 2.0},
    )
    margin_axis.set_xlim(time_left, time_right)
    margin_axis.set_ylim(lower_limit, upper_limit)
    margin_axis.set_xlabel("时间 $t$ / s")
    margin_axis.set_ylabel("遮蔽裕度 $H(t)$ / m")
    margin_axis.grid(axis="both", color="#d7d7d7", linewidth=0.55, alpha=0.65)
    margin_axis.spines[["top", "right"]].set_visible(False)

    # 左下：Π1=plane(M,C,Q*)。轴向压缩仅改变显示比例，不改变相交关系。
    schematic_missile = np.array([-5.4, 0.0])
    schematic_cloud = np.array([-0.5, 0.0])
    schematic_radius = 1.15
    center_to_missile = schematic_missile - schematic_cloud
    center_distance = np.linalg.norm(center_to_missile)
    tangent_base = (
        schematic_cloud
        + schematic_radius**2 / center_distance**2 * center_to_missile
    )
    perpendicular = np.array([-center_to_missile[1], center_to_missile[0]])
    tangent_offset = (
        schematic_radius
        * np.sqrt(center_distance**2 - schematic_radius**2)
        / center_distance**2
        * perpendicular
    )
    tangent_candidates = [tangent_base + tangent_offset, tangent_base - tangent_offset]
    tangent_point = max(tangent_candidates, key=lambda point: point[1])
    lower_tangent_point = min(tangent_candidates, key=lambda point: point[1])
    target_x = 5.15

    def extend_to_x(point_on_ray: np.ndarray, x_value: float) -> np.ndarray:
        scale = (x_value - schematic_missile[0]) / (
            point_on_ray[0] - schematic_missile[0]
        )
        return schematic_missile + scale * (point_on_ray - schematic_missile)

    critical_target = extend_to_x(tangent_point, target_x)
    lower_cone_edge = extend_to_x(lower_tangent_point, target_x)
    tangent_axis.add_patch(
        Polygon(
            [schematic_missile, lower_cone_edge, critical_target],
            closed=True,
            facecolor="#76a9d2",
            edgecolor="none",
            alpha=0.12,
        )
    )
    tangent_axis.add_patch(
        Circle(
            schematic_cloud,
            schematic_radius,
            facecolor="#76a9d2",
            edgecolor="#245b8a",
            linewidth=2.0,
            alpha=0.34,
        )
    )
    target_bottom = lower_cone_edge[1] + 0.38
    tangent_axis.add_patch(
        Rectangle(
            (target_x - 0.28, target_bottom),
            0.56,
            critical_target[1] - target_bottom,
            facecolor="#7bb68e",
            edgecolor="#286442",
            linewidth=1.8,
            alpha=0.38,
        )
    )
    tangent_axis.plot(
        [schematic_missile[0], critical_target[0]],
        [schematic_missile[1], critical_target[1]],
        color="#c84b31",
        linewidth=2.2,
    )
    tangent_axis.plot(
        [schematic_missile[0], lower_cone_edge[0]],
        [schematic_missile[1], lower_cone_edge[1]],
        color="#245b8a",
        linewidth=1.3,
        linestyle="--",
    )
    tangent_axis.scatter(
        [schematic_missile[0]], [schematic_missile[1]], s=58, color="#303030", zorder=5
    )
    tangent_axis.scatter(
        [schematic_cloud[0]], [schematic_cloud[1]], s=54, color="#245b8a", zorder=5
    )
    tangent_axis.scatter(
        [tangent_point[0]], [tangent_point[1]], s=50, color="#e68a2e", zorder=6
    )
    tangent_axis.scatter(
        [critical_target[0]], [critical_target[1]], s=58, color="#c84b31", zorder=6
    )
    tangent_axis.annotate(
        "$M(t_1)$：导弹位置",
        xy=schematic_missile,
        xytext=(-5.9, -1.18),
        arrowprops={"arrowstyle": "->", "color": "#303030"},
        ha="left",
    )
    tangent_axis.annotate(
        "$C(t_1)$：烟幕球心",
        xy=schematic_cloud,
        xytext=(-2.9, -1.85),
        arrowprops={"arrowstyle": "->", "color": "#245b8a"},
        color="#245b8a",
        ha="left",
    )
    tangent_axis.annotate(
        "$P_1$：视线与烟幕球的切点",
        xy=tangent_point,
        xytext=(-4.75, 1.72),
        arrowprops={"arrowstyle": "->", "color": "#e68a2e"},
        color="#9a541b",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.2},
    )
    tangent_axis.annotate(
        "$Q^*$：圆柱底部边缘临界点",
        xy=critical_target,
        xytext=(-0.55, 2.86),
        arrowprops={
            "arrowstyle": "->",
            "color": "#c84b31",
            "connectionstyle": "angle3,angleA=0,angleB=-90",
        },
        color="#8c2f22",
        ha="left",
        va="top",
    )
    tangent_axis.text(
        3.15,
        -0.35,
        "真目标圆柱\n在 $\\Pi_1$ 中的截面",
        ha="center",
        va="center",
        color="#286442",
    )
    tangent_axis.text(
        -0.5,
        -2.62,
        f"$D_{{\\max}}=R=10$ m，$s^*={start_state.projection_parameter:.5f}$",
        ha="center",
        color="#505050",
    )
    tangent_axis.set_xlim(-6.25, 5.9)
    tangent_axis.set_ylim(-2.9, 3.0)
    tangent_axis.set_aspect("equal", adjustable="box")
    tangent_axis.set_title(
        "(a) 开始边界：取过 $M(t_1)$、$C(t_1)$、$Q^*$ 的平面 $\\Pi_1$\n"
        "三维视线相切关系的二维截面（轴向压缩）",
        y=1.02,
    )
    tangent_axis.axis("off")

    # 右下：Π2=plane(M,C,Q0)。在截面中直接显示云团、导弹与目标的前后顺序。
    exit_cloud = np.array([-1.2, 0.0])
    exit_radius = 1.30
    exit_missile = np.array([exit_cloud[0] + exit_radius, 0.0])
    exit_target = np.array([6.75, 0.0])
    exit_axis.add_patch(
        Circle(
            exit_cloud,
            exit_radius,
            facecolor="#76a9d2",
            edgecolor="#245b8a",
            linewidth=2.0,
            alpha=0.34,
        )
    )
    exit_axis.add_patch(
        Rectangle(
            (6.42, -1.1),
            0.66,
            2.2,
            facecolor="#7bb68e",
            edgecolor="#286442",
            linewidth=1.8,
            alpha=0.38,
        )
    )
    exit_axis.plot(
        [-3.0, exit_missile[0]],
        [0.0, 0.0],
        color="#777777",
        linewidth=1.4,
        linestyle="--",
    )
    exit_axis.annotate(
        "",
        xy=(5.7, 0.0),
        xytext=exit_missile,
        arrowprops={"arrowstyle": "->", "color": "#c84b31", "lw": 2.4},
    )
    exit_axis.plot(
        [exit_missile[0], exit_target[0]],
        [exit_missile[1], exit_target[1]],
        color="#c84b31",
        linewidth=1.6,
    )
    exit_axis.plot(
        [exit_cloud[0], exit_missile[0]],
        [0.0, 0.0],
        color="#245b8a",
        linewidth=2.0,
    )
    exit_axis.scatter([exit_cloud[0]], [0.0], s=54, color="#245b8a", zorder=5)
    exit_axis.scatter([exit_missile[0]], [0.0], s=62, color="#c84b31", zorder=6)
    exit_axis.scatter([exit_target[0]], [0.0], s=48, color="#286442", zorder=6)
    exit_axis.annotate(
        "$C(t_2)$：烟幕球心",
        xy=exit_cloud,
        xytext=(-3.65, 1.95),
        arrowprops={"arrowstyle": "->", "color": "#245b8a"},
        color="#245b8a",
        ha="left",
    )
    exit_axis.annotate(
        "$M(t_2)$：导弹位置（出球点）",
        xy=exit_missile,
        xytext=(0.75, -2.38),
        arrowprops={"arrowstyle": "->", "color": "#c84b31"},
        color="#8c2f22",
        ha="left",
    )
    exit_axis.annotate(
        "$Q_0$：真目标代表点",
        xy=exit_target,
        xytext=(5.0, 1.85),
        arrowprops={"arrowstyle": "->", "color": "#286442"},
        color="#286442",
        ha="left",
    )
    exit_axis.text(
        2.85,
        0.28,
        "M1飞向目标",
        color="#8c2f22",
        ha="center",
    )
    exit_axis.text(
        -0.55,
        -1.82,
        "$R=\\|C-M\\|=10$ m",
        color="#245b8a",
        ha="center",
    )
    exit_axis.text(
        -3.0,
        -0.55,
        "烟幕位于导弹后方",
        color="#505050",
        ha="center",
    )
    exit_axis.text(
        6.75,
        -1.62,
        "真目标圆柱截面",
        color="#286442",
        ha="center",
    )
    exit_axis.set_xlim(-4.0, 8.15)
    exit_axis.set_ylim(-2.9, 3.0)
    exit_axis.set_aspect("equal", adjustable="box")
    exit_axis.set_title(
        "(b) 结束边界：取过 $M(t_2)$、$C(t_2)$、$Q_0$ 的平面 $\\Pi_2$\n"
        "导弹穿出烟幕球后的二维前后关系（轴向压缩）",
        y=1.02,
    )
    exit_axis.axis("off")

    output_directory = Path(__file__).resolve().parent
    png_path = output_directory / "q1_critical_states.png"
    figure.savefig(png_path, dpi=320, bbox_inches="tight", pad_inches=0.14)
    plt.close(figure)
    return png_path


def main() -> None:
    # 统一输出编码，避免 Windows 终端显示中文乱码。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    model = Q1Model(Q1Parameters())

    print("2025A 第一问：主体法模块化求解")
    print("=" * 52)
    print(f"M1 运动单位向量:  {format_vector(model.missile_direction)}")
    print(f"FY1 运动单位向量: {format_vector(model.uav_direction)}")
    print(f"烟幕弹投放时刻:   {model.p.release_time:.4f} s")
    print(f"烟幕弹投放点:     {format_vector(model.release_point)} m")
    print(f"烟幕弹起爆时刻:   {model.burst_time:.4f} s")
    print(f"烟幕弹起爆点:     {format_vector(model.burst_point)} m")
    print(
        "烟幕有效时间窗:   "
        f"[{model.valid_start:.4f}, {model.valid_end:.4f}] s"
    )
    print(f"完整圆柱表面采样点数: {len(model.cylinder_surface_points)}")
    print(f"上下边缘圆周采样点数: {len(model.cylinder_rim_points)}")

    point_intervals = model.find_effective_intervals(model.point_target_margin)
    print_intervals("一、点目标简化模型", point_intervals)

    cylinder_intervals = model.find_effective_intervals(model.cylinder_target_margin)
    print_intervals("二、完整圆柱严格遮蔽模型", cylinder_intervals)

    rim_intervals = model.find_effective_intervals(model.cylinder_rim_margin)
    print_intervals("三、上下边缘双圆周降维模型", rim_intervals)

    if not cylinder_intervals:
        raise RuntimeError("完整圆柱模型未得到有效遮蔽区间，无法分析临界状态。")

    final_interval = cylinder_intervals[0]
    critical_states = [
        model.diagnose_cylinder_boundary(final_interval[0], "遮蔽开始边界"),
        model.diagnose_cylinder_boundary(final_interval[1], "遮蔽结束边界"),
    ]

    point_duration = model.total_duration(point_intervals)
    cylinder_duration = model.total_duration(cylinder_intervals)
    rim_duration = model.total_duration(rim_intervals)

    print("\n三、结果汇总")
    print(f"  点目标基础结果:       {point_duration:.4f} s")
    print(f"  完整圆柱严格结果:     {cylinder_duration:.4f} s")
    print(f"  双圆周降维结果:       {rim_duration:.4f} s")
    print(f"  两种口径的时长差:     {point_duration - cylinder_duration:.4f} s")
    print(f"  表面与双圆周差:       {cylinder_duration - rim_duration:.10f} s")
    print(f"  第一问最终建议取值:   {cylinder_duration:.2f} s")

    print("\n四、有效遮蔽区间临界状态")
    for state in critical_states:
        print(
            f"  {state.boundary_name}: t={state.time:.10f} s, "
            f"D_max={state.maximum_distance:.10f} m, "
            f"s*={state.projection_parameter:.10f}, "
            f"||C-M||={state.missile_cloud_distance:.10f} m"
        )
        print(f"    最不利目标点: {format_vector(state.worst_point)} m")
        print(f"    点位说明: {state.worst_point_note}")
        print(f"    几何解释: {state.mechanism}")

    result_path = save_results_csv(
        model,
        point_intervals,
        cylinder_intervals,
        rim_intervals,
    )
    critical_table_path = save_critical_states_csv(critical_states)
    figure_png_path = plot_critical_state_figure(
        model,
        final_interval,
        critical_states,
    )
    print(f"\n结果表已保存至: {result_path}")
    print(f"临界状态表已保存至: {critical_table_path}")
    print(f"临界状态图已保存至: {figure_png_path}")


if __name__ == "__main__":
    main()
