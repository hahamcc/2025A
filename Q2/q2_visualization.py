"""问题二正式结果的可复核可视化。

本模块只读取问题二既有搜索结果，并用完整圆柱表面评价器重新计算关键几何。
运行方式：``python Q2/q2_visualization.py --all``。
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from PIL import Image, ImageStat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Q1.q1_visualization import (  # noqa: E402
    COLORS,
    annotate_3d,
    configure_matplotlib,
    draw_cylinder,
    draw_sphere,
    style_3d_axis,
    trim_white_margins,
)
from core.smoke_evaluator import (  # noqa: E402
    Deployment,
    MarginDiagnostic,
    SamplingConfig,
    ScenarioParameters,
    SmokeEvaluator,
    SmokeSimulation,
)

Q2_DIR = Path(__file__).resolve().parent
FIGURE_DIR = Q2_DIR / "figures"
MAX_BURST_TIME = 13.94
FULL_SAMPLING = SamplingConfig(
    angle_count=1440,
    height_count=41,
    radial_count=31,
    scan_step=0.05,
    root_tolerance=1.0e-10,
)
DE_COLORS = (
    COLORS["uav"],
    COLORS["bomb"],
    COLORS["target"],
    COLORS["false"],
    COLORS["cloud"],
    COLORS["neutral"],
)
DE_LINESTYLES = ("-", "--", "-.", ":", (0, (5, 2, 1, 2)), (0, (1, 1)))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError(f"没有可写入 {path.name} 的记录。")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, str], name: str) -> float:
    value = row.get(name, "")
    if value == "":
        raise RuntimeError(f"结果文件缺少 {name}。")
    return float(value)


def build_full_evaluator() -> SmokeEvaluator:
    return SmokeEvaluator(
        ScenarioParameters(max_burst_time=MAX_BURST_TIME), FULL_SAMPLING
    )


def selected_solution(
    evaluator: SmokeEvaluator,
) -> tuple[Deployment, SmokeSimulation, tuple[float, float], float]:
    rows = read_csv(Q2_DIR / "q2_best_solution.csv")
    selected = [row for row in rows if row.get("selected") == "True"]
    if len(selected) != 1:
        raise RuntimeError("q2_best_solution.csv 中应恰有一个 selected=True 的方案。")
    row = selected[0]
    deployment = Deployment(
        number(row, "theta_rad"),
        number(row, "uav_speed_mps"),
        number(row, "release_time_s"),
        number(row, "fuse_delay_s"),
    )
    if evaluator.validate(deployment) is not None:
        raise RuntimeError("最终方案未通过题设约束检查。")
    result = evaluator.evaluate(deployment, mode="surface")
    stored_duration = number(row, "duration_s")
    if not result.feasible or len(result.intervals) != 1:
        raise RuntimeError("完整圆柱表面复核未得到唯一遮蔽区间。")
    if abs(result.duration - stored_duration) > 1.0e-8:
        raise RuntimeError("最终结果与 q2_best_solution.csv 不一致，拒绝出图。")
    simulation = evaluator.simulation(deployment)
    if simulation is None:
        raise RuntimeError("无法建立最终方案的仿真对象。")
    return deployment, simulation, result.intervals[0], result.duration


def tidy(path: Path, padding: int = 20) -> Path:
    trim_white_margins(path, padding=padding)
    with Image.open(path) as image:
        dpi = image.info.get("dpi", (0.0, 0.0))
        if min(image.size) < 400 or not all(abs(float(value) - 300.0) < 5.0 for value in dpi):
            raise RuntimeError(f"图片 {path.name} 的尺寸或分辨率不符合要求。")
        if max(ImageStat.Stat(image.convert("L")).var) < 1.0:
            raise RuntimeError(f"图片 {path.name} 疑似为空白图。")
    return path


def save_figure(figure, name: str, padding: int = 20) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.10)
    plt.close(figure)
    return tidy(path, padding)


def light_grid(axis) -> None:
    axis.grid(True, color="#d7dade", alpha=0.68, linewidth=0.65)
    axis.set_axisbelow(True)


def draw_lhs_regions(deployment: Deployment) -> Path:
    rows = read_csv(Q2_DIR / "q2_coarse_search.csv")
    if len(rows) != 8000:
        raise RuntimeError("正式 LHS 粗搜索记录应为 8000 个样本。")
    theta = np.array([number(row, "theta_deg") for row in rows])
    burst = np.array([number(row, "burst_time_s") for row in rows])
    speed = np.array([number(row, "uav_speed_mps") for row in rows])
    duration = np.array([number(row, "duration_s") for row in rows])
    centers = [row for row in rows if row.get("selected_region", "")]
    if len(centers) != 6:
        raise RuntimeError("正式粗搜索应选出 6 个区域中心。")

    figure, axes = plt.subplots(1, 2, figsize=(14.8, 6.1), constrained_layout=True)
    panels = ((axes[0], burst, "起爆时刻 $t_e$ / s"), (axes[1], speed, "无人机速度 $v$ / (m·s$^{-1}$)"))
    zero = duration <= 0.0
    weak = (duration > 0.0) & (duration < 3.0)
    strong = duration >= 3.0
    for index, (axis, vertical, label) in enumerate(panels):
        axis.scatter(theta[zero], vertical[zero], s=8, color=COLORS["light"], alpha=0.24, linewidths=0, label="无有效遮蔽" if index == 0 else None)
        axis.scatter(theta[weak], vertical[weak], s=12, color=COLORS["cloud"], alpha=0.64, linewidths=0, label="$0<T_{\\rm eff}<3$ s" if index == 0 else None)
        axis.scatter(theta[strong], vertical[strong], s=18, color=COLORS["uav"], alpha=0.84, linewidths=0, label="$T_{\\rm eff}\\geq3$ s" if index == 0 else None)
        axis.scatter(
            [number(row, "theta_deg") for row in centers],
            [number(row, "burst_time_s") if index == 0 else number(row, "uav_speed_mps") for row in centers],
            s=92,
            facecolors="white",
            edgecolors=COLORS["bomb"],
            linewidths=2.0,
            label="6 个分散区域中心" if index == 0 else None,
            zorder=5,
        )
        final_y = deployment.burst_time if index == 0 else deployment.speed
        axis.scatter(np.degrees(deployment.heading), final_y, s=155, marker="*", color=COLORS["missile"], edgecolors="white", linewidths=0.7, label="完整表面最优方案" if index == 0 else None, zorder=6)
        axis.set_xlim(0.0, 360.0)
        axis.set_xticks(np.arange(0.0, 361.0, 60.0))
        axis.set_xlabel("航向角 $\\theta$ / (°)")
        axis.set_ylabel(label)
        light_grid(axis)
    axes[0].legend(loc="upper right", fontsize=10, frameon=False)
    return save_figure(figure, "q2_lhs_regions.png")


def draw_de_convergence(final_duration: float) -> Path:
    rows = read_csv(Q2_DIR / "q2_de_history.csv")
    grouped: dict[int, list[tuple[int, float]]] = {}
    for row in rows:
        grouped.setdefault(int(row["region_index"]), []).append((int(row["generation"]), number(row, "best_duration_s")))
    if sorted(grouped) != [1, 2, 3, 4, 5, 6]:
        raise RuntimeError("收敛记录必须对应 6 个区域。")

    figure, axis = plt.subplots(figsize=(12.8, 6.5), constrained_layout=True)
    for region, color, linestyle in zip(sorted(grouped), DE_COLORS, DE_LINESTYLES):
        data = sorted(grouped[region])
        generations, values = (np.array(value) for value in zip(*data))
        if len(values) < 2 or np.any(np.diff(values) < -1.0e-10):
            raise RuntimeError(f"区域 {region} 的历史最佳值记录不合法。")
        axis.plot(generations, values, color=color, linestyle=linestyle, linewidth=2.1, label=f"区域 {region}：{values[-1]:.4f} s")
    axis.axhline(final_duration, color=COLORS["missile"], linestyle="--", linewidth=1.8, label=f"完整表面复核：{final_duration:.4f} s")
    axis.set_xlabel("迭代代数")
    axis.set_ylabel("截至当前代的最佳遮蔽时长 $T_{\\rm eff}$ / s")
    light_grid(axis)
    axis.legend(ncol=2, loc="lower right", fontsize=10, frameon=False)
    return save_figure(figure, "q2_de_convergence.png")


def set_global_3d_axis(axis) -> None:
    axis.set_xlim(20600.0, -600.0)
    axis.set_ylim(-350.0, 650.0)
    axis.set_zlim(0.0, 2350.0)
    axis.set_box_aspect((2.55, 1.0, 1.18))
    axis.view_init(elev=20, azim=72)
    axis.set_xlabel("$x$ / m", labelpad=8)
    axis.set_ylabel("$y$ / m", labelpad=8)
    axis.set_zlabel("$z$ / m", labelpad=6)
    style_3d_axis(axis)


def draw_optimal_trajectory(
    simulation: SmokeSimulation,
    interval: tuple[float, float],
) -> Path:
    start, end = interval
    burst = simulation.burst_time
    missile_times = np.linspace(0.0, simulation.missile_impact_time, 320)
    uav_times = np.linspace(0.0, burst, 80)
    bomb_times = np.linspace(simulation.deployment.release_time, burst, 80)
    cloud_times = np.linspace(burst, end, 80)
    missile = np.vstack([simulation.missile_position(time) for time in missile_times])
    uav = np.vstack([simulation.uav_position(time) for time in uav_times])
    bomb = np.vstack([simulation.bomb_position(time) for time in bomb_times])
    cloud = np.vstack([simulation.cloud_center(time) for time in cloud_times])
    covered_missile = np.vstack([simulation.missile_position(time) for time in np.linspace(start, end, 100)])

    figure = plt.figure(figsize=(16.2, 6.6), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=(1.32, 1.0, 1.0))
    global_axis = figure.add_subplot(grid[0, 0], projection="3d")
    xy_axis = figure.add_subplot(grid[0, 1])
    xz_axis = figure.add_subplot(grid[0, 2])

    draw_cylinder(global_axis, simulation.target_point, simulation.parameters.target_radius, simulation.parameters.target_height)
    global_axis.scatter(0.0, 0.0, 0.0, s=115, marker="*", color=COLORS["false"], depthshade=False, label="假目标")
    global_axis.plot(*missile.T, color=COLORS["missile"], linewidth=1.7, alpha=0.75, label="M1 轨迹")
    global_axis.plot(*covered_missile.T, color=COLORS["missile"], linewidth=3.8, label="有效遮蔽段")
    global_axis.plot(*uav.T, color=COLORS["uav"], linestyle="-.", linewidth=2.1, label="FY1 轨迹")
    global_axis.plot(*bomb.T, color=COLORS["bomb"], linestyle=(0, (2, 2)), linewidth=2.0, label="干扰弹抛物段")
    global_axis.plot(*cloud.T, color=COLORS["cloud"], linestyle="--", linewidth=2.2, label="烟幕球心")
    draw_sphere(global_axis, simulation.cloud_center(0.5 * (start + end)), simulation.parameters.cloud_radius)
    for time, marker, label in ((burst, "D", "$t_e$"), (start, "o", "$t_1$"), (end, "s", "$t_2$")):
        point = simulation.cloud_center(time) if time == burst else simulation.missile_position(time)
        color = COLORS["cloud"] if time == burst else COLORS["missile"]
        global_axis.scatter(*point, s=50, marker=marker, color=color, edgecolor="white", linewidth=0.65, depthshade=False)
        annotate_3d(global_axis, label, point, (8, 16), color, fontsize=10)
    set_global_3d_axis(global_axis)
    global_axis.legend(loc="upper left", bbox_to_anchor=(0.01, 0.98), fontsize=8.5, frameon=False)

    def projection(axis, first: int, second: int, ylabel: str) -> None:
        axis.plot(missile[:, first], missile[:, second], color=COLORS["missile"], linewidth=1.7, alpha=0.75, label="M1 轨迹")
        axis.plot(covered_missile[:, first], covered_missile[:, second], color=COLORS["missile"], linewidth=4.0, label="有效遮蔽段")
        axis.plot(uav[:, first], uav[:, second], color=COLORS["uav"], linestyle="-.", linewidth=2.1, label="FY1 轨迹")
        axis.plot(bomb[:, first], bomb[:, second], color=COLORS["bomb"], linestyle=(0, (2, 2)), linewidth=2.0, label="干扰弹抛物段")
        axis.plot(cloud[:, first], cloud[:, second], color=COLORS["cloud"], linestyle="--", linewidth=2.2, label="烟幕球心")
        axis.scatter(simulation.target_point[first], simulation.target_point[second], color=COLORS["target"], s=42, label="真目标", zorder=4)
        axis.scatter(0.0, 0.0, color=COLORS["false"], marker="*", s=68, label="假目标", zorder=4)
        for time, marker, label in ((burst, "D", "$t_e$"), (start, "o", "$t_1$"), (end, "s", "$t_2$")):
            point = simulation.cloud_center(time) if time == burst else simulation.missile_position(time)
            color = COLORS["cloud"] if time == burst else COLORS["missile"]
            axis.scatter(point[first], point[second], s=42, marker=marker, color=color, edgecolor="white", linewidth=0.55, zorder=5)
            axis.annotate(label, (point[first], point[second]), xytext=(5, 7), textcoords="offset points", color=color, fontsize=10)
        axis.set_xlabel("$x$ / m")
        axis.set_ylabel(ylabel)
        axis.set_xlim(20600.0, -600.0)
        light_grid(axis)

    projection(xy_axis, 0, 1, "$y$ / m")
    projection(xz_axis, 0, 2, "$z$ / m")
    xz_axis.set_ylim(0.0, 2350.0)
    return save_figure(figure, "q2_optimal_trajectory.png")


def diagnostic_row(diagnostic: MarginDiagnostic, state: str) -> dict[str, str]:
    row = {
        "state": state,
        "time_s": f"{diagnostic.time:.10f}",
        "mode": diagnostic.mode,
        "max_distance_m": f"{diagnostic.max_distance:.10f}",
        "margin_m": f"{diagnostic.margin:.10f}",
        "closest_projection": f"{diagnostic.closest_projection:.10f}",
    }
    for label, vector in (("cloud", diagnostic.cloud_center), ("missile", diagnostic.missile_position), ("worst_target", diagnostic.target_point), ("closest", diagnostic.closest_point)):
        row.update({f"{label}_{axis}_m": f"{value:.10f}" for axis, value in zip(("x", "y", "z"), vector)})
    return row


def draw_state_plane(axis, diagnostic: MarginDiagnostic) -> None:
    missile = diagnostic.missile_position
    target = diagnostic.target_point
    cloud = diagnostic.cloud_center
    first_axis = target - missile
    first_axis /= np.linalg.norm(first_axis)
    perpendicular = cloud - missile - np.dot(cloud - missile, first_axis) * first_axis
    if np.linalg.norm(perpendicular) < 1.0e-12:
        perpendicular = np.array((0.0, 0.0, 1.0))
        perpendicular -= np.dot(perpendicular, first_axis) * first_axis
    second_axis = perpendicular / np.linalg.norm(perpendicular)

    def convert(point: np.ndarray) -> np.ndarray:
        difference = point - missile
        return np.array((np.dot(difference, first_axis), np.dot(difference, second_axis)))

    target_2d = convert(target)
    cloud_2d = convert(cloud)
    closest_2d = convert(diagnostic.closest_point)
    axis.add_patch(Circle(cloud_2d, radius=10.0, facecolor=COLORS["cloud"], edgecolor=COLORS["uav"], alpha=0.26, linewidth=1.5))
    line_length = float(target_2d[0])
    local_half_width = 42.0
    local_left = cloud_2d[0] - local_half_width
    local_right = cloud_2d[0] + local_half_width
    axis.plot([local_left, local_right], [0.0, 0.0], color=COLORS["missile"], linewidth=2.5)
    axis.plot([cloud_2d[0], closest_2d[0]], [cloud_2d[1], closest_2d[1]], color=COLORS["uav"], linestyle="--", linewidth=1.8)
    axis.scatter(*cloud_2d, color=COLORS["cloud"], edgecolor=COLORS["uav"], s=46, zorder=5)
    axis.scatter(*closest_2d, color=COLORS["uav"], s=34, zorder=5)
    boundary = abs(diagnostic.margin) < 1.0e-6
    interior = 1.0e-7 < diagnostic.closest_projection < 1.0 - 1.0e-7
    note = "内部切线临界" if boundary and interior else "端点临界" if boundary else "遮蔽中：最近点在线段内部" if interior else "遮蔽中：最近点在线段端点"
    axis.text(0.02, 0.05, f"{note}\n局部放大；完整视线段长 {line_length:.0f} m", transform=axis.transAxes, fontsize=9.5, color=COLORS["neutral"], va="bottom")
    axis.annotate("← 导弹端", (local_left + 2.0, 0.0), xytext=(0, 8), textcoords="offset points", color=COLORS["missile"], fontsize=9)
    axis.annotate("目标端 →", (local_right - 2.0, 0.0), xytext=(0, 8), textcoords="offset points", ha="right", color=COLORS["target"], fontsize=9)
    axis.set_xlim(local_left, local_right)
    axis.set_ylim(cloud_2d[1] - local_half_width, cloud_2d[1] + local_half_width)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("沿视线方向 / m")
    axis.set_ylabel("垂直于视线 / m")
    light_grid(axis)


def draw_occlusion_states(
    simulation: SmokeSimulation,
    interval: tuple[float, float],
) -> Path:
    start, end = interval
    middle = 0.5 * (start + end)
    end_time = min(simulation.valid_end, end + 0.40)
    times = np.arange(simulation.valid_start, end_time + 1.0e-12, 0.05)
    times = np.unique(np.append(times, (start, middle, end, end_time)))
    diagnostics = [simulation.margin_diagnostic(float(time), mode="surface") for time in times]
    margins = np.array([diagnostic.margin for diagnostic in diagnostics])
    margin_rows = [diagnostic_row(diagnostic, "margin_curve") for diagnostic in diagnostics]
    write_csv(Q2_DIR / "q2_margin_curve.csv", margin_rows)
    critical = [
        ("进入遮蔽 $t_1$", simulation.margin_diagnostic(start, mode="surface")),
        ("遮蔽中点", simulation.margin_diagnostic(middle, mode="surface")),
        ("退出遮蔽 $t_2$", simulation.margin_diagnostic(end, mode="surface")),
    ]
    write_csv(Q2_DIR / "q2_critical_states.csv", [diagnostic_row(item, state) for state, item in critical])
    if abs(critical[0][1].margin) > 1.0e-7 or abs(critical[2][1].margin) > 1.0e-7 or critical[1][1].margin <= 0.0:
        raise RuntimeError("完整表面遮蔽边界或中点裕量检查失败。")

    figure = plt.figure(figsize=(15.2, 9.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(1.05, 1.0))
    margin_axis = figure.add_subplot(grid[0, :])
    state_axes = [figure.add_subplot(grid[1, index]) for index in range(3)]
    margin_axis.plot(times, margins, color=COLORS["uav"], linewidth=2.4, label="完整表面遮蔽裕量 $H(t)$")
    margin_axis.axhline(0.0, color=COLORS["neutral"], linewidth=1.2)
    margin_axis.fill_between(times, 0.0, np.maximum(margins, 0.0), color=COLORS["target"], alpha=0.20, label="有效遮蔽")
    margin_axis.axvline(simulation.burst_time, color=COLORS["cloud"], linestyle="--", linewidth=1.8, label="$t_e$：烟幕起爆")
    margin_axis.scatter([start, end], [critical[0][1].margin, critical[2][1].margin], color=COLORS["missile"], s=48, zorder=5, label="$t_1,t_2$：遮蔽边界")
    margin_axis.annotate("$t_1$", (start, critical[0][1].margin), xytext=(-3, 10), textcoords="offset points", ha="right", color=COLORS["missile"])
    margin_axis.annotate("$t_2$", (end, critical[2][1].margin), xytext=(3, 10), textcoords="offset points", color=COLORS["missile"])
    margin_axis.set_xlabel("时刻 $t$ / s")
    margin_axis.set_ylabel("遮蔽裕量 $H(t)$ / m")
    light_grid(margin_axis)
    margin_axis.legend(loc="upper right", ncol=2, fontsize=10, frameon=False)
    for axis, (_, diagnostic) in zip(state_axes, critical):
        draw_state_plane(axis, diagnostic)
    return save_figure(figure, "q2_occlusion_states.png")


def draw_sensitivity(
    evaluator: SmokeEvaluator,
    baseline: Deployment,
) -> Path:
    specifications: list[tuple[str, str, str, str, list[float], Callable[[float], Deployment]]] = [
        ("heading", "航向角", "航向角扰动 $\\Delta\\theta$ / (°)", "°", [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0], lambda value: replace(baseline, heading=baseline.heading + np.radians(value))),
        ("speed", "飞行速度", "速度相对扰动 $\\Delta v/v^*$ / (%)", "%", [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0], lambda value: replace(baseline, speed=baseline.speed * (1.0 + value / 100.0))),
        ("release_time", "投放时刻", "投放时刻扰动 $\\Delta t_r$ / s", "s", [-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30], lambda value: replace(baseline, release_time=baseline.release_time + value)),
        ("fuse_delay", "引爆延时", "引爆延时扰动 $\\Delta\\tau$ / s", "s", [-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30], lambda value: replace(baseline, fuse_delay=baseline.fuse_delay + value)),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(14.8, 9.0), constrained_layout=True)
    rows: list[dict[str, str]] = []
    for index, (name, display_name, xlabel, unit, offsets, transform) in enumerate(specifications):
        axis = axes.flat[index]
        durations: list[float] = []
        feasible: list[bool] = []
        for offset in offsets:
            candidate = transform(offset)
            result = evaluator.evaluate(candidate, mode="surface")
            durations.append(result.duration if result.feasible else np.nan)
            feasible.append(result.feasible)
            rows.append({
                "parameter": name,
                "perturbation": f"{offset:.10f}",
                "unit": unit,
                "theta_rad": f"{candidate.heading:.12f}",
                "uav_speed_mps": f"{candidate.speed:.10f}",
                "release_time_s": f"{candidate.release_time:.10f}",
                "fuse_delay_s": f"{candidate.fuse_delay:.10f}",
                "feasible": str(result.feasible),
                "duration_s": "" if not result.feasible else f"{result.duration:.10f}",
                "reason": result.reason,
            })
        values = np.array(durations, dtype=float)
        offsets_array = np.array(offsets, dtype=float)
        valid = np.isfinite(values)
        axis.plot(offsets_array[valid], values[valid], color=COLORS["uav"], marker="o", markersize=5.0, linewidth=2.0)
        axis.axvline(0.0, color=COLORS["light"], linewidth=1.2)
        baseline_index = offsets.index(0.0)
        if valid[baseline_index]:
            axis.scatter(0.0, values[baseline_index], color=COLORS["missile"], s=62, zorder=5, label="基准最优解")
        if not np.all(valid):
            lower = np.nanmin(values[valid]) if np.any(valid) else 0.0
            axis.scatter(offsets_array[~valid], np.full(np.count_nonzero(~valid), lower), marker="x", color=COLORS["missile"], s=42, label="不可行")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("遮蔽时长 $T_{\\rm eff}$ / s")
        light_grid(axis)
        if index == 0 or not np.all(valid):
            axis.legend(loc="best", fontsize=10, frameon=False)
    write_csv(Q2_DIR / "q2_sensitivity.csv", rows)
    return save_figure(figure, "q2_sensitivity.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成问题二正式可视化与复核数据")
    parser.add_argument("--all", action="store_true", help="生成全部五张正式图")
    arguments = parser.parse_args()
    if not arguments.all:
        parser.error("请使用 --all 生成全部正式图。")
    configure_matplotlib()
    evaluator = build_full_evaluator()
    deployment, simulation, interval, duration = selected_solution(evaluator)
    paths = [
        draw_lhs_regions(deployment),
        draw_de_convergence(duration),
        draw_optimal_trajectory(simulation, interval),
        draw_occlusion_states(simulation, interval),
        draw_sensitivity(evaluator, deployment),
    ]
    print(f"完整表面复核遮蔽时长：{duration:.10f} s")
    for path in paths:
        print(f"已生成：{path}")


if __name__ == "__main__":
    main()
