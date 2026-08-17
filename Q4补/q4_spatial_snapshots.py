"""为问题四 baseline-900 正式方案绘制三阶段空间遮蔽快照图。"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Q1.q1_visualization import configure_matplotlib, trim_white_margins  # noqa: E402
from Q4 import q4_main as model  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "Q4补" / "runs" / "baseline_900_standard" / "q4_best_solution.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "Q4补" / "runs" / "baseline_900_standard" / "q4_02_spatial_snapshots.png"
DEFAULT_STATES = PROJECT_ROOT / "Q4补" / "runs" / "baseline_900_standard" / "q4_snapshot_states.csv"

COLORS = ("#C84B31", "#2878B5", "#2E8B57")
LINESTYLES = ("-", "--", "-.")
HATCHES = ("////", "\\\\", "xx")
MISSILE_COLOR = "#202A35"
TARGET_COLOR = "#2F7650"
FALSE_TARGET_COLOR = "#77622E"
GRID_COLOR = "#CBD2D9"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: row["uav"])
    if [row["uav"] for row in rows] != list(model.UAV_NAMES):
        raise RuntimeError("输入 CSV 必须恰好包含 FY1、FY2、FY3 三行。")
    return rows


def parse_intervals(text: str) -> tuple[tuple[float, float], ...]:
    intervals: list[tuple[float, float]] = []
    for item in text.split(";"):
        item = item.strip().strip("[]")
        if item:
            left, right = item.split(",")
            intervals.append((float(left), float(right)))
    if not intervals:
        raise RuntimeError("未读到联合遮蔽区间。")
    return tuple(intervals)


def match_intervals(
    rows: list[dict[str, str]], intervals: tuple[tuple[float, float], ...]
) -> tuple[tuple[float, float], ...]:
    """本正式方案无同时协同，按独立时长把三段联合区间对应至三架无人机。"""

    remaining = list(intervals)
    selected: list[tuple[float, float]] = []
    for row in rows:
        duration = float(row["individual_duration_s"])
        index = min(
            range(len(remaining)),
            key=lambda current: abs((remaining[current][1] - remaining[current][0]) - duration),
        )
        interval = remaining.pop(index)
        if abs((interval[1] - interval[0]) - duration) > 1.0e-5:
            raise RuntimeError("独立时长与联合区间不能唯一对应，不能绘制接力快照。")
        selected.append(interval)
    return tuple(selected)


def as_point(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.array(
        [float(row[f"{prefix}_x_m"]), float(row[f"{prefix}_y_m"]), float(row[f"{prefix}_z_m"])],
        dtype=float,
    )


def plan_from_row(row: dict[str, str], index: int) -> model.UavPlan:
    burst = float(row["burst_time_s"])
    speed = float(row["speed_mps"])
    return model.UavPlan(
        name=row["uav"],
        initial=tuple(float(value) for value in model.UAV_INITIALS[index]),
        heading=math.radians(float(row["theta_deg"])),
        speed=speed,
        horizontal_distance=speed * burst,
        burst_time=burst,
        fuse_delay=float(row["fuse_delay_s"]),
    )


def style_axis(axis: plt.Axes, *, local: bool = False) -> None:
    axis.grid(True, color=GRID_COLOR, linewidth=0.65 if local else 0.75, alpha=0.70)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=8.7 if local else 9.4)


def draw_target_xy(axis: plt.Axes) -> None:
    center = model.TARGET_BOTTOM_CENTER[:2]
    axis.add_patch(
        Circle(center, model.TARGET_RADIUS, facecolor=TARGET_COLOR, edgecolor=TARGET_COLOR, alpha=0.48, zorder=5)
    )
    axis.scatter([0.0], [0.0], marker="x", s=30, linewidth=1.5, color=FALSE_TARGET_COLOR, zorder=5)
    axis.annotate("真目标", center, xytext=(8, 8), textcoords="offset points", fontsize=8.6, color=TARGET_COLOR)
    axis.annotate("假目标", (0.0, 0.0), xytext=(8, -13), textcoords="offset points", fontsize=8.2, color=FALSE_TARGET_COLOR)


def draw_target_xz(axis: plt.Axes) -> None:
    axis.add_patch(
        Rectangle(
            (-model.TARGET_RADIUS, 0.0), 2.0 * model.TARGET_RADIUS, model.TARGET_HEIGHT,
            facecolor=TARGET_COLOR, edgecolor=TARGET_COLOR, alpha=0.48, zorder=5,
        )
    )
    axis.scatter([0.0], [0.0], marker="x", s=30, linewidth=1.5, color=FALSE_TARGET_COLOR, zorder=5)
    axis.annotate("真目标", (0.0, model.TARGET_HEIGHT), xytext=(8, 4), textcoords="offset points", fontsize=8.6, color=TARGET_COLOR)


def representative_target_points() -> tuple[np.ndarray, ...]:
    bottom = model.TARGET_BOTTOM_CENTER
    radius = model.TARGET_RADIUS
    top = bottom + np.array((0.0, 0.0, model.TARGET_HEIGHT))
    return (
        bottom + np.array((radius, 0.0, 0.0)),
        bottom + np.array((-radius, 0.0, 0.0)),
        top + np.array((radius, 0.0, 0.0)),
        top + np.array((-radius, 0.0, 0.0)),
    )


def draw_uav_path_xy(axis: plt.Axes, plan: model.UavPlan, color: str, linestyle: str) -> None:
    initial = np.asarray(plan.initial, dtype=float)
    release = plan.release_point
    burst = plan.burst_point
    axis.plot([initial[0], release[0]], [initial[1], release[1]], color=color, linewidth=1.8, linestyle=linestyle, alpha=0.82, zorder=3)
    axis.plot([release[0], burst[0]], [release[1], burst[1]], color=color, linewidth=1.5, linestyle=(0, (2, 2)), alpha=0.82, zorder=3)
    axis.scatter([initial[0]], [initial[1]], marker="o", s=22, facecolor="white", edgecolor=color, linewidth=1.0, zorder=5)
    axis.scatter([release[0]], [release[1]], marker="s", s=30, facecolor="white", edgecolor=color, linewidth=1.2, zorder=5)
    axis.scatter([burst[0]], [burst[1]], marker="*", s=68, facecolor=color, edgecolor="white", linewidth=0.8, zorder=6)


def draw_uav_path_xz(axis: plt.Axes, plan: model.UavPlan, color: str, linestyle: str) -> None:
    initial = np.asarray(plan.initial, dtype=float)
    release = plan.release_point
    delay_grid = np.linspace(0.0, plan.fuse_delay, 40)
    trajectory = release[None, :] + delay_grid[:, None] * plan.speed * plan.direction[None, :]
    trajectory[:, 2] = release[2] - 0.5 * model.GRAVITY * delay_grid**2
    axis.plot([initial[0], release[0]], [initial[2], release[2]], color=color, linewidth=1.8, linestyle=linestyle, alpha=0.82, zorder=3)
    axis.plot(trajectory[:, 0], trajectory[:, 2], color=color, linewidth=1.6, linestyle=(0, (2, 2)), alpha=0.86, zorder=3)
    axis.scatter([initial[0]], [initial[2]], marker="o", s=22, facecolor="white", edgecolor=color, linewidth=1.0, zorder=5)
    axis.scatter([release[0]], [release[2]], marker="s", s=30, facecolor="white", edgecolor=color, linewidth=1.2, zorder=5)
    axis.scatter([trajectory[-1, 0]], [trajectory[-1, 2]], marker="*", s=68, facecolor=color, edgecolor="white", linewidth=0.8, zorder=6)


def line_plane_basis(missile: np.ndarray, cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center_direction = model.TARGET_CENTER - missile
    e1 = center_direction / np.linalg.norm(center_direction)
    lateral = cloud - missile - np.dot(cloud - missile, e1) * e1
    if np.linalg.norm(lateral) < 1.0e-9:
        fallback = np.array((0.0, 0.0, 1.0))
        lateral = fallback - np.dot(fallback, e1) * e1
    e2 = lateral / np.linalg.norm(lateral)
    return e1, e2


def draw_local_inset(axis: plt.Axes, missile: np.ndarray, cloud: np.ndarray, color: str, hatch: str) -> float:
    """画出球心附近的真实尺度局部截面，返回球心到中心视线距离。"""

    inset = inset_axes(axis, width="48%", height="52%", loc="upper right", borderpad=1.0)
    e1, e2 = line_plane_basis(missile, cloud)
    relative_missile = missile - cloud
    m_local = np.array((np.dot(relative_missile, e1), np.dot(relative_missile, e2)))
    center_local = np.array(
        (
            np.dot(model.TARGET_CENTER - cloud, e1),
            np.dot(model.TARGET_CENTER - cloud, e2),
        )
    )
    distance = abs(float(m_local[1]))
    envelope_radius = model.TARGET_BOUND_RADIUS

    for offset in (-envelope_radius, -0.55 * envelope_radius, 0.0, 0.55 * envelope_radius, envelope_radius):
        endpoint = model.TARGET_CENTER + offset * e2
        endpoint_local = np.array((np.dot(endpoint - cloud, e1), np.dot(endpoint - cloud, e2)))
        parameter = (np.linspace(-26.0, 26.0, 120) - m_local[0]) / (endpoint_local[0] - m_local[0])
        values = m_local[1] + parameter * (endpoint_local[1] - m_local[1])
        inset.plot(np.linspace(-26.0, 26.0, 120), values, color=MISSILE_COLOR, linewidth=0.75, alpha=0.58, zorder=1)

    inset.add_patch(Circle((0.0, 0.0), model.CLOUD_RADIUS, facecolor=color, edgecolor=color, alpha=0.23, linewidth=1.45, hatch=hatch, zorder=3))
    inset.scatter([0.0], [0.0], color=color, s=17, zorder=4)
    inset.plot([-25.0, 25.0], [m_local[1], m_local[1]], color=MISSILE_COLOR, linestyle="--", linewidth=0.9, alpha=0.75, zorder=2)
    inset.annotate("M1方向", xy=(-23.5, m_local[1]), xytext=(-14.0, -21.0), textcoords="data", fontsize=6.9, color=MISSILE_COLOR,
                   arrowprops=dict(arrowstyle="->", color=MISSILE_COLOR, linewidth=0.65))
    inset.annotate("目标方向", xy=(23.5, m_local[1]), xytext=(6.0, 19.0), textcoords="data", fontsize=6.9, color=MISSILE_COLOR,
                   arrowprops=dict(arrowstyle="->", color=MISSILE_COLOR, linewidth=0.65))
    inset.text(0.03, 0.96, f"局部真实尺度\n球半径 10 m\n中心线距 {distance:.2f} m", transform=inset.transAxes,
               ha="left", va="top", fontsize=6.8, color="#3F4347",
               bbox=dict(facecolor="white", edgecolor="none", alpha=0.74, pad=1.0))
    inset.set_xlim(-27.0, 27.0)
    inset.set_ylim(-27.0, 27.0)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks((-20, 0, 20))
    inset.set_yticks((-20, 0, 20))
    style_axis(inset, local=True)
    return distance


def snapshot_state(
    row: dict[str, str], plan: model.UavPlan, index: int, interval: tuple[float, float]
) -> dict[str, float | str]:
    start, end = interval
    time = 0.5 * (start + end)
    if not plan.burst_time - 1.0e-12 <= time <= plan.burst_time + model.CLOUD_LIFETIME + 1.0e-12:
        raise RuntimeError(f"{plan.name} 的快照时刻不在烟幕有效窗内。")
    missile = model.MISSILE_INITIAL + time * model.MISSILE_VELOCITY
    cloud = model.JointEvaluator.cloud_centers(plan, np.array((time,)))[0]
    release = plan.release_point
    burst = plan.burst_point
    csv_release, csv_burst = as_point(row, "release"), as_point(row, "burst")
    if not np.allclose(release, csv_release, atol=1.0e-7) or not np.allclose(burst, csv_burst, atol=1.0e-7):
        raise RuntimeError(f"{plan.name} 的 CSV 点位与既有运动方程不一致。")
    direction = model.TARGET_CENTER - missile
    projection = float(np.dot(cloud - missile, direction) / np.dot(direction, direction))
    projection = float(np.clip(projection, 0.0, 1.0))
    closest = missile + projection * direction
    center_line_distance = float(np.linalg.norm(cloud - closest))
    return {
        "phase": index + 1,
        "uav": plan.name,
        "interval_start_s": start,
        "interval_end_s": end,
        "snapshot_time_s": time,
        "release_time_s": plan.release_time,
        "burst_time_s": plan.burst_time,
        "cloud_active_end_s": plan.burst_time + model.CLOUD_LIFETIME,
        "missile_x_m": missile[0],
        "missile_y_m": missile[1],
        "missile_z_m": missile[2],
        "cloud_x_m": cloud[0],
        "cloud_y_m": cloud[1],
        "cloud_z_m": cloud[2],
        "release_x_m": release[0],
        "release_y_m": release[1],
        "release_z_m": release[2],
        "burst_x_m": burst[0],
        "burst_y_m": burst[1],
        "burst_z_m": burst[2],
        "center_line_distance_m": center_line_distance,
    }


def draw_global_xy(axis: plt.Axes, state: dict[str, float | str], plan: model.UavPlan, index: int) -> float:
    color, linestyle = COLORS[index], LINESTYLES[index]
    missile = np.array((state["missile_x_m"], state["missile_y_m"], state["missile_z_m"]), dtype=float)
    cloud = np.array((state["cloud_x_m"], state["cloud_y_m"], state["cloud_z_m"]), dtype=float)
    draw_target_xy(axis)
    draw_uav_path_xy(axis, plan, color, linestyle)
    for point in representative_target_points():
        axis.plot([missile[0], point[0]], [missile[1], point[1]], color=MISSILE_COLOR, linewidth=0.7, alpha=0.35, zorder=1)
    axis.scatter([missile[0]], [missile[1]], marker=">", s=56, color=MISSILE_COLOR, zorder=6)
    axis.scatter([cloud[0]], [cloud[1]], marker="o", s=62, facecolor="white", edgecolor=color, linewidth=2.0, zorder=7)
    axis.annotate("M1", missile[:2], xytext=(5, 6), textcoords="offset points", fontsize=8.6, color=MISSILE_COLOR)
    axis.annotate("烟幕球心", cloud[:2], xytext=(5, -13), textcoords="offset points", fontsize=8.4, color=color)
    axis.text(0.025, 0.96, f"{state['uav']} 阶段  $t={float(state['snapshot_time_s']):.3f}$ s", transform=axis.transAxes,
            ha="left", va="top", fontsize=11.2, color=color, fontweight="bold")
    distance = draw_local_inset(axis, missile, cloud, color, HATCHES[index])
    axis.set_xlim(-600.0, 20600.0)
    axis.set_ylim(-3600.0, 2400.0)
    axis.set_xlabel("$x$ / m", fontsize=11.0)
    axis.set_ylabel("$y$ / m", fontsize=11.0)
    style_axis(axis)
    return distance


def draw_global_xz(axis: plt.Axes, state: dict[str, float | str], plan: model.UavPlan, index: int) -> None:
    color, linestyle = COLORS[index], LINESTYLES[index]
    missile = np.array((state["missile_x_m"], state["missile_y_m"], state["missile_z_m"]), dtype=float)
    cloud = np.array((state["cloud_x_m"], state["cloud_y_m"], state["cloud_z_m"]), dtype=float)
    draw_target_xz(axis)
    draw_uav_path_xz(axis, plan, color, linestyle)
    for point in representative_target_points():
        axis.plot([missile[0], point[0]], [missile[2], point[2]], color=MISSILE_COLOR, linewidth=0.7, alpha=0.35, zorder=1)
    burst = plan.burst_point
    axis.plot([burst[0], cloud[0]], [burst[2], cloud[2]], color=color, linestyle=(0, (2, 2)), linewidth=1.3, alpha=0.65, zorder=3)
    axis.scatter([missile[0]], [missile[2]], marker=">", s=56, color=MISSILE_COLOR, zorder=6)
    axis.scatter([cloud[0]], [cloud[2]], marker="o", s=62, facecolor="white", edgecolor=color, linewidth=2.0, zorder=7)
    axis.annotate("M1", (missile[0], missile[2]), xytext=(5, 6), textcoords="offset points", fontsize=8.6, color=MISSILE_COLOR)
    axis.annotate("烟幕球心", (cloud[0], cloud[2]), xytext=(5, -13), textcoords="offset points", fontsize=8.4, color=color)
    axis.set_xlim(-600.0, 20600.0)
    axis.set_ylim(-100.0, 2200.0)
    axis.set_xlabel("$x$ / m", fontsize=11.0)
    axis.set_ylabel("$z$ / m", fontsize=11.0)
    style_axis(axis)


def write_states(path: Path, states: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(states[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(states)


def draw(rows: list[dict[str, str]], output: Path, states_path: Path) -> Path:
    intervals = match_intervals(rows, parse_intervals(rows[0]["joint_intervals_s"]))
    plans = tuple(plan_from_row(row, index) for index, row in enumerate(rows))
    states = [snapshot_state(row, plan, index, interval) for index, (row, plan, interval) in enumerate(zip(rows, plans, intervals))]
    write_states(states_path, states)

    fig, axes = plt.subplots(2, 3, figsize=(18.2, 10.2), gridspec_kw={"height_ratios": (1.0, 0.82)})
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.075, top=0.855, wspace=0.17, hspace=0.22)
    local_distances = []
    for index, (state, plan) in enumerate(zip(states, plans)):
        local_distances.append(draw_global_xy(axes[0, index], state, plan, index))
        draw_global_xz(axes[1, index], state, plan, index)

    notes = (
        "颜色对应 FY1/FY2/FY3；小空心圆：无人机初始点　□：投放点　★：起爆点　大空心圆：当前烟幕球心\n"
        "局部窗按真实尺度绘制半径 10 m 烟幕球与代表视线束；正式有效性由完整圆柱表面判据 720×21×15 给出"
    )
    fig.text(0.985, 0.965, notes, ha="right", va="top", fontsize=10.3,
             bbox=dict(boxstyle="round,pad=0.42", facecolor="#F7F9FA", edgecolor="#AAB2BA"))
    for state, local_distance in zip(states, local_distances):
        if abs(float(state["center_line_distance_m"]) - local_distance) > 1.0e-6:
            raise RuntimeError("局部图与快照 CSV 的中心视线距离不一致。")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    trim_white_margins(output, padding=28)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制问题四 baseline-900 空间遮蔽快照图。")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES)
    args = parser.parse_args()
    configure_matplotlib()
    output = draw(read_rows(args.input), args.output, args.states)
    print(f"已生成：{output}")
    print(f"快照状态：{args.states}")


if __name__ == "__main__":
    main()
