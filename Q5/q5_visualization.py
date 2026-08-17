from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Q1.q1_visualization import (  # noqa: E402
    COLORS as Q1_COLORS,
    configure_matplotlib,
    draw_cylinder,
    style_3d_axis,
    trim_white_margins,
)
from Q5.q5_column_generation import load_best_plans  # noqa: E402
from Q5.q5_main import (  # noqa: E402
    CLOUD_LIFETIME,
    CLOUD_RADIUS,
    CLOUD_SINK_SPEED,
    GLOBAL_HORIZON,
    MISSILE_IMPACT_TIMES,
    MISSILE_INITIALS,
    MISSILE_NAMES,
    MISSILE_VELOCITIES,
    TARGET_BOTTOM_CENTER,
    TARGET_HEIGHT,
    TARGET_RADIUS,
    BombPlan,
    MultiMissileEvaluator,
    Strategy,
    UAV_INITIALS,
    UAV_NAMES,
)

Q5_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = Q5_DIR / "supplement"
DEFAULT_OUTPUT = Q5_DIR / "figures"

UAV_COLORS = {
    "FY1": "#C84B31",
    "FY2": "#2878B5",
    "FY3": "#2E8B57",
    "FY4": "#8F5BB3",
    "FY5": "#77622E",
}
MISSILE_COLORS = {"M1": "#C84B31", "M2": "#2878B5", "M3": "#2E8B57"}
BOMB_STYLES = {1: "-", 2: "--", 3: "-."}
BOMB_HATCHES = {1: "////", 2: "\\\\", 3: ".."}
JOINT_COLOR = "#E67E22"
NEUTRAL = "#3F4347"
LIGHT = "#9BA1A6"
GRID = "#CBD2D9"
SCALE = 1000.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def parse_intervals(value: str) -> tuple[tuple[float, float], ...]:
    result: list[tuple[float, float]] = []
    for item in value.split(";"):
        item = item.strip().strip("[]")
        if item:
            left, right = item.split(",")
            result.append((float(left), float(right)))
    return tuple(result)


def style_axis(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.75, alpha=0.78)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    trim_white_margins(path, padding=28)
    return path


def label_bomb(label: str) -> tuple[str, int]:
    uav, bomb = label.split("-B")
    return uav, int(bomb)


def strategy_from_source(source: Path) -> Strategy:
    return Strategy(load_best_plans(source / "q5_cg_best_solution.csv"))


def flat_labels(strategy: Strategy) -> tuple[str, ...]:
    return tuple(
        f"{plan.name}-B{bomb_index}"
        for plan in strategy.uavs
        for bomb_index, _ in enumerate(plan.bombs, start=1)
    )


def smoke_center(plan, bomb: BombPlan, time_value: float) -> np.ndarray:
    point = plan.burst_point(bomb).astype(float)
    point[2] -= CLOUD_SINK_SPEED * (time_value - bomb.burst_time)
    return point


def merge_mode_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda value: (value["missile"], number(value, "start_s"))):
        mode = row["mode"]
        coverers = row["individual_complete_coverers"]
        # In an overlap, use one persistent cloud as the visible relay label;
        # the underlying interval still represents the complete joint result.
        primary = coverers.split(";")[0] if coverers else ""
        start, end = number(row, "start_s"), number(row, "end_s")
        if (
            merged
            and merged[-1]["missile"] == row["missile"]
            and merged[-1]["mode"] == mode
            and merged[-1]["primary"] == primary
            and start <= float(merged[-1]["end_s"]) + 1.0e-7
        ):
            merged[-1]["end_s"] = end
        else:
            merged.append(
                {
                    "missile": row["missile"],
                    "mode": mode,
                    "primary": primary,
                    "start_s": start,
                    "end_s": end,
                }
            )
    return merged


def draw_timeline(source: Path, output: Path) -> Path:
    interval_rows = read_csv(source / "q5_cg_missile_intervals.csv")
    mode_rows = merge_mode_rows(read_csv(source / "q5_cg_occlusion_modes.csv"))
    summary = {row["missile"]: row for row in read_csv(source / "q5_cg_occlusion_mode_summary.csv")}
    durations = {row["missile"]: number(row, "duration_s") for row in interval_rows}
    intervals = {row["missile"]: parse_intervals(row["intervals_s"]) for row in interval_rows}
    strategy = strategy_from_source(source)
    bombs = {
        f"{plan.name}-B{bomb_index}": (plan, bomb)
        for plan in strategy.uavs
        for bomb_index, bomb in enumerate(plan.bombs, start=1)
    }

    # This deliberately follows the Question 3 visual grammar:
    # release square -> ballistic timeline -> burst star -> shaded 20 s cloud
    # window -> bottom joint-decision row.  A panel is used for each missile so
    # that all responsible smoke clouds remain legible rather than being packed
    # into a single 15-row chart.
    fig, axes = plt.subplots(3, 1, figsize=(17.2, 13.8), sharex=True, gridspec_kw={"hspace": 0.30})
    data_rows: list[dict[str, object]] = []
    blank_start, blank_end = 28.35, 53.35
    for axis, missile in zip(axes, MISSILE_NAMES):
        local = [item for item in mode_rows if item["missile"] == missile]
        labels: list[str] = []
        for item in local:
            if item["primary"] and item["primary"] not in labels:
                labels.append(str(item["primary"]))
        labels.sort(key=lambda label: min(
            float(item["start_s"]) for item in local if item["primary"] == label
        ))
        levels = {label: len(labels) - index for index, label in enumerate(labels)}
        joint_level = 0.0
        missile_color = MISSILE_COLORS[missile]

        axis.axvspan(blank_start, blank_end, facecolor="#EEF1F3", alpha=0.72, zorder=0)
        axis.text((blank_start + blank_end) / 2.0, len(labels) + 0.60, "无有效联合遮蔽阶段",
                  ha="center", va="center", color=LIGHT, fontsize=10.0)
        for left, right in intervals[missile]:
            axis.add_patch(
                Rectangle(
                    (left, joint_level - 0.31), right - left, 0.62,
                    facecolor=missile_color, edgecolor=missile_color, alpha=0.16,
                    linewidth=1.15, hatch="//", zorder=2,
                )
            )

        for label in labels:
            plan, bomb = bombs[label]
            uav, bomb_index = label_bomb(label)
            y = levels[label]
            release, burst = bomb.release_time, bomb.burst_time
            cloud_end = min(burst + CLOUD_LIFETIME, MISSILE_IMPACT_TIMES[MISSILE_NAMES.index(missile)])
            color = UAV_COLORS[uav]
            axis.axhline(y, color="#E5E8EB", linewidth=0.85, zorder=0)
            axis.plot([release, burst], [y, y], color=color, linewidth=2.25,
                      linestyle=BOMB_STYLES[bomb_index], solid_capstyle="round", zorder=4)
            axis.scatter(release, y, marker="s", s=48, facecolor="white", edgecolor=color,
                         linewidth=1.45, zorder=5)
            axis.scatter(burst, y, marker="*", s=155, color=color, edgecolor="white",
                         linewidth=0.75, zorder=6)
            if cloud_end > burst:
                axis.add_patch(
                    Rectangle(
                        (burst, y - 0.27), cloud_end - burst, 0.54,
                        facecolor=color, edgecolor=color, alpha=0.15,
                        linewidth=1.0, hatch=BOMB_HATCHES[bomb_index], zorder=1,
                    )
                )
            for item in [value for value in local if value["primary"] == label]:
                left, right = float(item["start_s"]), float(item["end_s"])
                axis.plot([left, right], [y, y], color=color, linewidth=6.0,
                          linestyle=BOMB_STYLES[bomb_index], solid_capstyle="butt", zorder=7)
                data_rows.append(
                    {
                        "missile": missile,
                        "mode": "single_cloud_relay",
                        "coverer": label,
                        "release_s": release,
                        "burst_s": burst,
                        "start_s": left,
                        "end_s": right,
                        "duration_s": right - left,
                    }
                )
            axis.plot([burst, burst], [joint_level + 0.35, y - 0.34], color=color,
                      linewidth=0.95, linestyle=(0, (3, 5)), alpha=0.48, zorder=3)

        for item in [value for value in local if value["mode"] == "spatial_joint"]:
            left, right = float(item["start_s"]), float(item["end_s"])
            axis.add_patch(
                Rectangle(
                    (left, joint_level - 0.31), right - left, 0.62,
                    facecolor=JOINT_COLOR, edgecolor=JOINT_COLOR, alpha=0.50,
                    linewidth=1.0, hatch="xx", zorder=8,
                )
            )
            axis.annotate(
                "空间联合\n0.315 s",
                xy=((left + right) / 2.0, joint_level + 0.30),
                xytext=(left + 1.15, joint_level + 0.78),
                ha="center", va="bottom", fontsize=9.0, color=JOINT_COLOR,
                arrowprops={"arrowstyle": "-", "color": JOINT_COLOR, "linewidth": 0.9},
                zorder=9,
            )
            data_rows.append(
                {
                    "missile": missile,
                    "mode": "spatial_joint",
                    "coverer": "",
                    "release_s": "",
                    "burst_s": "",
                    "start_s": left,
                    "end_s": right,
                    "duration_s": right - left,
                }
            )

        tick_positions = [joint_level] + [levels[label] for label in labels]
        tick_labels = ["联合判定"] + labels
        axis.set_yticks(tick_positions, tick_labels)
        for tick, label in zip(axis.get_yticklabels()[1:], labels):
            tick.set_color(UAV_COLORS[label_bomb(label)[0]])
            tick.set_fontweight("bold")
        axis.get_yticklabels()[0].set_color(missile_color)
        axis.get_yticklabels()[0].set_fontweight("bold")
        axis.text(0.985, 0.95, f"{missile} 联合有效：{durations[missile]:.3f} s",
                  transform=axis.transAxes, ha="right", va="top", color=missile_color,
                  fontsize=12.6, fontweight="bold")
        axis.set_ylim(-0.72, len(labels) + 0.96)
        axis.set_xlim(0.0, math.ceil(GLOBAL_HORIZON))
        axis.tick_params(axis="y", labelsize=10.5, pad=9)
        axis.tick_params(axis="x", labelsize=10.0, pad=5)
        style_axis(axis, grid_axis="x")

    total = sum(durations.values())
    legend = [
        Line2D([0], [0], marker="s", color=NEUTRAL, markerfacecolor="white", markersize=6.5,
               linestyle="None", label="投放"),
        Line2D([0], [0], marker="*", color=NEUTRAL, markersize=10,
               linestyle="None", label="起爆"),
        Patch(facecolor="#E8EDF0", edgecolor="#AAB4BE", hatch="//", label="该烟幕 20 s 有效窗"),
        Patch(facecolor="#E8EDF0", edgecolor="#AAB4BE", hatch="//", label="联合有效区间"),
        Patch(facecolor=JOINT_COLOR, edgecolor=JOINT_COLOR, hatch="xx", label="空间联合遮蔽"),
    ]
    fig.legend(handles=legend, ncol=5, loc="upper left", bbox_to_anchor=(0.10, 0.997),
               frameon=True, edgecolor="#AAB4BE", fontsize=9.6, columnspacing=1.25)
    fig.text(0.975, 0.986,
             f"M1：{durations['M1']:.3f} s    M2：{durations['M2']:.3f} s    "
             f"M3：{durations['M3']:.3f} s\n总遮蔽：{total:.4f} 导弹·秒",
             ha="right", va="top", fontsize=10.5,
             bbox={"boxstyle": "round,pad=0.42", "facecolor": "white", "edgecolor": "#AAB4BE", "linewidth": 1.0})
    axes[-1].set_xlabel("时间 $t$ / s", fontsize=14)
    write_csv(output / "data" / "q5_timeline_segments.csv", data_rows)
    return save(fig, output / "q5_01_three_channel_timeline.png")


def _draw_target_2d(ax: plt.Axes, projection: str) -> None:
    x0, y0, z0 = TARGET_BOTTOM_CENTER / SCALE
    if projection == "xy":
        ax.add_patch(Circle((x0, y0), TARGET_RADIUS / SCALE, facecolor=Q1_COLORS["target"], alpha=0.35, edgecolor=Q1_COLORS["target"]))
        ax.scatter([0.0], [0.0], marker="x", color=Q1_COLORS["false"], s=32, zorder=5)
    else:
        ax.add_patch(Rectangle((x0 - TARGET_RADIUS / SCALE, z0), 2 * TARGET_RADIUS / SCALE, TARGET_HEIGHT / SCALE,
                               facecolor=Q1_COLORS["target"], alpha=0.35, edgecolor=Q1_COLORS["target"]))


def draw_global_airspace(source: Path, output: Path) -> Path:
    strategy = strategy_from_source(source)
    fig = plt.figure(figsize=(17.0, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.12, 1.0))
    ax3 = fig.add_subplot(grid[:, 0], projection="3d")
    ax_xy = fig.add_subplot(grid[0, 1])
    ax_xz = fig.add_subplot(grid[1, 1])

    for missile_index, missile in enumerate(MISSILE_NAMES):
        times = np.linspace(0.0, MISSILE_IMPACT_TIMES[missile_index], 160)
        positions = (MISSILE_INITIALS[missile_index][None, :] + times[:, None] * MISSILE_VELOCITIES[missile_index][None, :]) / SCALE
        ax3.plot(*positions.T, color=NEUTRAL, linewidth=1.65, linestyle=("-", "--", "-.")[missile_index], alpha=0.9)
        ax_xy.plot(positions[:, 0], positions[:, 1], color=NEUTRAL, linewidth=1.45, linestyle=("-", "--", "-.")[missile_index], label=missile)
        ax_xz.plot(positions[:, 0], positions[:, 2], color=NEUTRAL, linewidth=1.45, linestyle=("-", "--", "-.")[missile_index], label=missile)
        ax_xy.text(positions[0, 0], positions[0, 1], missile, color=NEUTRAL, fontsize=9.5)

    for plan in strategy.uavs:
        color = UAV_COLORS[plan.name]
        t_end = max(bomb.release_time for bomb in plan.bombs)
        times = np.linspace(0.0, t_end, 80)
        points = np.asarray(plan.initial)[None, :] + plan.speed * times[:, None] * plan.direction[None, :]
        scaled = points / SCALE
        ax3.plot(*scaled.T, color=color, linewidth=2.15, zorder=5)
        ax_xy.plot(scaled[:, 0], scaled[:, 1], color=color, linewidth=1.85, label=plan.name)
        ax_xz.plot(scaled[:, 0], scaled[:, 2], color=color, linewidth=1.85)
        ax_xy.text(scaled[0, 0], scaled[0, 1], plan.name, color=color, fontsize=9.5, fontweight="bold")
        for bomb in plan.bombs:
            release, burst = plan.release_point(bomb) / SCALE, plan.burst_point(bomb) / SCALE
            ax3.scatter(*release, marker="s", facecolor="white", edgecolor=color, s=28, linewidth=1.0, zorder=7)
            ax3.scatter(*burst, marker="*", color=color, s=55, zorder=8)
            ax_xy.scatter(release[0], release[1], marker="s", facecolor="white", edgecolor=color, s=22, linewidth=0.9, zorder=7)
            ax_xy.scatter(burst[0], burst[1], marker="*", color=color, s=38, zorder=8)
            ax_xz.scatter(release[0], release[2], marker="s", facecolor="white", edgecolor=color, s=22, linewidth=0.9, zorder=7)
            ax_xz.scatter(burst[0], burst[2], marker="*", color=color, s=38, zorder=8)

    draw_cylinder(ax3, TARGET_BOTTOM_CENTER / SCALE, TARGET_RADIUS / SCALE, TARGET_HEIGHT / SCALE, Q1_COLORS["target"], alpha=0.35)
    ax3.scatter(0.0, 0.0, 0.0, marker="x", color=Q1_COLORS["false"], s=38)
    _draw_target_2d(ax_xy, "xy")
    _draw_target_2d(ax_xz, "xz")
    for axis, projection in ((ax_xy, "xy"), (ax_xz, "xz")):
        axis.set_xlabel("$x$ / km")
        axis.set_ylabel("$y$ / km" if projection == "xy" else "$z$ / km")
        style_axis(axis)
        axis.set_aspect("equal" if projection == "xy" else "auto")
    ax3.set_xlabel("$x$ / km")
    ax3.set_ylabel("$y$ / km")
    ax3.set_zlabel("$z$ / km")
    ax3.view_init(elev=24, azim=-58)
    ax3.set_box_aspect((20, 6, 2.7))
    style_3d_axis(ax3)
    # UAV and missile names are directly labelled at their initial positions.
    # Keep the legend to the two marker meanings so the horizontal projection
    # remains readable when all five aircraft are shown together.
    handles = [
        Line2D([0], [0], marker="s", color=NEUTRAL, markerfacecolor="white", linestyle="None", label="投放点"),
        Line2D([0], [0], marker="*", color=NEUTRAL, linestyle="None", label="起爆点"),
    ]
    ax_xy.legend(handles=handles, loc="upper left", fontsize=8.6, frameon=False)
    ax3.text2D(0.02, 0.97, "（a）三维总览", transform=ax3.transAxes, fontsize=11.5)
    ax_xy.text(0.02, 0.94, "（b）水平投影", transform=ax_xy.transAxes, fontsize=11.5)
    ax_xz.text(0.02, 0.94, "（c）竖直投影", transform=ax_xz.transAxes, fontsize=11.5)
    return save(fig, output / "q5_02_global_airspace.png")


def draw_ablation(source: Path, output: Path) -> Path:
    uav_rows = read_csv(source / "q5_cg_ablation_uavs.csv")
    bomb_rows = read_csv(source / "q5_cg_ablation_bombs.csv")
    fig, (ax_uav, ax_bomb) = plt.subplots(1, 2, figsize=(16.8, 7.2), gridspec_kw={"width_ratios": (1.05, 1.15)})
    components = ("M1", "M2", "M3")
    y = np.arange(len(UAV_NAMES))
    height = 0.17
    for offset, missile in zip((-0.25, -0.08, 0.09), components):
        values = [number(next(row for row in uav_rows if row["removed_id"] == name), f"loss_{missile}_s") for name in UAV_NAMES]
        ax_uav.barh(y + offset, values, height, color=MISSILE_COLORS[missile], alpha=0.78, label=f"{missile} 损失")
    totals = [number(next(row for row in uav_rows if row["removed_id"] == name), "loss_total_s") for name in UAV_NAMES]
    ax_uav.scatter(totals, y + 0.31, color=NEUTRAL, marker="D", s=35, zorder=5, label="总损失")
    for index, value in enumerate(totals):
        ax_uav.text(value + 0.18, index + 0.31, f"{value:.2f}", va="center", fontsize=9.5, color=NEUTRAL)
    ax_uav.set_yticks(y, UAV_NAMES)
    ax_uav.invert_yaxis()
    ax_uav.set_xlabel("撤除后的遮蔽时长损失 / s")
    ax_uav.text(0.02, 0.96, "（a）逐机消融", transform=ax_uav.transAxes, va="top", fontsize=11.5)
    style_axis(ax_uav, grid_axis="x")
    ax_uav.legend(frameon=False, fontsize=9, loc="lower right")

    rows = sorted(bomb_rows, key=lambda row: number(row, "loss_total_s"), reverse=True)
    labels = [row["removed_id"] for row in rows]
    values = np.array([number(row, "loss_total_s") for row in rows])
    primary = []
    for row in rows:
        losses = np.array([number(row, f"loss_{missile}_s") for missile in components])
        primary.append(components[int(np.argmax(losses))] if np.max(losses) > 1.0e-5 else "冗余")
    colors = [MISSILE_COLORS.get(name, LIGHT) for name in primary]
    y_b = np.arange(len(rows))
    ax_bomb.barh(y_b, values, color=colors, alpha=0.80)
    for index, (value, kind) in enumerate(zip(values, primary)):
        if value < 0.05:
            ax_bomb.text(max(value, 0.02) + 0.08, index, "冗余/备份", va="center", fontsize=8.7, color=LIGHT)
        elif value >= 1.3:
            ax_bomb.text(value + 0.06, index, f"{value:.2f}", va="center", fontsize=8.7, color=NEUTRAL)
    ax_bomb.set_yticks(y_b, labels)
    ax_bomb.invert_yaxis()
    ax_bomb.set_xlabel("撤除后的总遮蔽时长损失 / 导弹·秒")
    ax_bomb.text(0.02, 0.96, "（b）逐弹消融（按总贡献排序）", transform=ax_bomb.transAxes, va="top", fontsize=11.5)
    style_axis(ax_bomb, grid_axis="x")
    ax_bomb.text(0.02, -0.14, "注：各弹贡献不能相加，因有效遮蔽区间可能重叠。", transform=ax_bomb.transAxes, fontsize=9.2, color=NEUTRAL)
    return save(fig, output / "q5_03_ablation_contributions.png")


def draw_selection(source: Path, output: Path) -> Path:
    rows = read_csv(source / "q5_cg_convergence.csv")
    ultra = [row for row in rows if row["grid"] == "ultra"]
    cross = next(row for row in rows if row["grid"] == "cross_check")
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15.5, 6.7), gridspec_kw={"width_ratios": (1.0, 1.0)})
    labels = {
        "refined_master_lower": "最终推荐",
        "refined_master_balanced": "均衡候选",
        "refined_master_center_rank3": "中心候选",
    }
    markers = {"refined_master_lower": "*", "refined_master_balanced": "^", "refined_master_center_rank3": "s"}
    colors = {"refined_master_lower": "#C84B31", "refined_master_balanced": "#2E8B57", "refined_master_center_rank3": "#2878B5"}
    for row in ultra:
        source_name = row["source"]
        ax_left.scatter(number(row, "total_s"), number(row, "minimum_s"), s=180 if source_name == "refined_master_lower" else 95,
                        marker=markers.get(source_name, "o"), color=colors.get(source_name, LIGHT), zorder=4)
        ax_left.annotate(labels.get(source_name, source_name), (number(row, "total_s"), number(row, "minimum_s")),
                         xytext=(7, 7), textcoords="offset points", fontsize=10, color=colors.get(source_name, NEUTRAL))
    ax_left.set_xlabel("总有效遮蔽时长 / 导弹·秒")
    ax_left.set_ylabel("最短导弹遮蔽时长 / s")
    ax_left.text(0.02, 0.96, "（a）严格网格下的候选比较", transform=ax_left.transAxes, va="top", fontsize=11.5)
    style_axis(ax_left)

    grid_names = ("dense", "high", "ultra")
    chosen = []
    for name in grid_names:
        row = next(item for item in rows if item["grid"] == name and item["source"] == "refined_master_lower")
        chosen.append((row["grid_spec"], number(row, "total_s")))
    chosen.append((cross["grid_spec"], number(cross, "total_s")))
    x = np.arange(len(chosen))
    values = [item[1] for item in chosen]
    ax_right.plot(x, values, color="#C84B31", linewidth=2.2, marker="o", markersize=6)
    for index, value in enumerate(values):
        ax_right.text(index, value + 0.00020, f"{value:.6f}", ha="center", va="bottom", fontsize=9.2, color="#C84B31")
    ax_right.set_xticks(x, [item[0].replace("@", "\n@") for item in chosen])
    ax_right.set_ylabel("总有效遮蔽时长 / 导弹·秒")
    ax_right.text(0.02, 0.96, "（b）最终方案的网格加密复核", transform=ax_right.transAxes, va="top", fontsize=11.5)
    ax_right.margins(y=0.35)
    style_axis(ax_right)
    stability_rows = [{"grid_spec": name, "total_s": value} for name, value in chosen]
    write_csv(output / "data" / "q5_grid_stability.csv", stability_rows)
    return save(fig, output / "q5_04_selection_and_grid_check.png")


def draw_m3_bottleneck(source: Path, output: Path) -> Path:
    strategy = strategy_from_source(source)
    rows = read_csv(source / "q5_cg_reachability_m3.csv")
    fig = plt.figure(figsize=(16.5, 7.1), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.25, 0.92, 0.92))
    ax_xy = fig.add_subplot(grid[0, 0])
    middle = grid[0, 1].subgridspec(2, 1, hspace=0.18)
    ax_height = fig.add_subplot(middle[0, 0])
    ax_delay = fig.add_subplot(middle[1, 0])
    ax_reach = fig.add_subplot(grid[0, 2])

    m_index = 2
    times = np.linspace(0.0, MISSILE_IMPACT_TIMES[m_index], 180)
    m3 = (MISSILE_INITIALS[m_index][None, :] + times[:, None] * MISSILE_VELOCITIES[m_index][None, :]) / SCALE
    ax_xy.plot(m3[:, 0], m3[:, 1], color=NEUTRAL, linewidth=2.1, label="M3 轨迹")
    for plan in strategy.uavs:
        t_end = max(bomb.release_time for bomb in plan.bombs)
        pts = np.asarray(plan.initial)[None, :] + plan.speed * np.linspace(0.0, t_end, 60)[:, None] * plan.direction[None, :]
        points = pts / SCALE
        color = UAV_COLORS[plan.name]
        width = 2.5 if plan.name == "FY3" else 1.45
        ax_xy.plot(points[:, 0], points[:, 1], color=color, linewidth=width, alpha=0.92)
        ax_xy.scatter(points[0, 0], points[0, 1], color=color, s=26)
        ax_xy.text(points[0, 0], points[0, 1], plan.name, color=color, fontsize=9.6, fontweight="bold")
    ax_xy.annotate("FY3 横向偏移\n$y=-3.0$ km", xy=(UAV_INITIALS[2, 0] / SCALE, UAV_INITIALS[2, 1] / SCALE),
                   xytext=(1.5, -2.0), textcoords="data", fontsize=10, color=UAV_COLORS["FY3"],
                   arrowprops={"arrowstyle": "->", "color": UAV_COLORS["FY3"], "linewidth": 1.0})
    ax_xy.set_xlabel("$x$ / km")
    ax_xy.set_ylabel("$y$ / km")
    ax_xy.set_aspect("equal")
    ax_xy.text(0.02, 0.96, "（a）M3 通道的水平几何关系", transform=ax_xy.transAxes, va="top", fontsize=11.5)
    style_axis(ax_xy)

    names = [row["uav"] for row in rows]
    height_values = [number(row, "initial_z_m") for row in rows]
    delay_values = [number(row, "fuse_delay_limit_s") for row in rows]
    index = np.arange(len(names))
    ax_height.bar(index, height_values, color=[UAV_COLORS[name] for name in names], alpha=0.78)
    ax_height.set_xticks(index, [])
    ax_height.set_ylabel("初始高度 / m")
    ax_height.text(0.02, 0.92, "（b）初始高度", transform=ax_height.transAxes, va="top", fontsize=10.5)
    style_axis(ax_height, grid_axis="y")
    ax_delay.bar(index, delay_values, color=[UAV_COLORS[name] for name in names], alpha=0.78)
    ax_delay.set_xticks(index, names)
    ax_delay.set_ylabel("最大延迟 / s")
    ax_delay.text(0.02, 0.92, "最大自由落体延迟", transform=ax_delay.transAxes, va="top", fontsize=10.2)
    style_axis(ax_delay, grid_axis="y")

    reach_values = [number(row, "M3_library_candidate_s") for row in rows]
    ax_reach.barh(index, reach_values, color=[UAV_COLORS[name] for name in names], alpha=0.80)
    for idx, value in enumerate(reach_values):
        ax_reach.text(value + 0.06, idx, f"{value:.2f}", va="center", fontsize=9.2)
    ax_reach.set_yticks(index, names)
    ax_reach.invert_yaxis()
    ax_reach.set_xlabel("M3 单机候选库最佳时长 / s")
    ax_reach.text(0.02, 0.96, "（c）M3 单机可达性证据", transform=ax_reach.transAxes, va="top", fontsize=11.5)
    ax_reach.text(0.02, -0.13, "候选库结果仅用于说明可达性，\n不是多机系统的理论上界。", transform=ax_reach.transAxes, fontsize=8.8, color=NEUTRAL)
    style_axis(ax_reach, grid_axis="x")
    return save(fig, output / "q5_05_m3_bottleneck.png")


def _local_windows(intervals: tuple[tuple[float, float], ...], padding: float = 0.35) -> tuple[tuple[float, float], ...]:
    values: list[list[float]] = []
    for left, right in intervals:
        left, right = max(0.0, left - padding), min(GLOBAL_HORIZON, right + padding)
        if values and left <= values[-1][1] + 0.01:
            values[-1][1] = max(values[-1][1], right)
        else:
            values.append([left, right])
    return tuple((left, right) for left, right in values)


def margin_rows(strategy: Strategy, source: Path) -> list[dict[str, object]]:
    interval_rows = read_csv(source / "q5_cg_missile_intervals.csv")
    interval_map = {row["missile"]: parse_intervals(row["intervals_s"]) for row in interval_rows}
    evaluator = MultiMissileEvaluator(720, 21, 15, 0.005, time_batch_size=32, surface_batch_size=10_000)
    flat = evaluator.flatten(strategy)
    rows: list[dict[str, object]] = []
    for missile_index, missile in enumerate(MISSILE_NAMES):
        for left, right in _local_windows(interval_map[missile]):
            times = np.arange(left, right + 1.0e-12, 0.05)
            if abs(times[-1] - right) > 1.0e-9:
                times = np.append(times, right)
            values = np.full(len(times), -np.inf, dtype=float)
            for seg_left, seg_right, active in evaluator.event_segments(flat, missile_index):
                mask = (times >= seg_left - 1.0e-10) & (times <= seg_right + 1.0e-10)
                if np.any(mask):
                    values[mask] = evaluator.margin_batch(flat, missile_index, times[mask], active)
            rows.extend(
                {
                    "missile": missile, "window_left_s": left, "window_right_s": right,
                    "time_s": float(time_value), "margin_m": float(margin),
                }
                for time_value, margin in zip(times, values)
            )
    return rows


def draw_margin(source: Path, output: Path) -> Path:
    strategy = strategy_from_source(source)
    rows = margin_rows(strategy, source)
    write_csv(output / "data" / "q5_margin_curves.csv", rows)
    interval_rows = read_csv(source / "q5_cg_missile_intervals.csv")
    interval_map = {row["missile"]: parse_intervals(row["intervals_s"]) for row in interval_rows}
    modes = read_csv(source / "q5_cg_occlusion_modes.csv")
    fig, axes = plt.subplots(3, 1, figsize=(16.5, 8.0), sharex=True)
    for axis, missile in zip(axes, MISSILE_NAMES):
        local = [row for row in rows if row["missile"] == missile]
        grouped: dict[tuple[float, float], list[dict[str, object]]] = defaultdict(list)
        for row in local:
            grouped[(float(row["window_left_s"]), float(row["window_right_s"]))].append(row)
        for _, values in grouped.items():
            x = np.asarray([float(row["time_s"]) for row in values])
            y = np.asarray([float(row["margin_m"]) for row in values])
            axis.plot(x, y, color=MISSILE_COLORS[missile], linewidth=1.5)
            axis.fill_between(x, 0.0, np.maximum(y, 0.0), color=MISSILE_COLORS[missile], alpha=0.16, hatch="//")
        for left, right in interval_map[missile]:
            axis.axvspan(left, right, color=MISSILE_COLORS[missile], alpha=0.06)
        for item in modes:
            if item["missile"] == missile and item["mode"] == "spatial_joint":
                axis.axvspan(number(item, "start_s"), number(item, "end_s"), color=JOINT_COLOR, alpha=0.35, hatch="xx")
        axis.axhline(0.0, color=NEUTRAL, linewidth=1.1)
        axis.set_ylabel(f"{missile}\n$H(t)$ / m", color=MISSILE_COLORS[missile])
        axis.text(0.995, 0.86, r"$H(t)\geq0$：有效遮蔽", transform=axis.transAxes, ha="right", fontsize=9.2, color=NEUTRAL)
        axis.set_ylim(-2.5, 1.8)
        style_axis(axis)
    axes[-1].set_xlabel("时间 $t$ / s")
    axes[-1].set_xlim(0.0, math.ceil(GLOBAL_HORIZON))
    return save(fig, output / "q5_a1_joint_margin_curves.png")


def validate_pngs(paths: Iterable[Path]) -> None:
    from PIL import Image

    for path in paths:
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0, 0))[0]
            if image.width < 1000 or image.height < 600 or dpi < 295:
                raise RuntimeError(f"Figure verification failed: {path.name}, size={image.size}, dpi={dpi}")


def run(source: Path, output: Path, *, skip_margin: bool = False) -> list[Path]:
    configure_matplotlib()
    source, output = source.resolve(), output.resolve()
    required = (
        "q5_cg_best_solution.csv", "q5_cg_missile_intervals.csv", "q5_cg_occlusion_modes.csv",
        "q5_cg_occlusion_mode_summary.csv", "q5_cg_ablation_bombs.csv", "q5_cg_ablation_uavs.csv",
        "q5_cg_reachability_m3.csv", "q5_cg_convergence.csv",
    )
    missing = [name for name in required if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Q5 supplement files: {', '.join(missing)}")
    figures = [
        draw_timeline(source, output),
        draw_global_airspace(source, output),
        draw_ablation(source, output),
        draw_selection(source, output),
        draw_m3_bottleneck(source, output),
    ]
    if not skip_margin:
        figures.append(draw_margin(source, output))
    validate_pngs(figures)
    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Question 5 paper figures")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-margin", action="store_true", help="skip the expensive appendix margin diagnostic")
    args = parser.parse_args()
    figures = run(args.source_dir, args.output_dir, skip_margin=args.skip_margin)
    for path in figures:
        print(path)


if __name__ == "__main__":
    main()
