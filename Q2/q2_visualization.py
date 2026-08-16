from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image, ImageStat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Q1.q1_visualization import configure_matplotlib, trim_white_margins  # noqa: E402
from core.smoke_evaluator import (  # noqa: E402
    Deployment,
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

PALETTE = {
    "blue": "#2878B5",
    "orange": "#E67E22",
    "green": "#2E8B57",
    "purple": "#7B5EA7",
    "red": "#C44E52",
    "cyan": "#36A5A8",
    "gray": "#6C757D",
    "dark": "#263238",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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
    selected = [row for row in rows if row.get("selected", "").lower() == "true"]
    if len(selected) != 1:
        raise RuntimeError("q2_best_solution.csv 中必须恰有一个 selected=True 的方案。")
    row = selected[0]
    deployment = Deployment(
        heading=number(row, "theta_rad"),
        speed=number(row, "uav_speed_mps"),
        release_time=number(row, "release_time_s"),
        fuse_delay=number(row, "fuse_delay_s"),
    )
    result = evaluator.evaluate(deployment, mode="surface")
    if not result.feasible or result.release_point is None or result.burst_point is None:
        raise RuntimeError(f"最新方案不可行：{result.reason}")
    if len(result.intervals) != 1:
        raise RuntimeError(f"预期一个连续遮蔽区间，实际为 {result.intervals}")
    if abs(result.duration - number(row, "duration_s")) > 2.0e-7:
        raise RuntimeError("完整表面复核结果与 q2_best_solution.csv 不一致。")
    simulation = evaluator.simulation(deployment)
    if simulation is None:
        raise RuntimeError("无法建立最新方案仿真。")
    return deployment, simulation, result.intervals[0], result.duration


def style_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis, color="#CBD2D9", linewidth=0.8, alpha=0.68)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    trim_white_margins(path)
    with Image.open(path) as image:
        if image.width < 1200 or image.height < 650:
            raise RuntimeError(f"{path.name} 输出尺寸过小：{image.size}")
        if ImageStat.Stat(image.convert("L")).mean[0] > 252.8:
            raise RuntimeError(f"{path.name} 疑似空图。")


def draw_rim_principle() -> None:
    """仅保留目标上下双圆周三维主评价示意。"""
    fig = plt.figure(figsize=(9.6, 8.8))
    ax = fig.add_axes([0.08, 0.055, 0.84, 0.72], projection="3d")

    phi = np.linspace(0.0, 2.0 * np.pi, 241)
    z_levels = np.linspace(0.0, 10.0, 2)
    pp, zz = np.meshgrid(phi, z_levels)
    xx = 7.0 * np.cos(pp)
    yy = 200.0 + 7.0 * np.sin(pp)
    ax.plot_surface(xx, yy, zz, color=PALETTE["blue"], alpha=0.14, linewidth=0)
    ax.plot(
        7.0 * np.cos(phi), 200.0 + 7.0 * np.sin(phi), np.zeros_like(phi),
        color=PALETTE["blue"], linestyle="-", linewidth=3.0,
        label=r"下圆周 $\Gamma_0$（$z=0$ m）",
    )
    ax.plot(
        7.0 * np.cos(phi), 200.0 + 7.0 * np.sin(phi), np.full_like(phi, 10.0),
        color=PALETTE["purple"], linestyle="--", linewidth=3.0,
        label=r"上圆周 $\Gamma_{10}$（$z=10$ m）",
    )
    sample_phi = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    for height, color, marker in (
        (0.0, PALETTE["blue"], "o"),
        (10.0, PALETTE["purple"], "s"),
    ):
        ax.scatter(
            7.0 * np.cos(sample_phi), 200.0 + 7.0 * np.sin(sample_phi),
            np.full_like(sample_phi, height), s=27, color=color, marker=marker,
            edgecolor="white", linewidth=0.55, depthshade=False,
        )

    ax.set_xlabel(r"$x$/m", fontsize=13, labelpad=8)
    ax.set_ylabel(r"$y$/m", fontsize=13, labelpad=8)
    ax.set_zlabel(r"$z$/m", fontsize=13, labelpad=8)
    ax.set_xlim(-9, 9)
    ax.set_ylim(191, 209)
    ax.set_zlim(-1, 11)
    ax.set_box_aspect((18, 18, 12))
    ax.view_init(elev=23, azim=-48)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper left", fontsize=11.5, frameon=True, framealpha=0.96)

    fig.suptitle("主评价：目标上下双圆周", fontsize=20, fontweight="bold", y=0.978)
    fig.text(
        0.5, 0.885,
        r"目标外表面 $\partial T$：侧面与上下端面" "\n"
        r"DE内循环：在 $\Gamma_0\cup\Gamma_{10}$ 上进行高密度搜索" "\n"
        r"最终候选：在完整 $\partial T$ 上逐点复核",
        ha="center", va="top", fontsize=13, linespacing=1.42, color="black",
    )
    save_figure(fig, FIGURE_DIR / "q2_rim_principle.png")


def draw_de_table() -> None:
    rows = [
        row for row in read_csv(Q2_DIR / "q2_search_runs.csv")
        if row.get("record_type") == "directed_de_best"
    ]
    if len(rows) != 10:
        raise RuntimeError(f"DE汇总应有10轮，实际读取到{len(rows)}轮。")
    rows.sort(key=lambda row: int(row["source_seed"]))
    best = max(number(row, "duration_s") for row in rows)
    summary: list[dict[str, object]] = []
    cell_text: list[list[str]] = []
    for index, row in enumerate(rows, start=1):
        duration = number(row, "duration_s")
        gap_ms = 1000.0 * (best - duration)
        summary.append(
            {
                "run": index,
                "seed": row["source_seed"],
                "region": row["region_index"],
                "iterations": row["iterations"],
                "function_evaluations": row["function_evaluations"],
                "duration_s": f"{duration:.10f}",
                "gap_to_best_ms": f"{gap_ms:.6f}",
            }
        )
        cell_text.append(
            [
                str(index), row["source_seed"], row["region_index"], row["iterations"],
                row["function_evaluations"], f"{duration:.9f}", f"{gap_ms:.5f}",
            ]
        )
    write_csv(Q2_DIR / "q2_de_summary.csv", summary)

    durations = np.array([number(row, "duration_s") for row in rows])
    fig, ax = plt.subplots(figsize=(14.0, 5.8))
    ax.axis("off")
    columns = ["轮次", "随机种子", "搜索区域", "迭代数", "函数评价次数", r"$T_{eff}$/s", "距最优/ms"]
    table = ax.table(
        cellText=cell_text, colLabels=columns, cellLoc="center", colLoc="center",
        bbox=[0.025, 0.145, 0.95, 0.735],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12.0)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_linewidth(0.75)
        cell.set_edgecolor("#AAB7C4")
        if row_index == 0:
            cell.set_facecolor(PALETTE["blue"])
            cell.set_text_props(color="white", fontweight="bold", fontsize=12.5)
        elif row_index % 2 == 0:
            cell.set_facecolor("#F5F8FA")
    best_row = int(np.argmax(durations)) + 1
    for column_index in range(len(columns)):
        table[(best_row, column_index)].set_facecolor("#E9F6EF")
        table[(best_row, column_index)].set_text_props(
            fontweight="bold", color=PALETTE["green"]
        )

    ax.text(
        0.5, 0.925, "10轮独立DE搜索结果汇总",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=21, fontweight="bold",
    )
    ax.text(
        0.5, 0.075,
        f"10轮范围：{durations.min():.9f}–{durations.max():.9f} s    "
        f"极差：{np.ptp(durations) * 1000:.5f} ms    "
        f"均值±标准差：{durations.mean():.9f}±{durations.std(ddof=1):.9f} s",
        transform=ax.transAxes, ha="center", va="center", fontsize=13.0,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="#AAB7C4"),
    )
    save_figure(fig, FIGURE_DIR / "q2_de_runs_table.png")


def draw_strategy_spacetime(
    deployment: Deployment,
    simulation: SmokeSimulation,
    interval: tuple[float, float],
    duration: float,
) -> None:
    """最优策略水平关系与烟幕弹完整运动轨迹。"""
    parameters = simulation.parameters
    t_missile = np.linspace(0.0, simulation.missile_impact_time, 360)
    missile = np.vstack([simulation.missile_position(time) for time in t_missile])
    t_uav = np.linspace(0.0, deployment.burst_time, 220)
    uav = np.vstack([simulation.uav_position(time) for time in t_uav])
    t_bomb = np.linspace(deployment.release_time, deployment.burst_time, 100)
    bomb = np.vstack([simulation.bomb_position(time) for time in t_bomb])
    t_cloud = np.linspace(deployment.burst_time, interval[1], 180)
    cloud = np.vstack([simulation.cloud_center(time) for time in t_cloud])
    release_point = simulation.release_point
    burst_point = simulation.burst_point
    cloud_t2 = simulation.cloud_center(interval[1])

    fig = plt.figure(figsize=(18.0, 7.6))
    grid = fig.add_gridspec(
        1, 2, width_ratios=[1.20, 0.80],
        left=0.055, right=0.985, top=0.78, bottom=0.16, wspace=0.22,
    )
    xy = fig.add_subplot(grid[0, 0])
    xz = fig.add_subplot(grid[0, 1])

    xy.plot(
        missile[:, 0], missile[:, 1], color=PALETTE["orange"], linestyle="-",
        linewidth=2.5, label="导弹M1轨迹",
    )
    xy.plot(
        uav[:, 0], uav[:, 1], color=PALETTE["blue"], linestyle="-.",
        linewidth=2.6, label="无人机FY1轨迹",
    )
    xy.plot(
        bomb[:, 0], bomb[:, 1], color=PALETTE["purple"], linestyle=":",
        linewidth=3.3, label="_nolegend_",
    )
    xy.scatter(
        [release_point[0]], [release_point[1]], marker="o", s=88,
        color=PALETTE["purple"], edgecolor="white", linewidth=0.8,
        zorder=7, label="投放点 $P_r$",
    )
    xy.scatter(
        [burst_point[0]], [burst_point[1]], marker="D", s=91,
        color=PALETTE["cyan"], edgecolor="white", linewidth=0.8,
        zorder=7, label="起爆点 $P_e$",
    )
    xy.scatter(
        [0.0], [200.0], marker="o", s=105, color=PALETTE["green"],
        edgecolor="white", zorder=7, label="真目标",
    )
    xy.scatter(
        [0.0], [0.0], marker="*", s=155, color=PALETTE["red"],
        edgecolor="white", zorder=7, label="假目标",
    )
    xy.scatter(
        [parameters.missile_initial[0]], [parameters.missile_initial[1]],
        marker="^", s=92, color=PALETTE["orange"], zorder=7,
    )
    xy.annotate(
        "M1初始点\n(20000, 0)", (20000.0, 0.0), xytext=(-5, 50),
        textcoords="offset points", fontsize=11.5, ha="center",
        arrowprops=dict(arrowstyle="->", color=PALETTE["orange"], linewidth=1.5),
    )
    xy.annotate("真目标\n(0, 200)", (0.0, 200.0), xytext=(18, -22), textcoords="offset points", fontsize=11.5)
    xy.annotate("假目标\n(0, 0)", (0.0, 0.0), xytext=(20, 14), textcoords="offset points", fontsize=11.5)

    box_x0 = release_point[0] - 25.0
    box_y0 = min(release_point[1], burst_point[1]) - 5.0
    box_width = burst_point[0] - release_point[0] + 50.0
    box_height = abs(burst_point[1] - release_point[1]) + 10.0
    local_box = Rectangle(
        (box_x0, box_y0), box_width, box_height, fill=False,
        edgecolor=PALETTE["gray"], linewidth=1.8, linestyle="--", zorder=5,
    )
    xy.add_patch(local_box)

    zoom = inset_axes(xy, width="64%", height="48%", loc="center", borderpad=0.7)
    zoom.plot(uav[:, 0], uav[:, 1], color=PALETTE["blue"], linestyle="-", linewidth=2.4)
    zoom.plot(bomb[:, 0], bomb[:, 1], color=PALETTE["purple"], linestyle=":", linewidth=3.2)
    zoom.scatter([release_point[0]], [release_point[1]], marker="o", s=70, color=PALETTE["purple"], edgecolor="white", zorder=6)
    zoom.scatter([burst_point[0]], [burst_point[1]], marker="D", s=74, color=PALETTE["cyan"], edgecolor="white", zorder=6)
    zoom.annotate(
        f"投放点 $P_r$\n({release_point[0]:.3f}, {release_point[1]:.3f})",
        (release_point[0], release_point[1]), xytext=(-12, -37), textcoords="offset points",
        ha="right", fontsize=10.2,
    )
    zoom.annotate(
        f"起爆点 $P_e$\n({burst_point[0]:.3f}, {burst_point[1]:.3f})",
        (burst_point[0], burst_point[1]), xytext=(12, 13), textcoords="offset points",
        ha="left", fontsize=10.2,
    )
    zoom.set_xlim(release_point[0] - 18.0, burst_point[0] + 18.0)
    zoom.set_ylim(release_point[1] - 3.4, burst_point[1] + 3.4)
    zoom.set_title(
        f"虚线框区域放大：$\\theta={np.degrees(deployment.heading):.6f}^\\circ$",
        fontsize=11.2, pad=4,
    )
    zoom.tick_params(labelsize=9.5)
    zoom.grid(True, color="#CBD2D9", linewidth=0.7, alpha=0.65)
    for spine in zoom.spines.values():
        spine.set_linewidth(1.15)

    xy.add_artist(
        ConnectionPatch(
            xyA=(box_x0, box_y0 + box_height), coordsA=xy.transData,
            xyB=(0.0, 1.0), coordsB=zoom.transAxes,
            color=PALETTE["gray"], linestyle="--", linewidth=1.25, alpha=0.85,
        )
    )
    xy.add_artist(
        ConnectionPatch(
            xyA=(box_x0 + box_width, box_y0), coordsA=xy.transData,
            xyB=(1.0, 0.0), coordsB=zoom.transAxes,
            color=PALETTE["gray"], linestyle="--", linewidth=1.25, alpha=0.85,
        )
    )
    xy.set_xlim(-500, 20500)
    xy.set_ylim(-30, 230)
    xy.set_xlabel(r"$x$/m（正方向向右）", fontsize=13)
    xy.set_ylabel(r"$y$/m", fontsize=13)
    xy.tick_params(labelsize=11)
    xy.set_title("(a) 水平面全局航迹与投放—起爆位置", fontsize=15, pad=6)
    style_axis(xy)
    xy.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=4,
        fontsize=10.3, frameon=True,
    )

    xz.plot(
        bomb[:, 0], bomb[:, 2], color=PALETTE["purple"], linestyle=":",
        linewidth=3.4, label=r"烟幕弹平抛 $P_r\rightarrow P_e$",
    )
    xz.plot(
        cloud[:, 0], cloud[:, 2], color=PALETTE["cyan"], linestyle="--",
        linewidth=3.0, label="烟幕球心下沉 $C(t)$",
    )
    xz.scatter([release_point[0]], [release_point[2]], marker="o", s=84, color=PALETTE["purple"], edgecolor="white", zorder=6)
    xz.scatter([burst_point[0]], [burst_point[2]], marker="D", s=88, color=PALETTE["cyan"], edgecolor="white", zorder=6)
    xz.scatter([cloud_t2[0]], [cloud_t2[2]], marker="s", s=76, color=PALETTE["red"], edgecolor="white", zorder=7)
    bomb_drop = release_point[2] - burst_point[2]
    xz.annotate(
        f"投放点 $P_r$\n({release_point[0]:.2f}, {release_point[1]:.2f}, {release_point[2]:.2f})",
        (release_point[0], release_point[2]), xytext=(-7, -57), textcoords="offset points",
        ha="left", fontsize=9.8,
        arrowprops=dict(arrowstyle="->", color=PALETTE["purple"], linestyle=":"),
    )
    xz.annotate(
        f"起爆点 $P_e$\n({burst_point[0]:.2f}, {burst_point[1]:.2f}, {burst_point[2]:.2f})",
        (burst_point[0], burst_point[2]), xytext=(20, -62), textcoords="offset points",
        ha="left", fontsize=9.8,
        arrowprops=dict(arrowstyle="->", color=PALETTE["cyan"], linestyle="--"),
    )
    xz.annotate(
        f"遮蔽结束时球心 $C(t_2)$\n红色方形\n"
        f"({cloud_t2[0]:.2f}, {cloud_t2[1]:.2f}, {cloud_t2[2]:.2f})",
        (cloud_t2[0], cloud_t2[2]), xytext=(13, 18),
        textcoords="offset points", ha="left", fontsize=9.8,
        arrowprops=dict(arrowstyle="->", color=PALETTE["red"]),
    )

    bomb_box_y = burst_point[2] - 0.16
    left_guide_x = release_point[0]
    right_guide_x = burst_point[0]
    xz.vlines(
        [left_guide_x, right_guide_x], bomb_box_y, bomb_box_y + 0.34,
        colors=PALETTE["gray"], linestyles="--", linewidth=1.6, zorder=5,
    )
    bomb_zoom = xz.inset_axes([0.16, 0.23, 0.48, 0.30])
    bomb_x_relative = bomb[:, 0] - release_point[0]
    bomb_z_drop_mm = (bomb[:, 2] - release_point[2]) * 1000.0
    pe_x_relative = burst_point[0] - release_point[0]
    pe_z_drop_mm = (burst_point[2] - release_point[2]) * 1000.0
    bomb_zoom.plot(
        bomb_x_relative, bomb_z_drop_mm, color=PALETTE["purple"],
        linestyle=":", linewidth=3.0,
    )
    bomb_zoom.scatter([0.0], [0.0], marker="o", s=48, color=PALETTE["purple"], zorder=5)
    bomb_zoom.scatter([pe_x_relative], [pe_z_drop_mm], marker="D", s=52, color=PALETTE["cyan"], zorder=5)
    bomb_zoom.annotate("$P_r$", (0.0, 0.0), xytext=(8, -18), textcoords="offset points", fontsize=8.8)
    bomb_zoom.annotate("$P_e$", (pe_x_relative, pe_z_drop_mm), xytext=(-8, 9), textcoords="offset points", ha="right", fontsize=8.8)
    bomb_zoom.set_xlim(-0.06, pe_x_relative + 0.06)
    bomb_zoom.set_ylim(pe_z_drop_mm - 0.035, 0.035)
    bomb_zoom.set_xlabel(r"$x-x_r$/m", fontsize=8.4, labelpad=0)
    bomb_zoom.set_ylabel(r"$z-z_r$/mm", fontsize=8.4, labelpad=0)
    bomb_zoom.set_title(
        f"主图顶部虚线框内平抛轨迹放大（$\\Delta z={bomb_drop:.7f}$ m）",
        fontsize=8.7, pad=3,
    )
    bomb_zoom.tick_params(labelsize=7.8)
    bomb_zoom.grid(True, color="#CBD2D9", linewidth=0.65, alpha=0.65)
    xz.add_artist(
        ConnectionPatch(
            xyA=(left_guide_x, bomb_box_y), coordsA=xz.transData,
            xyB=(0.0, 1.0), coordsB=bomb_zoom.transAxes,
            color=PALETTE["gray"], linestyle="--", linewidth=1.15, alpha=0.85,
        )
    )
    xz.add_artist(
        ConnectionPatch(
            xyA=(right_guide_x, bomb_box_y), coordsA=xz.transData,
            xyB=(1.0, 1.0), coordsB=bomb_zoom.transAxes,
            color=PALETTE["gray"], linestyle="--", linewidth=1.15, alpha=0.85,
        )
    )
    xz.set_xlim(release_point[0] - 0.12, burst_point[0] + 0.42)
    xz.set_ylim(cloud_t2[2] - 1.0, release_point[2] + 0.55)
    xz.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    xz.set_xlabel(r"$x$/m", fontsize=13)
    xz.set_ylabel(r"$z$/m", fontsize=13)
    xz.tick_params(labelsize=10.5)
    xz.set_title("(b) 烟幕弹平抛与烟幕球心下沉", fontsize=15, pad=6)
    style_axis(xz)
    xz.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=2,
        fontsize=10.1, frameon=True,
    )

    fig.suptitle(
        "最优投放策略空间图\n"
        f"$\\theta={np.degrees(deployment.heading):.8f}^\\circ$,  "
        f"$v={deployment.speed:.6f}$ m/s,  $t_r={deployment.release_time:.6f}$ s,  "
        f"$\\tau={deployment.fuse_delay:.6f}$ s",
        fontsize=19, fontweight="bold", y=0.975,
    )
    save_figure(fig, FIGURE_DIR / "q2_strategy_spacetime.png")


def draw_event_table(
    deployment: Deployment,
    interval: tuple[float, float],
    duration: float,
) -> None:
    """独立、紧凑的关键事件时刻表。"""
    event_rows: list[dict[str, object]] = [
        {"event": "FY1开始飞行", "symbol": "t_0", "time_s": "0.000000", "note": "由初始点出发"},
        {"event": "投放烟幕弹", "symbol": "t_r", "time_s": f"{deployment.release_time:.6f}", "note": "烟幕弹开始平抛"},
        {"event": "烟幕弹起爆", "symbol": "t_e", "time_s": f"{deployment.burst_time:.6f}", "note": f"τ={deployment.fuse_delay:.6f} s"},
        {"event": "有效遮蔽开始", "symbol": "t_1", "time_s": f"{interval[0]:.6f}", "note": "完整圆柱表面判定"},
        {"event": "有效遮蔽结束", "symbol": "t_2", "time_s": f"{interval[1]:.6f}", "note": f"Teff={duration:.6f} s"},
        {"event": "烟幕有效期结束", "symbol": "t_e+20", "time_s": f"{deployment.burst_time + 20.0:.6f}", "note": "起爆后20 s"},
    ]
    write_csv(Q2_DIR / "q2_event_times.csv", event_rows)

    fig, ax = plt.subplots(figsize=(11.8, 4.25))
    ax.axis("off")
    event_cells = [
        [str(index), row["event"], f"${row['symbol']}$", row["time_s"], row["note"]]
        for index, row in enumerate(event_rows, start=1)
    ]
    table = ax.table(
        cellText=event_cells,
        colLabels=["序号", "关键事件", "符号", "发生时刻/s", "说明"],
        cellLoc="center", colLoc="center", bbox=[0.035, 0.055, 0.93, 0.78],
        colWidths=[0.07, 0.24, 0.12, 0.19, 0.38],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12.2)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor("#AAB7C4")
        cell.set_linewidth(0.75)
        if row_index == 0:
            cell.set_facecolor(PALETTE["blue"])
            cell.set_text_props(color="white", fontweight="bold", fontsize=12.6)
        elif row_index % 2 == 0:
            cell.set_facecolor("#F5F8FA")
    ax.text(
        0.5, 0.89, "(c) 关键事件时刻表", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=18, fontweight="bold",
    )
    save_figure(fig, FIGURE_DIR / "q2_event_times_table.png")


def draw_sensitivity_tornado(
    evaluator: SmokeEvaluator,
    deployment: Deployment,
    baseline_duration: float,
) -> list[dict[str, object]]:
    """单因素扰动龙卷风图；对称对数标尺放大近零变化。"""
    tests = [
        ("航向角", "−1°", replace(deployment, heading=deployment.heading - np.deg2rad(1.0))),
        ("航向角", "+1°", replace(deployment, heading=deployment.heading + np.deg2rad(1.0))),
        ("飞行速度", "−1 m/s", replace(deployment, speed=deployment.speed - 1.0)),
        ("飞行速度", "上边界140", replace(deployment, speed=140.0)),
        ("投放时刻", "−0.1 s", replace(deployment, release_time=deployment.release_time - 0.1)),
        ("投放时刻", "+0.1 s", replace(deployment, release_time=deployment.release_time + 0.1)),
        ("引爆延时", "下边界0", replace(deployment, fuse_delay=0.0)),
        ("引爆延时", "+0.1 s", replace(deployment, fuse_delay=deployment.fuse_delay + 0.1)),
    ]
    records: list[dict[str, object]] = []
    for factor, perturbation, candidate in tests:
        result = evaluator.evaluate(candidate, mode="surface")
        if not result.feasible:
            raise RuntimeError(f"敏感性扰动 {factor} {perturbation} 不可行：{result.reason}")
        records.append(
            {
                "factor": factor,
                "perturbation": perturbation,
                "theta_deg": f"{np.degrees(candidate.heading):.10f}",
                "speed_mps": f"{candidate.speed:.10f}",
                "release_time_s": f"{candidate.release_time:.10f}",
                "fuse_delay_s": f"{candidate.fuse_delay:.10f}",
                "duration_s": f"{result.duration:.10f}",
                "delta_duration_s": f"{result.duration - baseline_duration:.10f}",
            }
        )
    write_csv(Q2_DIR / "q2_sensitivity.csv", records)

    by_factor: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_factor.setdefault(str(record["factor"]), []).append(record)
    order = sorted(
        by_factor,
        key=lambda factor: max(
            abs(float(item["delta_duration_s"])) for item in by_factor[factor]
        ),
        reverse=True,
    )
    impacts = {
        factor: max(abs(float(item["delta_duration_s"])) for item in by_factor[factor])
        for factor in order
    }

    fig, ax = plt.subplots(figsize=(14.0, 6.55), constrained_layout=True)
    y_positions = np.arange(len(order))
    bar_height = 0.31
    for index, factor in enumerate(order):
        lower, upper = by_factor[factor]
        lower_delta = float(lower["delta_duration_s"])
        upper_delta = float(upper["delta_duration_s"])
        ax.barh(
            index + bar_height / 2, lower_delta, height=bar_height,
            color=PALETTE["blue"], edgecolor="#174A70", hatch="//", alpha=0.88,
            label="负向/下边界扰动" if index == 0 else None,
        )
        ax.barh(
            index - bar_height / 2, upper_delta, height=bar_height,
            color=PALETTE["orange"], edgecolor="#8C4A0E", hatch="\\\\", alpha=0.88,
            label="正向/上边界扰动" if index == 0 else None,
        )
        ax.annotate(
            f"{lower['perturbation']}：{float(lower['duration_s']):.6f} s",
            xy=(lower_delta, index + bar_height / 2), xytext=(-7, 0),
            textcoords="offset points", ha="right", va="center",
            fontsize=11.2, color=PALETTE["blue"],
        )
        if factor != "飞行速度":
            orange_label_x = -0.23
            ax.annotate(
                f"{upper['perturbation']}：{float(upper['duration_s']):.6f} s",
                xy=(orange_label_x, index - bar_height / 2), xytext=(0, 24),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=11.2, color=PALETTE["orange"],
            )
        else:
            ax.annotate(
                f"{upper['perturbation']}：{float(upper['duration_s']):.6f} s",
                xy=(upper_delta, index - bar_height / 2), xytext=(-7, 0),
                textcoords="offset points", ha="right", va="center",
                fontsize=11.2, color=PALETTE["orange"],
            )

    ax.set_xscale("symlog", linthresh=0.002, linscale=1.35, base=10)
    ax.set_xlim(-0.72, 0.0016)
    ax.set_xticks([-0.5, -0.1, -0.01, -0.002, -0.001, 0.0])
    ax.set_xticklabels(["−0.5", "−0.1", "−0.01", "−0.002", "−0.001", "0"])
    ax.axvline(0.0, color=PALETTE["dark"], linestyle="-.", linewidth=1.5)
    ax.set_yticks(y_positions, order)
    ax.invert_yaxis()
    ax.set_ylim(len(order) - 0.45, -0.68)
    ax.tick_params(axis="both", labelsize=12.5)
    ax.set_xlabel(
        r"遮蔽时长变化 $\Delta T$/s（对称对数标尺；近0区域线性放大）",
        fontsize=13.5,
    )
    ax.set_title(
        "单因素敏感性龙卷风图\n"
        f"基准完整表面遮蔽时长 $T_0={baseline_duration:.9f}$ s；其余三个变量保持不变",
        fontsize=18, fontweight="bold", pad=10,
    )
    style_axis(ax, "x")
    ax.legend(
        loc="lower left", bbox_to_anchor=(0.775, 0.025), ncol=1,
        frameon=True, fontsize=11.2,
    )
    ranking = "影响排序（按最大 |ΔT|）\n" + "\n".join(
        f"{rank}. {factor}  {impacts[factor]:.6f} s"
        for rank, factor in enumerate(order, start=1)
    )
    ax.text(
        0.79, 0.96, ranking, transform=ax.transAxes, ha="left", va="top",
        fontsize=11.5, linespacing=1.35,
        multialignment="left",
        bbox=dict(boxstyle="round,pad=0.42", facecolor="white", edgecolor="#AAB7C4", alpha=0.97),
    )
    ax.text(
        0.012, 0.025,
        "注：$v^*$接近140 m/s、$\\tau^*$接近0 s，故采用可行边界扰动；\n"
        "横轴采用对称对数标尺，仅放大近0变化的显示面积，不改变任何数值。",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=11.2,
        color="black", linespacing=1.32,
    )
    save_figure(fig, FIGURE_DIR / "q2_sensitivity_tornado.png")
    return records


def remove_obsolete_outputs() -> None:
    for name in (
        "q2_lhs_regions.png",
        "q2_de_convergence.png",
        "q2_optimal_trajectory.png",
        "q2_occlusion_states.png",
        "q2_sensitivity.png",
        "q2_snake_flowchart.png",
    ):
        path = FIGURE_DIR / name
        if path.exists():
            path.unlink()
    diagnostic = Q2_DIR / "q2_rim_diagnostic.csv"
    if diagnostic.exists():
        diagnostic.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="生成问题二最终图表")
    parser.add_argument("--all", action="store_true", help="生成全部正式图表")
    args = parser.parse_args()
    if not args.all:
        parser.error("请使用 --all 生成全部正式图表。")

    configure_matplotlib()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    evaluator = build_full_evaluator()
    deployment, simulation, interval, duration = selected_solution(evaluator)

    draw_rim_principle()
    draw_de_table()
    draw_strategy_spacetime(deployment, simulation, interval, duration)
    draw_event_table(deployment, interval, duration)
    sensitivity = draw_sensitivity_tornado(evaluator, deployment, duration)
    remove_obsolete_outputs()

    ranking: dict[str, float] = {}
    for item in sensitivity:
        factor = str(item["factor"])
        ranking[factor] = max(
            ranking.get(factor, 0.0), abs(float(item["delta_duration_s"]))
        )
    print("问题二最终图表已生成：")
    for path in sorted(FIGURE_DIR.glob("q2_*.png")):
        print(f"  {path.name}")
    print("敏感性影响排序：")
    for factor, impact in sorted(
        ranking.items(), key=lambda pair: pair[1], reverse=True
    ):
        print(f"  {factor}: max|ΔT|={impact:.10f} s")


if __name__ == "__main__":
    main()
