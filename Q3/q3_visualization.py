from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Circle, Ellipse, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Q1.q1_visualization import configure_matplotlib, trim_white_margins  # noqa: E402
from Q3.q3_main import UniformJointEvaluator  # noqa: E402
from core.multi_smoke_evaluator import ThreeDeployment  # noqa: E402

Q3_DIR = Path(__file__).resolve().parent
FIGURE_DIR = Q3_DIR / "figures"
BEST_CSV = Q3_DIR / "q3_best_solution.csv"
DE_CSV = Q3_DIR / "q3_de_history.csv"

COLORS = ("#C84B31", "#2878B5", "#2E8B57")
LINESTYLES = ("-", "--", "-.")
MARKERS = ("o", "s", "^")
HATCHES = ("////", "\\\\", "xx")
JOINT_COLOR = "#E67E22"
JOINT_STYLE = (0, (6, 2, 1.5, 2))
MISSILE_COLOR = "#202A35"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise RuntimeError(f"{path_label(BEST_CSV)} 缺少字段 {key}")
    return float(value)


def path_label(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def selected_row() -> dict[str, str]:
    rows = read_csv(BEST_CSV)
    selected = [row for row in rows if row.get("selected", "").lower() == "true"]
    if len(selected) != 1:
        raise RuntimeError("q3_best_solution.csv 中必须恰有一行 selected=True")
    return selected[0]


def parse_intervals(text: str) -> tuple[tuple[float, float], ...]:
    intervals: list[tuple[float, float]] = []
    for item in text.split(";"):
        item = item.strip().strip("[]")
        if not item:
            continue
        left, right = item.split(",")
        intervals.append((float(left), float(right)))
    return tuple(intervals)


def deployment_from_row(row: dict[str, str]) -> ThreeDeployment:
    return ThreeDeployment(
        heading=number(row, "theta_rad"),
        speed=number(row, "uav_speed_mps"),
        release_times=tuple(number(row, f"release_time_{i}_s") for i in range(1, 4)),
        fuse_delays=tuple(number(row, f"fuse_delay_{i}_s") for i in range(1, 4)),
    )


def style_axis(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color="#CBD2D9", linewidth=0.75, alpha=0.70)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, filename: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    trim_white_margins(path, padding=28)
    return path


def interval_duration(intervals: tuple[tuple[float, float], ...]) -> float:
    return float(sum(right - left for left, right in intervals))


def calculate_individual_intervals(
    deployment: ThreeDeployment,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """用与正式模型相同的完整表面判据，逐枚复核独立遮蔽区间。"""

    evaluator = UniformJointEvaluator(360, 11, 11, 0.01, 1.0e-7)
    intervals = []
    for release, delay in zip(deployment.release_times, deployment.fuse_delays):
        single = ThreeDeployment(
            heading=deployment.heading,
            speed=deployment.speed,
            release_times=(release,),
            fuse_delays=(delay,),
        )
        result = evaluator.evaluate(single)
        if not result.feasible:
            raise RuntimeError(f"单枚烟幕复核失败：{result.reason}")
        intervals.append(result.intervals)
    return tuple(intervals)


def draw_timeline(
    row: dict[str, str],
    deployment: ThreeDeployment,
    individual_intervals: tuple[tuple[tuple[float, float], ...], ...],
) -> Path:
    joint_intervals = parse_intervals(row["joint_intervals_s"])
    joint_duration = number(row, "joint_duration_s")
    fig, ax = plt.subplots(figsize=(16.8, 6.6))
    y_positions = (3.0, 2.0, 1.0)
    contribution_spans: list[tuple[int, float, float, float]] = []

    for i, (release, burst, intervals, y) in enumerate(
        zip(
            deployment.release_times,
            deployment.burst_times,
            individual_intervals,
            y_positions,
        )
    ):
        color, linestyle, marker = COLORS[i], LINESTYLES[i], MARKERS[i]
        ax.plot(
            [release, burst], [y, y], color=color, linestyle=linestyle,
            linewidth=2.7, alpha=0.95,
        )
        ax.scatter(
            [release], [y], s=95, facecolor="white", edgecolor=color,
            linewidth=2.2, marker=marker, zorder=5,
        )
        ax.scatter(
            [burst], [y], s=185, facecolor=color, edgecolor="white",
            linewidth=1.0, marker="*", zorder=6,
        )
        release_offset = (14, 15) if i == 0 else (0, 15)
        release_align = "left" if i == 0 else "center"
        ax.annotate(
            f"投放 {release:.3f}", (release, y), xytext=release_offset,
            textcoords="offset points", ha=release_align, color=color, fontsize=12.2,
        )
        burst_offset = -30 if i in (1, 2) else -22
        ax.annotate(
            f"起爆 {burst:.3f}", (burst, y), xytext=(0, burst_offset),
            textcoords="offset points", ha="center", color=color, fontsize=12.2,
        )
        for left, right in intervals:
            ax.barh(
                y, right - left, left=left, height=0.34,
                color=color, alpha=0.18, edgecolor=color, linewidth=1.6,
                hatch=HATCHES[i], zorder=2,
            )
            contribution_spans.append((i, y, left, right))
        duration = interval_duration(intervals)
        if intervals:
            center = 0.5 * (intervals[0][0] + intervals[-1][1])
            ax.text(
                center, y + 0.24, f"独立有效 {duration:.4f} s",
                ha="center", va="bottom", fontsize=12.3, color=color,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.5),
            )

    for left, right in joint_intervals:
        ax.barh(
            0.0, right - left, left=left, height=0.42,
            color=JOINT_COLOR, alpha=0.38, edgecolor=JOINT_COLOR,
            linewidth=2.1, hatch="..", zorder=3,
        )
    # 将每枚烟幕的独立有效区间投影到联合判定层，明确三段接力来源。
    for i, source_y, left, right in contribution_spans:
        color = COLORS[i]
        ax.plot(
            [left, left], [source_y - 0.19, 0.18], color=color,
            linestyle=(0, (3, 3)), linewidth=1.45, alpha=0.52, zorder=2,
        )
        ax.plot(
            [right, right], [source_y - 0.19, 0.18], color=color,
            linestyle=(0, (3, 3)), linewidth=1.45, alpha=0.52, zorder=2,
        )
        ax.plot(
            [left, right], [0.12, 0.12], color=color,
            linestyle=LINESTYLES[i], linewidth=3.1, alpha=0.95, zorder=5,
        )
    first, last = joint_intervals[0][0], joint_intervals[-1][1]
    ax.annotate(
        "", xy=(last, -0.42), xytext=(first, -0.42),
        arrowprops=dict(arrowstyle="<->", color=JOINT_COLOR, linewidth=2.1),
    )
    ax.text(
        0.5 * (first + last), -0.55,
        f"联合有效总时长 = {joint_duration:.6f} s",
        ha="center", va="top", color="#A34F00", fontsize=14.0, fontweight="bold",
    )
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(["烟幕弹 1", "烟幕弹 2", "烟幕弹 3", "联合判定"], fontsize=13.0)
    ax.set_ylim(-0.82, 3.62)
    ax.set_xlim(0.0, 14.0)
    ax.text(
        0.995, 0.98,
        f"航向 {math.degrees(deployment.heading):.4f}°　速度 {deployment.speed:.4f} m/s\n"
        "空心点：投放　★：起爆　阴影：该层有效遮蔽区间",
        transform=ax.transAxes, ha="right", va="top", multialignment="left", fontsize=12.2,
        bbox=dict(boxstyle="round,pad=0.42", facecolor="#F7F9FA", edgecolor="#AAB2BA"),
    )
    style_axis(ax, "x")
    ax.set_xlabel("时间 $t$ / s", fontsize=16.0)
    ax.tick_params(axis="x", labelsize=12.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1.0))
    return save_figure(fig, "q3_01_timeline_relay.png")


def draw_joint_criterion() -> Path:
    """仅解释计算逻辑，不混入某一组优化结果。"""

    fig, axes = plt.subplots(1, 3, figsize=(19.2, 7.35))
    ax_a, ax_b, ax_c = axes

    # (a) 球心到“有限”视线段的距离。
    ax_a.set_aspect("equal")
    ax_a.set_xlim(-0.7, 9.2)
    ax_a.set_ylim(-2.5, 3.4)
    ax_a.axis("off")
    missile = np.array([0.0, 0.0])
    target_point = np.array([8.0, 0.0])
    cloud = np.array([4.5, 2.1])
    foot = np.array([4.5, 0.0])
    ax_a.plot(
        [missile[0], target_point[0]], [0.0, 0.0],
        color=MISSILE_COLOR, linewidth=2.5,
    )
    ax_a.plot(
        [target_point[0], 9.0], [0.0, 0.0],
        color="#AAB2BA", linestyle=(0, (2, 2)), linewidth=1.3,
    )
    ax_a.scatter(*missile, s=120, marker=">", color=MISSILE_COLOR, zorder=6)
    ax_a.scatter(*target_point, s=90, marker="o", color="#235F36", zorder=6)
    ax_a.add_patch(
        Circle(cloud, 0.92, facecolor=COLORS[0], edgecolor=COLORS[0], alpha=0.18, linewidth=2.0)
    )
    ax_a.scatter(*cloud, s=70, color=COLORS[0], marker=MARKERS[0], zorder=6)
    ax_a.plot(
        [cloud[0], foot[0]], [cloud[1], foot[1]],
        color=COLORS[0], linestyle=LINESTYLES[0], linewidth=2.1,
    )
    ax_a.plot(
        [foot[0], foot[0] + 0.22, foot[0] + 0.22],
        [foot[1] + 0.22, foot[1] + 0.22, foot[1]],
        color="#59636C", linewidth=1.1,
    )
    ax_a.text(-0.05, -0.48, "$M(t)$\n导弹", ha="center", va="top", fontsize=12.5)
    ax_a.text(8.0, -0.48, "$Q$\n目标表面点", ha="center", va="top", fontsize=12.5)
    ax_a.text(4.5, 3.12, "$C_j(t)$：第 $j$ 枚烟幕球心", ha="center", color=COLORS[0], fontsize=12.5)
    ax_a.text(4.72, 1.03, "$d_j(t,Q)$", color=COLORS[0], fontsize=13.0)
    ax_a.text(4.5, -0.46, "$H_j$", ha="center", va="top", fontsize=12.0)
    ax_a.annotate(
        "只在 $M(t)$ 到 $Q$ 的\n有限线段上寻最近点",
        xy=foot, xytext=(0.45, 2.15),
        arrowprops=dict(arrowstyle="->", color="#6C757D", linewidth=1.2),
        ha="left", fontsize=11.3, color="#4F5962",
    )
    ax_a.text(
        4.25, -1.65,
        r"$d_j(t,Q)=\min_{0\leq\lambda\leq1}$"
        "\n"
        r"$\left\|M(t)+\lambda(Q-M(t))-C_j(t)\right\|$",
        ha="center", va="center", fontsize=13.0,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#F6F8FA", edgecolor="#9AA3AB"),
    )

    # (b) 对烟幕取 min，再对完整表面取 max。
    ax_b.set_xlim(0.0, 10.0)
    ax_b.set_ylim(0.0, 10.0)
    ax_b.axis("off")
    cloud_positions = ((1.5, 8.55), (5.0, 8.75), (8.5, 8.55))
    for i, position in enumerate(cloud_positions):
        ax_b.add_patch(
            Circle(position, 0.62, facecolor=COLORS[i], edgecolor=COLORS[i], alpha=0.18,
                   linewidth=1.8, linestyle=LINESTYLES[i])
        )
        ax_b.scatter(*position, s=55, color=COLORS[i], marker=MARKERS[i], zorder=5)
        ax_b.text(position[0], 9.43, f"烟幕 {i + 1}  $C_{i + 1}(t)$", ha="center", fontsize=11.7, color=COLORS[i])
        ax_b.annotate(
            f"$d_{i + 1}(t,Q)$", xy=(5.0, 7.05), xytext=position,
            arrowprops=dict(arrowstyle="->", color=COLORS[i], linewidth=1.5, linestyle=LINESTYLES[i]),
            ha="center", va="center", fontsize=11.0, color=COLORS[i],
        )
    ax_b.text(
        5.0, 6.7, r"每个点先选最有效烟幕：$g_t(Q)=\min_{j\in\mathcal{J}(t)}d_j(t,Q)$",
        ha="center", va="center", fontsize=12.0,
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#FFF8E8", edgecolor="#D6A64A"),
    )
    ax_b.annotate("", xy=(5.0, 5.55), xytext=(5.0, 6.25), arrowprops=dict(arrowstyle="->", linewidth=1.6))
    body = Rectangle((3.55, 2.0), 2.9, 3.1, facecolor="#DDEEE1", edgecolor="#2D6841", linewidth=1.6)
    ax_b.add_patch(body)
    ax_b.add_patch(Ellipse((5.0, 5.1), 2.9, 0.72, facecolor="#EAF5EC", edgecolor="#2D6841", linewidth=1.6))
    ax_b.add_patch(Ellipse((5.0, 2.0), 2.9, 0.72, facecolor="#D0E6D5", edgecolor="#2D6841", linewidth=1.6))
    sample_positions = (
        (3.58, 2.4), (3.58, 3.3), (3.58, 4.2), (4.25, 5.22),
        (5.0, 5.42), (5.78, 5.23), (6.42, 4.4), (6.42, 3.45),
        (6.42, 2.55), (4.25, 1.82), (5.0, 1.65), (5.75, 1.82),
    )
    sample_owners = (0, 0, 1, 0, 1, 2, 2, 2, 1, 0, 1, 2)
    for position, owner in zip(sample_positions, sample_owners):
        ax_b.scatter(*position, s=68, color=COLORS[owner], marker=MARKERS[owner], edgecolor="white", linewidth=0.7, zorder=6)
    worst = (6.42, 4.4)
    ax_b.scatter(*worst, s=165, marker="X", color="black", edgecolor="white", linewidth=0.9, zorder=7)
    ax_b.annotate(
        "$Q^*$：全表面最难遮蔽的点", xy=worst, xytext=(7.0, 5.25),
        arrowprops=dict(arrowstyle="->", color="black", linewidth=1.1), fontsize=11.0,
    )
    ax_b.text(
        5.0, 0.82,
        r"全表面再取最不利点：$G(t)=\max_{Q\in\partial K}g_t(Q)$"
        "\n"
        r"$G(t)\leq R=10\ \mathrm{m}\ \Longleftrightarrow$ 该时刻完整遮蔽",
        ha="center", va="center", fontsize=12.3, linespacing=1.55,
        bbox=dict(boxstyle="round,pad=0.43", facecolor="#F6F8FA", edgecolor="#87919A"),
    )

    # (c) 自适应面片：通过、失败、临界细分。
    ax_c.set_xlim(0.0, 12.0)
    ax_c.set_ylim(0.0, 10.0)
    ax_c.axis("off")
    x0, y0, width, height = 0.55, 3.3, 6.55, 5.2
    cols, rows = 6, 3
    dx, dy = width / cols, height / rows
    ax_c.add_patch(Rectangle((x0, y0), width, height, facecolor="#F7F9FA", edgecolor="#59636C", linewidth=1.5))
    for k in range(1, cols):
        ax_c.plot([x0 + k * dx] * 2, [y0, y0 + height], color="#9DA6AE", linewidth=0.8)
    for k in range(1, rows):
        ax_c.plot([x0, x0 + width], [y0 + k * dy] * 2, color="#9DA6AE", linewidth=0.8)
    pass_patch = (x0 + dx, y0)
    refine_patch = (x0 + 3 * dx, y0 + dy)
    fail_patch = (x0 + 5 * dx, y0 + 2 * dy)
    ax_c.add_patch(Rectangle(pass_patch, dx, dy, facecolor="#8BC49A", edgecolor="#2E8B57", alpha=0.72, linewidth=1.8, hatch="////"))
    ax_c.add_patch(Rectangle(refine_patch, dx, dy, facecolor="#F4D27A", edgecolor="#B47B00", alpha=0.78, linewidth=1.8))
    ax_c.add_patch(Rectangle(fail_patch, dx, dy, facecolor="#E69B91", edgecolor="#B53C2E", alpha=0.75, linewidth=1.8, hatch="xx"))
    rx, ry = refine_patch
    ax_c.plot([rx + dx / 2] * 2, [ry, ry + dy], color="#8D6500", linewidth=1.2)
    ax_c.plot([rx, rx + dx], [ry + dy / 2] * 2, color="#8D6500", linewidth=1.2)
    centers = (
        (pass_patch[0] + dx / 2, pass_patch[1] + dy / 2),
        (refine_patch[0] + dx / 2, refine_patch[1] + dy / 2),
        (fail_patch[0] + dx / 2, fail_patch[1] + dy / 2),
    )
    for center, color in zip(centers, ("#236B3A", "#8D6500", "#A63A2B")):
        ax_c.scatter(*center, s=42, color=color, edgecolor="white", linewidth=0.5, zorder=5)
    ax_c.text(x0 + width / 2, 8.8, "圆柱侧面展开后的初始面片", ha="center", fontsize=12.0)
    rules = (
        (8.0, 4.15, "确认通过", r"$g_t(Q_c)+\rho\leq R$", "#2E8B57", centers[0]),
        (8.0, 6.15, "继续细分", r"$g_t(Q_c)\leq R<g_t(Q_c)+\rho$", "#B47B00", centers[1]),
        (8.0, 8.15, "立即失败", r"$g_t(Q_c)>R$", "#B53C2E", centers[2]),
    )
    for tx, ty, title, formula, color, center in rules:
        ax_c.annotate(
            f"{title}\n{formula}", xy=center, xytext=(tx, ty),
            arrowprops=dict(arrowstyle="->", color=color, linewidth=1.5),
            ha="left", va="center", fontsize=11.3, color=color,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor=color, alpha=0.95),
        )
    for cx, label in ((2.05, "上底面"), (5.55, "下底面")):
        cap_center = (cx, 1.55)
        ax_c.add_patch(Circle(cap_center, 0.9, facecolor="#EEF3F6", edgecolor="#59636C", linewidth=1.2))
        for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
            ax_c.plot(
                [cap_center[0], cap_center[0] + 0.9 * np.cos(angle)],
                [cap_center[1], cap_center[1] + 0.9 * np.sin(angle)],
                color="#9DA6AE", linewidth=0.65,
            )
        ax_c.add_patch(Circle(cap_center, 0.45, facecolor="none", edgecolor="#9DA6AE", linewidth=0.65))
        ax_c.text(cx, 0.38, label, ha="center", fontsize=11.0)
    ax_c.text(3.8, 2.72, "上下底面按半径—角度同样分片", ha="center", fontsize=11.3, color="#4F5962")
    ax_c.text(
        9.5, 1.55, "$Q_c$：面片代表点\n$\\rho$：面片内点到 $Q_c$ 的距离上界\n只细分临界面片，避免全表面统一加密",
        ha="center", va="center", fontsize=10.9, linespacing=1.55,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F6F8FA", edgecolor="#9AA3AB"),
    )
    fig.text(
        0.5, 0.012,
        "有限视线段距离  →  每个表面点选择最有效烟幕  →  搜索最不利表面点  →  自适应认证完整表面",
        ha="center", fontsize=13.0, color="#3F4850",
    )
    fig.subplots_adjust(left=0.012, right=0.995, bottom=0.075, top=0.90, wspace=0.018)
    # 三个子图标题使用相同的图级纵坐标；(c) 标题横向对准方形网格中心。
    title_y = 0.94
    box_a, box_b, box_c = (axis.get_position() for axis in (ax_a, ax_b, ax_c))
    title_positions = (
        (0.5 * (box_a.x0 + box_a.x1), "(a) 单个目标点：有限视线段距离"),
        (0.5 * (box_b.x0 + box_b.x1), "(b) 三烟幕联合：先取最小，再取最大"),
        (box_c.x0 + ((x0 + width / 2.0) / 12.0) * box_c.width, "(c) 完整表面：自适应面片认证"),
    )
    for title_x, title in title_positions:
        fig.text(title_x, title_y, title, ha="center", va="center", fontsize=15.0)
    return save_figure(fig, "q3_02_joint_criterion.png")


def draw_convergence(row: dict[str, str]) -> Path:
    history = [item for item in read_csv(DE_CSV) if item.get("profile") == "standard"]
    grouped: dict[int, list[tuple[int, float]]] = {}
    for item in history:
        seed = int(item["seed"])
        grouped.setdefault(seed, []).append(
            (int(item["generation"]), float(item["best_joint_duration_s"]))
        )

    fine = number(row, "dense_uniform_duration_s")
    fig, ax_de = plt.subplots(figsize=(14.2, 8.0))
    final_values: list[float] = []
    max_generation = 0

    for order, seed in enumerate(sorted(grouped)):
        data = sorted(grouped[seed])
        generations = np.array([item[0] for item in data])
        values = np.array([item[1] for item in data])
        final_values.append(float(values[-1]))
        max_generation = max(max_generation, int(generations[-1]))
        ax_de.plot(
            generations, values, color=COLORS[order], linestyle=LINESTYLES[order],
            linewidth=2.7, marker=MARKERS[order], markevery=max(1, len(data) // 10),
            markersize=5.8, label=f"随机种子 {seed}：{values[-1]:.6f} s",
        )
        ax_de.scatter(
            generations[-1], values[-1], s=88, color=COLORS[order],
            marker=MARKERS[order], edgecolor="white", linewidth=0.7, zorder=5,
        )
    ax_de.axhline(
        fine, color=JOINT_COLOR, linestyle=JOINT_STYLE, linewidth=2.1,
        label=f"最终高精度复核 {fine:.6f} s",
    )
    ax_de.set_xlabel("DE 迭代代数", fontsize=16.0)
    ax_de.set_ylabel("当前最优联合有效遮蔽时长 / s", fontsize=16.0)
    style_axis(ax_de)
    ax_de.legend(loc="lower right", frameon=True, fontsize=12.0)
    ax_de.tick_params(axis="both", labelsize=12.5)
    spread_ms = 1000.0 * (max(final_values) - min(final_values))
    ax_de.text(
        0.5, 7.575,
        "三个随机种子独立初始化\n"
        f"末值极差仅 {spread_ms:.3f} ms\n"
        "说明最优时长对随机初值较稳定",
        transform=ax_de.get_yaxis_transform(), ha="center", va="center", fontsize=12.2,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#B6BDC4", alpha=0.90),
    )
    ax_de.set_xlim(-2, max_generation + 4)

    # 同一坐标系的局部放大，仅用于看清三条末段轨迹的聚集程度。
    ax_zoom = ax_de.inset_axes([0.57, 0.30, 0.39, 0.34], zorder=10)
    ax_zoom.set_facecolor("white")
    zoom_left = max(0, max_generation - 38)
    for order, seed in enumerate(sorted(grouped)):
        data = sorted(grouped[seed])
        generations = np.array([item[0] for item in data])
        values = np.array([item[1] for item in data])
        ax_zoom.plot(
            generations, values, color=COLORS[order], linestyle=LINESTYLES[order],
            linewidth=1.9, marker=MARKERS[order], markevery=max(1, len(data) // 8),
            markersize=4.2,
        )
    ax_zoom.axhline(fine, color=JOINT_COLOR, linestyle=JOINT_STYLE, linewidth=1.25)
    ax_zoom.set_xlim(zoom_left, max_generation + 2)
    zoom_values = np.array([*final_values, fine])
    ax_zoom.set_ylim(float(np.min(zoom_values)) - 5.0e-4, float(np.max(zoom_values)) + 5.0e-4)
    ax_zoom.set_title("末段局部放大", fontsize=12.2)
    ax_zoom.set_xlabel("迭代代数", fontsize=11.0)
    ax_zoom.set_ylabel("时长 / s", fontsize=11.0)
    ax_zoom.tick_params(labelsize=10.0)
    ax_zoom.grid(True, color="#D4DADF", linewidth=0.6, alpha=0.72)
    ax_de.indicate_inset_zoom(ax_zoom, edgecolor="#77818A", alpha=0.75)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.97)
    return save_figure(fig, "q3_03_de_convergence.png")


def main() -> None:
    configure_matplotlib()
    row = selected_row()
    deployment = deployment_from_row(row)
    individual_intervals = calculate_individual_intervals(deployment)

    paths = [
        draw_timeline(row, deployment, individual_intervals),
        draw_joint_criterion(),
        draw_convergence(row),
    ]
    obsolete = FIGURE_DIR / "q3_03_de_grid_convergence.png"
    if obsolete.exists():
        obsolete.unlink()
    print("问题三可视化已生成：")
    for path in paths:
        print(f"  {path_label(path)}")


if __name__ == "__main__":
    main()
