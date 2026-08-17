"""为问题四 baseline-900 正式方案绘制接力时间轴图。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = PROJECT_ROOT / "Q4补" / "runs" / "baseline_900_standard" / "q4_best_solution.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "Q4补" / "runs" / "baseline_900_standard" / "q4_01_timeline_relay.png"

COLORS = ("#C84B31", "#2878B5", "#2E8B57")
LINESTYLES = ("-", "--", "-.")
MARKERS = ("o", "s", "^")
HATCHES = ("////", "\\\\", "xx")
JOINT_COLOR = "#E67E22"


def configure_fonts() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "sans-serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 3:
        raise RuntimeError(f"{path} 应包含 FY1、FY2、FY3 三行，实际为 {len(rows)} 行。")
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


def match_independent_intervals(
    rows: list[dict[str, str]], joint_intervals: tuple[tuple[float, float], ...]
) -> tuple[tuple[float, float], ...]:
    """按严格复核时长，将每架无人机对应到其独立遮蔽时间段。"""

    remaining = list(joint_intervals)
    matched: list[tuple[float, float]] = []
    for row in rows:
        duration = float(row["individual_duration_s"])
        index = min(range(len(remaining)), key=lambda i: abs((remaining[i][1] - remaining[i][0]) - duration))
        interval = remaining.pop(index)
        if abs((interval[1] - interval[0]) - duration) > 1.0e-5:
            raise RuntimeError("独立遮蔽时长无法与联合区间唯一对应；请改用评价器逐机复核。")
        matched.append(interval)
    return tuple(matched)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="x", color="#CBD2D9", linewidth=0.75, alpha=0.70)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def trim_white_margins(path: Path, padding: int = 28) -> None:
    from PIL import Image, ImageChops

    image = Image.open(path).convert("RGB")
    background = Image.new("RGB", image.size, "white")
    bbox = ImageChops.difference(image, background).getbbox()
    if bbox is None:
        return
    left, upper, right, lower = bbox
    cropped = image.crop(
        (
            max(0, left - padding),
            max(0, upper - padding),
            min(image.width, right + padding),
            min(image.height, lower + padding),
        )
    )
    cropped.save(path, dpi=(300, 300))


def draw(rows: list[dict[str, str]], output: Path) -> Path:
    joint_intervals = parse_intervals(rows[0]["joint_intervals_s"])
    independent_intervals = match_independent_intervals(rows, joint_intervals)
    joint_duration = float(rows[0]["joint_duration_s"])

    fig, ax = plt.subplots(figsize=(16.8, 6.6))
    y_positions = (3.0, 2.0, 1.0)
    contribution_spans: list[tuple[int, float, float, float]] = []

    for index, (row, interval, y) in enumerate(zip(rows, independent_intervals, y_positions)):
        color, linestyle, marker = COLORS[index], LINESTYLES[index], MARKERS[index]
        release = float(row["release_time_s"])
        burst = float(row["burst_time_s"])
        duration = float(row["individual_duration_s"])
        ax.plot([release, burst], [y, y], color=color, linestyle=linestyle, linewidth=2.7, alpha=0.95)
        ax.scatter([release], [y], s=95, facecolor="white", edgecolor=color, linewidth=2.2, marker=marker, zorder=5)
        ax.scatter([burst], [y], s=185, facecolor=color, edgecolor="white", linewidth=1.0, marker="*", zorder=6)

        release_offset = (12, 15) if index == 0 else (0, 15)
        release_align = "left" if index == 0 else "center"
        ax.annotate(
            f"投放 {release:.3f}", (release, y), xytext=release_offset,
            textcoords="offset points", ha=release_align, color=color, fontsize=12.2,
        )
        burst_offset = -30 if index in (1, 2) else -22
        ax.annotate(
            f"起爆 {burst:.3f}", (burst, y), xytext=(0, burst_offset),
            textcoords="offset points", ha="center", color=color, fontsize=12.2,
        )

        left, right = interval
        ax.barh(y, right - left, left=left, height=0.34, color=color, alpha=0.18,
                edgecolor=color, linewidth=1.6, hatch=HATCHES[index], zorder=2)
        contribution_spans.append((index, y, left, right))
        ax.text(
            0.5 * (left + right), y + 0.24, f"独立有效 {duration:.4f} s",
            ha="center", va="bottom", fontsize=12.3, color=color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.5),
        )

    for left, right in joint_intervals:
        ax.barh(0.0, right - left, left=left, height=0.42, color=JOINT_COLOR, alpha=0.38,
                edgecolor=JOINT_COLOR, linewidth=2.1, hatch="..", zorder=3)

    for index, source_y, left, right in contribution_spans:
        color = COLORS[index]
        ax.plot([left, left], [source_y - 0.19, 0.18], color=color, linestyle=(0, (3, 3)), linewidth=1.45, alpha=0.52, zorder=2)
        ax.plot([right, right], [source_y - 0.19, 0.18], color=color, linestyle=(0, (3, 3)), linewidth=1.45, alpha=0.52, zorder=2)
        ax.plot([left, right], [0.12, 0.12], color=color, linestyle=LINESTYLES[index], linewidth=3.1, alpha=0.95, zorder=5)

    first, last = joint_intervals[0][0], joint_intervals[-1][1]
    ax.annotate("", xy=(last, -0.42), xytext=(first, -0.42),
                arrowprops=dict(arrowstyle="<->", color=JOINT_COLOR, linewidth=2.1))
    ax.text(0.5 * (first + last), -0.55, f"联合有效总时长 = {joint_duration:.6f} s",
            ha="center", va="top", color="#A34F00", fontsize=14.0, fontweight="bold")

    summary_lines = []
    for row in rows:
        summary_lines.append(
            f"{row['uav']}：航向 {float(row['theta_deg']):.4f}°　速度 {float(row['speed_mps']):.4f} m/s"
        )
    summary_lines.append("空心标记：投放　★：起爆　阴影：该层有效遮蔽区间")
    ax.text(0.995, 0.98, "\n".join(summary_lines), transform=ax.transAxes,
            ha="right", va="top", multialignment="left", fontsize=11.2,
            bbox=dict(boxstyle="round,pad=0.42", facecolor="#F7F9FA", edgecolor="#AAB2BA"))

    x_right = max(last + 1.8, max(float(row["burst_time_s"]) for row in rows) + 2.0)
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(["烟幕弹 1（FY1）", "烟幕弹 2（FY2）", "烟幕弹 3（FY3）", "联合判定"], fontsize=13.0)
    ax.set_ylim(-0.82, 3.62)
    ax.set_xlim(0.0, x_right)
    style_axis(ax)
    ax.set_xlabel("时间 $t$ / s", fontsize=16.0)
    ax.tick_params(axis="x", labelsize=12.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    trim_white_margins(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制问题四 baseline-900 接力时间轴图。")
    parser.add_argument("--input", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    configure_fonts()
    output = draw(read_rows(args.input), args.output)
    print(f"已生成：{output}")


if __name__ == "__main__":
    main()
