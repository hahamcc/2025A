from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from mpl_toolkits.mplot3d import proj3d
from PIL import Image, ImageChops


# =============================================================================
# 第一问题面参数
# =============================================================================
GRAVITY = 9.8
MISSILE_SPEED = 300.0
UAV_SPEED = 120.0
RELEASE_TIME = 1.5
FUSE_DELAY = 3.6

MISSILE_INITIAL = np.array([20000.0, 0.0, 2000.0])
UAV_INITIAL = np.array([17800.0, 0.0, 1800.0])
FALSE_TARGET = np.array([0.0, 0.0, 0.0])
TRUE_TARGET_BOTTOM = np.array([0.0, 200.0, 0.0])
TARGET_RADIUS = 7.0
TARGET_HEIGHT = 10.0

MISSILE_DIRECTION = -MISSILE_INITIAL / np.linalg.norm(MISSILE_INITIAL)
UAV_DIRECTION = np.array([-1.0, 0.0, 0.0])
MISSILE_VELOCITY = MISSILE_SPEED * MISSILE_DIRECTION
UAV_VELOCITY = UAV_SPEED * UAV_DIRECTION

RELEASE_POINT = UAV_INITIAL + UAV_VELOCITY * RELEASE_TIME
BURST_TIME = RELEASE_TIME + FUSE_DELAY
BURST_POINT = RELEASE_POINT + UAV_VELOCITY * FUSE_DELAY
BURST_POINT[2] = RELEASE_POINT[2] - 0.5 * GRAVITY * FUSE_DELAY**2

EXPECTED_RELEASE_POINT = np.array([17620.0, 0.0, 1800.0])
EXPECTED_BURST_POINT = np.array([17188.0, 0.0, 1736.496])
if not np.allclose(RELEASE_POINT, EXPECTED_RELEASE_POINT, atol=1e-9):
    raise ValueError(f"投放点计算错误：{RELEASE_POINT}")
if not np.allclose(BURST_POINT, EXPECTED_BURST_POINT, atol=1e-9):
    raise ValueError(f"起爆点计算错误：{BURST_POINT}")

COLORS = {
    "missile": "#c84b31",
    "uav": "#245b8a",
    "target": "#2f7650",
    "false": "#77622e",
    "bomb": "#8f5bb3",
    "cloud": "#75a8cf",
    "neutral": "#3f4347",
    "light": "#9ba1a6",
}


def configure_matplotlib() -> None:
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if font_name in installed_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["figure.dpi"] = 130


def trim_white_margins(path: Path, padding: int = 24) -> None:
    """裁去纯白外边缘，同时给文字和坐标轴保留少量安全边距。"""

    with Image.open(path) as source:
        image = source.convert("RGB")
        difference = ImageChops.difference(
            image,
            Image.new("RGB", image.size, "white"),
        )
        bounding_box = difference.getbbox()
        if bounding_box is None:
            return
        left, top, right, bottom = bounding_box
        crop_box = (
            max(0, left - padding),
            max(0, top - padding),
            min(image.width, right + padding),
            min(image.height, bottom + padding),
        )
        image.crop(crop_box).save(path, dpi=source.info.get("dpi", (300, 300)))


def style_3d_axis(axis) -> None:
    axis.grid(True, alpha=0.20)
    for current_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        current_axis.pane.set_facecolor((0.96, 0.97, 0.98, 0.20))
        current_axis.pane.set_edgecolor((0.72, 0.74, 0.76, 0.45))


def annotate_3d(
    axis,
    text: str,
    point: np.ndarray,
    offset: tuple[float, float],
    color: str,
    horizontal_alignment: str = "left",
    fontsize: float = 11,
) -> None:
    """把三维点投影到画布后，在空白位置用引线标注。"""

    projected_x, projected_y, _ = proj3d.proj_transform(*point, axis.get_proj())
    axis.annotate(
        text,
        xy=(projected_x, projected_y),
        xytext=offset,
        textcoords="offset points",
        ha=horizontal_alignment,
        va="center",
        fontsize=fontsize,
        color=color,
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "linewidth": 1.0,
            "shrinkA": 2,
            "shrinkB": 3,
        },
        zorder=20,
    )


def draw_sphere(axis, center: np.ndarray, radius: float, alpha: float = 0.18) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 48)
    v = np.linspace(0.0, np.pi, 26)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    axis.plot_surface(
        x,
        y,
        z,
        color=COLORS["cloud"],
        alpha=alpha,
        linewidth=0,
        shade=False,
    )


def draw_cylinder(
    axis,
    bottom_center: np.ndarray,
    radius: float,
    height: float,
    color: str = COLORS["target"],
    alpha: float = 0.22,
) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 64)
    z_values = np.linspace(bottom_center[2], bottom_center[2] + height, 14)
    theta_mesh, z_mesh = np.meshgrid(theta, z_values)
    x_mesh = bottom_center[0] + radius * np.cos(theta_mesh)
    y_mesh = bottom_center[1] + radius * np.sin(theta_mesh)
    axis.plot_surface(
        x_mesh,
        y_mesh,
        z_mesh,
        color=color,
        alpha=alpha,
        linewidth=0,
        shade=False,
    )
    for z in (bottom_center[2], bottom_center[2] + height):
        radii = np.linspace(0.0, radius, 12)
        radius_mesh, cap_theta = np.meshgrid(radii, theta)
        cap_x = bottom_center[0] + radius_mesh * np.cos(cap_theta)
        cap_y = bottom_center[1] + radius_mesh * np.sin(cap_theta)
        cap_z = np.full_like(cap_x, z)
        axis.plot_surface(
            cap_x,
            cap_y,
            cap_z,
            color=color,
            alpha=0.16,
            linewidth=0,
            shade=False,
        )


def bomb_curve() -> np.ndarray:
    times = np.linspace(RELEASE_TIME, BURST_TIME, 100)
    elapsed = times - RELEASE_TIME
    positions = RELEASE_POINT[None, :] + elapsed[:, None] * UAV_VELOCITY[None, :]
    positions[:, 2] = RELEASE_POINT[2] - 0.5 * GRAVITY * elapsed**2
    return positions


def draw_scene_overview(output_directory: Path) -> Path:
    """图1：全局三维坐标系及两个关键区域的三维局部放大。"""

    figure = plt.figure(figsize=(16.0, 6.4), constrained_layout=True)
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.62, 0.76, 1.08),
        wspace=0.02,
    )
    scene_axis = figure.add_subplot(grid[0, 0], projection="3d")
    target_axis = figure.add_subplot(grid[0, 1], projection="3d")
    process_axis = figure.add_subplot(grid[0, 2], projection="3d")
    figure.suptitle("第一问三维坐标场景与主体运动关系", fontsize=19)
    figure.text(
        0.22,
        0.91,
        "(a) 全局三维坐标系（轴向显示比例调整）",
        ha="center",
        fontsize=15,
    )
    figure.text(
        0.59,
        0.91,
        "(b) 目标区三维局部放大",
        ha="center",
        fontsize=15,
    )
    figure.text(
        0.84,
        0.91,
        "(c) FY1—投放—起爆过程三维放大",
        ha="center",
        fontsize=15,
    )

    # ------------------------------------------------------------------
    # (a) 全局三维场景：坐标值保持真实，三个轴的显示比例单独调整。
    # ------------------------------------------------------------------
    scene_axis.plot(
        [MISSILE_INITIAL[0], FALSE_TARGET[0]],
        [MISSILE_INITIAL[1], FALSE_TARGET[1]],
        [MISSILE_INITIAL[2], FALSE_TARGET[2]],
        color=COLORS["missile"],
        linestyle="-",
        linewidth=2.5,
    )
    uav_track_end = np.array([14500.0, 0.0, 1800.0])
    scene_axis.plot(
        [UAV_INITIAL[0], uav_track_end[0]],
        [UAV_INITIAL[1], uav_track_end[1]],
        [UAV_INITIAL[2], uav_track_end[2]],
        color=COLORS["uav"],
        linestyle="-.",
        linewidth=2.5,
    )
    curve = bomb_curve()
    scene_axis.plot(
        curve[:, 0],
        curve[:, 1],
        curve[:, 2],
        color=COLORS["bomb"],
        linestyle=(0, (2, 2)),
        linewidth=2.0,
    )

    scene_axis.quiver(
        *MISSILE_INITIAL,
        *MISSILE_DIRECTION,
        length=3200,
        normalize=True,
        color=COLORS["missile"],
        linewidth=2.2,
        arrow_length_ratio=0.12,
    )
    scene_axis.quiver(
        *UAV_INITIAL,
        *UAV_DIRECTION,
        length=2700,
        normalize=True,
        color=COLORS["uav"],
        linewidth=2.2,
        arrow_length_ratio=0.13,
    )

    scene_axis.scatter(
        *MISSILE_INITIAL,
        s=78,
        color=COLORS["missile"],
        edgecolor="white",
        linewidth=0.9,
        depthshade=False,
    )
    scene_axis.scatter(
        *UAV_INITIAL,
        s=76,
        marker="s",
        color=COLORS["uav"],
        edgecolor="white",
        linewidth=0.9,
        depthshade=False,
    )
    scene_axis.scatter(
        *FALSE_TARGET,
        s=120,
        marker="*",
        color=COLORS["false"],
        depthshade=False,
    )
    scene_axis.scatter(
        *TRUE_TARGET_BOTTOM,
        s=72,
        color=COLORS["target"],
        depthshade=False,
    )

    scene_axis.text2D(
        0.03,
        0.94,
        r"M1：$M_1(0)=(20000,0,2000)$ m，$\mathbf{e}_M=(-0.9950,0,-0.0995)$，$v_M=300$ m/s",
        transform=scene_axis.transAxes,
        color=COLORS["missile"],
        fontsize=11.5,
    )
    scene_axis.text2D(
        0.03,
        0.895,
        r"FY1：$FY_1(0)=(17800,0,1800)$ m，$\mathbf{e}_F=(-1,0,0)$，$v_F=120$ m/s",
        transform=scene_axis.transAxes,
        color=COLORS["uav"],
        fontsize=11.5,
    )
    scene_axis.text2D(
        0.03,
        0.845,
        r"坐标约定：$\odot\,+y$ 指向纸外（朝向观察者）",
        transform=scene_axis.transAxes,
        color=COLORS["neutral"],
        fontsize=11,
    )
    scene_axis.set_xlim(20600, -600)
    scene_axis.set_ylim(-350, 650)
    scene_axis.set_zlim(0, 2350)
    scene_axis.set_box_aspect((2.55, 1.0, 1.18))
    scene_axis.view_init(elev=20, azim=72)
    scene_axis.set_xlabel("$x$ / m", labelpad=9)
    scene_axis.set_ylabel("$y$ / m", labelpad=9)
    scene_axis.set_zlabel("$z$ / m", labelpad=7)
    style_3d_axis(scene_axis)
    annotate_3d(
        scene_axis,
        "$M_1(0)$：来袭导弹",
        MISSILE_INITIAL,
        (-12, 28),
        COLORS["missile"],
        horizontal_alignment="right",
    )
    annotate_3d(
        scene_axis,
        "$FY_1(0)$：无人机",
        UAV_INITIAL,
        (-10, -30),
        COLORS["uav"],
        horizontal_alignment="right",
    )
    annotate_3d(
        scene_axis,
        "假目标 $O$",
        FALSE_TARGET,
        (-20, 26),
        COLORS["false"],
        horizontal_alignment="right",
    )
    annotate_3d(
        scene_axis,
        "真目标圆柱 $T$",
        TRUE_TARGET_BOTTOM,
        (18, -24),
        COLORS["target"],
    )

    # ------------------------------------------------------------------
    # (b) 目标区三维放大：明确真假目标相距 200 m 及真实圆柱尺寸。
    # ------------------------------------------------------------------
    draw_cylinder(
        target_axis,
        TRUE_TARGET_BOTTOM,
        TARGET_RADIUS,
        TARGET_HEIGHT,
        alpha=0.34,
    )
    target_axis.scatter(
        *FALSE_TARGET,
        s=100,
        marker="*",
        color=COLORS["false"],
        depthshade=False,
    )
    target_axis.plot(
        [0, 0],
        [0, 200],
        [0, 0],
        color=COLORS["neutral"],
        linestyle="--",
        linewidth=1.4,
    )
    target_axis.text2D(
        0.05,
        0.82,
        "圆柱尺寸：$r=7$ m，$h=10$ m",
        transform=target_axis.transAxes,
        color=COLORS["target"],
        fontsize=10,
    )
    target_axis.text(
        -13,
        95,
        2,
        "200 m",
        color=COLORS["neutral"],
        fontsize=10,
    )
    target_axis.set_xlim(35, -25)
    target_axis.set_ylim(-25, 230)
    target_axis.set_zlim(0, 20)
    target_axis.set_box_aspect((0.9, 2.5, 0.72))
    target_axis.view_init(elev=24, azim=72)
    target_axis.set_xlabel("$x$ / m", labelpad=5)
    target_axis.set_ylabel("$y$ / m", labelpad=5)
    target_axis.set_zlabel("$z$ / m", labelpad=3)
    style_3d_axis(target_axis)
    annotate_3d(
        target_axis,
        "假目标 $O$",
        FALSE_TARGET,
        (-18, 22),
        COLORS["false"],
        horizontal_alignment="right",
        fontsize=10.5,
    )
    annotate_3d(
        target_axis,
        "真目标圆柱 $T$",
        TRUE_TARGET_BOTTOM + np.array([0.0, 0.0, TARGET_HEIGHT / 2.0]),
        (20, -8),
        COLORS["target"],
        fontsize=10.5,
    )

    # ------------------------------------------------------------------
    # (c) 投放—起爆区域三维放大：分开显示三个相邻位置。
    # ------------------------------------------------------------------
    process_axis.plot(
        [UAV_INITIAL[0], 16950],
        [0, 0],
        [UAV_INITIAL[2], UAV_INITIAL[2]],
        color=COLORS["uav"],
        linestyle="-.",
        linewidth=2.2,
    )
    process_axis.plot(
        curve[:, 0],
        curve[:, 1],
        curve[:, 2],
        color=COLORS["bomb"],
        linestyle=(0, (2, 2)),
        linewidth=2.2,
    )
    process_axis.scatter(
        *UAV_INITIAL,
        s=66,
        marker="s",
        color=COLORS["uav"],
        depthshade=False,
    )
    process_axis.scatter(
        *RELEASE_POINT,
        s=58,
        marker="D",
        color=COLORS["bomb"],
        depthshade=False,
    )
    process_axis.scatter(
        *BURST_POINT,
        s=65,
        color=COLORS["cloud"],
        edgecolor=COLORS["uav"],
        linewidth=0.9,
        depthshade=False,
    )
    process_axis.quiver(
        *UAV_INITIAL,
        *UAV_DIRECTION,
        length=650,
        normalize=True,
        color=COLORS["uav"],
        linewidth=2.0,
        arrow_length_ratio=0.17,
    )
    process_axis.text2D(
        0.04,
        0.90,
        "■ FY1初始：$(17800,0,1800)$ m\n"
        "◆ $P_r$投放：$t=1.5$ s，$(17620,0,1800)$ m\n"
        "● $P_b$起爆：$t=5.1$ s，$(17188,0,1736.5)$ m",
        transform=process_axis.transAxes,
        color=COLORS["neutral"],
        fontsize=10.5,
        va="top",
    )
    process_axis.set_xlim(17950, 16920)
    process_axis.set_ylim(-60, 60)
    process_axis.set_zlim(1670, 1855)
    process_axis.set_box_aspect((2.4, 0.75, 1.05))
    process_axis.view_init(elev=24, azim=72)
    process_axis.set_xlabel("$x$ / m", labelpad=5)
    process_axis.set_ylabel("$y$ / m", labelpad=4)
    process_axis.set_zlabel("$z$ / m", labelpad=3)
    style_3d_axis(process_axis)
    annotate_3d(
        process_axis,
        "FY1初始位置",
        UAV_INITIAL,
        (-8, -28),
        COLORS["uav"],
        horizontal_alignment="right",
        fontsize=10.5,
    )
    annotate_3d(
        process_axis,
        "$P_r$：烟幕弹投放点",
        RELEASE_POINT,
        (8, 28),
        COLORS["bomb"],
        fontsize=10.5,
    )
    annotate_3d(
        process_axis,
        "$P_b$：烟幕弹起爆点",
        BURST_POINT,
        (14, -30),
        COLORS["uav"],
        fontsize=10.5,
    )

    path = output_directory / "q1_scene_overview.png"
    figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.10)
    plt.close(figure)
    return path


def draw_global_scene(output_directory: Path) -> Path:
    """独立图(a)：全局三维坐标场景与主体运动关系。"""

    figure = plt.figure(figsize=(12.8, 7.0), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    figure.suptitle("三维坐标场景与主体运动关系", fontsize=21)

    axis.plot(
        [MISSILE_INITIAL[0], FALSE_TARGET[0]],
        [MISSILE_INITIAL[1], FALSE_TARGET[1]],
        [MISSILE_INITIAL[2], FALSE_TARGET[2]],
        color=COLORS["missile"],
        linestyle="-",
        linewidth=3.0,
    )
    uav_track_end = np.array([14500.0, 0.0, 1800.0])
    axis.plot(
        [UAV_INITIAL[0], uav_track_end[0]],
        [UAV_INITIAL[1], uav_track_end[1]],
        [UAV_INITIAL[2], uav_track_end[2]],
        color=COLORS["uav"],
        linestyle="-.",
        linewidth=3.0,
    )
    missile_arrow_start = MISSILE_INITIAL + 0.36 * (
        FALSE_TARGET - MISSILE_INITIAL
    )
    axis.quiver(
        *missile_arrow_start,
        *MISSILE_DIRECTION,
        length=2800,
        normalize=True,
        color=COLORS["missile"],
        linewidth=2.5,
        arrow_length_ratio=0.12,
    )
    axis.quiver(
        *UAV_INITIAL,
        *UAV_DIRECTION,
        length=2500,
        normalize=True,
        color=COLORS["uav"],
        linewidth=2.5,
        arrow_length_ratio=0.13,
    )
    axis.scatter(
        *MISSILE_INITIAL,
        s=92,
        color=COLORS["missile"],
        edgecolor="white",
        linewidth=1.0,
        depthshade=False,
    )
    axis.scatter(
        *UAV_INITIAL,
        s=90,
        marker="s",
        color=COLORS["uav"],
        edgecolor="white",
        linewidth=1.0,
        depthshade=False,
    )
    axis.scatter(
        *FALSE_TARGET,
        s=150,
        marker="*",
        color=COLORS["false"],
        depthshade=False,
    )
    axis.scatter(
        *TRUE_TARGET_BOTTOM,
        s=88,
        color=COLORS["target"],
        depthshade=False,
    )

    axis.text2D(
        0.035,
        0.925,
        r"M1：$M_1(0)=(20000,0,2000)$ m，$\mathbf{e}_M=(-0.9950,0,-0.0995)$，$v_M=300$ m/s",
        transform=axis.transAxes,
        color=COLORS["missile"],
        fontsize=14,
    )
    axis.text2D(
        0.035,
        0.865,
        r"FY1：$FY_1(0)=(17800,0,1800)$ m，$\mathbf{e}_F=(-1,0,0)$，$v_F=120$ m/s",
        transform=axis.transAxes,
        color=COLORS["uav"],
        fontsize=14,
    )

    axis.set_xlim(20600, 0)
    axis.set_ylim(0, 650)
    axis.set_yticks([200, 400, 600])
    axis.set_zlim(0, 2350)
    axis.set_box_aspect((2.55, 1.0, 1.18))
    axis.view_init(elev=20, azim=72)
    axis.set_xlabel("$x$ / m", labelpad=10)
    axis.set_ylabel("$y$ / m", labelpad=10)
    axis.set_zlabel("$z$ / m", labelpad=8)
    axis.tick_params(labelsize=12)
    style_3d_axis(axis)

    annotate_3d(
        axis,
        "$M_1(0)$：来袭导弹",
        MISSILE_INITIAL,
        (-14, 34),
        COLORS["missile"],
        horizontal_alignment="right",
        fontsize=14,
    )
    annotate_3d(
        axis,
        "$FY_1(0)$：无人机",
        UAV_INITIAL,
        (20, -42),
        COLORS["uav"],
        horizontal_alignment="left",
        fontsize=14,
    )
    annotate_3d(
        axis,
        "假目标 $O$",
        FALSE_TARGET,
        (28, 42),
        COLORS["false"],
        horizontal_alignment="left",
        fontsize=14,
    )
    annotate_3d(
        axis,
        "真目标圆柱 $T$",
        TRUE_TARGET_BOTTOM,
        (42, 26),
        COLORS["target"],
        fontsize=14,
    )

    path = output_directory / "q1_scene_overview.png"
    figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.10)
    plt.close(figure)
    return path


def draw_target_area(output_directory: Path) -> Path:
    """独立图(b)：真假目标的三维局部关系。"""

    figure = plt.figure(figsize=(9.2, 6.4))
    axis = figure.add_axes([0.05, 0.00, 0.92, 0.90], projection="3d")
    figure.suptitle("目标区三维局部关系", fontsize=21, y=0.84)

    draw_cylinder(
        axis,
        TRUE_TARGET_BOTTOM,
        TARGET_RADIUS,
        TARGET_HEIGHT,
        alpha=0.38,
    )
    axis.scatter(
        *FALSE_TARGET,
        s=150,
        marker="*",
        color=COLORS["false"],
        depthshade=False,
    )
    axis.plot(
        [0, 0],
        [0, 200],
        [0, 0],
        color=COLORS["neutral"],
        linestyle="--",
        linewidth=1.8,
    )
    axis.text(
        -11,
        100,
        1.0,
        "200 m",
        color=COLORS["neutral"],
        fontsize=13,
    )

    axis.set_xlim(35, -25)
    axis.set_ylim(-25, 230)
    axis.set_zlim(0, 20)
    axis.set_xticks([-20, -10, 0])
    axis.set_zticks([0, 5, 10, 15, 20])
    axis.set_yticks([0, 50, 100, 150, 200])
    axis.set_box_aspect((1.0, 2.4, 0.9))
    axis.view_init(elev=24, azim=72)
    axis.set_xlabel("$x$ / m", labelpad=8)
    axis.set_ylabel("$y$ / m", labelpad=5)
    axis.set_zlabel("$z$ / m", labelpad=8)
    axis.tick_params(labelsize=12, pad=2)
    style_3d_axis(axis)

    annotate_3d(
        axis,
        "假目标 $O$",
        FALSE_TARGET,
        (24, 28),
        COLORS["false"],
        fontsize=14,
    )
    annotate_3d(
        axis,
        "真目标圆柱 $T$\n$r=7$ m，$h=10$ m",
        TRUE_TARGET_BOTTOM + np.array([0.0, 0.0, 1.5]),
        (38, 0),
        COLORS["target"],
        horizontal_alignment="left",
        fontsize=13.5,
    )

    path = output_directory / "q1_target_area_3d.png"
    figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)
    trim_white_margins(path)
    return path


def draw_release_burst_process(output_directory: Path) -> Path:
    """独立图(c)：FY1、投放点与起爆点的三维局部关系。"""

    figure = plt.figure(figsize=(10.8, 6.2))
    axis = figure.add_axes([0.05, 0.00, 0.92, 0.90], projection="3d")
    figure.suptitle("FY1投放—起爆过程三维局部关系", fontsize=21, y=0.84)
    curve = bomb_curve()

    axis.plot(
        [UAV_INITIAL[0], 16950],
        [0, 0],
        [UAV_INITIAL[2], UAV_INITIAL[2]],
        color=COLORS["uav"],
        linestyle="-.",
        linewidth=3.0,
    )
    axis.plot(
        curve[:, 0],
        curve[:, 1],
        curve[:, 2],
        color=COLORS["bomb"],
        linestyle=(0, (2, 2)),
        linewidth=2.8,
    )
    axis.scatter(
        *UAV_INITIAL,
        s=92,
        marker="s",
        color=COLORS["uav"],
        depthshade=False,
    )
    axis.scatter(
        *RELEASE_POINT,
        s=82,
        marker="D",
        color=COLORS["bomb"],
        depthshade=False,
    )
    axis.scatter(
        *BURST_POINT,
        s=88,
        color=COLORS["cloud"],
        edgecolor=COLORS["uav"],
        linewidth=1.0,
        depthshade=False,
    )

    # 将关键点正交投影到前侧 x 轴，消除三维透视造成的横坐标错觉。
    z_floor = 1670.0
    y_front = 60.0
    for point, color in (
        (RELEASE_POINT, COLORS["bomb"]),
        (BURST_POINT, COLORS["uav"]),
    ):
        axis.plot(
            [point[0], point[0], point[0]],
            [point[1], point[1], y_front],
            [point[2], z_floor, z_floor],
            color=color,
            linestyle=(0, (2, 2)),
            linewidth=1.5,
            alpha=0.82,
        )
        axis.scatter(
            point[0],
            y_front,
            z_floor,
            s=70,
            marker="|",
            color=color,
            depthshade=False,
        )
    axis.quiver(
        *UAV_INITIAL,
        *UAV_DIRECTION,
        length=300,
        normalize=True,
        color=COLORS["uav"],
        linewidth=2.2,
        arrow_length_ratio=0.20,
    )

    axis.set_xlim(17950, 16920)
    axis.set_ylim(-70, 70)
    axis.set_zlim(1670, 1855)
    axis.set_xticks([17000, 17188, 17400, 17620, 17800])
    axis.set_yticks([-60, 0, 60])
    axis.set_zticks([1680, 1720, 1760, 1800, 1840])
    axis.set_box_aspect((2.35, 0.78, 1.05))
    axis.view_init(elev=24, azim=72)
    axis.set_xlabel("$x$ / m", labelpad=8)
    axis.set_ylabel("")
    axis.set_zlabel("$z$ / m", labelpad=8)
    axis.tick_params(labelsize=12, pad=2)
    style_3d_axis(axis)
    axis.text2D(
        -0.015,
        0.15,
        "$y$ / m",
        transform=axis.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=13,
    )

    annotate_3d(
        axis,
        "FY1初始位置",
        UAV_INITIAL,
        (-10, -36),
        COLORS["uav"],
        horizontal_alignment="right",
        fontsize=14,
    )
    annotate_3d(
        axis,
        "$P_r=(17620,0,1800)$ m\n烟幕弹投放点，$t=1.5$ s",
        RELEASE_POINT,
        (12, 38),
        COLORS["bomb"],
        fontsize=13,
    )
    annotate_3d(
        axis,
        "$P_b=(17188,0,1736.496)$ m\n烟幕弹起爆点，$t=5.1$ s",
        BURST_POINT,
        (20, -42),
        COLORS["uav"],
        fontsize=13,
    )

    path = output_directory / "q1_release_burst_3d.png"
    figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)
    trim_white_margins(path)
    return path


def closest_point_on_segment(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    direction = end - start
    raw_parameter = np.dot(point - start, direction) / np.dot(direction, direction)
    parameter = float(np.clip(raw_parameter, 0.0, 1.0))
    closest = start + parameter * direction
    return closest, float(np.linalg.norm(point - closest)), parameter


def draw_cylinder_sampling(axis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.array([4.7, 1.8, 0.0])
    radius = 1.05
    height = 2.6
    angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    heights = np.linspace(0, height, 7)

    draw_cylinder(axis, center, radius, height, alpha=0.10)

    side_points = np.array(
        [
            [center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta), z]
            for z in heights
            for theta in angles
        ]
    )
    axis.scatter(
        side_points[:, 0],
        side_points[:, 1],
        side_points[:, 2],
        s=7,
        color=COLORS["target"],
        alpha=0.74,
        depthshade=False,
    )

    cap_points = []
    for z in (0.0, height):
        for current_radius in np.linspace(0, radius, 5):
            for theta in angles:
                cap_points.append(
                    [
                        center[0] + current_radius * np.cos(theta),
                        center[1] + current_radius * np.sin(theta),
                        z,
                    ]
                )
    cap_points = np.asarray(cap_points)
    axis.scatter(
        cap_points[:, 0],
        cap_points[:, 1],
        cap_points[:, 2],
        s=6,
        color="#4f9d70",
        alpha=0.58,
        depthshade=False,
    )
    return center, side_points, cap_points


def draw_visibility_model(output_directory: Path) -> Path:
    """图2：三维有限视线段判定与三维完整圆柱表面校核。"""

    figure = plt.figure(figsize=(15.2, 7.8))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 1.08),
        wspace=0.02,
        left=0.04,
        right=0.99,
        bottom=0.05,
        top=0.76,
    )
    point_axis = figure.add_subplot(grid[0, 0], projection="3d")
    cylinder_axis = figure.add_subplot(grid[0, 1], projection="3d")
    figure.suptitle("从单个目标点到完整圆柱表面", fontsize=22, y=0.985)
    figure.text(
        0.25,
        0.915,
        "(a) 单点遮蔽：有限视线段与烟幕球",
        ha="center",
        fontsize=17,
    )
    figure.text(
        0.75,
        0.915,
        "(b) 完整圆柱校核：取所有采样点中的最大值",
        ha="center",
        fontsize=17,
    )

    # ------------------------------------------------------------------
    # (a) 单目标点：球心到有限视线段的最短距离。
    # ------------------------------------------------------------------
    missile = np.array([-6.0, -3.0, 3.0])
    cloud = np.array([0.0, 1.0, 2.0])
    target = np.array([6.0, 2.0, 1.0])
    radius = 1.5
    closest, _, _ = closest_point_on_segment(cloud, missile, target)

    draw_sphere(point_axis, cloud, radius, alpha=0.20)
    point_axis.plot(
        [missile[0], target[0]],
        [missile[1], target[1]],
        [missile[2], target[2]],
        color=COLORS["missile"],
        linewidth=3.0,
    )
    point_axis.plot(
        [cloud[0], closest[0]],
        [cloud[1], closest[1]],
        [cloud[2], closest[2]],
        color=COLORS["uav"],
        linestyle=(0, (6, 3)),
        linewidth=3.0,
    )
    point_axis.scatter(*missile, s=80, color="#303030", depthshade=False)
    point_axis.scatter(*cloud, s=68, color=COLORS["uav"], depthshade=False)
    point_axis.scatter(*target, s=80, color=COLORS["target"], depthshade=False)
    point_axis.scatter(*closest, s=68, color="#e68a2e", depthshade=False)

    figure.text(
        0.045,
        0.845,
        "━━━━  有限视线段 $\\overline{M(t)Q_i}$\n"
        "– – –  球心到视线段的最短距离 $d_i(t)$\n"
        "● $P_i$：最短距离在线段上的垂足",
        fontsize=13,
        color=COLORS["neutral"],
        va="top",
    )
    point_axis.text2D(
        0.50,
        0.03,
        r"$d_i(t)=\mathrm{dist}\!\left(C(t),\overline{M(t)Q_i}\right)\leq10$ m"
        r" $\Rightarrow Q_i$ 被遮蔽",
        transform=point_axis.transAxes,
        ha="center",
        fontsize=14,
        color=COLORS["neutral"],
    )
    point_axis.set_xlim(7.0, -6.8)
    point_axis.set_ylim(-4.0, 3.6)
    point_axis.set_zlim(0.0, 5.0)
    point_axis.set_box_aspect((1.55, 1.0, 0.85))
    point_axis.view_init(elev=23, azim=72)
    point_axis.set_xlabel("局部 $x$ 坐标 / m", labelpad=8)
    point_axis.set_ylabel("局部 $y$ 坐标 / m", labelpad=10)
    point_axis.set_zlabel("高度 $z$ / m", labelpad=7)
    point_axis.tick_params(labelsize=11)
    style_3d_axis(point_axis)
    annotate_3d(
        point_axis,
        "$M(t)$：导弹位置",
        missile,
        (18, 34),
        COLORS["neutral"],
        horizontal_alignment="left",
        fontsize=13,
    )
    annotate_3d(
        point_axis,
        "$C(t)$：烟幕球心",
        cloud,
        (-30, -40),
        COLORS["uav"],
        horizontal_alignment="right",
        fontsize=13,
    )
    annotate_3d(
        point_axis,
        "$Q_i$：目标表面点",
        target,
        (12, -26),
        COLORS["target"],
        fontsize=13,
    )
    annotate_3d(
        point_axis,
        "$P_i$：线段最近点",
        closest,
        (28, 34),
        "#9a541b",
        fontsize=13,
    )

    # ------------------------------------------------------------------
    # (b) 完整圆柱：侧面、上端面和下端面全部采样。
    # ------------------------------------------------------------------
    missile_3d = np.array([-5.6, -3.5, 3.7])
    cloud_3d = np.array([-0.8, -0.5, 2.25])
    draw_sphere(cylinder_axis, cloud_3d, 1.5, alpha=0.17)
    cylinder_center, side_points, cap_points = draw_cylinder_sampling(cylinder_axis)

    representative_indices = [0, 28, 65, 105, 150, 202]
    for point in side_points[representative_indices]:
        cylinder_axis.plot(
            [missile_3d[0], point[0]],
            [missile_3d[1], point[1]],
            [missile_3d[2], point[2]],
            color="#858585",
            alpha=0.62,
            linewidth=1.5,
            linestyle=(0, (6, 3)),
        )

    all_points = np.vstack((side_points, cap_points))
    distances = np.array(
        [
            closest_point_on_segment(cloud_3d, missile_3d, point)[1]
            for point in all_points
        ]
    )
    worst_point = all_points[int(np.argmax(distances))]
    worst_closest, _, _ = closest_point_on_segment(
        cloud_3d,
        missile_3d,
        worst_point,
    )

    cylinder_axis.plot(
        [missile_3d[0], worst_point[0]],
        [missile_3d[1], worst_point[1]],
        [missile_3d[2], worst_point[2]],
        color=COLORS["missile"],
        linewidth=2.8,
        linestyle="-.",
    )
    cylinder_axis.plot(
        [cloud_3d[0], worst_closest[0]],
        [cloud_3d[1], worst_closest[1]],
        [cloud_3d[2], worst_closest[2]],
        color=COLORS["uav"],
        linestyle="--",
        linewidth=1.9,
    )
    cylinder_axis.scatter(
        *missile_3d,
        s=75,
        color="#303030",
        depthshade=False,
    )
    cylinder_axis.scatter(
        *cloud_3d,
        s=64,
        color=COLORS["uav"],
        depthshade=False,
    )
    cylinder_axis.scatter(
        *worst_point,
        s=78,
        color=COLORS["missile"],
        edgecolor="white",
        linewidth=0.9,
        depthshade=False,
    )
    figure.text(
        0.545,
        0.845,
        "– – –  普通采样点对应的视线段\n"
        "—·—·—  最不利目标点对应的视线段\n"
        "┄┄  球心到最不利视线的距离",
        fontsize=13,
        color=COLORS["neutral"],
        va="top",
    )
    cylinder_axis.text2D(
        0.50,
        0.03,
        r"$D_{\max}(t)=\max_i d_i(t)\leq10$ m"
        r" $\Rightarrow$ 完整圆柱被遮蔽",
        transform=cylinder_axis.transAxes,
        ha="center",
        fontsize=14,
        color=COLORS["neutral"],
    )
    cylinder_axis.set_xlim(6.1, -6.3)
    cylinder_axis.set_ylim(-4.3, 3.8)
    cylinder_axis.set_zlim(0, 5.0)
    cylinder_axis.set_box_aspect((1.45, 1.0, 0.78))
    cylinder_axis.view_init(elev=21, azim=72)
    cylinder_axis.set_xlabel("局部 $x$ 坐标 / m", labelpad=8)
    cylinder_axis.set_ylabel("局部 $y$ 坐标 / m", labelpad=10)
    cylinder_axis.set_zlabel("高度 $z$ / m", labelpad=7)
    cylinder_axis.tick_params(labelsize=11)
    style_3d_axis(cylinder_axis)
    annotate_3d(
        cylinder_axis,
        "$M(t)$：导弹位置",
        missile_3d,
        (22, 32),
        COLORS["neutral"],
        horizontal_alignment="left",
        fontsize=13,
    )
    annotate_3d(
        cylinder_axis,
        "$C(t)$：烟幕球心",
        cloud_3d,
        (-28, -48),
        COLORS["uav"],
        horizontal_alignment="right",
        fontsize=13,
    )
    annotate_3d(
        cylinder_axis,
        "$Q^*$：最不利目标点",
        worst_point,
        (0, 46),
        COLORS["missile"],
        horizontal_alignment="center",
        fontsize=13,
    )
    annotate_3d(
        cylinder_axis,
        "侧面及上下端面的离散采样点",
        side_points[165],
        (-58, -76),
        COLORS["target"],
        horizontal_alignment="right",
        fontsize=13,
    )

    path = output_directory / "q1_visibility_model.png"
    figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.10)
    plt.close(figure)
    return path


def main() -> None:
    configure_matplotlib()
    output_directory = Path(__file__).resolve().parent
    scene_path = draw_global_scene(output_directory)
    target_path = draw_target_area(output_directory)
    process_path = draw_release_burst_process(output_directory)
    visibility_path = draw_visibility_model(output_directory)
    print(f"全局场景图已保存：{scene_path}")
    print(f"目标区局部图已保存：{target_path}")
    print(f"投放—起爆过程图已保存：{process_path}")
    print(f"遮蔽判定图已保存：{visibility_path}")


if __name__ == "__main__":
    main()
