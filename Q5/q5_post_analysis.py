from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from .q5_column_generation import Q5_DIR, load_best_plans
    from .q5_main import (
        CLOUD_LIFETIME,
        FUSE_DELAY_LIMITS,
        MISSILE_NAMES,
        MultiMissileEvaluator,
        Strategy,
        UavPlan,
        UAV_INITIALS,
        UAV_NAMES,
        BombPlan,
        write_csv,
    )
except ImportError:
    from q5_column_generation import Q5_DIR, load_best_plans
    from q5_main import (
        CLOUD_LIFETIME,
        FUSE_DELAY_LIMITS,
        MISSILE_NAMES,
        MultiMissileEvaluator,
        Strategy,
        UavPlan,
        UAV_INITIALS,
        UAV_NAMES,
        BombPlan,
        write_csv,
    )


def _bomb_label(plan: UavPlan, bomb_index: int) -> str:
    return f"{plan.name}-B{bomb_index + 1}"


def _flat_labels(strategy: Strategy) -> tuple[str, ...]:
    return tuple(
        _bomb_label(plan, bomb_index)
        for plan in strategy.uavs
        for bomb_index, _ in enumerate(plan.bombs)
    )


def _remove_bomb(strategy: Strategy, uav_name: str, bomb_index: int) -> Strategy:
    revised = []
    for plan in strategy.uavs:
        if plan.name != uav_name:
            revised.append(plan)
            continue
        bombs = tuple(bomb for index, bomb in enumerate(plan.bombs) if index != bomb_index)
        if bombs:
            revised.append(UavPlan(plan.name, plan.initial, plan.heading, plan.speed, bombs))
    return Strategy(tuple(revised))


def _remove_uav(strategy: Strategy, uav_name: str) -> Strategy:
    return Strategy(tuple(plan for plan in strategy.uavs if plan.name != uav_name))


def _ablation_rows(strategy: Strategy, evaluator: MultiMissileEvaluator) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    baseline = evaluator.evaluate(strategy)
    bomb_rows: list[dict[str, object]] = []
    uav_rows: list[dict[str, object]] = []
    for plan in strategy.uavs:
        for bomb_index, _ in enumerate(plan.bombs):
            result = evaluator.evaluate(_remove_bomb(strategy, plan.name, bomb_index))
            bomb_rows.append(
                {
                    "removed_type": "bomb",
                    "removed_id": _bomb_label(plan, bomb_index),
                    "remaining_M1_s": result.missile_durations[0],
                    "remaining_M2_s": result.missile_durations[1],
                    "remaining_M3_s": result.missile_durations[2],
                    "remaining_total_s": result.total_duration,
                    "loss_M1_s": baseline.missile_durations[0] - result.missile_durations[0],
                    "loss_M2_s": baseline.missile_durations[1] - result.missile_durations[1],
                    "loss_M3_s": baseline.missile_durations[2] - result.missile_durations[2],
                    "loss_total_s": baseline.total_duration - result.total_duration,
                }
            )
        result = evaluator.evaluate(_remove_uav(strategy, plan.name))
        uav_rows.append(
            {
                "removed_type": "uav",
                "removed_id": plan.name,
                "remaining_M1_s": result.missile_durations[0],
                "remaining_M2_s": result.missile_durations[1],
                "remaining_M3_s": result.missile_durations[2],
                "remaining_total_s": result.total_duration,
                "loss_M1_s": baseline.missile_durations[0] - result.missile_durations[0],
                "loss_M2_s": baseline.missile_durations[1] - result.missile_durations[1],
                "loss_M3_s": baseline.missile_durations[2] - result.missile_durations[2],
                "loss_total_s": baseline.total_duration - result.total_duration,
            }
        )
    return bomb_rows, uav_rows


def _partition_edges(left: float, right: float, step: float) -> np.ndarray:
    edges = np.arange(left, right, step, dtype=float)
    if edges.size == 0 or abs(edges[0] - left) > 1.0e-12:
        edges = np.insert(edges, 0, left)
    if abs(edges[-1] - right) > 1.0e-12:
        edges = np.append(edges, right)
    return edges


def _append_mode_runs(
    rows: list[dict[str, object]],
    missile: str,
    edges: np.ndarray,
    modes: list[str],
    coverers: list[str],
) -> None:
    if not modes:
        return
    start = 0
    for index in range(1, len(modes) + 1):
        changed = index == len(modes) or modes[index] != modes[start] or coverers[index] != coverers[start]
        if changed:
            rows.append(
                {
                    "missile": missile,
                    "mode": modes[start],
                    "start_s": edges[start],
                    "end_s": edges[index],
                    "duration_s": edges[index] - edges[start],
                    "individual_complete_coverers": coverers[start],
                }
            )
            start = index


def occlusion_mode_rows(strategy: Strategy, evaluator: MultiMissileEvaluator) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Partition exact joint intervals into single-cloud relay and spatial-joint cells."""
    result = evaluator.evaluate(strategy)
    flat = evaluator.flatten(strategy)
    labels = _flat_labels(strategy)
    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for missile_index, missile in enumerate(MISSILE_NAMES):
        local_rows: list[dict[str, object]] = []
        for segment_left, segment_right, active in evaluator.event_segments(flat, missile_index):
            if not active:
                continue
            for interval_left, interval_right in result.missile_intervals[missile_index]:
                left = max(segment_left, interval_left)
                right = min(segment_right, interval_right)
                if right - left <= 1.0e-10:
                    continue
                edges = _partition_edges(left, right, evaluator.scan_step)
                middles = 0.5 * (edges[:-1] + edges[1:])
                joint = evaluator.status_batch(flat, missile_index, middles, active)
                singleton = np.zeros((len(active), len(middles)), dtype=bool)
                for row_index, cloud_index in enumerate(active):
                    singleton[row_index] = evaluator.status_batch(flat, missile_index, middles, (cloud_index,))
                modes: list[str] = []
                coverers: list[str] = []
                for cell_index, valid in enumerate(joint):
                    single_labels = tuple(labels[active[row]] for row in np.flatnonzero(singleton[:, cell_index]))
                    if not valid:
                        modes.append("unclassified_numerical_gap")
                    elif single_labels:
                        modes.append("single_cloud_relay")
                    else:
                        modes.append("spatial_joint")
                    coverers.append(";".join(single_labels))
                _append_mode_runs(local_rows, missile, edges, modes, coverers)
        rows.extend(local_rows)
        single_duration = sum(row["duration_s"] for row in local_rows if row["mode"] == "single_cloud_relay")
        spatial_duration = sum(row["duration_s"] for row in local_rows if row["mode"] == "spatial_joint")
        gap_duration = sum(row["duration_s"] for row in local_rows if row["mode"] == "unclassified_numerical_gap")
        summary.append(
            {
                "missile": missile,
                "joint_duration_s": result.missile_durations[missile_index],
                "single_cloud_relay_s": single_duration,
                "spatial_joint_s": spatial_duration,
                "numerical_gap_s": gap_duration,
                "partition_error_s": single_duration + spatial_duration + gap_duration - result.missile_durations[missile_index],
            }
        )
    return rows, summary


def _m3_reachability_rows(single_library: Path, evaluator: MultiMissileEvaluator) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = {name: [] for name in UAV_NAMES}
    with single_library.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["missile"] == "M3":
                grouped[row["uav"]].append(row)
    rows: list[dict[str, object]] = []
    for uav_index, name in enumerate(UAV_NAMES):
        candidate = min(grouped[name], key=lambda row: int(row["rank"]))
        plan = UavPlan(
            name,
            tuple(float(value) for value in UAV_INITIALS[uav_index]),
            float(np.radians(float(candidate["heading_deg"]))),
            float(candidate["speed_mps"]),
            (BombPlan(float(candidate["release_time_s"]), float(candidate["fuse_delay_s"])),),
        )
        duration = evaluator.evaluate(Strategy((plan,))).missile_durations[2]
        rows.append(
            {
                "uav": name,
                "initial_y_m": UAV_INITIALS[uav_index][1],
                "initial_z_m": UAV_INITIALS[uav_index][2],
                "fuse_delay_limit_s": FUSE_DELAY_LIMITS[uav_index],
                "library_rank": candidate["rank"],
                "heading_deg": candidate["heading_deg"],
                "speed_mps": candidate["speed_mps"],
                "burst_time_s": candidate["burst_time_s"],
                "M3_library_candidate_s": duration,
            }
        )
    return rows


def run(best_path: Path, output_dir: Path, grid: tuple[int, int, int, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    strategy = Strategy(load_best_plans(best_path))
    evaluator = MultiMissileEvaluator(*grid, time_batch_size=64)
    baseline = evaluator.evaluate(strategy)
    if not baseline.feasible:
        raise RuntimeError(f"Final strategy is infeasible: {baseline.reason}")
    bomb_rows, uav_rows = _ablation_rows(strategy, evaluator)
    mode_rows, mode_summary = occlusion_mode_rows(strategy, evaluator)
    reachability_rows = _m3_reachability_rows(Q5_DIR / "q5_single_library.csv", evaluator)
    write_csv(output_dir / "q5_cg_ablation_bombs.csv", bomb_rows)
    write_csv(output_dir / "q5_cg_ablation_uavs.csv", uav_rows)
    write_csv(output_dir / "q5_cg_occlusion_modes.csv", mode_rows)
    write_csv(output_dir / "q5_cg_occlusion_mode_summary.csv", mode_summary)
    write_csv(output_dir / "q5_cg_reachability_m3.csv", reachability_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Question 5 ablation and occlusion-mechanism analysis")
    parser.add_argument("--best", type=Path, default=Q5_DIR / "q5_cg_best_solution.csv")
    parser.add_argument("--output-dir", type=Path, default=Q5_DIR)
    parser.add_argument("--grid", default="720,21,15,0.005")
    args = parser.parse_args()
    values = tuple(float(value) for value in args.grid.split(","))
    if len(values) != 4:
        raise ValueError("--grid requires angle,height,radial,time_step")
    grid = (int(values[0]), int(values[1]), int(values[2]), values[3])
    run(args.best.resolve(), args.output_dir.resolve(), grid)


if __name__ == "__main__":
    main()
