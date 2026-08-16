"""问题三：三枚烟幕联合完整遮蔽的标准差分进化求解。"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from openpyxl import load_workbook
from scipy.optimize import differential_evolution
from scipy.stats import qmc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.multi_smoke_evaluator import (  # noqa: E402
    AdaptiveSurfaceConfig,
    JointEvaluationResult,
    MultiSmokeEvaluator,
    ThreeDeployment,
    ThreeSmokeSimulation,
    TimeDiagnostic,
    UniformReview,
)
from core.smoke_evaluator import ScenarioParameters  # noqa: E402

Q3_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = PROJECT_ROOT / "Resources" / "result1.xlsx"
MAX_BURST_TIME = 13.9423
DIMENSION = 8
BOUNDS = ((0.0, 2.0 * np.pi), (70.0, 140.0), *((0.0, 1.0),) * 6)
QUICK_SEEDS = (20250831, 20250832)
STANDARD_SEEDS = (20250831, 20250832, 20250833, 20250834, 20250835)


@dataclass(frozen=True)
class SearchProfile:
    name: str
    seeds: tuple[int, ...]
    population_size: int
    maxiter: int
    adaptive: AdaptiveSurfaceConfig
    rerank_adaptive: AdaptiveSurfaceConfig
    final_adaptive: AdaptiveSurfaceConfig
    top_per_seed: int = 10
    rerank_count: int = 10
    final_count: int = 3


PROFILES = {
    "quick": SearchProfile(
        name="quick",
        seeds=QUICK_SEEDS,
        population_size=80,
        maxiter=40,
        adaptive=AdaptiveSurfaceConfig(
            rho_min=1.0, max_depth=6, max_patches=5_000, scan_step=0.20, root_tolerance=1.0e-4
        ),
        rerank_adaptive=AdaptiveSurfaceConfig(
            rho_min=0.5, max_depth=8, max_patches=20_000, scan_step=0.10, root_tolerance=1.0e-5
        ),
        final_adaptive=AdaptiveSurfaceConfig(
            rho_min=0.25, max_depth=10, max_patches=50_000, scan_step=0.05, root_tolerance=1.0e-6
        ),
        rerank_count=6,
        final_count=2,
    ),
    "standard": SearchProfile(
        name="standard",
        seeds=STANDARD_SEEDS,
        population_size=80,
        maxiter=250,
        adaptive=AdaptiveSurfaceConfig(
            rho_min=0.5, max_depth=8, max_patches=20_000, scan_step=0.10, root_tolerance=1.0e-5
        ),
        rerank_adaptive=AdaptiveSurfaceConfig(
            rho_min=0.25, max_depth=10, max_patches=50_000, scan_step=0.05, root_tolerance=1.0e-6
        ),
        final_adaptive=AdaptiveSurfaceConfig(
            rho_min=0.10, max_depth=12, max_patches=100_000, scan_step=0.02, root_tolerance=1.0e-6
        ),
    ),
}


@dataclass(frozen=True)
class Candidate:
    vector: np.ndarray
    deployment: ThreeDeployment
    result: JointEvaluationResult
    source: str
    seed: int | str
    rank: int


@dataclass(frozen=True)
class SearchRun:
    seed: int
    elapsed_seconds: float
    iterations: int
    evaluations: int
    success: bool
    message: str
    best: Candidate
    candidates: tuple[Candidate, ...]
    history: tuple[float, ...]


@dataclass(frozen=True)
class FinalReview:
    candidate: Candidate
    result: JointEvaluationResult
    independent_duration: float
    individual_durations: tuple[float, ...]
    uniform_review: UniformReview | None = None
    dense_uniform_review: UniformReview | None = None


def decode_vector(vector: Iterable[float]) -> ThreeDeployment:
    """将 DE 单位区间变量还原为严格可行的物理策略。"""

    values = tuple(float(value) for value in vector)
    if len(values) != DIMENSION:
        raise ValueError("问题三 DE 决策向量必须包含八个变量。")
    heading, speed, u1, u2, u3, u4, u5, u6 = values
    u1, u2, u3, u4, u5, u6 = (float(np.clip(value, 0.0, 1.0)) for value in (u1, u2, u3, u4, u5, u6))
    release1 = (MAX_BURST_TIME - 2.0) * u1
    slack = MAX_BURST_TIME - release1 - 2.0
    gap12 = 1.0 + slack * u2
    gap23 = 1.0 + slack * (1.0 - u2) * u3
    releases = (release1, release1 + gap12, release1 + gap12 + gap23)
    delays = tuple((MAX_BURST_TIME - release) * unit for release, unit in zip(releases, (u4, u5, u6)))
    return ThreeDeployment(
        heading=float(np.clip(heading, 0.0, 2.0 * np.pi)),
        speed=float(np.clip(speed, 70.0, 140.0)),
        release_times=tuple(float(value) for value in releases),
        fuse_delays=tuple(float(value) for value in delays),
    )


def encode_deployment(deployment: ThreeDeployment) -> np.ndarray:
    """将物理策略编码回单位区间，用于已知可行种子。"""

    if len(deployment.release_times) != 3:
        raise ValueError("问题三种子必须包含三枚烟幕弹。")
    release1, release2, release3 = deployment.release_times
    gap12, gap23 = release2 - release1, release3 - release2
    slack = MAX_BURST_TIME - release1 - 2.0
    if slack < -1.0e-10 or min(gap12, gap23) < 1.0 - 1.0e-10:
        raise ValueError("种子方案不满足问题三投放间隔。")
    u1 = release1 / (MAX_BURST_TIME - 2.0)
    u2 = 0.0 if slack <= 1.0e-12 else (gap12 - 1.0) / slack
    remaining = slack - (gap12 - 1.0)
    u3 = 0.0 if remaining <= 1.0e-12 else (gap23 - 1.0) / remaining
    units = [u1, u2, u3]
    for release, delay in zip(deployment.release_times, deployment.fuse_delays):
        units.append(0.0 if MAX_BURST_TIME - release <= 1.0e-12 else delay / (MAX_BURST_TIME - release))
    return np.array(
        [deployment.heading, deployment.speed, *np.clip(units, 0.0, 1.0)], dtype=float
    )


def is_feasible_deployment(deployment: ThreeDeployment) -> bool:
    if len(deployment.release_times) != 3:
        return False
    releases = deployment.release_times
    return (
        0.0 <= deployment.heading <= 2.0 * np.pi
        and 70.0 <= deployment.speed <= 140.0
        and releases[1] - releases[0] >= 1.0 - 1.0e-10
        and releases[2] - releases[1] >= 1.0 - 1.0e-10
        and all(release >= 0.0 and delay >= 0.0 for release, delay in zip(releases, deployment.fuse_delays))
        and all(burst <= MAX_BURST_TIME + 1.0e-10 for burst in deployment.burst_times)
    )


def build_evaluator(config: AdaptiveSurfaceConfig) -> MultiSmokeEvaluator:
    return MultiSmokeEvaluator(
        ScenarioParameters(max_burst_time=MAX_BURST_TIME), config
    )


class DurationObjective:
    def __init__(self, evaluator: MultiSmokeEvaluator) -> None:
        self.evaluator = evaluator
        self.cache: dict[tuple[float, ...], JointEvaluationResult] = {}

    @staticmethod
    def key(vector: Iterable[float]) -> tuple[float, ...]:
        return tuple(round(float(value), 12) for value in vector)

    def result_for(self, vector: Iterable[float]) -> JointEvaluationResult:
        key = self.key(vector)
        if key not in self.cache:
            self.cache[key] = self.evaluator.evaluate(decode_vector(key))
        return self.cache[key]

    def __call__(self, vector: np.ndarray) -> float:
        result = self.result_for(vector)
        return -result.duration if result.feasible else 1.0e6


def q1_anchor() -> ThreeDeployment:
    return ThreeDeployment(
        heading=math.pi,
        speed=120.0,
        release_times=(1.5, 2.5, 3.5),
        fuse_delays=(3.6, 3.6, 3.6),
    )


def q2_anchor() -> ThreeDeployment:
    return ThreeDeployment(
        heading=0.103436697315,
        speed=138.7915231228,
        release_times=(0.2944979344, 1.2944979344, 2.2944979344),
        fuse_delays=(0.5391236072, 0.5391236072, 0.5391236072),
    )


def build_initial_population(seed: int, population_size: int) -> np.ndarray:
    if population_size < 8:
        raise ValueError("DE 初始种群至少需要八个个体。")
    unit = qmc.LatinHypercube(d=DIMENSION, seed=seed).random(population_size)
    population = np.column_stack((2.0 * np.pi * unit[:, 0], 70.0 + 70.0 * unit[:, 1], unit[:, 2:]))
    anchors = [
        encode_deployment(q2_anchor()),
        encode_deployment(q1_anchor()),
    ]
    rng = np.random.default_rng(seed + 1000)
    for base in (anchors[0], anchors[1]):
        varied = base.copy()
        varied[0] = (varied[0] + rng.normal(0.0, 0.10)) % (2.0 * np.pi)
        varied[1] = np.clip(varied[1] + rng.normal(0.0, 6.0), 70.0, 140.0)
        varied[2:] = np.clip(varied[2:] + rng.normal(0.0, 0.08, 6), 0.0, 1.0)
        anchors.append(varied)
    population[: len(anchors)] = np.asarray(anchors)
    return population


def make_candidate(
    vector: Iterable[float], result: JointEvaluationResult, source: str, seed: int | str, rank: int
) -> Candidate:
    values = np.asarray(tuple(float(value) for value in vector), dtype=float)
    return Candidate(values, decode_vector(values), result, source, seed, rank)


def run_seed(profile: SearchProfile, seed: int) -> SearchRun:
    evaluator = build_evaluator(profile.adaptive)
    objective = DurationObjective(evaluator)
    history: list[float] = []

    def callback(intermediate_result) -> bool:
        current = max(0.0, -float(intermediate_result.fun))
        history.append(max(history[-1], current) if history else current)
        return False

    started = time.perf_counter()
    result = differential_evolution(
        objective,
        bounds=BOUNDS,
        strategy="best1bin",
        maxiter=profile.maxiter,
        popsize=10,
        mutation=0.7,
        recombination=0.9,
        seed=seed,
        init=build_initial_population(seed, profile.population_size),
        tol=0.01,
        atol=0.0,
        polish=False,
        updating="immediate",
        workers=1,
        callback=callback,
    )
    elapsed = time.perf_counter() - started
    population = np.asarray(result.population, dtype=float)
    energies = np.asarray(result.population_energies, dtype=float)
    order = np.argsort(energies)
    candidates = tuple(
        make_candidate(
            population[index],
            objective.result_for(population[index]),
            "de_population",
            seed,
            rank,
        )
        for rank, index in enumerate(order[: profile.top_per_seed], start=1)
    )
    best_result = objective.result_for(result.x)
    best = make_candidate(result.x, best_result, "de_best", seed, 0)
    if not history:
        history.append(best_result.duration)
    return SearchRun(
        seed=seed,
        elapsed_seconds=elapsed,
        iterations=int(result.nit),
        evaluations=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
        best=best,
        candidates=candidates,
        history=tuple(history),
    )


def candidate_key(candidate: Candidate) -> tuple[float, ...]:
    return tuple(round(float(value), 6) for value in candidate.vector)


def deduplicate_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    unique: dict[tuple[float, ...], Candidate] = {}
    for candidate in candidates:
        key = candidate_key(candidate)
        previous = unique.get(key)
        if previous is None or candidate.result.duration > previous.result.duration:
            unique[key] = candidate
    return sorted(unique.values(), key=lambda item: (-item.result.duration, candidate_key(item)))


def rerank_candidates(candidates: Sequence[Candidate], profile: SearchProfile) -> list[Candidate]:
    evaluator = build_evaluator(profile.rerank_adaptive)
    reranked = [
        make_candidate(
            candidate.vector,
            evaluator.evaluate(candidate.deployment),
            "rerank",
            candidate.seed,
            rank,
        )
        for rank, candidate in enumerate(candidates[: profile.rerank_count], start=1)
    ]
    return sorted(reranked, key=lambda item: (-item.result.duration, candidate_key(item)))


def independent_statistics(
    deployment: ThreeDeployment, config: AdaptiveSurfaceConfig
) -> tuple[float, tuple[float, ...]]:
    evaluator = build_evaluator(config)
    intervals: list[tuple[float, float]] = []
    durations: list[float] = []
    for release, delay in zip(deployment.release_times, deployment.fuse_delays):
        single = evaluator.evaluate(
            ThreeDeployment(deployment.heading, deployment.speed, (release,), (delay,))
        )
        intervals.extend(single.intervals)
        durations.append(single.duration)
    merged = ThreeSmokeSimulation.merge_intervals(intervals)
    return ThreeSmokeSimulation.total_duration(merged), tuple(durations)


def intervals_close(
    first: Sequence[tuple[float, float]], second: Sequence[tuple[float, float]], tolerance: float
) -> bool:
    return len(first) == len(second) and all(
        abs(a0 - b0) <= tolerance and abs(a1 - b1) <= tolerance
        for (a0, a1), (b0, b1) in zip(first, second)
    )


def final_review(candidates: Sequence[Candidate], profile: SearchProfile) -> list[FinalReview]:
    evaluator = build_evaluator(profile.final_adaptive)
    reviewed: list[FinalReview] = []
    for rank, candidate in enumerate(candidates[: profile.final_count], start=1):
        result = evaluator.evaluate(candidate.deployment, collect_diagnostics=True)
        independent, individual = independent_statistics(candidate.deployment, profile.final_adaptive)
        reviewed.append(
            FinalReview(
                candidate=make_candidate(candidate.vector, result, "final_adaptive", candidate.seed, rank),
                result=result,
                independent_duration=independent,
                individual_durations=individual,
            )
        )
    reviewed.sort(key=lambda item: (-item.result.duration, candidate_key(item.candidate)))
    if not reviewed:
        return reviewed
    winner = reviewed[0]
    simulation = evaluator.simulation(winner.candidate.deployment)
    assert simulation is not None
    uniform = simulation.uniform_review(720, 21, 15)
    dense: UniformReview | None = None
    if (
        abs(uniform.duration - winner.result.duration) > 0.02
        or not intervals_close(uniform.intervals, winner.result.intervals, 0.02)
    ):
        dense = simulation.uniform_review(1440, 41, 31)
    reviewed[0] = FinalReview(
        candidate=winner.candidate,
        result=winner.result,
        independent_duration=winner.independent_duration,
        individual_durations=winner.individual_durations,
        uniform_review=uniform,
        dense_uniform_review=dense,
    )
    return reviewed


def format_intervals(intervals: Sequence[tuple[float, float]]) -> str:
    return "; ".join(f"[{start:.10f}, {end:.10f}]" for start, end in intervals)


def point_columns(prefix: str, point: np.ndarray) -> dict[str, str]:
    return {f"{prefix}_{axis}_m": f"{value:.10f}" for axis, value in zip(("x", "y", "z"), point)}


def candidate_row(candidate: Candidate) -> dict[str, str]:
    deployment, result = candidate.deployment, candidate.result
    row = {
        "source": candidate.source,
        "seed": str(candidate.seed),
        "rank": str(candidate.rank),
        "theta_rad": f"{deployment.heading:.12f}",
        "theta_deg": f"{np.degrees(deployment.heading):.10f}",
        "uav_speed_mps": f"{deployment.speed:.10f}",
        "feasible": str(result.feasible),
        "reason": result.reason,
        "joint_intervals_s": format_intervals(result.intervals),
        "joint_duration_s": f"{result.duration:.10f}",
        "diagnostic_checked_patches": str(result.checked_patches),
        "diagnostic_uncertain_checks": str(result.uncertain_checks),
    }
    for index, (release, delay, burst) in enumerate(
        zip(deployment.release_times, deployment.fuse_delays, deployment.burst_times), start=1
    ):
        point_index = index - 1
        row.update(
            {
                f"release_time_{index}_s": f"{release:.10f}",
                f"fuse_delay_{index}_s": f"{delay:.10f}",
                f"burst_time_{index}_s": f"{burst:.10f}",
            }
        )
        row.update(point_columns(f"release_point_{index}", result.release_points[point_index]))
        row.update(point_columns(f"burst_point_{index}", result.burst_points[point_index]))
    return row


def write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError(f"没有可写入 {path.name} 的记录。")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_outputs(
    profile: SearchProfile,
    runs: Sequence[SearchRun],
    reviewed: Sequence[FinalReview],
    output_dir: Path = Q3_DIR,
) -> tuple[Path, ...]:
    """保存本次搜索已有的结果；分布式子任务可只传入 ``runs``。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    search_rows: list[dict[str, str]] = []
    history_rows: list[dict[str, str]] = []
    for run in runs:
        row = candidate_row(run.best)
        row.update(
            {
                "profile": profile.name,
                "record_type": "de_best",
                "elapsed_seconds": f"{run.elapsed_seconds:.6f}",
                "iterations": str(run.iterations),
                "function_evaluations": str(run.evaluations),
                "success": str(run.success),
                "message": run.message,
            }
        )
        search_rows.append(row)
        for candidate in run.candidates:
            candidate_row_data = candidate_row(candidate)
            candidate_row_data.update({"profile": profile.name, "record_type": "de_population"})
            search_rows.append(candidate_row_data)
        history_rows.extend(
            {
                "profile": profile.name,
                "seed": str(run.seed),
                "generation": str(generation),
                "best_joint_duration_s": f"{duration:.10f}",
            }
            for generation, duration in enumerate(run.history, start=1)
        )
    best_rows: list[dict[str, str]] = []
    diagnostics: list[dict[str, str]] = []
    for rank, item in enumerate(reviewed, start=1):
        row = candidate_row(item.candidate)
        row.update(
            {
                "profile": profile.name,
                "final_rank": str(rank),
                "selected": str(rank == 1),
                "independent_duration_s": f"{item.independent_duration:.10f}",
                "synergy_gain_s": f"{item.result.duration - item.independent_duration:.10f}",
                "individual_durations_s": "; ".join(f"{value:.10f}" for value in item.individual_durations),
                "uniform_duration_s": "" if item.uniform_review is None else f"{item.uniform_review.duration:.10f}",
                "uniform_intervals_s": "" if item.uniform_review is None else format_intervals(item.uniform_review.intervals),
                "dense_uniform_duration_s": "" if item.dense_uniform_review is None else f"{item.dense_uniform_review.duration:.10f}",
            }
        )
        best_rows.append(row)
        for diagnostic in item.result.diagnostics:
            diagnostic_row = {
                "final_rank": str(rank),
                "time_s": f"{diagnostic.time:.10f}",
                "status": diagnostic.status,
                "active_indices": ",".join(str(index + 1) for index in diagnostic.active_indices),
                "checked_patches": str(diagnostic.checked_patches),
                "passed_patches": str(diagnostic.passed_patches),
                "uncertain_patches": str(diagnostic.uncertain_patches),
                "witness_distance_m": "" if diagnostic.witness_distance is None else f"{diagnostic.witness_distance:.10f}",
            }
            if diagnostic.witness_point is not None:
                diagnostic_row.update(point_columns("witness_point", diagnostic.witness_point))
            diagnostics.append(diagnostic_row)
    paths: list[Path] = []
    if search_rows:
        search_path = output_dir / "q3_search_runs.csv"
        write_csv(search_path, search_rows)
        paths.append(search_path)
    if history_rows:
        history_path = output_dir / "q3_de_history.csv"
        write_csv(history_path, history_rows)
        paths.append(history_path)
    if best_rows:
        best_path = output_dir / "q3_best_solution.csv"
        diagnostics_path = output_dir / "q3_surface_diagnostics.csv"
        write_csv(best_path, best_rows)
        write_csv(diagnostics_path, diagnostics or [{"status": "no_final_candidate"}])
        paths.extend((best_path, diagnostics_path))
    if reviewed:
        paths.append(write_result_workbook(reviewed[0], output_dir))
    return tuple(paths)


def normalize_header(value: object) -> str:
    return "".join(str(value or "").split()).replace("：", ":")


def header_column(worksheet, keywords: Sequence[str]) -> int:
    for cell in worksheet[1]:
        value = normalize_header(cell.value)
        if all(keyword in value for keyword in keywords):
            return cell.column
    raise KeyError(f"模板缺少列：{'/'.join(keywords)}")


def write_result_workbook(review: FinalReview, output_dir: Path = Q3_DIR) -> Path:
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"未找到结果模板：{TEMPLATE_PATH}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "result1.xlsx"
    shutil.copy2(TEMPLATE_PATH, output)
    workbook = load_workbook(output)
    worksheet = workbook.active
    columns = {
        "heading": header_column(worksheet, ("无人机", "方向")),
        "speed": header_column(worksheet, ("无人机", "速度")),
        "number": header_column(worksheet, ("烟幕", "编号")),
        "release_x": header_column(worksheet, ("投放点", "x")),
        "release_y": header_column(worksheet, ("投放点", "y")),
        "release_z": header_column(worksheet, ("投放点", "z")),
        "burst_x": header_column(worksheet, ("起爆点", "x")),
        "burst_y": header_column(worksheet, ("起爆点", "y")),
        "burst_z": header_column(worksheet, ("起爆点", "z")),
        "duration": header_column(worksheet, ("有效", "时长")),
    }
    deployment, result = review.candidate.deployment, review.result
    for index in range(3):
        row = index + 2
        worksheet.cell(row, columns["heading"]).value = float(np.degrees(deployment.heading))
        worksheet.cell(row, columns["speed"]).value = float(deployment.speed)
        worksheet.cell(row, columns["number"]).value = index + 1
        release = result.release_points[index]
        burst = result.burst_points[index]
        worksheet.cell(row, columns["release_x"]).value = float(release[0])
        worksheet.cell(row, columns["release_y"]).value = float(release[1])
        worksheet.cell(row, columns["release_z"]).value = float(release[2])
        worksheet.cell(row, columns["burst_x"]).value = float(burst[0])
        worksheet.cell(row, columns["burst_y"]).value = float(burst[1])
        worksheet.cell(row, columns["burst_z"]).value = float(burst[2])
        worksheet.cell(row, columns["duration"]).value = float(review.individual_durations[index])
    workbook.save(output)
    return output


def solve(profile: SearchProfile) -> tuple[list[SearchRun], list[FinalReview]]:
    runs: list[SearchRun] = []
    for seed in profile.seeds:
        print(f"正在运行 DE：档位={profile.name}，随机种子={seed}...", flush=True)
        run = run_seed(profile, seed)
        runs.append(run)
        print(
            f"  完成：{run.elapsed_seconds:.1f} s，"
            f"低精度联合遮蔽={run.best.result.duration:.6f} s。",
            flush=True,
        )
    pool = deduplicate_candidates(
        candidate for run in runs for candidate in (run.best, *run.candidates)
    )
    print("正在执行中精度重排与最终完整表面复核...", flush=True)
    reranked = rerank_candidates(pool, profile)
    return runs, final_review(reranked, profile)


def print_final_result(reviewed: Sequence[FinalReview]) -> None:
    if not reviewed:
        raise RuntimeError("没有候选方案通过最终联合评价。")
    result = reviewed[0]
    deployment = result.candidate.deployment
    print("\n问题三推荐方案（自适应完整圆柱表面联合复核）")
    print(f"  航向角: {deployment.heading:.10f} rad / {np.degrees(deployment.heading):.8f}°")
    print(f"  飞行速度: {deployment.speed:.8f} m/s")
    for index, (release, delay, burst) in enumerate(
        zip(deployment.release_times, deployment.fuse_delays, deployment.burst_times), start=1
    ):
        print(f"  弹 {index}: 投放 {release:.8f} s，引爆延迟 {delay:.8f} s，起爆 {burst:.8f} s")
        print(f"       投放点 {np.array2string(result.result.release_points[index - 1], precision=5)}")
        print(f"       起爆点 {np.array2string(result.result.burst_points[index - 1], precision=5)}")
    print(f"  联合遮蔽区间: {format_intervals(result.result.intervals)} s")
    print(f"  联合遮蔽总时长: {result.result.duration:.10f} s")
    print(f"  独立遮蔽并集时长: {result.independent_duration:.10f} s")
    print(f"  协同增益: {result.result.duration - result.independent_duration:.10f} s")
    if result.uniform_review is not None:
        print(f"  720×21×15 均匀网格复核: {result.uniform_review.duration:.10f} s")
    if result.dense_uniform_review is not None:
        print(f"  1440×41×31 加密复核: {result.dense_uniform_review.duration:.10f} s")


def parse_seed_list(text: str) -> tuple[int, ...]:
    """解析 ``20250831,20250832`` 形式的独立 DE 种子列表。"""

    try:
        seeds = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("随机种子必须是逗号分隔的整数。") from error
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("至少提供一个且不能重复的随机种子。")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="问题三三烟幕联合遮蔽标准 DE 求解")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="standard")
    parser.add_argument(
        "--seeds",
        type=parse_seed_list,
        help="仅运行指定的独立 DE 种子，例如 20250831,20250832；用于多人并行。",
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="只运行 DE 并保存候选/历史，不进行重复的最终复核。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="结果目录；多人并行时必须为每人指定不同目录。",
    )
    arguments = parser.parse_args()
    profile = PROFILES[arguments.profile]
    if arguments.seeds is not None:
        invalid = set(arguments.seeds) - set(profile.seeds)
        if invalid:
            parser.error(f"指定种子不属于 {profile.name} 档：{sorted(invalid)}")
        profile = replace(profile, seeds=arguments.seeds)
    output_dir = arguments.output_dir or Q3_DIR
    started = time.perf_counter()
    if arguments.search_only:
        runs = [run_seed(profile, seed) for seed in profile.seeds]
        reviewed: list[FinalReview] = []
    else:
        runs, reviewed = solve(profile)
        print_final_result(reviewed)
    for path in save_outputs(profile, runs, reviewed, output_dir):
        print(f"结果已保存至: {path}")
    print(f"总运行时间: {time.perf_counter() - started:.2f} s")


if __name__ == "__main__":
    main()
