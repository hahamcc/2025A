"""问题二：全域 LHS 粗搜索、分区 DE 精搜索与逐级复核。"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution
from scipy.stats import qmc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.smoke_evaluator import (  # noqa: E402
    Deployment,
    EvaluationResult,
    SamplingConfig,
    ScenarioParameters,
    SmokeEvaluator,
)

Q2_DIR = Path(__file__).resolve().parent
MAX_BURST_TIME = 13.94
COARSE_SEED = 20250818
DIRECTED_SEEDS = (20250821, 20250822, 20250823, 20250824, 20250825, 20250826)
BOUNDS = (
    (0.0, 2.0 * np.pi),
    (70.0, 140.0),
    (0.0, MAX_BURST_TIME),
    (0.0, MAX_BURST_TIME),
)
Q1_ANCHORS = np.array(
    [
        (np.pi, 120.0, 1.5, 3.6),
        (np.pi - 0.06, 120.0, 1.5, 3.6),
        (np.pi + 0.06, 120.0, 1.5, 3.6),
        (np.pi, 120.0, 1.2, 3.9),
    ],
    dtype=float,
)


@dataclass(frozen=True)
class SearchProfile:
    name: str
    coarse_samples: int
    coarse_angle_count: int
    coarse_scan_step: float
    angle_count: int
    scan_step: float
    popsize: int
    maxiter: int
    region_count: int
    high_rim_count: int
    surface_count: int


PROFILES = {
    "quick": SearchProfile(
        name="quick",
        coarse_samples=1000,
        coarse_angle_count=90,
        coarse_scan_step=0.10,
        angle_count=90,
        scan_step=0.10,
        popsize=20,
        maxiter=60,
        region_count=2,
        high_rim_count=6,
        surface_count=3,
    ),
    "standard": SearchProfile(
        name="standard",
        coarse_samples=8000,
        coarse_angle_count=90,
        coarse_scan_step=0.10,
        angle_count=180,
        scan_step=0.05,
        popsize=20,
        maxiter=150,
        region_count=6,
        high_rim_count=20,
        surface_count=10,
    ),
}


@dataclass(frozen=True)
class Candidate:
    deployment: Deployment
    source: str
    source_seed: int | str
    source_rank: int
    low_result: EvaluationResult
    region_index: int | None = None


@dataclass(frozen=True)
class CoarseSearch:
    samples: tuple[Candidate, ...]
    region_centers: tuple[Candidate, ...]
    elapsed_seconds: float


@dataclass(frozen=True)
class SearchRun:
    seed: int
    region_index: int
    region_center: Candidate
    elapsed_seconds: float
    iterations: int
    evaluations: int
    success: bool
    message: str
    best_result: EvaluationResult
    candidates: tuple[Candidate, ...]
    history: tuple[float, ...]


class DurationObjective:
    """正式目标始终是 -T_eff；缓存只避免重复计算。"""

    def __init__(self, evaluator: SmokeEvaluator) -> None:
        self.evaluator = evaluator
        self._cache: dict[tuple[float, float, float, float], EvaluationResult] = {}

    @staticmethod
    def _key(vector: Iterable[float]) -> tuple[float, float, float, float]:
        values = tuple(round(float(value), 12) for value in vector)
        if len(values) != 4:
            raise ValueError("决策向量必须包含四个变量。")
        return values  # type: ignore[return-value]

    def result_for(self, vector: Iterable[float]) -> EvaluationResult:
        key = self._key(vector)
        if key not in self._cache:
            self._cache[key] = self.evaluator.evaluate(
                Deployment(*key), mode="rim"
            )
        return self._cache[key]

    def __call__(self, vector: np.ndarray) -> float:
        result = self.result_for(vector)
        return -result.duration if result.feasible else 1.0e6


def build_evaluator(angle_count: int, scan_step: float) -> SmokeEvaluator:
    return SmokeEvaluator(
        ScenarioParameters(max_burst_time=MAX_BURST_TIME),
        SamplingConfig(
            angle_count=angle_count,
            height_count=41,
            radial_count=31,
            scan_step=scan_step,
            root_tolerance=1.0e-10,
        ),
    )


def deployment_from_vector(vector: Iterable[float]) -> Deployment:
    values = tuple(float(value) for value in vector)
    if len(values) != 4:
        raise ValueError("决策向量必须包含四个变量。")
    return Deployment(*values)


def vector_from_deployment(deployment: Deployment) -> np.ndarray:
    return np.array(
        (
            deployment.heading,
            deployment.speed,
            deployment.release_time,
            deployment.fuse_delay,
        ),
        dtype=float,
    )


def lhs_feasible_vectors(sample_count: int, seed: int) -> np.ndarray:
    """在完整可行域内生成样本，后三维中的时间部分均匀覆盖三角形。"""

    if sample_count <= 0:
        raise ValueError("LHS 样本数必须为正。")
    unit = qmc.LatinHypercube(d=4, seed=seed).random(sample_count)
    triangle = unit[:, 2:4].copy()
    reflected = triangle.sum(axis=1) > 1.0
    triangle[reflected] = 1.0 - triangle[reflected]
    return np.column_stack(
        (
            2.0 * np.pi * unit[:, 0],
            70.0 + 70.0 * unit[:, 1],
            MAX_BURST_TIME * triangle[:, 0],
            MAX_BURST_TIME * triangle[:, 1],
        )
    )


def is_feasible_vector(vector: Iterable[float]) -> bool:
    heading, speed, release_time, fuse_delay = (float(value) for value in vector)
    return (
        0.0 <= heading <= 2.0 * np.pi
        and 70.0 <= speed <= 140.0
        and release_time >= 0.0
        and fuse_delay >= 0.0
        and release_time + fuse_delay <= MAX_BURST_TIME + 1.0e-12
    )


def project_times(release_time: float, fuse_delay: float) -> tuple[float, float]:
    release_time = max(0.0, release_time)
    fuse_delay = max(0.0, fuse_delay)
    total = release_time + fuse_delay
    if total > MAX_BURST_TIME:
        scale = MAX_BURST_TIME / total
        release_time *= scale
        fuse_delay *= scale
    return release_time, fuse_delay


def perturb_vector(
    vector: Iterable[float] | Deployment, rng: np.random.Generator, *, local: bool
) -> np.ndarray:
    if isinstance(vector, Deployment):
        vector = vector_from_deployment(vector)
    heading, speed, release_time, fuse_delay = (float(value) for value in vector)
    heading_scale = 0.20 if local else 0.05
    speed_scale = 12.0 if local else 3.0
    time_scale = 0.80 if local else 0.20
    release_time, fuse_delay = project_times(
        release_time + rng.normal(0.0, time_scale),
        fuse_delay + rng.normal(0.0, time_scale),
    )
    return np.array(
        (
            (heading + rng.normal(0.0, heading_scale)) % (2.0 * np.pi),
            np.clip(speed + rng.normal(0.0, speed_scale), 70.0, 140.0),
            release_time,
            fuse_delay,
        )
    )


def build_directed_initial_population(
    seed: int,
    center: Candidate,
    region_centers: Sequence[Candidate],
    population_size: int,
) -> np.ndarray:
    """构建 32+12+4+32 的可行初始种群，不改变 DE 的完整搜索边界。"""

    if population_size != 80:
        raise ValueError("正式定向 DE 的初始种群固定为 80 个个体。")
    if not region_centers:
        raise ValueError("至少需要一个区域中心。")

    rng = np.random.default_rng(seed)
    local = [vector_from_deployment(center.deployment)]
    local.extend(
        perturb_vector(center.deployment, rng, local=True) for _ in range(31)
    )
    interregional = [
        perturb_vector(region_centers[index % len(region_centers)].deployment, rng, local=False)
        for index in range(12)
    ]
    global_lhs = lhs_feasible_vectors(32, seed + 1000)
    population = np.vstack((local, interregional, Q1_ANCHORS, global_lhs))
    if population.shape != (population_size, 4):
        raise AssertionError("初始种群规模错误。")
    if not all(is_feasible_vector(row) for row in population):
        raise AssertionError("初始种群出现不可行方案。")
    return population


def candidate_key(candidate: Candidate) -> tuple[float, float, float, float]:
    deployment = candidate.deployment
    return (
        round(deployment.heading, 6),
        round(deployment.speed, 5),
        round(deployment.release_time, 5),
        round(deployment.fuse_delay, 5),
    )


def normalized_distance(first: Candidate, second: Candidate) -> float:
    a = first.deployment
    b = second.deployment
    angular_delta = abs(a.heading - b.heading) % (2.0 * np.pi)
    angular_delta = min(angular_delta, 2.0 * np.pi - angular_delta) / np.pi
    scaled = np.array(
        (
            angular_delta,
            (a.speed - b.speed) / 70.0,
            (a.release_time - b.release_time) / MAX_BURST_TIME,
            (a.fuse_delay - b.fuse_delay) / MAX_BURST_TIME,
        )
    )
    return float(np.linalg.norm(scaled))


def select_region_centers(
    samples: Sequence[Candidate], region_count: int
) -> list[Candidate]:
    """优先高遮蔽、再保证分散；不足时以最大最小距离补足。"""

    positive = sorted(
        (candidate for candidate in samples if candidate.low_result.duration > 0.0),
        key=lambda candidate: (-candidate.low_result.duration, candidate.source_rank),
    )
    preferred = [
        candidate for candidate in positive if candidate.low_result.duration >= 3.0
    ]
    pool = preferred if len(preferred) >= region_count else positive
    selected: list[Candidate] = []
    if pool:
        selected.append(pool[0])
        for candidate in pool[1:]:
            if len(selected) == region_count:
                break
            if min(normalized_distance(candidate, chosen) for chosen in selected) >= 0.30:
                selected.append(candidate)
        remaining = [candidate for candidate in pool if candidate not in selected]
        while remaining and len(selected) < region_count:
            candidate = max(
                remaining,
                key=lambda item: (
                    min(normalized_distance(item, chosen) for chosen in selected),
                    item.low_result.duration,
                ),
            )
            selected.append(candidate)
            remaining.remove(candidate)

    if len(selected) < region_count:
        fallback_evaluator = build_evaluator(90, 0.10)
        for index, vector in enumerate(Q1_ANCHORS, start=1):
            result = fallback_evaluator.evaluate(deployment_from_vector(vector), mode="rim")
            selected.append(
                Candidate(
                    deployment=result.deployment,
                    source="q1_fallback",
                    source_seed="Q1",
                    source_rank=index,
                    low_result=result,
                )
            )
            if len(selected) == region_count:
                break
    return [
        Candidate(
            deployment=candidate.deployment,
            source=candidate.source,
            source_seed=candidate.source_seed,
            source_rank=candidate.source_rank,
            low_result=candidate.low_result,
            region_index=index,
        )
        for index, candidate in enumerate(selected, start=1)
    ]


def run_coarse_search(profile: SearchProfile) -> CoarseSearch:
    evaluator = build_evaluator(profile.coarse_angle_count, profile.coarse_scan_step)
    vectors = lhs_feasible_vectors(profile.coarse_samples, COARSE_SEED)
    started = time.perf_counter()
    samples = tuple(
        Candidate(
            deployment=deployment_from_vector(vector),
            source="coarse_lhs",
            source_seed=COARSE_SEED,
            source_rank=index,
            low_result=evaluator.evaluate(deployment_from_vector(vector), mode="rim"),
        )
        for index, vector in enumerate(vectors, start=1)
    )
    centers = tuple(select_region_centers(samples, profile.region_count))
    return CoarseSearch(samples, centers, time.perf_counter() - started)


def run_q1_regression() -> None:
    deployment = Deployment(np.pi, 120.0, 1.5, 3.6)
    evaluator = build_evaluator(1440, 0.05)
    expected = (8.056445489404712, 9.448088158713638)
    for result in (
        evaluator.evaluate(deployment, mode="surface"),
        evaluator.evaluate(deployment, mode="rim"),
    ):
        if not result.feasible or len(result.intervals) != 1:
            raise AssertionError("Q1 回归失败：未得到唯一有效区间。")
        start, end = result.intervals[0]
        if abs(start - expected[0]) > 1.0e-8 or abs(end - expected[1]) > 1.0e-8:
            raise AssertionError("Q1 回归失败：区间发生变化。")
        if abs(result.duration - 1.391642669308926) > 1.0e-8:
            raise AssertionError("Q1 回归失败：时长发生变化。")


def run_seed(
    profile: SearchProfile,
    seed: int,
    evaluator: SmokeEvaluator,
    center: Candidate,
    region_centers: Sequence[Candidate],
) -> SearchRun:
    objective = DurationObjective(evaluator)
    initial_population = build_directed_initial_population(
        seed, center, region_centers, profile.popsize * len(BOUNDS)
    )
    time_constraint = NonlinearConstraint(
        lambda vector: MAX_BURST_TIME - vector[2] - vector[3], 0.0, np.inf
    )
    history: list[float] = []

    def capture_generation(intermediate_result) -> bool:
        """保存每一代结束时的历史最优遮蔽时长。"""

        current = max(0.0, -float(intermediate_result.fun))
        if history:
            current = max(history[-1], current)
        history.append(current)
        return False

    started = time.perf_counter()
    result = differential_evolution(
        objective,
        bounds=BOUNDS,
        constraints=(time_constraint,),
        seed=seed,
        popsize=profile.popsize,
        maxiter=profile.maxiter,
        init=initial_population,
        tol=0.01,
        atol=0.0,
        polish=False,
        updating="immediate",
        workers=1,
        callback=capture_generation,
    )
    elapsed_seconds = time.perf_counter() - started
    population = np.asarray(result.population, dtype=float)
    energies = np.asarray(result.population_energies, dtype=float)
    ordered = np.flatnonzero(np.isfinite(energies))
    ordered = ordered[np.argsort(energies[ordered])]
    candidates = [
        Candidate(
            deployment=(low_result := objective.result_for(population[index])).deployment,
            source="directed_de_population",
            source_seed=seed,
            source_rank=rank,
            low_result=low_result,
            region_index=center.region_index,
        )
        for rank, index in enumerate(ordered[:10], start=1)
        if (low_result := objective.result_for(population[index])).feasible
    ]
    best_result = objective.result_for(result.x)
    if not history:
        history.append(best_result.duration if best_result.feasible else 0.0)
    if best_result.feasible and all(
        candidate_key(candidate)
        != (
            round(best_result.deployment.heading, 6),
            round(best_result.deployment.speed, 5),
            round(best_result.deployment.release_time, 5),
            round(best_result.deployment.fuse_delay, 5),
        )
        for candidate in candidates
    ):
        candidates.insert(
            0,
            Candidate(
                deployment=best_result.deployment,
                source="directed_de_best",
                source_seed=seed,
                source_rank=0,
                low_result=best_result,
                region_index=center.region_index,
            ),
        )
    return SearchRun(
        seed=seed,
        region_index=center.region_index or 0,
        region_center=center,
        elapsed_seconds=elapsed_seconds,
        iterations=int(result.nit),
        evaluations=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
        best_result=best_result,
        candidates=tuple(candidates),
        history=tuple(history),
    )


def deduplicate_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    unique: dict[tuple[float, float, float, float], Candidate] = {}
    for candidate in candidates:
        key = candidate_key(candidate)
        current = unique.get(key)
        if current is None or candidate.low_result.duration > current.low_result.duration:
            unique[key] = candidate
    return sorted(
        unique.values(),
        key=lambda candidate: (-candidate.low_result.duration, candidate_key(candidate)),
    )


def rerank_candidates(
    candidates: Sequence[Candidate], high_rim_count: int
) -> list[tuple[Candidate, EvaluationResult]]:
    evaluator = build_evaluator(1440, 0.05)
    reranked = [
        (candidate, evaluator.evaluate(candidate.deployment, mode="rim"))
        for candidate in candidates[:high_rim_count]
    ]
    return sorted(reranked, key=lambda item: (-item[1].duration, candidate_key(item[0])))


def final_surface_review(
    reranked: Sequence[tuple[Candidate, EvaluationResult]], surface_count: int
) -> list[tuple[Candidate, EvaluationResult, EvaluationResult]]:
    evaluator = build_evaluator(1440, 0.05)
    reviewed = [
        (candidate, rim_result, evaluator.evaluate(candidate.deployment, mode="surface"))
        for candidate, rim_result in reranked[:surface_count]
    ]
    return sorted(reviewed, key=lambda item: (-item[2].duration, candidate_key(item[0])))


def format_intervals(intervals: tuple[tuple[float, float], ...]) -> str:
    return "; ".join(f"[{start:.10f}, {end:.10f}]" for start, end in intervals)


def vector_columns(prefix: str, vector: np.ndarray | None) -> dict[str, str]:
    if vector is None:
        return {f"{prefix}_{axis}_m": "" for axis in ("x", "y", "z")}
    return {f"{prefix}_{axis}_m": f"{value:.10f}" for axis, value in zip(("x", "y", "z"), vector)}


def result_row(
    result: EvaluationResult,
    *,
    profile: str,
    record_type: str,
    candidate: Candidate | None = None,
    elapsed_seconds: float | str = "",
    iterations: int | str = "",
    evaluations: int | str = "",
    success: bool | str = "",
    message: str = "",
) -> dict[str, str]:
    deployment = result.deployment
    row = {
        "record_type": record_type,
        "profile": profile,
        "source": "" if candidate is None else candidate.source,
        "source_seed": "" if candidate is None else str(candidate.source_seed),
        "source_rank": "" if candidate is None else str(candidate.source_rank),
        "region_index": "" if candidate is None or candidate.region_index is None else str(candidate.region_index),
        "theta_rad": f"{deployment.heading:.12f}",
        "theta_deg": f"{np.degrees(deployment.heading):.10f}",
        "uav_speed_mps": f"{deployment.speed:.10f}",
        "release_time_s": f"{deployment.release_time:.10f}",
        "fuse_delay_s": f"{deployment.fuse_delay:.10f}",
        "burst_time_s": f"{deployment.burst_time:.10f}",
        "feasible": str(result.feasible),
        "reason": result.reason,
        "valid_start_s": "" if result.valid_start is None else f"{result.valid_start:.10f}",
        "valid_end_s": "" if result.valid_end is None else f"{result.valid_end:.10f}",
        "intervals_s": format_intervals(result.intervals),
        "duration_s": f"{result.duration:.10f}",
        "elapsed_seconds": "" if elapsed_seconds == "" else f"{float(elapsed_seconds):.6f}",
        "iterations": str(iterations),
        "function_evaluations": str(evaluations),
        "success": str(success),
        "message": message,
    }
    row.update(vector_columns("release_point", result.release_point))
    row.update(vector_columns("burst_point", result.burst_point))
    return row


def write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError(f"没有可写入 {path.name} 的记录。")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def solve(
    profile: SearchProfile,
) -> tuple[CoarseSearch, list[SearchRun], list[tuple[Candidate, EvaluationResult, EvaluationResult]]]:
    run_q1_regression()
    coarse = run_coarse_search(profile)
    evaluator = build_evaluator(profile.angle_count, profile.scan_step)
    runs = [
        run_seed(profile, seed, evaluator, center, coarse.region_centers)
        for seed, center in zip(DIRECTED_SEEDS[: profile.region_count], coarse.region_centers)
    ]
    candidate_pool: list[Candidate] = list(coarse.region_centers)
    candidate_pool.extend(candidate for run in runs for candidate in run.candidates)
    reranked = rerank_candidates(
        deduplicate_candidates(candidate_pool), profile.high_rim_count
    )
    reviewed = final_surface_review(reranked, profile.surface_count)
    return coarse, runs, reviewed


def save_outputs(
    profile: SearchProfile,
    coarse: CoarseSearch,
    runs: Sequence[SearchRun],
    reviewed: Sequence[tuple[Candidate, EvaluationResult, EvaluationResult]],
) -> tuple[Path, Path, Path, Path]:
    selected_keys = {candidate_key(center): center.region_index for center in coarse.region_centers}
    coarse_rows = []
    for sample in coarse.samples:
        row = result_row(sample.low_result, profile=profile.name, record_type="coarse_lhs", candidate=sample)
        row["selected_region"] = str(selected_keys.get(candidate_key(sample), ""))
        coarse_rows.append(row)
    search_rows = []
    for run in runs:
        center_row = result_row(
            run.region_center.low_result,
            profile=profile.name,
            record_type="region_center",
            candidate=run.region_center,
        )
        center_row.update({
            "elapsed_seconds": f"{run.elapsed_seconds:.6f}",
            "iterations": str(run.iterations),
            "function_evaluations": str(run.evaluations),
            "success": str(run.success),
            "message": run.message,
        })
        search_rows.append(center_row)
        search_rows.append(
            result_row(
                run.best_result,
                profile=profile.name,
                record_type="directed_de_best",
                candidate=Candidate(
                    run.best_result.deployment,
                    "directed_de_best",
                    run.seed,
                    0,
                    run.best_result,
                    run.region_index,
                ),
                elapsed_seconds=run.elapsed_seconds,
                iterations=run.iterations,
                evaluations=run.evaluations,
                success=run.success,
                message=run.message,
            )
        )
        search_rows.extend(
            result_row(
                candidate.low_result,
                profile=profile.name,
                record_type="final_population_candidate",
                candidate=candidate,
            )
            for candidate in run.candidates
        )
    best_rows = []
    for rank, (candidate, rim_result, surface_result) in enumerate(reviewed, start=1):
        row = result_row(
            surface_result,
            profile=profile.name,
            record_type="surface_review",
            candidate=candidate,
        )
        row["surface_rank"] = str(rank)
        row["high_rim_duration_s"] = f"{rim_result.duration:.10f}"
        row["selected"] = str(rank == 1)
        best_rows.append(row)
    history_rows = []
    for run in runs:
        history_rows.extend(
            {
                "profile": profile.name,
                "region_index": str(run.region_index),
                "seed": str(run.seed),
                "generation": str(generation),
                "best_duration_s": f"{duration:.10f}",
            }
            for generation, duration in enumerate(run.history, start=1)
        )
    coarse_path = Q2_DIR / "q2_coarse_search.csv"
    search_path = Q2_DIR / "q2_search_runs.csv"
    best_path = Q2_DIR / "q2_best_solution.csv"
    history_path = Q2_DIR / "q2_de_history.csv"
    write_csv(coarse_path, coarse_rows)
    write_csv(search_path, search_rows)
    write_csv(best_path, best_rows)
    write_csv(history_path, history_rows)
    return coarse_path, search_path, best_path, history_path


def print_final_result(
    coarse: CoarseSearch,
    reviewed: Sequence[tuple[Candidate, EvaluationResult, EvaluationResult]],
) -> None:
    positive = sum(sample.low_result.duration > 0.0 for sample in coarse.samples)
    strong = sum(sample.low_result.duration >= 3.0 for sample in coarse.samples)
    print("\n全域 LHS 粗搜索")
    print(f"  样本数: {len(coarse.samples)}")
    print(f"  正遮蔽样本: {positive} ({positive / len(coarse.samples):.2%})")
    print(f"  遮蔽不少于 3 s 的样本: {strong} ({strong / len(coarse.samples):.2%})")
    print(f"  区域中心数: {len(coarse.region_centers)}")
    if not reviewed:
        raise RuntimeError("没有候选方案通过完整表面复核。")
    candidate, _, result = reviewed[0]
    deployment = candidate.deployment
    print("\n问题二最终方案（完整圆柱外表面复核）")
    print(f"  航向角: {deployment.heading:.10f} rad / {np.degrees(deployment.heading):.8f}°")
    print(f"  飞行速度: {deployment.speed:.8f} m/s")
    print(f"  投放时刻: {deployment.release_time:.8f} s")
    print(f"  引爆延时: {deployment.fuse_delay:.8f} s")
    print(f"  起爆时刻: {deployment.burst_time:.8f} s")
    print(f"  投放点: {np.array2string(result.release_point, precision=6)} m")
    print(f"  起爆点: {np.array2string(result.burst_point, precision=6)} m")
    print(f"  有效遮蔽区间: {format_intervals(result.intervals)} s")
    print(f"  总有效遮蔽时长: {result.duration:.10f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description="问题二全域 LHS + 多区域 DE 求解")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="standard")
    arguments = parser.parse_args()
    profile = PROFILES[arguments.profile]
    print("正在执行 Q1 共享内核回归检查……")
    started = time.perf_counter()
    coarse, runs, reviewed = solve(profile)
    print_final_result(coarse, reviewed)
    paths = save_outputs(profile, coarse, runs, reviewed)
    for path in paths:
        print(f"结果已保存至: {path}")
    print(f"总运行时间: {time.perf_counter() - started:.2f} s")


if __name__ == "__main__":
    main()
