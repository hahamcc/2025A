"""Independent Q4 experiment: block refinement + exploratory/final DE.

This program deliberately does not modify ``Q4/`` or the existing baseline.
It uses the same 12-dimensional physical model and the same complete-cylinder
joint objective as the baseline, but changes only the search schedule:

    baseline seeds -> rand1bin exploration -> three 4-D block refinements
    -> best1bin joint refinement -> the original multi-resolution verification.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np
from scipy.optimize import differential_evolution

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Q4 import q4_main as base  # noqa: E402
from mode_seed_factory import PopulationBundle, make_baseline_population  # noqa: E402
import q4_mode_hybrid_main as io  # noqa: E402


@dataclass(frozen=True)
class StageProfile:
    name: str
    explore_maxiter: int
    block_cycles: int
    block_population: int
    block_maxiter: int
    refine_maxiter: int


STAGE_PROFILES = {
    "quick": StageProfile("quick", 12, 1, 16, 8, 16),
    # Per seed: 180*72 + 2*3*31*24 + 350*72 = 42,768 evaluations at most,
    # essentially the same order as a 600-generation baseline (43,272).
    "standard": StageProfile("standard", 180, 2, 24, 30, 350),
}


@dataclass(frozen=True)
class PhaseOutcome:
    name: str
    best_vector: np.ndarray
    best_result: base.Evaluation
    population: np.ndarray
    energies: np.ndarray
    history: tuple[float, ...]
    elapsed_seconds: float
    evaluations: int
    iterations: int
    strategy: str


class BlockObjective:
    """Optimise one UAV's four variables while holding the other eight fixed."""

    def __init__(self, objective: base.Objective, context: np.ndarray, uav_index: int) -> None:
        self.objective = objective
        self.context = np.asarray(context, dtype=float).copy()
        self.uav_index = int(uav_index)

    def full_vector(self, values: np.ndarray) -> np.ndarray:
        vector = self.context.copy()
        start = 4 * self.uav_index
        vector[start : start + 4] = values
        return vector

    def __call__(self, values: np.ndarray) -> float:
        return self.objective(self.full_vector(values))


def parse_seeds(text: str) -> tuple[int, ...]:
    return io.parse_seeds(text)


def _record_history(objective: base.Objective, history: list[float]):
    def callback(vector: np.ndarray, convergence: float) -> bool:
        del convergence
        duration = objective.result_for(vector).duration
        history.append(max(history[-1], duration) if history else duration)
        return False

    return callback


def run_joint_phase(
    name: str,
    evaluator: base.JointEvaluator,
    initial: np.ndarray,
    maxiter: int,
    seed: int,
    strategy: str,
    recombination: float,
    mutation: float | tuple[float, float] = (0.5, 1.0),
) -> PhaseOutcome:
    objective = base.Objective(evaluator, (0, 1, 2))
    history: list[float] = []
    started = time.perf_counter()
    result = differential_evolution(
        objective,
        bounds=base.JOINT_BOUNDS,
        strategy=strategy,
        init=np.asarray(initial, dtype=float),
        maxiter=maxiter,
        mutation=mutation,
        recombination=recombination,
        seed=seed,
        polish=False,
        tol=1.0e-5,
        atol=0.0,
        callback=_record_history(objective, history),
        updating="immediate",
        workers=1,
    )
    best = objective.result_for(result.x)
    return PhaseOutcome(
        name=name,
        best_vector=np.asarray(result.x, dtype=float),
        best_result=best,
        population=np.asarray(result.population, dtype=float),
        energies=np.asarray(result.population_energies, dtype=float),
        history=tuple(history or [best.duration]),
        elapsed_seconds=time.perf_counter() - started,
        evaluations=int(result.nfev),
        iterations=int(result.nit),
        strategy=strategy,
    )


def _block_initial(center: np.ndarray, uav_index: int, count: int, seed: int) -> np.ndarray:
    bounds = base.variable_bounds((uav_index,))
    population = base.lhs_population(bounds, count, seed)
    population[0] = center[4 * uav_index : 4 * uav_index + 4]
    return population


def run_block_refinement(
    evaluator: base.JointEvaluator,
    start: np.ndarray,
    profile: StageProfile,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, object]], int, float]:
    """Cyclic coordinate-DE. Each successful block update becomes the next context."""
    objective = base.Objective(evaluator, (0, 1, 2))
    current = np.asarray(start, dtype=float).copy()
    current_result = objective.result_for(current)
    rows: list[dict[str, object]] = []
    evaluations = 0
    elapsed = 0.0
    for cycle in range(profile.block_cycles):
        for uav_index in range(3):
            block = BlockObjective(objective, current, uav_index)
            history: list[float] = []

            def callback(values: np.ndarray, convergence: float) -> bool:
                del convergence
                trial = block.full_vector(values)
                duration = objective.result_for(trial).duration
                history.append(max(history[-1], duration) if history else duration)
                return False

            started = time.perf_counter()
            result = differential_evolution(
                block,
                bounds=base.variable_bounds((uav_index,)),
                strategy="best1bin",
                init=_block_initial(current, uav_index, profile.block_population, seed + 1000 * (cycle + 1) + uav_index),
                maxiter=profile.block_maxiter,
                mutation=(0.5, 1.0),
                recombination=0.65,
                seed=seed + 10_000 * (cycle + 1) + uav_index,
                polish=False,
                tol=1.0e-5,
                atol=0.0,
                callback=callback,
                updating="immediate",
                workers=1,
            )
            elapsed += time.perf_counter() - started
            evaluations += int(result.nfev)
            trial = block.full_vector(np.asarray(result.x, dtype=float))
            trial_result = objective.result_for(trial)
            before_duration = current_result.duration
            accepted = trial_result.duration >= current_result.duration - 1.0e-12
            if accepted:
                current, current_result = trial, trial_result
            rows.append(
                {
                    "cycle": cycle + 1,
                    "uav": base.UAV_NAMES[uav_index],
                    "before_duration_s": before_duration,
                    "after_duration_s": trial_result.duration,
                    "accepted": accepted,
                    "iterations": int(result.nit),
                    "function_evaluations": int(result.nfev),
                    "elapsed_seconds": time.perf_counter() - started,
                    "best_history_duration_s": history[-1] if history else trial_result.duration,
                }
            )
    return current, rows, evaluations, elapsed


def _jitter(vector: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lower = np.array([item[0] for item in base.JOINT_BOUNDS], dtype=float)
    upper = np.array([item[1] for item in base.JOINT_BOUNDS], dtype=float)
    scale = upper - lower
    values = np.repeat(vector[None, :], count, axis=0)
    values += rng.normal(0.0, 0.035, size=values.shape) * scale
    return np.clip(values, lower, upper)


def build_refinement_population(
    baseline: PopulationBundle,
    exploration: PhaseOutcome,
    block_best: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Mix global coverage, exploratory survivors and block-refined local variants."""
    population = np.asarray(baseline.population, dtype=float).copy()
    order = np.argsort(exploration.energies)
    top_count = min(18, len(order))
    population[:top_count] = exploration.population[order[:top_count]]
    jitter_count = min(18, len(population) - top_count)
    population[top_count : top_count + jitter_count] = _jitter(block_best, jitter_count, seed + 30_000)
    population[0] = block_best
    return population


def run_one_seed(
    base_profile: base.SearchProfile,
    stage_profile: StageProfile,
    bundle: PopulationBundle,
    seed: int,
    time_batch_size: int,
) -> tuple[base.SearchRun, list[dict[str, object]]]:
    evaluator = io.BatchedJointEvaluator(*base_profile.search_grid, time_batch_size=time_batch_size)
    exploration = run_joint_phase(
        "exploration", evaluator, bundle.population, stage_profile.explore_maxiter,
        seed, "rand1bin", 0.65,
    )
    block_best, block_rows, block_evaluations, block_elapsed = run_block_refinement(
        evaluator, exploration.best_vector, stage_profile, seed,
    )
    refinement_initial = build_refinement_population(bundle, exploration, block_best, seed)
    refinement = run_joint_phase(
        "joint_refinement", evaluator, refinement_initial, stage_profile.refine_maxiter,
        seed + 50_000, "best1bin", 0.9,
    )
    candidates = []
    for index in np.argsort(refinement.energies)[: base_profile.top_per_seed]:
        vector = refinement.population[index].copy()
        objective = base.Objective(evaluator, (0, 1, 2))
        candidates.append(base.JointCandidate(vector, objective.result_for(vector), "block_joint_population", seed))
    best = base.JointCandidate(refinement.best_vector, refinement.best_result, "block_joint_best", seed)
    cumulative: list[float] = []
    for duration in (*exploration.history, *refinement.history):
        cumulative.append(max(cumulative[-1], duration) if cumulative else duration)
    run = base.SearchRun(
        seed=seed,
        elapsed_seconds=exploration.elapsed_seconds + block_elapsed + refinement.elapsed_seconds,
        iterations=exploration.iterations + stage_profile.block_cycles * 3 * stage_profile.block_maxiter + refinement.iterations,
        evaluations=exploration.evaluations + block_evaluations + refinement.evaluations,
        success=False,
        message="Completed staged schedule; the configured phase limits were reached.",
        best=best,
        candidates=tuple(candidates),
        history=tuple(cumulative),
    )
    stage_rows = [
        {
            "seed": seed, "phase": exploration.name, "de_strategy": exploration.strategy,
            "duration_s": exploration.best_result.duration, "iterations": exploration.iterations,
            "function_evaluations": exploration.evaluations, "elapsed_seconds": exploration.elapsed_seconds,
        },
        {
            "seed": seed, "phase": "block_refinement", "de_strategy": "six 4-D best1bin blocks",
            "duration_s": base.Objective(evaluator, (0, 1, 2)).result_for(block_best).duration,
            "iterations": stage_profile.block_cycles * 3 * stage_profile.block_maxiter,
            "function_evaluations": block_evaluations, "elapsed_seconds": block_elapsed,
        },
        {
            "seed": seed, "phase": refinement.name, "de_strategy": refinement.strategy,
            "duration_s": refinement.best_result.duration, "iterations": refinement.iterations,
            "function_evaluations": refinement.evaluations, "elapsed_seconds": refinement.elapsed_seconds,
        },
    ]
    for row in block_rows:
        row["seed"] = seed
        row["phase"] = "block_detail"
    return run, [*stage_rows, *block_rows]


def write_summary(path: Path, verification: base.Evaluation, independent: float, runs: Sequence[base.SearchRun]) -> None:
    lines = [
        "# Q4 分块协同搜索运行摘要",
        "",
        "搜索流程：单机候选组合和全域 LHS → rand1bin 分散探索 → 三机四维循环微调 → best1bin 联合精修 → 完整表面复核。",
        f"- 最终联合完整遮蔽时长：`{verification.duration:.10f}` s",
        f"- 独立完整遮蔽区间并集：`{independent:.10f}` s",
        f"- 空间协同增益：`{verification.duration - independent:.10f}` s",
        "",
        "该实验的分块阶段只用于改善初值和局部结构；最终目标始终是三烟幕联合完整遮蔽时长。",
        "",
        "| Seed | 分阶段搜索时长 (s) | 总函数评价次数 | 耗时 (s) |",
        "|---:|---:|---:|---:|",
    ]
    lines.extend(f"| {run.seed} | {run.best.result.duration:.6f} | {run.evaluations} | {run.elapsed_seconds:.1f} |" for run in runs)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(base_profile: base.SearchProfile, stage_profile: StageProfile, output_dir: Path, time_batch_size: int, check_only: bool) -> None:
    base.run_regression_checks()
    libraries = tuple(base.build_single_library(base_profile, index, 20250900 + index) for index in range(3))
    bundles = [make_baseline_population(base_profile, libraries, seed) for seed in base_profile.seeds]
    if any(bundle.population.shape != (base_profile.population_size, base.DIMENSION) for bundle in bundles):
        raise AssertionError("Initial population shape is incorrect.")
    if check_only:
        print("Check passed: Q1 regression and every staged initial population is feasible.")
        return
    completed = [run_one_seed(base_profile, stage_profile, bundle, seed, time_batch_size) for bundle, seed in zip(bundles, base_profile.seeds)]
    runs = [item[0] for item in completed]
    stage_rows = [row for _, rows in completed for row in rows]
    pool = base.deduplicate(candidate for run_item in runs for candidate in (run_item.best, *run_item.candidates))
    middle = io.rerank(pool, base_profile.rerank_grid, base_profile.rerank_count, "middle_rerank", time_batch_size)
    final = io.rerank(middle, base_profile.final_grid, base_profile.final_count, "final_review", time_batch_size)
    verifier = io.BatchedJointEvaluator(*base_profile.verification_grid, time_batch_size=time_batch_size)
    verification = verifier.evaluate(final[0].result.strategy)
    independent, individual, independent_intervals = base.independent_statistics(verification.strategy, verifier)
    io.save_outputs(output_dir, "block_hybrid", base_profile, bundles, runs, verification, independent, individual, independent_intervals)
    io.write_csv(output_dir / "q4_stage_runs.csv", stage_rows)
    write_summary(output_dir / "run_summary.md", verification, independent, runs)
    print(f"Completed block-hybrid: T_joint={verification.duration:.10f} s")
    print(f"Outputs: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent Q4 block-refined staged DE experiment")
    parser.add_argument("--profile", choices=tuple(base.PROFILES), default="standard")
    parser.add_argument("--seeds", type=parse_seeds)
    parser.add_argument("--explore-iter", type=int)
    parser.add_argument("--block-cycles", type=int)
    parser.add_argument("--block-iter", type=int)
    parser.add_argument("--refine-iter", type=int)
    parser.add_argument("--time-batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default="Q4补/runs/block_hybrid_standard")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.time_batch_size < 1:
        parser.error("time-batch-size must be positive")
    if any(value is not None and value < 1 for value in (args.explore_iter, args.block_cycles, args.block_iter, args.refine_iter)):
        parser.error("Every iteration setting must be positive")
    base_profile = base.PROFILES[args.profile]
    if args.seeds is not None:
        base_profile = replace(base_profile, seeds=args.seeds)
    stage = STAGE_PROFILES[args.profile]
    stage = replace(
        stage,
        explore_maxiter=args.explore_iter or stage.explore_maxiter,
        block_cycles=args.block_cycles or stage.block_cycles,
        block_maxiter=args.block_iter or stage.block_maxiter,
        refine_maxiter=args.refine_iter or stage.refine_maxiter,
    )
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if not args.check_only:
        io.ensure_output_dir(output_dir, args.overwrite)
    run(base_profile, stage, output_dir, args.time_batch_size, args.check_only)


if __name__ == "__main__":
    main()
