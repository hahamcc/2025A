from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Q4 import q4_main as base  # noqa: E402
import mode_seed_factory as seeds  # noqa: E402
import q4_block_hybrid_main as block  # noqa: E402
import q4_mode_hybrid_main as io  # noqa: E402


@dataclass(frozen=True)
class TwoStageProfile:
    explore_population: int
    explore_maxiter: int
    refine_population: int
    refine_maxiter: int
    middle_count: int


PROFILES = {
    "quick": TwoStageProfile(48, 20, 36, 24, 12),
    # 144*(250+1)+96*(350+1)=69,840 evaluations per seed, close to the
    # 64,872 evaluations used by the 900-generation baseline.
    "standard": TwoStageProfile(144, 250, 96, 350, 24),
}


def parse_seeds(text: str) -> tuple[int, ...]:
    return io.parse_seeds(text)


def library_vectors(libraries: Sequence[Sequence[base.SingleCandidate]], count: int, seed: int) -> list[np.ndarray]:
    combinations = list(itertools.product(*libraries))
    rng = np.random.default_rng(seed + 20_000)
    rng.shuffle(combinations)
    return [np.concatenate([item.encoded for item in combo]).astype(float) for combo in combinations[:count]]


def exploration_bundle(
    stage: TwoStageProfile,
    libraries: Sequence[Sequence[base.SingleCandidate]],
    seed: int,
) -> seeds.PopulationBundle:
    """Half diverse single-library combinations, half full-domain LHS."""
    library_count = stage.explore_population // 2
    global_count = stage.explore_population - library_count
    vectors: list[tuple[np.ndarray, str, str, bool]] = []
    for vector in library_vectors(libraries, library_count, seed):
        vectors.append((vector, "single_library", "single_library", False))
    for vector in base.lhs_population(base.JOINT_BOUNDS, global_count, seed + 10_000):
        vectors.append((vector, "global_lhs", "global_lhs", False))
    records = seeds._make_records(seed, vectors)
    return seeds.PopulationBundle(
        np.vstack([record.vector for record in records]),
        records,
        {"single_library": library_count, "global_lhs": global_count},
    )


def coarse_candidates(
    phase: block.PhaseOutcome,
    evaluator: base.JointEvaluator,
    count: int,
    seed: int,
) -> list[base.JointCandidate]:
    candidates = []
    for index in np.argsort(phase.energies)[:count]:
        vector = phase.population[index].copy()
        candidates.append(
            base.JointCandidate(vector, evaluator.evaluate(base.decode_strategy(vector)), "exploration_coarse", seed)
        )
    return candidates


def jittered_vectors(parents: Sequence[np.ndarray], count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lower = np.array([left for left, _ in base.JOINT_BOUNDS], dtype=float)
    upper = np.array([right for _, right in base.JOINT_BOUNDS], dtype=float)
    scale = upper - lower
    values = []
    for index in range(count):
        parent = np.asarray(parents[index % len(parents)], dtype=float)
        trial = parent + rng.normal(0.0, 0.025, size=parent.shape) * scale
        values.append(np.clip(trial, lower, upper))
    return np.vstack(values)


def refinement_population(
    stage: TwoStageProfile,
    middle: Sequence[base.JointCandidate],
    libraries: Sequence[Sequence[base.SingleCandidate]],
    seed: int,
) -> np.ndarray:
    """Retain medium-grid survivors while preserving local and global diversity."""
    survivor_count = min(24, len(middle), stage.refine_population // 3)
    survivors = [candidate.encoded for candidate in middle[:survivor_count]]
    if not survivors:
        raise RuntimeError("No medium-grid candidate available for joint refinement.")
    jitter_count = min(36, stage.refine_population - survivor_count)
    library_count = min(24, stage.refine_population - survivor_count - jitter_count)
    global_count = stage.refine_population - survivor_count - jitter_count - library_count
    pieces = [np.vstack(survivors), jittered_vectors(survivors, jitter_count, seed + 30_000)]
    if library_count:
        pieces.append(np.vstack(library_vectors(libraries, library_count, seed + 40_000)))
    if global_count:
        pieces.append(base.lhs_population(base.JOINT_BOUNDS, global_count, seed + 50_000))
    population = np.vstack(pieces)
    if population.shape != (stage.refine_population, base.DIMENSION):
        raise AssertionError("Refinement population shape is incorrect.")
    return population


def phase_rows(seed: int, explore: block.PhaseOutcome, refine: block.PhaseOutcome) -> list[dict[str, object]]:
    return [
        {
            "seed": seed, "phase": "diverse_exploration", "de_strategy": explore.strategy,
            "population": len(explore.population), "recombination": 0.55,
            "duration_s": explore.best_result.duration, "iterations": explore.iterations,
            "function_evaluations": explore.evaluations, "elapsed_seconds": explore.elapsed_seconds,
        },
        {
            "seed": seed, "phase": "joint_refinement", "de_strategy": refine.strategy,
            "population": len(refine.population), "recombination": 0.70,
            "duration_s": refine.best_result.duration, "iterations": refine.iterations,
            "function_evaluations": refine.evaluations, "elapsed_seconds": refine.elapsed_seconds,
        },
    ]


def run_one_seed(
    base_profile: base.SearchProfile,
    stage: TwoStageProfile,
    bundle: seeds.PopulationBundle,
    libraries: Sequence[Sequence[base.SingleCandidate]],
    seed: int,
    time_batch_size: int,
) -> tuple[base.SearchRun, list[dict[str, object]]]:
    evaluator = io.BatchedJointEvaluator(*base_profile.search_grid, time_batch_size=time_batch_size)
    explore = block.run_joint_phase(
        "diverse_exploration", evaluator, bundle.population, stage.explore_maxiter,
        seed, "rand1bin", 0.55, (0.70, 1.40),
    )
    middle = io.rerank(
        coarse_candidates(explore, evaluator, stage.middle_count, seed),
        base_profile.rerank_grid, stage.middle_count, "exploration_middle_rerank", time_batch_size,
    )
    refine_initial = refinement_population(stage, middle, libraries, seed)
    refine = block.run_joint_phase(
        "joint_refinement", evaluator, refine_initial, stage.refine_maxiter,
        seed + 50_000, "best1bin", 0.70, (0.45, 0.90),
    )
    objective = base.Objective(evaluator, (0, 1, 2))
    candidates = tuple(
        base.JointCandidate(refine.population[index].copy(), objective.result_for(refine.population[index]), "two_stage_population", seed)
        for index in np.argsort(refine.energies)[: base_profile.top_per_seed]
    )
    best = base.JointCandidate(refine.best_vector, refine.best_result, "two_stage_best", seed)
    history: list[float] = []
    for duration in (*explore.history, *refine.history):
        history.append(max(history[-1], duration) if history else duration)
    run = base.SearchRun(
        seed=seed,
        elapsed_seconds=explore.elapsed_seconds + refine.elapsed_seconds,
        iterations=explore.iterations + refine.iterations,
        evaluations=explore.evaluations + refine.evaluations,
        success=False,
        message="Completed the configured two-stage schedule.",
        best=best,
        candidates=candidates,
        history=tuple(history),
    )
    return run, phase_rows(seed, explore, refine)


def write_summary(path: Path, verification: base.Evaluation, independent: float, runs: Sequence[base.SearchRun]) -> None:
    lines = [
        "# Q4 two-stage DE summary",
        "",
        "Schedule: diverse rand1bin exploration -> medium-grid reranking -> best1bin joint refinement -> complete-surface verification.",
        f"- Joint duration: `{verification.duration:.10f}` s",
        f"- Independent-union duration: `{independent:.10f}` s",
        f"- Spatial synergy gain: `{verification.duration - independent:.10f}` s",
        "",
        "| Seed | Search duration (s) | Evaluations | Elapsed (s) |",
        "|---:|---:|---:|---:|",
    ]
    lines.extend(f"| {run.seed} | {run.best.result.duration:.6f} | {run.evaluations} | {run.elapsed_seconds:.1f} |" for run in runs)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(base_profile: base.SearchProfile, stage: TwoStageProfile, output_dir: Path, time_batch_size: int, check_only: bool) -> None:
    base.run_regression_checks()
    libraries = tuple(base.build_single_library(base_profile, index, 20250900 + index) for index in range(3))
    bundles = [exploration_bundle(stage, libraries, seed) for seed in base_profile.seeds]
    if check_only:
        print("Check passed: Q1 regression and every two-stage exploration population is feasible.")
        return
    completed = [run_one_seed(base_profile, stage, bundle, libraries, seed, time_batch_size) for bundle, seed in zip(bundles, base_profile.seeds)]
    runs = [item[0] for item in completed]
    rows = [row for _, items in completed for row in items]
    pool = base.deduplicate(candidate for run_item in runs for candidate in (run_item.best, *run_item.candidates))
    middle = io.rerank(pool, base_profile.rerank_grid, base_profile.rerank_count, "final_middle_rerank", time_batch_size)
    finals = io.rerank(middle, base_profile.final_grid, base_profile.final_count, "final_review", time_batch_size)
    verifier = io.BatchedJointEvaluator(*base_profile.verification_grid, time_batch_size=time_batch_size)
    verification = verifier.evaluate(finals[0].result.strategy)
    independent, individual, intervals = base.independent_statistics(verification.strategy, verifier)
    io.save_outputs(output_dir, "two_stage", base_profile, bundles, runs, verification, independent, individual, intervals)
    io.write_csv(output_dir / "q4_stage_runs.csv", rows)
    write_summary(output_dir / "run_summary.md", verification, independent, runs)
    print(f"Completed two-stage DE: T_joint={verification.duration:.10f} s")
    print(f"Outputs: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent Q4 two-stage differential evolution experiment")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="standard")
    parser.add_argument("--seeds", type=parse_seeds)
    parser.add_argument("--time-batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default="Q4补/runs/two_stage_standard")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.time_batch_size < 1:
        parser.error("time-batch-size must be positive")
    base_profile = base.PROFILES[args.profile]
    if args.seeds is not None:
        base_profile = replace(base_profile, seeds=args.seeds)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if not args.check_only:
        io.ensure_output_dir(output_dir, args.overwrite)
    run(base_profile, PROFILES[args.profile], output_dir, args.time_batch_size, args.check_only)


if __name__ == "__main__":
    main()
