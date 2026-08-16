from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Q3.q3_main import (
    Q3_DIR,
    Candidate,
    PROFILES,
    SearchProfile,
    candidate_row,
    deduplicate_candidates,
    encode_deployment,
    final_review,
    make_candidate,
    print_final_result,
    rerank_candidates,
    save_outputs,
    UniformJointEvaluator,
    write_csv,
)
from core.multi_smoke_evaluator import ThreeDeployment


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def deployment_from_row(row: dict[str, str]) -> ThreeDeployment:
    return ThreeDeployment(
        heading=float(row["theta_rad"]),
        speed=float(row["uav_speed_mps"]),
        release_times=tuple(float(row[f"release_time_{index}_s"]) for index in range(1, 4)),
        fuse_delays=tuple(float(row[f"fuse_delay_{index}_s"]) for index in range(1, 4)),
    )


def worker_candidates(rows: Iterable[dict[str, str]], profile: SearchProfile) -> list[Candidate]:
    """只读取每个种子最终种群的前十名，重新按统一低精度口径评价。"""

    angle_count, height_count, radial_count, scan_step = profile.search_grid
    evaluator = UniformJointEvaluator(
        angle_count,
        height_count,
        radial_count,
        scan_step,
        root_tolerance=max(1.0e-5, scan_step * 1.0e-3),
    )
    candidates: list[Candidate] = []
    for row in rows:
        if row.get("record_type") != "de_population":
            continue
        deployment = deployment_from_row(row)
        result = evaluator.evaluate(deployment)
        candidates.append(
            make_candidate(
                encode_deployment(deployment),
                result,
                "distributed_de_population",
                row["seed"],
                int(row["rank"]),
            )
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总问题三并行 DE 子任务并统一复核")
    parser.add_argument(
        "worker_dirs",
        nargs="+",
        type=Path,
        help="每个子任务的输出目录，例如 Q3/distributed/me Q3/distributed/writer",
    )
    parser.add_argument("--profile", choices=("standard",), default="standard")
    parser.add_argument(
        "--output-dir", type=Path, default=Q3_DIR, help="汇总后的正式结果目录"
    )
    arguments = parser.parse_args()
    profile = PROFILES[arguments.profile]

    all_search_rows: list[dict[str, str]] = []
    all_history_rows: list[dict[str, str]] = []
    all_candidates: list[Candidate] = []
    seen_seeds: set[str] = set()
    for directory in arguments.worker_dirs:
        search_path = directory / "q3_search_runs.csv"
        history_path = directory / "q3_de_history.csv"
        if not search_path.exists() or not history_path.exists():
            parser.error(f"子任务目录缺少搜索结果或历史文件：{directory}")
        search_rows = read_csv_rows(search_path)
        history_rows = read_csv_rows(history_path)
        seeds = {row["seed"] for row in search_rows if row.get("seed")}
        duplicate = seen_seeds & seeds
        if duplicate:
            parser.error(f"不同目录中出现重复随机种子：{sorted(duplicate)}")
        seen_seeds.update(seeds)
        all_search_rows.extend(search_rows)
        all_history_rows.extend(history_rows)
        all_candidates.extend(worker_candidates(search_rows, profile))

    required = {str(seed) for seed in profile.seeds}
    if seen_seeds != required:
        parser.error(
            "Standard 配置中的全部种子必须齐全且仅出现一次；"
            f"当前={sorted(seen_seeds)}，应为={sorted(required)}"
        )
    pool = deduplicate_candidates(all_candidates)
    reranked = rerank_candidates(pool, profile)
    reviewed = final_review(reranked, profile)
    print_final_result(reviewed)

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(arguments.output_dir / "q3_search_runs.csv", all_search_rows)
    write_csv(arguments.output_dir / "q3_de_history.csv", all_history_rows)
    for path in save_outputs(profile, (), reviewed, arguments.output_dir):
        print(f"正式复核结果已保存至: {path}")


if __name__ == "__main__":
    main()
