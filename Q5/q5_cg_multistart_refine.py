from __future__ import annotations

import argparse
import time
from pathlib import Path

from q5_column_generation import PROFILES, exact_rank, load_best_plans, refine_strategy, save_best_solution
from q5_main import Strategy, write_csv


Q5_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="对策略列/MILP结果执行多起点分块连续精修")
    parser.add_argument("--starts", type=int, default=3)
    args = parser.parse_args()
    if args.starts < 1:
        parser.error("starts必须为正整数。")
    profile = PROFILES["standard"]
    base_path = Q5_DIR / "q5_cg_best_solution.csv"
    base = Strategy(load_best_plans(base_path))
    candidates = [("archive_base", base)]
    history_rows = []
    started = time.perf_counter()
    for start in range(1, args.starts + 1):
        print(f"正在执行多起点分块精修{start}/{args.starts}……", flush=True)
        refined, history = refine_strategy(base, profile, 20256000 + start)
        candidates.append((f"multistart_{start}", refined))
        history_rows.extend({"start": start, **row} for row in history)
    print("正在执行180×9×9@0.02多起点重排……", flush=True)
    dense = exact_rank(candidates, (180, 9, 9, 0.02))
    print("正在执行360×13×11@0.01高密度重排……", flush=True)
    high = exact_rank([(source, strategy) for source, strategy, _ in dense], (360, 13, 11, 0.01))
    print("正在执行720×21×15@0.005超密复核……", flush=True)
    ultra = exact_rank([(source, strategy) for source, strategy, _ in high[:2]], (720, 21, 15, 0.005))
    source, strategy, result = ultra[0]
    rows = []
    for grid_name, reviewed in (("dense", dense), ("high", high), ("ultra", ultra)):
        for rank, (item_source, _, item_result) in enumerate(reviewed, start=1):
            rows.append(
                {
                    "grid": grid_name,
                    "rank": rank,
                    "source": item_source,
                    "M1_s": item_result.missile_durations[0],
                    "M2_s": item_result.missile_durations[1],
                    "M3_s": item_result.missile_durations[2],
                    "total_s": item_result.total_duration,
                    "minimum_s": item_result.minimum_duration,
                    "grid_spec": f"{item_result.grid[0]}x{item_result.grid[1]}x{item_result.grid[2]}@{item_result.grid[3]}",
                }
            )
    write_csv(Q5_DIR / "q5_cg_multistart_history.csv", history_rows)
    write_csv(Q5_DIR / "q5_cg_multistart_convergence.csv", rows)
    save_best_solution(base_path, result, strategy)
    print(f"最优档案来源：{source}")
    print(f"三导弹时长：{result.missile_durations}")
    print(f"总时长：{result.total_duration:.10f}s；最差导弹：{result.minimum_duration:.10f}s")
    print(f"最优档案已更新：{base_path}")
    print(f"运行时间：{time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
