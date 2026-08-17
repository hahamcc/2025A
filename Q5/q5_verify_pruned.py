from __future__ import annotations

import argparse
import csv
from pathlib import Path

from q5_column_generation import load_best_plans, save_best_solution
from q5_main import MultiMissileEvaluator, Strategy, UavPlan, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="删除零边际烟幕后执行超密完整圆柱复核")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--contributions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-output", type=Path, required=True)
    args = parser.parse_args()
    full = Strategy(load_best_plans(args.input))
    used: set[tuple[str, int]] = set()
    with args.contributions.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["used"].lower() == "true":
                used.add((row["uav"], int(row["bomb"])))
    plans = []
    for plan in full.uavs:
        bombs = tuple(
            bomb for bomb_index, bomb in enumerate(plan.bombs, start=1)
            if (plan.name, bomb_index) in used
        )
        if bombs:
            plans.append(UavPlan(plan.name, plan.initial, plan.heading, plan.speed, bombs))
    strategy = Strategy(tuple(plans))
    evaluator = MultiMissileEvaluator(720, 21, 15, 0.005)
    result = evaluator.evaluate(strategy)
    save_best_solution(args.output, result, strategy)
    interval_rows = [
        {
            "missile": f"M{index + 1}",
            "duration_s": result.missile_durations[index],
            "intervals_s": "; ".join(f"[{left:.10f}, {right:.10f}]" for left, right in result.missile_intervals[index]),
        }
        for index in range(3)
    ]
    write_csv(args.interval_output, interval_rows)
    print(f"保留烟幕弹数量：{sum(len(plan.bombs) for plan in strategy.uavs)}")
    print(f"超密分项：{result.missile_durations}")
    print(f"超密总时长：{result.total_duration:.10f}s；最差导弹：{result.minimum_duration:.10f}s")
    print(f"精简方案已保存至：{args.output}")


if __name__ == "__main__":
    main()
