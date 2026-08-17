from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from q5_main import (
    BombPlan,
    MultiMissileEvaluator,
    Strategy,
    UavPlan,
    UAV_INITIALS,
    UAV_NAMES,
    write_csv,
)


def load_strategy(path: Path) -> Strategy:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["uav"]].append(row)
    plans = []
    for uav_index, name in enumerate(UAV_NAMES):
        rows = sorted(grouped[name], key=lambda row: int(row["bomb"]))
        if not rows:
            continue
        bombs = tuple(
            BombPlan(float(row["release_time_s"]), float(row["fuse_delay_s"]))
            for row in rows
        )
        plans.append(
            UavPlan(
                name=name,
                initial=tuple(float(value) for value in UAV_INITIALS[uav_index]),
                heading=float(rows[0]["heading_rad"]),
                speed=float(rows[0]["speed_mps"]),
                bombs=bombs,
            )
        )
    return Strategy(tuple(plans))


def remove_bomb(strategy: Strategy, uav_index: int, bomb_index: int) -> Strategy:
    plans = []
    for current_uav, plan in enumerate(strategy.uavs):
        if current_uav != uav_index:
            plans.append(plan)
            continue
        bombs = tuple(bomb for index, bomb in enumerate(plan.bombs) if index != bomb_index)
        if bombs:
            plans.append(UavPlan(plan.name, plan.initial, plan.heading, plan.speed, bombs))
    return Strategy(tuple(plans))


def main() -> None:
    parser = argparse.ArgumentParser(description="复核既有问题五方案中每枚烟幕弹的边际贡献")
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("q5_best_solution.csv"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("q5_bomb_contributions.csv"))
    parser.add_argument("--angles", type=int, default=180)
    parser.add_argument("--heights", type=int, default=9)
    parser.add_argument("--radials", type=int, default=9)
    parser.add_argument("--step", type=float, default=0.02)
    args = parser.parse_args()
    strategy = load_strategy(args.input)
    evaluator = MultiMissileEvaluator(args.angles, args.heights, args.radials, args.step)
    base = evaluator.evaluate(strategy)
    rows = []
    for uav_index, plan in enumerate(strategy.uavs):
        for bomb_index, _ in enumerate(plan.bombs):
            reduced = evaluator.evaluate(remove_bomb(strategy, uav_index, bomb_index))
            marginal = tuple(base.missile_durations[m] - reduced.missile_durations[m] for m in range(3))
            target = max(range(3), key=lambda m: marginal[m])
            rows.append(
                {
                    "uav": plan.name,
                    "bomb": bomb_index + 1,
                    "marginal_M1_s": marginal[0],
                    "marginal_M2_s": marginal[1],
                    "marginal_M3_s": marginal[2],
                    "marginal_total_s": sum(marginal),
                    "primary_missile": f"M{target + 1}" if marginal[target] > 1.0e-6 else "",
                    "used": marginal[target] > 1.0e-6,
                }
            )
    used_keys = {(str(row["uav"]), int(row["bomb"]) - 1) for row in rows if row["used"]}
    pruned_plans = []
    for plan in strategy.uavs:
        bombs = tuple(
            bomb for bomb_index, bomb in enumerate(plan.bombs)
            if (plan.name, bomb_index) in used_keys
        )
        if bombs:
            pruned_plans.append(UavPlan(plan.name, plan.initial, plan.heading, plan.speed, bombs))
    pruned = evaluator.evaluate(Strategy(tuple(pruned_plans)))
    write_csv(args.output, rows)
    print(f"基准总时长：{base.total_duration:.10f}，分项：{base.missile_durations}")
    print(f"删除零边际弹后：{pruned.total_duration:.10f}，分项：{pruned.missile_durations}")
    for row in rows:
        print(
            f"{row['uav']}-弹{row['bomb']}: 边际总贡献{float(row['marginal_total_s']):.10f}s，"
            f"主任务{row['primary_missile'] or '未使用'}"
        )
    print(f"贡献表已保存至：{args.output}")


if __name__ == "__main__":
    main()
