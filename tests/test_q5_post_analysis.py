from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Q5.q5_main import BombPlan, MultiMissileEvaluator, Strategy, UavPlan, UAV_INITIALS
from Q5.q5_post_analysis import _ablation_rows, occlusion_mode_rows


class Q5PostAnalysisTest(unittest.TestCase):
    def test_ablation_and_mode_partition_are_recomputable(self) -> None:
        strategy = Strategy(
            (UavPlan("FY1", tuple(UAV_INITIALS[0]), math.pi, 120.0, (BombPlan(1.5, 3.6),)),)
        )
        evaluator = MultiMissileEvaluator(48, 5, 5, 0.05, time_batch_size=2)
        bomb_rows, uav_rows = _ablation_rows(strategy, evaluator)
        self.assertEqual(len(bomb_rows), 1)
        self.assertEqual(len(uav_rows), 1)
        mode_rows, summary_rows = occlusion_mode_rows(strategy, evaluator)
        self.assertTrue(mode_rows)
        for row in summary_rows:
            self.assertAlmostEqual(row["partition_error_s"], 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
