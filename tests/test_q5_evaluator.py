from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Q5.q5_main import (
    BombPlan,
    DIMENSION,
    GLOBAL_HORIZON,
    GROUP_DIMENSION,
    MISSILE_IMPACT_TIMES,
    MultiMissileEvaluator,
    Strategy,
    UavPlan,
    UAV_INITIALS,
    decode_strategy,
    validate_strategy,
)


class Q5EvaluatorTest(unittest.TestCase):
    def test_decoder_enforces_three_bombs_and_release_gaps(self) -> None:
        rng = np.random.default_rng(20250950)
        for vector in rng.random((100, DIMENSION)):
            for uav_index in range(5):
                vector[uav_index * GROUP_DIMENSION] *= 2.0 * np.pi
            strategy = decode_strategy(vector)
            self.assertIsNone(validate_strategy(strategy))
            for plan in strategy.uavs:
                self.assertEqual(len(plan.bombs), 3)
                releases = [bomb.release_time for bomb in plan.bombs]
                self.assertGreaterEqual(releases[1] - releases[0], 1.0 - 1.0e-10)
                self.assertGreaterEqual(releases[2] - releases[1], 1.0 - 1.0e-10)
                self.assertLessEqual(max(bomb.burst_time for bomb in plan.bombs), GLOBAL_HORIZON + 1.0e-10)

    def test_missile_impact_order(self) -> None:
        self.assertGreater(MISSILE_IMPACT_TIMES[0], MISSILE_IMPACT_TIMES[1])
        self.assertGreater(MISSILE_IMPACT_TIMES[1], MISSILE_IMPACT_TIMES[2])

    def test_q1_regression_for_m1(self) -> None:
        strategy = Strategy((UavPlan("FY1", tuple(UAV_INITIALS[0]), math.pi, 120.0, (BombPlan(1.5, 3.6),)),))
        result = MultiMissileEvaluator(180, 9, 9, 0.02).evaluate(strategy)
        self.assertTrue(result.feasible)
        self.assertAlmostEqual(result.missile_durations[0], 1.3916426693, places=3)

    def test_all_smokes_are_available_to_every_missile(self) -> None:
        plan = UavPlan("FY1", tuple(UAV_INITIALS[0]), math.pi, 120.0, (BombPlan(1.5, 3.6),))
        result = MultiMissileEvaluator(1, 1, 1, 0.05).evaluate(Strategy((plan,)))
        self.assertEqual(len(result.missile_durations), 3)
        self.assertGreater(result.missile_durations[0], 0.0)

    def test_time_batched_status_matches_unbatched_status(self) -> None:
        strategy = Strategy((UavPlan("FY1", tuple(UAV_INITIALS[0]), math.pi, 120.0, (BombPlan(1.5, 3.6),)),))
        direct = MultiMissileEvaluator(48, 5, 5, 0.05)
        batched = MultiMissileEvaluator(48, 5, 5, 0.05, time_batch_size=3, surface_batch_size=11)
        flat = direct.flatten(strategy)
        times = np.arange(5.0, 8.0, 0.02)
        active = direct.active_indices(flat, 0, 6.0)
        self.assertTrue(active)
        np.testing.assert_array_equal(
            direct.status_batch(flat, 0, times, active),
            batched.status_batch(flat, 0, times, active),
        )
        direct_result = direct.evaluate(strategy)
        batched_result = batched.evaluate(strategy)
        self.assertEqual(direct_result.missile_intervals, batched_result.missile_intervals)

    def test_margin_sign_matches_complete_surface_status(self) -> None:
        strategy = Strategy((UavPlan("FY1", tuple(UAV_INITIALS[0]), math.pi, 120.0, (BombPlan(1.5, 3.6),)),))
        evaluator = MultiMissileEvaluator(48, 5, 5, 0.05, time_batch_size=3, surface_batch_size=11)
        flat = evaluator.flatten(strategy)
        times = np.arange(5.0, 8.0, 0.02)
        active = evaluator.active_indices(flat, 0, 6.0)
        margin = evaluator.margin_batch(flat, 0, times, active)
        status = evaluator.status_batch(flat, 0, times, active)
        np.testing.assert_array_equal(margin >= -1.0e-10, status)


if __name__ == "__main__":
    unittest.main()
