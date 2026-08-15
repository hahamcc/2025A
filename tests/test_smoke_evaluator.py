"""共享评价内核和问题二全域搜索链路的回归测试。"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.smoke_evaluator import Deployment, SamplingConfig, SmokeEvaluator
from Q2.q2_main import (
    COARSE_SEED,
    MAX_BURST_TIME,
    PROFILES,
    Candidate,
    build_directed_initial_population,
    is_feasible_vector,
    lhs_feasible_vectors,
    run_coarse_search,
)


class SmokeEvaluatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = SmokeEvaluator(
            sampling=SamplingConfig(angle_count=1440, scan_step=0.05)
        )
        self.q1_deployment = Deployment(math.pi, 120.0, 1.5, 3.6)

    def test_q1_full_surface_regression(self) -> None:
        result = self.evaluator.evaluate(self.q1_deployment, mode="surface")
        self.assertTrue(result.feasible)
        self.assertEqual(len(result.intervals), 1)
        self.assertAlmostEqual(result.intervals[0][0], 8.056445489404712, places=8)
        self.assertAlmostEqual(result.intervals[0][1], 9.448088158713638, places=8)
        self.assertAlmostEqual(result.duration, 1.391642669308926, places=8)

    def test_q1_dual_rim_regression(self) -> None:
        result = self.evaluator.evaluate(self.q1_deployment, mode="rim")
        self.assertTrue(result.feasible)
        self.assertAlmostEqual(result.duration, 1.391642669308926, places=8)

    def test_margin_diagnostic_matches_full_surface_margin(self) -> None:
        simulation = self.evaluator.simulation(self.q1_deployment)
        self.assertIsNotNone(simulation)
        assert simulation is not None
        diagnostic = simulation.margin_diagnostic(8.5, mode="surface")
        self.assertAlmostEqual(
            diagnostic.margin,
            simulation.cylinder_target_margin(8.5),
            places=10,
        )
        self.assertAlmostEqual(
            diagnostic.max_distance + diagnostic.margin,
            10.0,
            places=10,
        )
        self.assertGreaterEqual(diagnostic.closest_projection, 0.0)
        self.assertLessEqual(diagnostic.closest_projection, 1.0)

    def test_q2_time_constraint_rejects_infeasible_solution(self) -> None:
        result = self.evaluator.evaluate(
            Deployment(math.pi, 120.0, 10.0, 4.0), mode="rim"
        )
        self.assertFalse(result.feasible)
        self.assertIn("13.94", result.reason)

    def test_standard_lhs_is_reproducible_and_feasible(self) -> None:
        first = lhs_feasible_vectors(PROFILES["standard"].coarse_samples, COARSE_SEED)
        second = lhs_feasible_vectors(PROFILES["standard"].coarse_samples, COARSE_SEED)
        self.assertEqual(first.shape, (8000, 4))
        self.assertTrue((first == second).all())
        self.assertTrue(all(is_feasible_vector(row) for row in first))

    def test_quick_coarse_region_centers_are_reproducible(self) -> None:
        first = run_coarse_search(PROFILES["quick"])
        second = run_coarse_search(PROFILES["quick"])
        self.assertEqual(len(first.region_centers), PROFILES["quick"].region_count)
        self.assertEqual(
            [center.deployment for center in first.region_centers],
            [center.deployment for center in second.region_centers],
        )

    def test_directed_initial_population_is_feasible_and_contains_q1_anchor(self) -> None:
        result = self.evaluator.evaluate(self.q1_deployment, mode="rim")
        center = Candidate(
            deployment=self.q1_deployment,
            source="test",
            source_seed="test",
            source_rank=1,
            low_result=result,
            region_index=1,
        )
        population = build_directed_initial_population(20250821, center, [center], 80)
        self.assertEqual(population.shape, (80, 4))
        self.assertTrue(all(is_feasible_vector(row) for row in population))
        self.assertTrue(
            any(
                abs(row[0] - math.pi) < 1.0e-12
                and abs(row[1] - 120.0) < 1.0e-12
                and abs(row[2] - 1.5) < 1.0e-12
                and abs(row[3] - 3.6) < 1.0e-12
                for row in population
            )
        )


if __name__ == "__main__":
    unittest.main()
