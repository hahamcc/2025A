"""问题三联合遮蔽内核的回归与可行性测试。"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.multi_smoke_evaluator import (
    AdaptiveSurfaceConfig,
    MultiSmokeEvaluator,
    SurfacePatch,
    ThreeDeployment,
)
from core.smoke_evaluator import ScenarioParameters
from Q3.q3_main import (
    MAX_BURST_TIME,
    UniformJointEvaluator,
    build_initial_population,
    decode_vector,
    independent_statistics,
    is_feasible_deployment,
    q2_anchor,
)


class MultiSmokeEvaluatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = ScenarioParameters(max_burst_time=MAX_BURST_TIME)

    def test_unit_vector_decodes_to_all_timing_constraints(self) -> None:
        rng = np.random.default_rng(20250830)
        for unit in rng.random((200, 6)):
            deployment = decode_vector((1.0, 100.0, *unit))
            self.assertTrue(is_feasible_deployment(deployment))
            self.assertGreaterEqual(
                deployment.release_times[1] - deployment.release_times[0], 1.0
            )
            self.assertGreaterEqual(
                deployment.release_times[2] - deployment.release_times[1], 1.0
            )
            self.assertTrue(
                all(value <= MAX_BURST_TIME + 1.0e-10 for value in deployment.burst_times)
            )

    def test_single_smoke_uniform_surface_regression_matches_q1(self) -> None:
        evaluator = MultiSmokeEvaluator(
            self.parameters,
            AdaptiveSurfaceConfig(scan_step=0.05, root_tolerance=1.0e-10),
        )
        simulation = evaluator.simulation(
            ThreeDeployment(math.pi, 120.0, (1.5,), (3.6,))
        )
        self.assertIsNotNone(simulation)
        assert simulation is not None
        review = simulation.uniform_review(1440, 41, 31)
        self.assertAlmostEqual(review.intervals[0][0], 8.056445489404712, places=8)
        self.assertAlmostEqual(review.intervals[0][1], 9.448088158713638, places=8)
        self.assertAlmostEqual(review.duration, 1.391642669308926, places=8)

    def test_joint_duration_is_not_smaller_than_independent_union(self) -> None:
        config = AdaptiveSurfaceConfig(rho_min=1.0, scan_step=0.20)
        evaluator = MultiSmokeEvaluator(self.parameters, config)
        deployment = q2_anchor()
        joint = evaluator.evaluate(deployment)
        independent, _ = independent_statistics(deployment, config)
        self.assertGreaterEqual(joint.duration + 1.0e-9, independent)

    def test_patch_rho_bounds_random_points(self) -> None:
        rng = np.random.default_rng(20250830)
        patches = (
            SurfacePatch("side", 0.2, 1.1, 1.0, 7.0),
            SurfacePatch("cap", 1.8, 2.5, 2.0, 7.0, 10.0),
        )
        x0, y0, _ = self.parameters.target_bottom_center
        for patch in patches:
            center = patch.center(self.parameters)
            for _ in range(1_000):
                phi = rng.uniform(patch.phi0, patch.phi1)
                value = rng.uniform(patch.lower, patch.upper)
                if patch.kind == "side":
                    point = np.array(
                        (
                            x0 + self.parameters.target_radius * np.cos(phi),
                            y0 + self.parameters.target_radius * np.sin(phi),
                            value,
                        )
                    )
                else:
                    point = np.array(
                        (
                            x0 + value * np.cos(phi),
                            y0 + value * np.sin(phi),
                            patch.cap_height,
                        )
                    )
                self.assertLessEqual(
                    np.linalg.norm(point - center), patch.rho_bound(self.parameters) + 1.0e-12
                )

    def test_only_active_smoke_clouds_participate(self) -> None:
        evaluator = MultiSmokeEvaluator(self.parameters, AdaptiveSurfaceConfig())
        deployment = ThreeDeployment(
            0.1, 120.0, (0.0, 2.0, 4.0), (0.5, 0.5, 0.5)
        )
        simulation = evaluator.simulation(deployment)
        self.assertIsNotNone(simulation)
        assert simulation is not None
        self.assertEqual(simulation.active_indices(0.49), ())
        self.assertEqual(simulation.active_indices(0.50), (0,))
        self.assertEqual(simulation.active_indices(2.50), (0, 1))
        self.assertEqual(simulation.active_indices(24.51), ())

    def test_uncertain_surface_is_not_marked_valid(self) -> None:
        evaluator = MultiSmokeEvaluator(
            self.parameters,
            AdaptiveSurfaceConfig(rho_min=100.0, max_depth=0, scan_step=0.2),
        )
        simulation = evaluator.simulation(
            ThreeDeployment(math.pi, 120.0, (1.5,), (3.6,))
        )
        self.assertIsNotNone(simulation)
        assert simulation is not None
        state = simulation.check_time(8.5, (0,))
        self.assertNotEqual(state.status, "valid")

    def test_initial_population_is_reproducible_and_feasible(self) -> None:
        first = build_initial_population(20250831, 80)
        second = build_initial_population(20250831, 80)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape, (80, 8))
        self.assertTrue(all(is_feasible_deployment(decode_vector(row)) for row in first))

    def test_three_smoke_uniform_grid_converges_on_external_regression_candidate(self) -> None:
        """回归参数只验证评价器，不进入搜索初始种群或最终输出。"""

        deployment = ThreeDeployment(
            heading=np.deg2rad(179.6562501381396),
            speed=139.99986918044473,
            release_times=(0.00194328774, 3.57715515, 5.53790100),
            fuse_delays=(3.61993501, 5.43525310, 6.08136076),
        )
        sparse = UniformJointEvaluator(36, 5, 5, 0.05, 1.0e-7).evaluate(
            deployment
        )
        dense = UniformJointEvaluator(180, 9, 9, 0.02, 1.0e-7).evaluate(
            deployment
        )
        self.assertGreater(sparse.duration, 7.0)
        self.assertAlmostEqual(sparse.duration, dense.duration, places=4)


if __name__ == "__main__":
    unittest.main()
