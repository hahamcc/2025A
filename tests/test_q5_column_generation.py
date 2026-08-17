from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Q5.q5_column_generation import (
    PROFILES,
    build_master_grid,
    build_master_matrices,
    build_patch_grid,
    make_column,
    softmax_gap_decode,
    solve_balanced_master,
    solve_integer_master,
    select_diverse_refinement_candidates,
)
from Q5.q5_main import MIN_RELEASE_GAP, Strategy


class Q5ColumnGenerationTest(unittest.TestCase):
    def test_patch_radii_cover_random_side_points(self) -> None:
        patches = build_patch_grid(12, 3, 3)
        self.assertTrue(np.all(patches.radii > 0.0))
        self.assertEqual(len(patches.centers), 12 * (3 + 2 * 3))

    def test_softmax_gap_decoder_preserves_bomb_identity_and_spacing(self) -> None:
        rng = np.random.default_rng(20250817)
        for uav_index in range(5):
            for _ in range(100):
                vector = np.r_[rng.uniform(0.0, 2.0 * np.pi), rng.random(), rng.uniform(-8.0, 8.0, 4), rng.random(3)]
                plan = softmax_gap_decode(vector, uav_index)
                releases = [bomb.release_time for bomb in plan.bombs]
                self.assertGreaterEqual(releases[1] - releases[0], MIN_RELEASE_GAP - 1.0e-10)
                self.assertGreaterEqual(releases[2] - releases[1], MIN_RELEASE_GAP - 1.0e-10)

    def test_small_master_selects_one_column_per_uav(self) -> None:
        profile = PROFILES["quick"]
        tiny_profile = type(profile)(
            **{**profile.__dict__, "time_step": 2.0, "angle_cells": 6, "height_cells": 2, "radial_cells": 2}
        )
        grid = build_master_grid(tiny_profile)
        columns = []
        for uav_index in range(5):
            first = softmax_gap_decode(np.array((0.1, 0.5, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6)), uav_index)
            second = softmax_gap_decode(np.array((3.0, 0.8, 1.0, 0.0, -1.0, 0.5, 0.5, 0.5, 0.5)), uav_index)
            columns.append([make_column(uav_index, first, "test", grid), make_column(uav_index, second, "test", grid)])
        matrices = build_master_matrices(columns, grid, "center")
        solution = solve_integer_master(matrices, "center", 30.0, 0.0)
        self.assertEqual(len(solution.selected_global_indices), 5)
        self.assertTrue(all(index in group for index, group in zip(solution.selected_global_indices, matrices.column_groups)))
        balanced = solve_balanced_master(
            matrices,
            grid,
            solution.sampled_duration,
            tiny_profile.time_step,
            30.0,
            0.0,
        )
        self.assertEqual(len(balanced.selected_global_indices), 5)
        self.assertGreaterEqual(balanced.sampled_duration, solution.sampled_duration - tiny_profile.time_step - 1.0e-8)

    def test_diverse_refinement_keeps_balanced_representative(self) -> None:
        reviewed = []
        vectors = (
            (0.1, 0.2, 0.0, 0.2, -0.2, 0.0, 0.1, 0.3, 0.5),
            (1.4, 0.7, 1.0, -1.0, 0.2, 0.0, 0.3, 0.5, 0.7),
            (2.8, 0.4, -1.0, 1.0, 0.0, 0.4, 0.2, 0.4, 0.6),
            (4.2, 0.9, 0.5, 0.5, -0.5, -0.5, 0.4, 0.6, 0.8),
            (5.6, 0.5, -0.8, -0.2, 0.8, 0.2, 0.6, 0.4, 0.2),
        )
        for index, vector in enumerate(vectors):
            plans = tuple(softmax_gap_decode(np.asarray(vector), uav_index) for uav_index in range(5))
            source = "master_balanced" if index == 2 else f"candidate_{index}"
            reviewed.append(
                (source, Strategy(plans), SimpleNamespace(total_duration=10.0 - 0.1 * index, minimum_duration=2.0 + 0.1 * index, positive_count=3))
            )
        selected = select_diverse_refinement_candidates(reviewed, count=4, max_total_loss=1.0)
        self.assertEqual(len(selected), 4)
        self.assertIn("master_balanced", [item[0] for item in selected])


if __name__ == "__main__":
    unittest.main()
