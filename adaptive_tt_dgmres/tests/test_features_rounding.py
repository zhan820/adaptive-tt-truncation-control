"""Small CPU checks of the spectral representation and audited local budgets."""
import math
import unittest

import numpy as np
import torch
import tntorch as tn

from adaptive_tt_dgmres.features import TT_FEATURE_NAMES, tt_interface_tokens
from adaptive_tt_dgmres.rounding import BudgetRounder, RoundingToleranceError
from adaptive_tt_dgmres.backend.paper_tt_gmres import tt_randn, tt_round_capped, tt_scale


class FeaturesRoundingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)

    @staticmethod
    def tensor(shape=(3, 4, 3)):
        return tn.Tensor(torch.randn(shape, dtype=torch.float64), eps=1e-14)

    def test_spectral_features_match_dense_cuts_and_gauge(self):
        tensor = self.tensor()
        original = [core.clone() for core in tensor.cores]
        tokens = np.asarray(tt_interface_tokens(tensor))
        self.assertEqual(tokens.shape, (2, len(TT_FEATURE_NAMES)))
        dense = tensor.torch()
        for cut in range(1, len(tensor.cores)):
            singular = torch.linalg.svdvals(dense.reshape(math.prod(dense.shape[:cut]), -1))
            self.assertAlmostEqual(tokens[cut - 1, 3], float(singular[0] ** 2 / singular.square().sum()), places=12)
        gauged = tensor.clone()
        for cut in range(len(gauged.cores) - 1):
            rank = gauged.cores[cut].shape[-1]
            gauge = torch.eye(rank, dtype=torch.float64) + 0.1 * torch.randn(rank, rank, dtype=torch.float64)
            gauged.cores[cut] = gauged.cores[cut] @ gauge
            gauged.cores[cut + 1] = torch.einsum("ab,bnc->anc", torch.linalg.inv(gauge), gauged.cores[cut + 1])
        np.testing.assert_allclose(tokens, tt_interface_tokens(gauged), atol=2e-12, rtol=2e-12)
        for old, new in zip(original, tensor.cores):
            self.assertTrue(torch.equal(old, new))

    def test_zero_and_single_mode_are_finite(self):
        for shape in [(5,), (3, 4, 2)]:
            tensor = tn.Tensor([torch.zeros((1, n, 1), dtype=torch.float64) for n in shape])
            rows = np.asarray(tt_interface_tokens(tensor))
            self.assertEqual(rows.shape, (max(1, len(shape) - 1), 10))
            self.assertTrue(np.isfinite(rows).all())
            np.testing.assert_equal(rows[:, 3:9], 0.0)
            rounder = BudgetRounder()
            rounded = rounder(tensor, 1e-8, phase="zero")
            self.assertEqual(float(rounded.torch().norm()), 0.0)
            self.assertEqual(rounder.logs[0]["actual_relative_error"], 0.0)

    def test_tracking_only_preserves_exact_backend_output(self):
        tensor = self.tensor()
        direct = tt_round_capped(tensor, 1e-4, None)
        rounder = BudgetRounder(adaptive=False)
        rounder.start_cycle(4, 0.1, 1e-4)
        tracked = rounder(tensor, 1e-4, phase="matvec")
        self.assertEqual(rounder.logs[0]["effective_eps"], 1e-4)
        for expected, actual in zip(direct.cores, tracked.cores):
            self.assertTrue(torch.equal(expected, actual))

    def test_hard_rank_cap_cannot_masquerade_as_accuracy(self):
        tensor = tn.Tensor(torch.eye(4, dtype=torch.float64), eps=1e-14)
        rounder = BudgetRounder(adaptive=False)
        with self.assertRaises(RoundingToleranceError) as caught:
            rounder(tensor, 1e-10, max_rank=1, phase="deliberate_cap")
        self.assertTrue(caught.exception.record["rank_cap_hit"])
        self.assertGreater(caught.exception.record["actual_relative_error"], 0.5)
        self.assertFalse(caught.exception.record["within_requested"])

    def test_adaptive_budget_changes_tolerance_and_tracks_actual_error(self):
        tensor = tn.Tensor(torch.diag(torch.tensor([1.0, 0.02, 0.0001], dtype=torch.float64)), eps=1e-14)
        rounder = BudgetRounder()
        rounder.start_cycle(4, 0.1, 0.1)
        rounder(tensor, 0.1, phase="matvec")
        first = rounder.logs[-1]
        self.assertAlmostEqual(first["effective_eps"], 0.01)
        self.assertGreater(first["actual_relative_error"], 0)
        rounder(tensor, 0.1, phase="orthogonalization")
        self.assertGreater(rounder.logs[-1]["effective_eps"], first["effective_eps"])
        summary = rounder.cycle_summary()
        self.assertAlmostEqual(summary["budget_total"] - summary["budget_remaining"], summary["actual_relative_error_sum"])
        self.assertFalse(summary["is_residual_certificate"])

    def test_cancellation_does_not_silently_relax_requested_accuracy(self):
        # This pure TT represents a small residual as the difference of large
        # terms. A 1e-12 relative request can be below the representational
        # rounding floor even though its absolute error is tiny.
        torch.manual_seed(4)
        large = tt_randn([5, 5, 5], 2, dtype=torch.float64)
        small = tt_randn([5, 5, 5], 2, dtype=torch.float64)
        residual = (large + tt_scale(small, 1e-4)) - large
        strict = BudgetRounder(adaptive=False)
        with self.assertRaises(RoundingToleranceError):
            strict(residual, 1e-12, phase="cycle_residual")
        failed = strict.logs[-1]
        self.assertFalse(failed["rank_cap_hit"])
        self.assertFalse(failed["measured_within_requested"])
        self.assertEqual(failed["audit_acceptance"], "rejected")
        self.assertLess(failed["actual_absolute_error"], 1e-12)

        # Using the declared algorithm tolerance is a separate, explicit
        # request; the independent solve-residual gate is not changed here.
        practical = BudgetRounder(adaptive=False)
        practical(residual, 1e-8, phase="cycle_residual")
        self.assertTrue(practical.logs[-1]["measured_within_requested"])
        self.assertEqual(practical.logs[-1]["noise_slack_relative"], failed["noise_slack_relative"])


if __name__ == "__main__":
    unittest.main()
