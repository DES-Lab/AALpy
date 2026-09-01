import unittest

from aalpy.learning_algs.stochastic.DifferenceChecker import (
    AdvancedHoeffdingChecker,
    ChiSquareChecker,
    HoeffdingChecker,
    compute_epsilon,
)


class HoeffdingCheckerTest(unittest.TestCase):

    def test_identical_distributions_are_not_different(self):
        checker = HoeffdingChecker(alpha=0.05)
        c1 = {'x': 50, 'y': 50}
        c2 = {'x': 500, 'y': 500}
        self.assertFalse(checker.are_cells_different(c1, c2))

    def test_disjoint_output_supports_are_different(self):
        checker = HoeffdingChecker(alpha=0.05)
        self.assertTrue(checker.are_cells_different({'x': 10}, {'y': 10}))

    def test_clearly_different_ratios_with_large_samples_are_different(self):
        checker = HoeffdingChecker(alpha=0.05)
        c1 = {'x': 950, 'y': 50}
        c2 = {'x': 50, 'y': 950}
        self.assertTrue(checker.are_cells_different(c1, c2))

    def test_small_samples_with_different_ratios_can_be_indistinguishable(self):
        # With very few samples the Hoeffding bound is wide, so even a 1 vs 0 count difference
        # should not be flagged as a statistically significant difference.
        checker = HoeffdingChecker(alpha=0.05)
        self.assertFalse(checker.are_cells_different({'x': 1, 'y': 0}, {'x': 0, 'y': 1}))

    def test_empty_cells_are_not_different(self):
        checker = HoeffdingChecker(alpha=0.05)
        self.assertFalse(checker.are_cells_different({}, {}))

    def test_difference_value_and_use_diff_value_default_to_unsupported(self):
        checker = HoeffdingChecker()
        self.assertFalse(checker.use_diff_value())
        self.assertIsNone(checker.difference_value({'x': 1}, {'x': 1}))


class ComputeEpsilonTest(unittest.TestCase):

    def test_epsilon_decreases_with_more_samples(self):
        small_n = compute_epsilon(0.05, 10)
        large_n = compute_epsilon(0.05, 10000)
        self.assertGreater(small_n, large_n)

    def test_epsilon_increases_with_smaller_alpha(self):
        loose = compute_epsilon(0.5, 100)
        strict = compute_epsilon(0.01, 100)
        self.assertGreater(strict, loose)


class AdvancedHoeffdingCheckerTest(unittest.TestCase):

    def test_identical_distributions_are_not_different(self):
        checker = AdvancedHoeffdingChecker(alpha=0.05)
        self.assertFalse(checker.are_cells_different({'x': 100, 'y': 100}, {'x': 1000, 'y': 1000}))

    def test_clearly_different_ratios_are_different(self):
        checker = AdvancedHoeffdingChecker(alpha=0.05)
        self.assertTrue(checker.are_cells_different({'x': 950, 'y': 50}, {'x': 50, 'y': 950}))

    def test_use_diff_value_reflects_constructor_flag(self):
        self.assertFalse(AdvancedHoeffdingChecker(use_diff=False).use_diff_value())
        self.assertTrue(AdvancedHoeffdingChecker(use_diff=True).use_diff_value())

    def test_difference_value_is_symmetric_and_zero_for_identical_cells(self):
        checker = AdvancedHoeffdingChecker()
        cell = {'x': 10, 'y': 20}
        self.assertEqual(checker.difference_value(cell, dict(cell)), 0)

    def test_difference_value_grows_with_more_disagreement(self):
        checker = AdvancedHoeffdingChecker()
        small_diff = checker.difference_value({'x': 51, 'y': 49}, {'x': 49, 'y': 51})
        large_diff = checker.difference_value({'x': 99, 'y': 1}, {'x': 1, 'y': 99})
        self.assertLess(small_diff, large_diff)

    def test_difference_value_with_one_empty_cell_uses_epsilon_bound(self):
        checker = AdvancedHoeffdingChecker(alpha=0.05)
        value = checker.difference_value({}, {'x': 10})
        self.assertGreater(value, 0)

    def test_difference_value_both_empty_is_zero(self):
        checker = AdvancedHoeffdingChecker()
        self.assertEqual(checker.difference_value({}, {}), 0)


class ChiSquareCheckerTest(unittest.TestCase):

    def test_invalid_alpha_raises(self):
        with self.assertRaises(ValueError):
            ChiSquareChecker(alpha=0.1234)

    def test_valid_alphas_are_accepted(self):
        for alpha in (0.05, 0.01, 0.001):
            ChiSquareChecker(alpha=alpha)

    def test_identical_distributions_are_not_different(self):
        checker = ChiSquareChecker(alpha=0.05)
        self.assertFalse(checker.are_cells_different({'x': 100, 'y': 100}, {'x': 1000, 'y': 1000}))

    def test_clearly_different_ratios_are_different(self):
        checker = ChiSquareChecker(alpha=0.05)
        self.assertTrue(checker.are_cells_different({'x': 950, 'y': 50}, {'x': 50, 'y': 950}))

    def test_empty_cell_is_never_different(self):
        checker = ChiSquareChecker(alpha=0.05)
        self.assertFalse(checker.are_cells_different({}, {'x': 10}))
        self.assertFalse(checker.are_cells_different({'x': 10}, {}))

    def test_single_shared_output_key_has_zero_degrees_of_freedom_and_is_not_different(self):
        checker = ChiSquareChecker(alpha=0.05)
        self.assertFalse(checker.are_cells_different({'x': 5}, {'x': 500}))

    def test_disjoint_supports_falls_back_to_hoeffding_and_flags_large_samples_as_different(self):
        checker = ChiSquareChecker(alpha=0.05)
        self.assertTrue(checker.are_cells_different({'x': 1000}, {'y': 1000}))

    def test_use_diff_value_reflects_constructor_flag(self):
        self.assertFalse(ChiSquareChecker(use_diff_value=False).use_diff_value())
        self.assertTrue(ChiSquareChecker(use_diff_value=True).use_diff_value())

    def test_difference_value_zero_for_single_degree_of_freedom(self):
        checker = ChiSquareChecker()
        self.assertEqual(checker.difference_value({'x': 5}, {'x': 500}), 0)

    def test_difference_value_grows_with_disagreement(self):
        checker = ChiSquareChecker()
        small = checker.difference_value({'x': 51, 'y': 49}, {'x': 49, 'y': 51})
        large = checker.difference_value({'x': 99, 'y': 1}, {'x': 1, 'y': 99})
        self.assertLess(small, large)


if __name__ == '__main__':
    unittest.main()
