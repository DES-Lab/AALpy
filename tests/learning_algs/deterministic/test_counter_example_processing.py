import unittest

from aalpy.SULs import AutomatonSUL
from aalpy.automata import Dfa, DfaState
from aalpy.learning_algs.deterministic.CounterExampleProcessing import (
    counterexample_successfully_processed, exponential_cex_processing, linear_cex_processing,
    longest_prefix_cex_processing, rs_cex_processing)
from aalpy.utils import get_Angluin_dfa


def wrong_two_state_hypothesis():
    """
    A deliberately wrong 2-state hypothesis for get_Angluin_dfa() that conflates the ground truth's
    q2 and q3 states with q1 (both non-accepting), used to produce genuine counterexamples for the
    cex-processing strategies below.
    h0 (accepting, prefix=())    --a--> h1   --b--> h0
    h1 (non-accepting, prefix=('a',)) --a--> h0   --b--> h1
    """
    h0 = DfaState('h0', is_accepting=True)
    h1 = DfaState('h1', is_accepting=False)
    h0.transitions = {'a': h1, 'b': h0}
    h1.transitions = {'a': h0, 'b': h1}
    h0.prefix = tuple()
    h1.prefix = ('a',)
    return Dfa(h0, [h0, h1])


class TestCounterexampleSuccessfullyProcessed(unittest.TestCase):
    def test_returns_false_while_still_a_counterexample(self):
        sul = AutomatonSUL(get_Angluin_dfa())
        hyp = wrong_two_state_hypothesis()
        self.assertFalse(counterexample_successfully_processed(sul, ('b', 'b', 'b'), hyp))

    def test_returns_true_once_outputs_agree(self):
        sul = AutomatonSUL(get_Angluin_dfa())
        hyp = wrong_two_state_hypothesis()
        # 'a' alone: dfa q0 --a--> q1 (False); hyp h0 --a--> h1 (False). Outputs already agree.
        self.assertTrue(counterexample_successfully_processed(sul, ('a',), hyp))


class TestLongestPrefixCexProcessing(unittest.TestCase):
    def test_trims_longest_matching_prefix_and_returns_suffixes(self):
        prefixes = [tuple(), ('a',), ('b',)]
        cex = ('a', 'b', 'a')
        # Longest matching prefix of cex among `prefixes` is ('a',); remaining trimmed suffix is
        # ('b', 'a'), whose own (reversed) suffixes are [('b', 'a'), ('a',)].
        result = longest_prefix_cex_processing(list(prefixes), cex, closedness='suffix')
        self.assertEqual(result, [('b', 'a'), ('a',)])

    def test_prefix_closedness_returns_prefixes_of_trimmed_suffix(self):
        prefixes = [tuple(), ('a',), ('b',)]
        cex = ('a', 'b', 'a')
        result = longest_prefix_cex_processing(list(prefixes), cex, closedness='prefix')
        self.assertEqual(result, [('b', 'a'), ('b',)])

    def test_no_matching_prefix_uses_whole_counterexample(self):
        prefixes = [('c',)]
        cex = ('a', 'b')
        result = longest_prefix_cex_processing(list(prefixes), cex, closedness='suffix')
        self.assertEqual(result, [('a', 'b'), ('b',)])


class TestRsCexProcessing(unittest.TestCase):
    def test_finds_single_distinguishing_suffix(self):
        sul = AutomatonSUL(get_Angluin_dfa())
        hyp = wrong_two_state_hypothesis()
        cex = ('b', 'b', 'b')
        suffix = rs_cex_processing(sul, cex, hyp, suffix_closedness=False)
        self.assertEqual(suffix, [('b', 'b')])

    def test_suffix_closedness_adds_all_suffixes(self):
        sul = AutomatonSUL(get_Angluin_dfa())
        hyp = wrong_two_state_hypothesis()
        cex = ('b', 'b', 'b')
        suffixes = rs_cex_processing(sul, cex, hyp, suffix_closedness=True)
        self.assertEqual(suffixes, [('b', 'b'), ('b',)])

    def test_result_is_a_genuine_suffix_of_the_counterexample(self):
        sul = AutomatonSUL(get_Angluin_dfa())
        hyp = wrong_two_state_hypothesis()
        cex = ('a', 'b', 'a')
        suffix = rs_cex_processing(sul, cex, hyp, suffix_closedness=False)[0]
        self.assertEqual(cex[len(cex) - len(suffix):], suffix)


class TestLinearCexProcessing(unittest.TestCase):
    def test_forward_and_backward_scans_can_find_different_witnesses(self):
        """
        Regression test: linear_cex_processing used to unconditionally overwrite its `direction`
        parameter with 'fwd' right after validating it (a leftover from development, visible in git
        history but never cleaned up for this function even though the analogous line was removed
        from exponential_cex_processing). This silently made cex_processing='linear_bwd' behave
        exactly like 'linear_fwd'. Fixed by deleting the stray override in CounterExampleProcessing.py.
        """
        sul = AutomatonSUL(get_Angluin_dfa())
        hyp = wrong_two_state_hypothesis()
        cex = ('b', 'b', 'b')

        forward = linear_cex_processing(sul, cex, hyp, direction='fwd', suffix_closedness=False)
        backward = linear_cex_processing(sul, cex, hyp, direction='bwd', suffix_closedness=False)

        self.assertEqual(forward, [('b', 'b')])
        self.assertEqual(backward, [('b',)])
        self.assertNotEqual(forward, backward)

    def test_invalid_direction_raises(self):
        sul = AutomatonSUL(get_Angluin_dfa())
        hyp = wrong_two_state_hypothesis()
        with self.assertRaises(AssertionError):
            linear_cex_processing(sul, ('a', 'b'), hyp, direction='sideways')


class TestExponentialCexProcessing(unittest.TestCase):
    def test_forward_scan_finds_a_valid_suffix(self):
        sul = AutomatonSUL(get_Angluin_dfa())
        hyp = wrong_two_state_hypothesis()
        cex = ('b', 'b', 'b')
        suffix = exponential_cex_processing(sul, cex, hyp, direction='fwd', suffix_closedness=False)
        self.assertEqual(suffix, [('b', 'b')])

    def test_forward_result_agrees_with_rs_processing(self):
        # Exponential search falls back to Rivest-Schapire binary search once it has bracketed the
        # divergence point, so on a fixed cex/hypothesis pair both should settle on the same suffix.
        sul = AutomatonSUL(get_Angluin_dfa())
        hyp = wrong_two_state_hypothesis()
        cex = ('a', 'b', 'a')
        rs_suffix = rs_cex_processing(sul, cex, hyp, suffix_closedness=False)
        exp_suffix = exponential_cex_processing(sul, cex, hyp, direction='fwd', suffix_closedness=False)
        self.assertEqual(rs_suffix, exp_suffix)


if __name__ == '__main__':
    unittest.main()
