import unittest

from aalpy.SULs import AutomatonSUL
from aalpy.automata import Mdp, MdpState
from aalpy.learning_algs.stochastic.StochasticCexProcessing import stochastic_longest_prefix, stochastic_rs


class StochasticLongestPrefixTest(unittest.TestCase):

    def test_no_matching_prefix_uses_whole_cex(self):
        # mdp-style cex: output, input, output, input, output, ...
        cex = ('o0', 'a', 'o1', 'b', 'o2')
        prefixes = [('does', 'not', 'match')]
        suffixes = stochastic_longest_prefix(cex, prefixes)
        # suffixes are ordered shortest (index 0) to longest (last == the whole, untrimmed cex)
        self.assertEqual(suffixes[0], ('o2',))
        self.assertEqual(suffixes[-1], cex)
        self.assertEqual(len(suffixes), 3)

    def test_matching_prefix_trims_cex(self):
        cex = ('o0', 'a', 'o1', 'b', 'o2')
        # a prefix matching the leading input 'a' (comparing at odd indices, i.e. inputs) trims the cex down
        # to ('b', 'o2'), which (having even length) yields a single length-1 suffix.
        prefixes = [('o0', 'a', 'o1')]
        suffixes = stochastic_longest_prefix(cex, prefixes)
        self.assertEqual(suffixes, [('o2',)])

    def test_longest_matching_prefix_is_preferred(self):
        cex = ('o0', 'a', 'o1', 'b', 'o2', 'c', 'o3')
        short_prefix = ('o0', 'a', 'o1')
        long_prefix = ('o0', 'a', 'o1', 'b', 'o2')
        # the short prefix alone would trim to ('b', 'o2', 'c', 'o3') (even length -> 2 suffixes); since the
        # long prefix is tried first (prefixes are sorted by length, descending) and also matches, it wins,
        # trimming to ('c', 'o3') (even length -> a single suffix).
        suffixes_long_preferred = stochastic_longest_prefix(cex, [short_prefix, long_prefix])
        self.assertEqual(suffixes_long_preferred, [('o3',)])

        suffixes_short_only = stochastic_longest_prefix(cex, [short_prefix])
        self.assertEqual(suffixes_short_only, [('o3',), ('o2', 'c', 'o3')])

    def test_full_match_returns_empty_tuple(self):
        cex = ('o0',)
        prefixes = [('o0',)]
        result = stochastic_longest_prefix(cex, prefixes)
        self.assertEqual(result, ())

    def test_suffixes_are_all_suffixes_of_trimmed_cex_with_odd_lengths(self):
        cex = ('o0', 'a', 'o1', 'b', 'o2', 'c', 'o3')
        suffixes = stochastic_longest_prefix(cex, [])
        for suf in suffixes:
            self.assertEqual(cex[len(cex) - len(suf):], suf)
            self.assertEqual(len(suf) % 2, 1)


def _hypothesis_mdp():
    """
    2-state deterministic MDP hypothesis: s0(A) --a--> s1(B) --a--> s0(A). Prefixes are assigned in the same
    (initial_output, i1, o1, i2, o2, ...) format used by SamplingBasedObservationTable.generate_hypothesis.
    """
    s0 = MdpState('s0', output='A')
    s1 = MdpState('s1', output='B')
    s0.transitions['a'].append((s1, 1.0))
    s1.transitions['a'].append((s0, 1.0))
    s0.prefix = ('A',)
    s1.prefix = ('A', 'a', 'B')
    return Mdp(s0, [s0, s1])


def _ground_truth_sul():
    """Ground truth: single-state deterministic MDP whose output is always 'X', diverging immediately from
    the alternating A/B hypothesis above."""
    g0 = MdpState('g0', output='X')
    g0.transitions['a'].append((g0, 1.0))
    return AutomatonSUL(Mdp(g0, [g0]))


class StochasticRsTest(unittest.TestCase):

    def test_rs_returns_suffix_pinpointing_first_divergence(self):
        hypothesis = _hypothesis_mdp()
        sul = _ground_truth_sul()
        # real trace observed on the ground truth SUL (all outputs 'X'), in mdp cex format (o0, i1, o1, i2, o2)
        cex = ('X', 'a', 'X', 'a', 'X')

        suffixes = stochastic_rs(sul, cex, hypothesis)

        self.assertEqual(suffixes, [('X',), ('X', 'a', 'X')])
        for suf in suffixes:
            self.assertEqual(len(suf) % 2, 1)


if __name__ == '__main__':
    unittest.main()
