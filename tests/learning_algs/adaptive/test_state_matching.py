import unittest

from aalpy.SULs import AutomatonSUL
from aalpy.automata import MealyMachine, MealyState
from aalpy.learning_algs.adaptive.AdaptiveObservationTree import AdaptiveObservationTree
from aalpy.learning_algs.adaptive.StateMatching import ApproximateStateMatching, TotalStateMatching


def two_state_mealy(out_a1='x', out_b1='y', out_a2='y', out_b2='x'):
    """
    2-state Mealy machine over {a, b}.
    s0 --a/out_a1--> s1   s0 --b/out_b1--> s0
    s1 --a/out_a2--> s0   s1 --b/out_b2--> s1
    """
    s0 = MealyState('s0')
    s1 = MealyState('s1')
    s0.transitions = {'a': s1, 'b': s0}
    s0.output_fun = {'a': out_a1, 'b': out_b1}
    s1.transitions = {'a': s0, 'b': s1}
    s1.output_fun = {'a': out_a2, 'b': out_b2}
    mm = MealyMachine(s0, [s0, s1])
    mm.compute_prefixes()
    return mm


def make_tree(target, reference, matching_type):
    sul = AutomatonSUL(target)
    return AdaptiveObservationTree(['a', 'b'], sul, [reference], 'mealy', None, 'SepSeq',
                                   rebuilding=False, state_matching=matching_type)


class TestPureHelpers(unittest.TestCase):
    def setUp(self):
        target = two_state_mealy()
        reference = two_state_mealy()
        self.tree = make_tree(target, reference, 'Approximate')
        self.matcher = self.tree.state_matcher

    def test_is_prefix_of_true(self):
        self.assertTrue(self.matcher.is_prefix_of(('a',), ('a', 'b')))
        self.assertTrue(self.matcher.is_prefix_of((), ('a', 'b')))
        self.assertTrue(self.matcher.is_prefix_of(('a', 'b'), ('a', 'b')))

    def test_is_prefix_of_false(self):
        self.assertFalse(self.matcher.is_prefix_of(('a', 'b'), ('a',)))
        self.assertFalse(self.matcher.is_prefix_of(('b',), ('a', 'b')))

    def test_find_longest_common_part_full_match(self):
        common, rest = self.matcher.find_longest_common_part(('a', 'b'), ('a', 'b'))
        self.assertEqual(common, ('a', 'b'))
        self.assertEqual(rest, ())

    def test_find_longest_common_part_partial_match(self):
        common, rest = self.matcher.find_longest_common_part(('a', 'a'), ('a', 'b', 'a'))
        self.assertEqual(common, ('a',))
        self.assertEqual(rest, ('b', 'a'))

    def test_validate_reference_input(self):
        ref_state = self.tree.combined_model.states[0]
        self.assertTrue(self.matcher.validate_reference_input((), ref_state))
        self.assertTrue(self.matcher.validate_reference_input(('a', 'b'), ref_state))
        self.assertFalse(self.matcher.validate_reference_input(('z',), ref_state))


class TestApproximateStateMatchingScoring(unittest.TestCase):
    def test_perfect_match_scores_one_for_matching_reference_state(self):
        target = two_state_mealy()
        reference = two_state_mealy()
        tree = make_tree(target, reference, 'Approximate')

        tree.insert_observation(['a', 'b'], ['x', 'x'])

        matcher = tree.state_matcher
        best = matcher.best_match[tree.root]
        self.assertEqual(matcher.best_score[tree.root], 1.0)
        self.assertEqual([s.state_id for s in best], ['s(0,0)'])

    def test_worse_match_still_ranks_below_perfect_one(self):
        target = two_state_mealy()
        reference = two_state_mealy()
        tree = make_tree(target, reference, 'Approximate')

        tree.insert_observation(['a', 'b'], ['x', 'x'])
        matcher = tree.state_matcher

        r0, r1 = tree.combined_model.states
        self.assertEqual(matcher.get_score(tree.root, r0), 1.0)
        self.assertEqual(matcher.get_score(tree.root, r1), 0.0)

    def test_completely_unrelated_sul_yields_empty_best_match(self):
        target = two_state_mealy(out_a1='p', out_b1='q', out_a2='q', out_b2='p')
        reference = two_state_mealy(out_a1='x', out_b1='y', out_a2='y', out_b2='x')
        tree = make_tree(target, reference, 'Approximate')

        tree.insert_observation(['a', 'b'], ['p', 'p'])
        matcher = tree.state_matcher

        self.assertEqual(matcher.best_score[tree.root], 0)
        self.assertEqual(matcher.best_match[tree.root], [])

    def test_get_score_is_zero_when_no_observations_made(self):
        target = two_state_mealy()
        reference = two_state_mealy()
        tree = make_tree(target, reference, 'Approximate')
        matcher = tree.state_matcher

        for ref_state in tree.combined_model.states:
            self.assertEqual(matcher.get_score(tree.root, ref_state), 0)


class TestTotalStateMatchingScoring(unittest.TestCase):
    def test_perfect_match_scores_one_and_excludes_mismatching_state(self):
        target = two_state_mealy()
        reference = two_state_mealy()
        tree = make_tree(target, reference, 'Total')

        tree.insert_observation(['a', 'b'], ['x', 'x'])
        matcher = tree.state_matcher

        best = matcher.best_match[tree.root]
        self.assertEqual([s.state_id for s in best], ['s(0,0)'])

    def test_single_mismatch_zeroes_out_total_match_permanently(self):
        target = two_state_mealy(out_a1='p', out_b1='q', out_a2='q', out_b2='p')
        reference = two_state_mealy(out_a1='x', out_b1='y', out_a2='y', out_b2='x')
        tree = make_tree(target, reference, 'Total')

        tree.insert_observation(['a'], ['p'])
        matcher = tree.state_matcher

        for ref_state in tree.combined_model.states:
            self.assertEqual(matcher.matchings[tree.root][ref_state], 0)
        self.assertEqual(matcher.best_match[tree.root], [])

    def test_add_entry_basis_dfa_style_output_matching(self):
        # Total matching for dfa/moore compares the state's own output at initialization time.
        from aalpy.automata import DfaState, Dfa
        q0 = DfaState('q0', is_accepting=True)
        q1 = DfaState('q1', is_accepting=False)
        q0.transitions = {'a': q1}
        q1.transitions = {'a': q0}
        dfa = Dfa(q0, [q0, q1])
        dfa.compute_prefixes()

        combined_accept = DfaState('ref_accept', is_accepting=True)
        combined_reject = DfaState('ref_reject', is_accepting=False)
        matcher = TotalStateMatching(['a'], None)
        matcher.combined_model = type('C', (), {'states': [combined_accept, combined_reject]})()

        basis_state = type('B', (), {'output': True})()
        matcher.add_entry_basis(basis_state, 'dfa')
        self.assertEqual(matcher.matchings[basis_state][combined_accept], 1)
        self.assertEqual(matcher.matchings[basis_state][combined_reject], 0)


if __name__ == '__main__':
    unittest.main()
