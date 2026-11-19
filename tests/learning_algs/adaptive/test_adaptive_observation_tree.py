import unittest

from aalpy.SULs import AutomatonSUL
from aalpy.automata import MealyMachine, MealyState
from aalpy.learning_algs.adaptive.AdaptiveObservationTree import AdaptiveObservationTree


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


def three_state_mealy_extra_letter():
    """
    3-state Mealy machine over {a, b, c}, used as a reference with only partial input overlap tests.
    s0 --a/x--> s1   s0 --b/y--> s0   s0 --c/z--> s2
    s1 --a/y--> s0   s1 --b/x--> s1   s1 --c/z--> s2
    s2 --a/z--> s2   s2 --b/z--> s2   s2 --c/z--> s2
    """
    s0 = MealyState('s0')
    s1 = MealyState('s1')
    s2 = MealyState('s2')
    s0.transitions = {'a': s1, 'b': s0, 'c': s2}
    s0.output_fun = {'a': 'x', 'b': 'y', 'c': 'z'}
    s1.transitions = {'a': s0, 'b': s1, 'c': s2}
    s1.output_fun = {'a': 'y', 'b': 'x', 'c': 'z'}
    s2.transitions = {'a': s2, 'b': s2, 'c': s2}
    s2.output_fun = {'a': 'z', 'b': 'z', 'c': 'z'}
    mm = MealyMachine(s0, [s0, s1, s2])
    mm.compute_prefixes()
    return mm


def make_tree(alphabet, sul_automaton, references, rebuilding=True, state_matching='Approximate'):
    sul = AutomatonSUL(sul_automaton)
    return AdaptiveObservationTree(alphabet, sul, references, 'mealy', None, 'SepSeq',
                                   rebuilding=rebuilding, state_matching=state_matching)


class TestNoReferences(unittest.TestCase):
    def test_empty_references_disables_state_matching(self):
        target = two_state_mealy()
        tree = make_tree(['a', 'b'], target, [], rebuilding=True, state_matching='Approximate')
        self.assertIsNone(tree.state_matching)
        self.assertEqual(tree.rebuild_states, 0)
        self.assertEqual(tree.matching_states, 0)
        self.assertEqual(tree.basis, [tree.root])

    def test_reference_with_no_overlapping_inputs_is_dropped(self):
        target = two_state_mealy()
        reference = two_state_mealy()
        # reference only defined over 'c', no overlap with tree alphabet {'a', 'b'}
        s0 = MealyState('r0')
        s0.transitions = {'c': s0}
        s0.output_fun = {'c': 'z'}
        reference_no_overlap = MealyMachine(s0, [s0])
        reference_no_overlap.compute_prefixes()

        tree = make_tree(['a', 'b'], target, [reference_no_overlap], rebuilding=False, state_matching=None)
        self.assertIsNone(tree.state_matching)
        self.assertEqual(tree.references, [])


class TestCombinedModel(unittest.TestCase):
    def test_combined_model_contains_states_of_all_references(self):
        target = two_state_mealy()
        ref1 = two_state_mealy()
        ref2 = two_state_mealy()
        tree = make_tree(['a', 'b'], target, [ref1, ref2], rebuilding=False, state_matching=None)

        self.assertEqual(len(tree.combined_model.states), 4)
        state_ids = {s.state_id for s in tree.combined_model.states}
        self.assertEqual(state_ids, {'s(0,0)', 's(0,1)', 's(1,0)', 's(1,1)'})

    def test_combined_model_preserves_output_function(self):
        target = two_state_mealy()
        reference = two_state_mealy()
        tree = make_tree(['a', 'b'], target, [reference], rebuilding=False, state_matching=None)

        combined_initial = tree.combined_model.states[0]
        self.assertEqual(combined_initial.output_fun['a'], 'x')
        self.assertEqual(combined_initial.output_fun['b'], 'y')

    def test_combined_model_restricted_to_shared_alphabet(self):
        target = two_state_mealy()
        reference = three_state_mealy_extra_letter()
        tree = make_tree(['a', 'b'], target, [reference], rebuilding=False, state_matching=None)

        for state in tree.combined_model.states:
            self.assertEqual(set(state.transitions.keys()), {'a', 'b'})

    def test_prefix_map_contains_shortest_access_sequences(self):
        target = two_state_mealy()
        reference = two_state_mealy()
        tree = make_tree(['a', 'b'], target, [reference], rebuilding=False, state_matching=None)

        prefixes = tree.prefixes_map[0]
        self.assertIn((), prefixes)
        self.assertEqual(len(prefixes), 2)

    def test_characterization_map_has_entry_per_state(self):
        target = two_state_mealy()
        reference = two_state_mealy()
        tree = make_tree(['a', 'b'], target, [reference], rebuilding=False, state_matching=None)

        self.assertEqual(len(tree.characterization_map), len(tree.combined_model.states))
        for identifiers in tree.characterization_map.values():
            self.assertTrue(len(identifiers) >= 1)

    def test_all_references_dropped_yields_no_combined_model(self):
        target = two_state_mealy()
        s0 = MealyState('r0')
        s0.transitions = {'c': s0}
        s0.output_fun = {'c': 'z'}
        reference_no_overlap = MealyMachine(s0, [s0])
        reference_no_overlap.compute_prefixes()

        sul = AutomatonSUL(target)
        tree = AdaptiveObservationTree(['a', 'b'], sul, [reference_no_overlap], 'mealy', None, 'SepSeq',
                                       rebuilding=False, state_matching='Approximate')
        self.assertIsNone(tree.combined_model)
        self.assertIsNone(tree.state_matching)


class TestFindDistinguishingSeqPartial(unittest.TestCase):
    def test_finds_witness_between_distinct_states(self):
        target = two_state_mealy()
        reference = two_state_mealy()
        tree = make_tree(['a', 'b'], target, [reference], rebuilding=False, state_matching=None)

        combined = tree.combined_model
        s0, s1 = combined.states[0], combined.states[1]
        witness = tree.find_distinguishing_seq_partial(combined, s0, s1, ['a', 'b'])
        self.assertIsNotNone(witness)

        out_from_s0 = combined.execute_sequence(s0, witness)
        out_from_s1 = combined.execute_sequence(s1, witness)
        self.assertNotEqual(out_from_s0, out_from_s1)

    def test_no_witness_for_identical_states(self):
        target = two_state_mealy()
        reference = two_state_mealy()
        tree = make_tree(['a', 'b'], target, [reference], rebuilding=False, state_matching=None)

        combined = tree.combined_model
        s0 = combined.states[0]
        witness = tree.find_distinguishing_seq_partial(combined, s0, s0, ['a', 'b'])
        self.assertIsNone(witness)


class TestRebuildObsTree(unittest.TestCase):
    def test_perfect_reference_match_rebuilds_states(self):
        target = two_state_mealy()
        reference = two_state_mealy()
        tree = make_tree(['a', 'b'], target, [reference], rebuilding=True, state_matching=None)

        self.assertGreaterEqual(tree.rebuild_states, 1)
        self.assertGreater(len(tree.basis), 1)

    def test_no_rebuilding_keeps_root_only_basis(self):
        target = two_state_mealy()
        reference = two_state_mealy()
        tree = make_tree(['a', 'b'], target, [reference], rebuilding=False, state_matching=None)

        self.assertEqual(tree.rebuild_states, 0)
        self.assertEqual(tree.basis, [tree.root])

    def test_unrelated_reference_does_not_force_bogus_rebuild(self):
        target = two_state_mealy(out_a1='p', out_b1='q', out_a2='q', out_b2='p')
        reference = two_state_mealy(out_a1='x', out_b1='y', out_a2='y', out_b2='x')
        tree = make_tree(['a', 'b'], target, [reference], rebuilding=True, state_matching=None)

        # basis states found via rebuilding must still be pairwise apart in the real SUL
        for i, s1 in enumerate(tree.basis):
            for s2 in tree.basis[i + 1:]:
                from aalpy.learning_algs.deterministic.Apartness import Apartness
                self.assertTrue(Apartness.states_are_apart(s1, s2, tree))


class TestInsertObservationAndMatching(unittest.TestCase):
    def test_insert_observation_without_matching_extends_tree(self):
        target = two_state_mealy()
        tree = make_tree(['a', 'b'], target, [], rebuilding=False, state_matching=None)
        tree.insert_observation(['a', 'b'], ['x', 'x'])
        self.assertEqual(tree.get_observation(['a', 'b']), ['x', 'x'])

    def test_insert_observation_mismatched_lengths_raises(self):
        target = two_state_mealy()
        tree = make_tree(['a', 'b'], target, [], rebuilding=False, state_matching=None)
        with self.assertRaises(ValueError):
            tree.insert_observation(['a', 'b'], ['x'])

    def test_insert_observation_with_matching_updates_best_match(self):
        target = two_state_mealy()
        reference = two_state_mealy()
        tree = make_tree(['a', 'b'], target, [reference], rebuilding=False, state_matching='Approximate')

        self.assertIn(tree.root, tree.state_matcher.matchings)
        self.assertNotIn(tree.root, tree.state_matcher.best_match)

        tree.insert_observation(['a', 'b'], ['x', 'x'])
        self.assertIn(tree.root, tree.state_matcher.best_match)
        best = tree.state_matcher.best_match[tree.root]
        self.assertEqual(len(best), 1)
        self.assertEqual(best[0].state_id, 's(0,0)')

    def test_promote_frontier_state_updates_matching_for_new_basis(self):
        target = two_state_mealy()
        reference = two_state_mealy()
        tree = make_tree(['a', 'b'], target, [reference], rebuilding=False, state_matching='Approximate')

        tree.insert_observation(['a', 'a'], ['x', 'y'])
        tree.update_frontier_and_basis()

        for basis_state in tree.basis:
            self.assertIn(basis_state, tree.state_matcher.best_match)


if __name__ == '__main__':
    unittest.main()
