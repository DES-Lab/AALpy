import unittest

from aalpy.SULs import AutomatonSUL
from aalpy.automata import Dfa, DfaState, MealyState
from aalpy.learning_algs.deterministic.Apartness import Apartness
from aalpy.learning_algs.deterministic.ObservationTree import ObservationTree
from aalpy.utils import get_Angluin_dfa


def dfa_tree():
    sul = AutomatonSUL(get_Angluin_dfa())
    return ObservationTree(['a', 'b'], sul, 'dfa', None, 'SepSeq')


def mealy_tree():
    sul = AutomatonSUL(get_Angluin_dfa())
    return ObservationTree(['a', 'b'], sul, 'mealy', None, 'SepSeq')


class TestStatesAreApartMoore(unittest.TestCase):
    def test_identical_node_is_never_apart_from_itself(self):
        tree = dfa_tree()
        self.assertFalse(Apartness.states_are_apart(tree.root, tree.root, tree))

    def test_nodes_with_same_output_and_no_observed_difference_are_not_apart(self):
        tree = dfa_tree()
        tree.insert_observation(['a'], [False])
        tree.insert_observation(['b'], [False])
        node_a = tree.get_successor(['a'])
        node_b = tree.get_successor(['b'])
        self.assertFalse(Apartness.states_are_apart(node_a, node_b, tree))

    def test_nodes_apart_via_immediate_output_difference(self):
        tree = dfa_tree()
        tree.insert_observation(['a'], [False])
        tree.insert_observation(['b'], [True])
        node_a = tree.get_successor(['a'])
        node_b = tree.get_successor(['b'])
        self.assertTrue(Apartness.states_are_apart(node_a, node_b, tree))
        self.assertEqual(Apartness.compute_witness(node_a, node_b, tree), [])

    def test_nodes_apart_via_one_step_successor_difference(self):
        tree = dfa_tree()
        tree.insert_observation(['a'], [False])
        tree.insert_observation(['b'], [False])
        node_a = tree.get_successor(['a'])
        node_b = tree.get_successor(['b'])
        self.assertFalse(Apartness.states_are_apart(node_a, node_b, tree))

        # Both have output False, but diverge one step further via input 'a'.
        tree.insert_observation(['a', 'a'], [False, True])
        tree.insert_observation(['b', 'a'], [False, False])
        self.assertTrue(Apartness.states_are_apart(node_a, node_b, tree))
        self.assertEqual(Apartness.compute_witness(node_a, node_b, tree), ['a'])

    def test_nodes_apart_only_via_a_longer_suffix(self):
        tree = dfa_tree()
        tree.insert_observation(['a'], [True])
        tree.insert_observation(['b'], [True])
        node_a = tree.get_successor(['a'])
        node_b = tree.get_successor(['b'])

        tree.insert_observation(['a', 'a'], [True, True])
        tree.insert_observation(['b', 'a'], [True, True])
        self.assertFalse(Apartness.states_are_apart(node_a, node_b, tree))

        # Only diverge two steps deep, via 'a', 'a'.
        tree.insert_observation(['a', 'a', 'a'], [True, True, True])
        tree.insert_observation(['b', 'a', 'a'], [True, True, False])
        self.assertTrue(Apartness.states_are_apart(node_a, node_b, tree))
        self.assertEqual(Apartness.compute_witness(node_a, node_b, tree), ['a', 'a'])

    def test_no_witness_when_not_apart(self):
        tree = dfa_tree()
        tree.insert_observation(['a'], [False])
        tree.insert_observation(['b'], [False])
        node_a = tree.get_successor(['a'])
        node_b = tree.get_successor(['b'])
        self.assertIsNone(Apartness.compute_witness(node_a, node_b, tree))


class TestStatesAreApartMealy(unittest.TestCase):
    def test_apart_via_differing_output_on_same_input(self):
        tree = mealy_tree()
        tree.insert_observation(['a'], [1])
        tree.insert_observation(['b'], [0])
        node_a = tree.get_successor(['a'])
        node_b = tree.get_successor(['b'])
        # node_a/node_b themselves have no outgoing observations yet, so they only become apart once
        # their own successors disagree on some input.
        tree.insert_observation(['a', 'a'], [1, 5])
        tree.insert_observation(['b', 'a'], [0, 7])
        self.assertTrue(Apartness.states_are_apart(node_a, node_b, tree))
        self.assertEqual(Apartness.compute_witness(node_a, node_b, tree), ['a'])

    def test_not_apart_when_undetermined_inputs_produce_no_conflict(self):
        tree = mealy_tree()
        tree.insert_observation(['a'], [1])
        tree.insert_observation(['b'], [1])
        node_a = tree.get_successor(['a'])
        node_b = tree.get_successor(['b'])
        # Neither node has an observed 'a'/'b'-successor yet, so no output pair can conflict.
        self.assertFalse(Apartness.states_are_apart(node_a, node_b, tree))


class TestComputeWitnessInTreeAndHypothesisStates(unittest.TestCase):
    def test_dfa_root_matching_hypothesis_state_returns_none(self):
        tree = dfa_tree()
        h0 = DfaState('h0', is_accepting=True)
        h0.transitions = {'a': h0, 'b': h0}
        self.assertIsNone(Apartness.compute_witness_in_tree_and_hypothesis_states(tree, tree.root, h0))

    def test_dfa_immediate_output_mismatch_returns_empty_witness(self):
        tree = dfa_tree()
        tree.insert_observation(['a'], [False])
        node_a = tree.get_successor(['a'])
        h0 = DfaState('h0', is_accepting=True)
        h0.transitions = {'a': h0, 'b': h0}
        # node_a.output is False but h0.is_accepting is True: they already differ at this node.
        witness = Apartness.compute_witness_in_tree_and_hypothesis_states(tree, node_a, h0)
        self.assertEqual(witness, [])

    def test_dfa_mismatch_found_one_step_deeper(self):
        tree = dfa_tree()
        tree.insert_observation(['a'], [True])
        tree.insert_observation(['a', 'b'], [True, False])
        node_a = tree.get_successor(['a'])
        h0 = DfaState('h0', is_accepting=True)
        h0.transitions = {'a': h0, 'b': h0}
        # node_a agrees with h0 (both True); its 'b'-successor (False) disagrees with h0 (True).
        witness = Apartness.compute_witness_in_tree_and_hypothesis_states(tree, node_a, h0)
        self.assertEqual(witness, ['b'])

    def test_mealy_output_mismatch_detected(self):
        tree = mealy_tree()
        m0 = MealyState('m0')
        m0.transitions = {'a': m0, 'b': m0}
        m0.output_fun = {'a': 1, 'b': 0}
        # root's observed 'a'-output (0) disagrees with m0.output_fun['a'] (1); the witness is the
        # single input that exposes the mismatch, i.e. the transfer sequence to root's 'a'-successor.
        tree.insert_observation(['a'], [0])
        witness = Apartness.compute_witness_in_tree_and_hypothesis_states(tree, tree.root, m0)
        self.assertEqual(witness, ['a'])

    def test_mealy_no_mismatch_returns_none(self):
        tree = mealy_tree()
        m0 = MealyState('m0')
        m0.transitions = {'a': m0, 'b': m0}
        m0.output_fun = {'a': 1, 'b': 0}
        tree.insert_observation(['a'], [1])
        tree.insert_observation(['b'], [0])
        self.assertIsNone(Apartness.compute_witness_in_tree_and_hypothesis_states(tree, tree.root, m0))


if __name__ == '__main__':
    unittest.main()
