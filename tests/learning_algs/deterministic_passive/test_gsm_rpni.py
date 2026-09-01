import unittest
from itertools import product

from aalpy.automata import Dfa, DfaState, MooreMachine, MooreState, MealyMachine, MealyState
from aalpy.learning_algs.deterministic_passive.GsmRPNI import GsmRPNI
from aalpy.learning_algs.deterministic_passive.ClassicRPNI import ClassicRPNI
from aalpy.utils.ModelChecking import bisimilar


def even_a_dfa():
    q0 = DfaState('q0', is_accepting=True)
    q1 = DfaState('q1', is_accepting=False)
    q0.transitions = {'a': q1, 'b': q0}
    q1.transitions = {'a': q0, 'b': q1}
    return Dfa(q0, [q0, q1])


def full_sample(automaton, depth=3):
    data = []
    if isinstance(automaton, (Dfa, MooreMachine)):
        data.append(((), automaton.initial_state.output))
    alphabet = automaton.get_input_alphabet()
    for level in range(1, depth + 1):
        for seq in product(alphabet, repeat=level):
            automaton.reset_to_initial()
            outputs = automaton.execute_sequence(automaton.initial_state, seq)
            data.append((seq, outputs[-1]))
    return data


class TestGsmRpniDfa(unittest.TestCase):
    def test_learns_minimal_dfa_from_complete_sample(self):
        ground_truth = even_a_dfa()
        data = full_sample(ground_truth, depth=3)
        learned = GsmRPNI(data, 'dfa', print_info=False).run_rpni()
        self.assertEqual(len(learned.states), 2)
        self.assertTrue(bisimilar(learned, ground_truth))

    def test_merges_states_with_identical_output_and_behavior(self):
        data = [((), True), (('a',), True), (('b',), True), (('a', 'b'), True), (('b', 'a'), True)]
        learned = GsmRPNI(data, 'dfa', print_info=False).run_rpni()
        self.assertEqual(len(learned.states), 1)

    def test_dfa_internally_uses_moore_representation(self):
        gsm = GsmRPNI([((), True)], 'dfa', print_info=False)
        self.assertEqual(gsm.automaton_type, 'moore')
        self.assertEqual(gsm.final_automaton_type, 'dfa')


class TestGsmRpniMoore(unittest.TestCase):
    def test_learns_from_complete_sample(self):
        q0 = MooreState('q0', 0)
        q1 = MooreState('q1', 1)
        q0.transitions = {'a': q1, 'b': q0}
        q1.transitions = {'a': q0, 'b': q1}
        ground_truth = MooreMachine(q0, [q0, q1])

        data = full_sample(ground_truth, depth=3)
        learned = GsmRPNI(data, 'moore', print_info=False).run_rpni()
        self.assertEqual(len(learned.states), 2)
        self.assertTrue(bisimilar(learned, ground_truth))


class TestGsmRpniMealy(unittest.TestCase):
    def test_learns_from_complete_sample(self):
        q0 = MealyState('q0')
        q1 = MealyState('q1')
        q0.transitions = {'a': q1, 'b': q0}
        q0.output_fun = {'a': 'x', 'b': 'y'}
        q1.transitions = {'a': q0, 'b': q1}
        q1.output_fun = {'a': 'y', 'b': 'x'}
        ground_truth = MealyMachine(q0, [q0, q1])

        data = full_sample(ground_truth, depth=3)
        learned = GsmRPNI(data, 'mealy', print_info=False).run_rpni()
        self.assertEqual(len(learned.states), 2)
        self.assertTrue(bisimilar(learned, ground_truth))


class TestGsmRpniNonDeterministicData(unittest.TestCase):
    def test_root_node_none_for_conflicting_data(self):
        data = [((), True), ((), False)]
        rpni = GsmRPNI(data, 'dfa', print_info=False)
        self.assertIsNone(rpni.root_node)


class TestGsmAndClassicAgree(unittest.TestCase):
    """
    Both underlying RPNI strategies should produce bisimilar (typically identical) results for the
    same consistent, complete sample, even though they compute compatibility differently
    (partition-based vs copy-and-fold).
    """

    def test_agree_on_dfa_sample(self):
        ground_truth = even_a_dfa()
        data = full_sample(ground_truth, depth=4)
        gsm_learned = GsmRPNI(list(data), 'dfa', print_info=False).run_rpni()
        classic_learned = ClassicRPNI(list(data), 'dfa', print_info=False).run_rpni()
        self.assertTrue(bisimilar(gsm_learned, classic_learned))

    def test_agree_on_mealy_sample(self):
        q0 = MealyState('q0')
        q1 = MealyState('q1')
        q0.transitions = {'a': q1, 'b': q0}
        q0.output_fun = {'a': 'x', 'b': 'y'}
        q1.transitions = {'a': q0, 'b': q1}
        q1.output_fun = {'a': 'y', 'b': 'x'}
        ground_truth = MealyMachine(q0, [q0, q1])
        data = full_sample(ground_truth, depth=4)
        gsm_learned = GsmRPNI(list(data), 'mealy', print_info=False).run_rpni()
        classic_learned = ClassicRPNI(list(data), 'mealy', print_info=False).run_rpni()
        self.assertTrue(bisimilar(gsm_learned, classic_learned))


if __name__ == '__main__':
    unittest.main()
