import unittest
from itertools import product
from pathlib import Path

import aalpy
from aalpy.automata import Dfa, MooreMachine, MealyMachine
from aalpy.learning_algs import run_RPNI
from aalpy.utils import load_automaton_from_file
from aalpy.utils.ModelChecking import compare_automata

DOT_MODELS_DIR = Path(__file__).resolve().parents[3] / 'DotModels'

correct_automata = {
    Dfa: load_automaton_from_file(str(DOT_MODELS_DIR / 'SimpleABC' / 'simple_abc_dfa.dot'), automaton_type='dfa'),
    MooreMachine: load_automaton_from_file(str(DOT_MODELS_DIR / 'SimpleABC' / 'simple_abc_moore.dot'),
                                           automaton_type='moore'),
    MealyMachine: load_automaton_from_file(str(DOT_MODELS_DIR / 'SimpleABC' / 'simple_abc_mealy.dot'),
                                           automaton_type='mealy'),
}

automata_type = {Dfa: 'dfa', MooreMachine: 'moore', MealyMachine: 'mealy'}


def prove_equivalence(learned_automaton):
    correct_automaton = correct_automata[learned_automaton.__class__]

    # only works if the correct automaton is already minimal
    if len(learned_automaton.states) != len(correct_automaton.states):
        return False

    return correct_automaton == learned_automaton


def generate_data(ground_truth, depth=5, step=1):
    data = []
    if isinstance(ground_truth, (aalpy.automata.Dfa, aalpy.automata.MooreMachine)):
        data.append(((), ground_truth.initial_state.output))

    alphabet = ground_truth.get_input_alphabet()
    for level in range(1, depth + 1, step):
        for seq in product(alphabet, repeat=level):
            ground_truth.reset_to_initial()
            outputs = ground_truth.execute_sequence(ground_truth.initial_state, seq)
            data.append((seq, outputs[-1]))

    return data


class TestRunRpniAllConfigurations(unittest.TestCase):
    """
    Ported from the legacy tests/test_deterministic_passive.py: learns each reference automaton (loaded
    from DotModels/SimpleABC) via run_RPNI for both the 'gsm' and 'classic' algorithms, on complete and
    input-incomplete sample data, and checks that the learned model equals the reference.
    """

    def test_all_configuration_combinations_complete_data(self):
        algorithms = ['gsm', 'classic']

        for automata_class, correct_automaton in correct_automata.items():
            data = generate_data(correct_automaton, depth=3)
            for algorithm in algorithms:
                learned_model = run_RPNI(data, automaton_type=automata_type[automata_class],
                                         algorithm=algorithm, print_info=False)

                if not prove_equivalence(learned_model):
                    cex = compare_automata(learned_model, correct_automaton)
                    self.fail(f'{algorithm}/{automata_type[automata_class]}: learned model does not match '
                              f'reference. Counterexamples: {cex}')

    def test_all_configuration_combinations_input_incomplete_data(self):
        algorithms = ['gsm', 'classic']

        for automata_class, correct_automaton in correct_automata.items():
            data = generate_data(correct_automaton, depth=3, step=2)
            if automata_type[automata_class] == 'moore':
                data += [(('a', 'a', 'a', 'a'), 1), (('b', 'b', 'b', 'b'), 2), (('c', 'c', 'c', 'c'), 3)]
            for algorithm in algorithms:
                learned_model = run_RPNI(data, automaton_type=automata_type[automata_class],
                                         algorithm=algorithm, print_info=False)

                if not prove_equivalence(learned_model):
                    cex = compare_automata(learned_model, correct_automaton)
                    self.fail(f'{algorithm}/{automata_type[automata_class]}: learned model does not match '
                              f'reference. Counterexamples: {cex}')

    def test_returns_none_for_nondeterministic_data(self):
        data = [((), True), ((), False)]
        learned_model = run_RPNI(data, automaton_type='dfa', algorithm='classic', print_info=False)
        self.assertIsNone(learned_model)

    def test_input_completeness_sink_state(self):
        data = [((), True), (('a',), False), (('b',), True)]
        learned_model = run_RPNI(data, automaton_type='dfa', algorithm='gsm',
                                 input_completeness='sink_state', print_info=False)
        self.assertTrue(learned_model.is_input_complete())

    def test_input_completeness_self_loop(self):
        data = [((), True), (('a',), False), (('b',), True)]
        learned_model = run_RPNI(data, automaton_type='dfa', algorithm='gsm',
                                 input_completeness='self_loop', print_info=False)
        self.assertTrue(learned_model.is_input_complete())


if __name__ == '__main__':
    unittest.main()
