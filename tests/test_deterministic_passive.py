import unittest
from itertools import product

import aalpy
from aalpy.automata import Dfa, MooreMachine, MealyMachine
from aalpy.learning_algs import run_RPNI
from aalpy.utils import load_automaton_from_file
# from aalpy.utils.ModelChecking import bisimilar
from aalpy.utils.ModelChecking import compare_automata

correct_automata = {Dfa: load_automaton_from_file('../DotModels/SimpleABC/simple_abc_dfa.dot', automaton_type='dfa'),
                    MooreMachine: load_automaton_from_file('../DotModels/SimpleABC/simple_abc_moore.dot', automaton_type='moore'),
                    MealyMachine: load_automaton_from_file('../DotModels/SimpleABC/simple_abc_mealy.dot', automaton_type='mealy')}


class DeterministicPassiveTest(unittest.TestCase):

    def prove_equivalence(self, learned_automaton):

        correct_automaton = correct_automata[learned_automaton.__class__]

        # only work if correct automaton is already minimal
        if len(learned_automaton.states) != len(correct_automaton.states):
            print(len(learned_automaton.states), len(correct_automaton.states))
            return False

        return correct_automaton == learned_automaton  # bisimilar

    def generate_data(self, ground_truth, depth=5, step=1):
        data = []
        if isinstance(ground_truth, aalpy.automata.Dfa) or isinstance(ground_truth, aalpy.automata.MooreMachine):
            data.append(((), ground_truth.initial_state.output))
    
        alphabet = ground_truth.get_input_alphabet()
        for level in range(1, depth + 1, step):
            for seq in product(alphabet, repeat=level):
                ground_truth.reset_to_initial()
                outputs = ground_truth.execute_sequence(ground_truth.initial_state, seq)
                data.append((seq, outputs[-1]))

        return data

    def test_all_configuration_combinations(self):
        automata_type = {Dfa: 'dfa', MooreMachine: 'moore', MealyMachine: 'mealy'}
        algorithms = ['gsm', 'classic']

        for automata in correct_automata:
            correct_automaton = correct_automata[automata]
            alphabet = correct_automaton.get_input_alphabet()
            data = self.generate_data(correct_automaton, depth=3)
            for algorithm in algorithms:
                learned_model = run_RPNI(data,
                                         automaton_type=automata_type[automata],
                                         algorithm=algorithm,
                                         print_info=False)

                is_eq = self.prove_equivalence(learned_model)
                if not is_eq:
                    print("Learned:")
                    print(learned_model)
                    print(algorithm, automata_type[automata])
                    cex = compare_automata(learned_model, correct_automaton)
                    print(cex)
                    assert False

        assert True

    def test_all_configuration_combinations_input_incomplete_data(self):
        automata_type = {Dfa: 'dfa', MooreMachine: 'moore', MealyMachine: 'mealy'}
        algorithms = ['gsm', 'classic']

        for automata in correct_automata:
            correct_automaton = correct_automata[automata]
            alphabet = correct_automaton.get_input_alphabet()
            data = self.generate_data(correct_automaton, depth=3, step=2)
            if automata_type[automata] == 'moore':
                data += [(('a', 'a', 'a', 'a'), 1), (('b', 'b', 'b', 'b'), 2), (('c', 'c', 'c', 'c'), 3)]
            for algorithm in algorithms:
                learned_model = run_RPNI(data,
                                         automaton_type=automata_type[automata],
                                         algorithm=algorithm,
                                         print_info=False)

                is_eq = self.prove_equivalence(learned_model)
                if not is_eq:
                    print("Learned:")
                    print(learned_model)
                    print(algorithm, automata_type[automata])
                    cex = compare_automata(learned_model, correct_automaton)
                    print(cex)
                    assert False

        assert True
