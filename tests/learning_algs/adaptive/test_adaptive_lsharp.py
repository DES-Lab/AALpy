import random
import unittest

from aalpy.SULs import AutomatonSUL
from aalpy.automata import MealyMachine, MealyState
from aalpy.learning_algs import run_adaptive_Lsharp
from aalpy.oracles import WpMethodEqOracle
from aalpy.utils import generate_random_deterministic_automata
from aalpy.utils.ModelChecking import bisimilar


def two_state_mealy():
    """
    2-state Mealy machine over {a, b}.
    s0 --a/x--> s1   s0 --b/y--> s0
    s1 --a/y--> s0   s1 --b/x--> s1
    """
    s0 = MealyState('s0')
    s1 = MealyState('s1')
    s0.transitions = {'a': s1, 'b': s0}
    s0.output_fun = {'a': 'x', 'b': 'y'}
    s1.transitions = {'a': s0, 'b': s1}
    s1.output_fun = {'a': 'y', 'b': 'x'}
    mm = MealyMachine(s0, [s0, s1])
    mm.compute_prefixes()
    return mm


def three_state_mealy_variant():
    """
    3-state variant of two_state_mealy: s1's 'b' output changed and a new state s2 introduced,
    simulating a slightly changed system that reuses most of the original behaviour.
    s0 --a/x--> s1   s0 --b/y--> s0
    s1 --a/y--> s2   s1 --b/z--> s1
    s2 --a/y--> s0   s2 --b/x--> s2
    """
    s0 = MealyState('s0')
    s1 = MealyState('s1')
    s2 = MealyState('s2')
    s0.transitions = {'a': s1, 'b': s0}
    s0.output_fun = {'a': 'x', 'b': 'y'}
    s1.transitions = {'a': s2, 'b': s1}
    s1.output_fun = {'a': 'y', 'b': 'z'}
    s2.transitions = {'a': s0, 'b': s2}
    s2.output_fun = {'a': 'y', 'b': 'x'}
    mm = MealyMachine(s0, [s0, s1, s2])
    mm.compute_prefixes()
    return mm


def learn_with_adaptive(target, references, state_matching='Approximate', rebuilding=True):
    alphabet = target.get_input_alphabet()
    sul = AutomatonSUL(target)
    eq_oracle = WpMethodEqOracle(alphabet, sul, max_number_of_states=len(target.states) + 1)
    return run_adaptive_Lsharp(alphabet, sul, references, eq_oracle, automaton_type='mealy',
                               extension_rule=None, separation_rule='SepSeq',
                               rebuilding=rebuilding, state_matching=state_matching,
                               print_level=0, return_data=True)


class TestColdStartFallback(unittest.TestCase):
    def test_empty_references_falls_back_to_plain_lsharp(self):
        target = two_state_mealy()
        learned, info = learn_with_adaptive(target, [], state_matching=None, rebuilding=True)

        self.assertTrue(bisimilar(learned, target))


class TestLearnsWithPerfectReference(unittest.TestCase):
    def test_learns_minimal_bisimilar_model_with_self_as_reference(self):
        for matching in (None, 'Total', 'Approximate'):
            with self.subTest(matching=matching):
                target = two_state_mealy()
                reference = two_state_mealy()
                learned, info = learn_with_adaptive(target, [reference], state_matching=matching)

                self.assertTrue(bisimilar(learned, target))
                self.assertEqual(learned.size, len(target.states))

    def test_random_automata_learned_correctly_with_perfect_reference(self):
        random.seed(42)
        for i in range(3):
            num_states = random.randint(3, 6)
            target = generate_random_deterministic_automata(
                'mealy', num_states=num_states, input_alphabet_size=3, output_alphabet_size=3)
            reference = target.copy()

            learned, info = learn_with_adaptive(target, [reference], state_matching='Approximate')
            self.assertTrue(bisimilar(learned, target), f'run {i} with {num_states} states failed')


class TestAdaptiveReuseAcrossRelatedSystems(unittest.TestCase):
    def test_reference_from_one_system_helps_learn_a_related_system(self):
        # A represents a previously learned/known model; B is a related system whose behaviour has
        # diverged (state added, an output changed). Adaptive learning should still converge correctly.
        model_a = two_state_mealy()
        model_b = three_state_mealy_variant()

        learned_b, info = learn_with_adaptive(model_b, [model_a], state_matching='Approximate')

        self.assertTrue(bisimilar(learned_b, model_b))
        self.assertFalse(bisimilar(learned_b, model_a))

    def test_reused_reference_reduces_or_matches_plain_learning_effort(self):
        model_a = two_state_mealy()
        model_b = three_state_mealy_variant()

        _, info_with_reference = learn_with_adaptive(model_b, [model_a], state_matching='Approximate')
        _, info_without_reference = learn_with_adaptive(model_b, [], state_matching=None, rebuilding=True)

        self.assertLessEqual(info_with_reference['queries_learning'],
                             info_without_reference['queries_learning'] + 5)

    def test_random_pairs_of_related_automata(self):
        random.seed(7)
        for i in range(3):
            num_states = random.randint(3, 5)
            base = generate_random_deterministic_automata(
                'mealy', num_states=num_states, input_alphabet_size=3, output_alphabet_size=3)

            # simulate a slightly changed system: mutate a single transition's output
            mutated = base.copy()
            mutated_state = mutated.states[0]
            an_input = mutated.get_input_alphabet()[0]
            current_output = mutated_state.output_fun[an_input]
            other_outputs = [o for o in {'o1', 'o2', 'o3'} if o != current_output]
            mutated_state.output_fun[an_input] = other_outputs[0]

            learned, info = learn_with_adaptive(mutated, [base], state_matching='Approximate')
            self.assertTrue(bisimilar(learned, mutated), f'run {i} with {num_states} states failed')


if __name__ == '__main__':
    unittest.main()
