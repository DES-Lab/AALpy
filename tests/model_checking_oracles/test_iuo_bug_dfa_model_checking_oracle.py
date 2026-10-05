from typing import Any, Tuple
import unittest

from aalpy.automata import Dfa, DfaState, MealyMachine, MealyState
from aalpy.model_checking_oracles import IUOBugDfaModelCheckingOracle
from aalpy.utils.AutomatonGenerators import generate_random_dfa
from aalpy.utils.ModelChecking import bisimilar


def precise_word_mealy(word: Tuple[str], alphabet=('a', 'b')):
    """
    Mealy machine that outputs len(word) - 1 `o`s followed by an `x` and then only 'o's for the word
    and only `o`s for any other input sequence.
    """
    length = len(word)
    sink = MealyState('sink')
    sink.transitions = {a: sink for a in alphabet}
    sink.output_fun = {a: 'o' for a in alphabet}
    states = [MealyState(f's_{word[:i]}') for i in range(length)] + [sink]
    for i in range(length):
        states[i].transitions = {a: sink for a in alphabet}
        states[i].output_fun = {a: 'o' for a in alphabet}
        if i < length - 1:
            states[i].transitions[word[i]] = states[i + 1]
        else:
            states[i].output_fun[word[i]] = 'x'
    return MealyMachine(states[0], states)

def precise_word_dfa(word: Tuple[str], alphabet=('a', 'b')):
    """
    Dfa which accepts the input sequences that are prefixed by the given word.
    """
    length = len(word)
    sink = DfaState('sink', False)
    sink.transitions = {a: sink for a in alphabet}
    states = [DfaState(f's_{word[:i]}', i == length) for i in range(length + 1)] + [sink]
    for i in range(length):
        states[i].transitions = {a: sink for a in alphabet}
        states[i].transitions[word[i]] = states[i + 1]
    states[length].transitions = {a: sink for a in alphabet}
    return Dfa(states[0], states)


def dfa_input_from_mealy_input(input: Any) -> Tuple[str,Any]:
    return ('I', input)

def dfa_output_from_mealy_output(output: Any) -> Tuple[str,Any]:
    return ('O', output)

def is_dfa_input(letter: Tuple[str,Any]) -> bool:
    return letter[0] == 'I'

def mealy_letter_from_dfa_letter(letter: Tuple[str,Any]) -> Any:
    return letter[1]


class IUOBugDfaModelCheckingOracleTests(unittest.TestCase):

    def test_correct_counterexample_is_found_when_present(self):
        mealy = precise_word_mealy(
            word=('a', 'b', 'a', 'b'),
            alphabet=('a', 'b')
        )
        prop_bug_dfa = precise_word_dfa(
            word=(
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('x')
            ),
            alphabet=(
                dfa_input_from_mealy_input('a'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('o'),
                dfa_output_from_mealy_output('x')
            )
        )

        oracle = IUOBugDfaModelCheckingOracle(
            bug_dfa=prop_bug_dfa,
            mealy_input_to_dfa_input=dfa_input_from_mealy_input,
            mealy_output_to_dfa_output=dfa_output_from_mealy_output,
            is_dfa_input=is_dfa_input,
            dfa_letter_to_mealy_letter=mealy_letter_from_dfa_letter
        )

        self.assertEqual(oracle.find_cex(mealy), ('a', 'b', 'a', 'b'))

        self.assertEqual(oracle.num_checks_performed, 1)
        self.assertEqual(oracle.num_counterexamples_found, 1)

    def test_no_counterexample_is_found_when_absent(self):
        mealy = MealyMachine.from_state_setup({
            'q': {'a': ('o', 'q')}
        })
        prop_bug_dfa = Dfa.from_state_setup({
            'p': (False, {
                dfa_input_from_mealy_input('a'): 'p',
                dfa_output_from_mealy_output('o'): 'p'
            }),
        })

        oracle = IUOBugDfaModelCheckingOracle(
            bug_dfa=prop_bug_dfa,
            mealy_input_to_dfa_input=dfa_input_from_mealy_input,
            mealy_output_to_dfa_output=dfa_output_from_mealy_output,
            is_dfa_input=is_dfa_input,
            dfa_letter_to_mealy_letter=mealy_letter_from_dfa_letter
        )

        self.assertIsNone(oracle.find_cex(mealy))

        self.assertEqual(oracle.num_checks_performed, 1)
        self.assertEqual(oracle.num_counterexamples_found, 0)

    def test_counterexample_can_be_found_with_fallback_letter(self):
        mealy = precise_word_mealy(
            word=('a', 'b', 'a', 'b'),
            alphabet=('a', 'b')
        )
        prop_bug_dfa = precise_word_dfa(
            word=(
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('fallback_o'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('fallback_o'),
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('fallback_o'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('x')
            ),
            alphabet=(
                dfa_input_from_mealy_input('a'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('x'),
                dfa_output_from_mealy_output('fallback_o')
            )
        )

        oracle = IUOBugDfaModelCheckingOracle(
            bug_dfa=prop_bug_dfa,
            mealy_input_to_dfa_input=dfa_input_from_mealy_input,
            mealy_output_to_dfa_output=dfa_output_from_mealy_output,
            is_dfa_input=is_dfa_input,
            dfa_letter_to_mealy_letter=mealy_letter_from_dfa_letter,
            bug_dfa_fallback_letter=dfa_output_from_mealy_output('fallback_o')
        )

        self.assertEqual(oracle.find_cex(mealy), ('a', 'b', 'a', 'b'))

        self.assertEqual(oracle.num_checks_performed, 1)
        self.assertEqual(oracle.num_counterexamples_found, 1)

    def test_counterexample_cannot_be_found_when_fallback_output_is_needed_and_missing(self):
        mealy = precise_word_mealy(
            word=('a', 'b', 'a', 'b'),
            alphabet=('a', 'b')
        )
        prop_bug_dfa = precise_word_dfa(
            word=(
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('fallback_o'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('fallback_o'),
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('fallback_o'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('x')
            ),
            alphabet=(
                dfa_input_from_mealy_input('a'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('x'),
                dfa_output_from_mealy_output('fallback_o')
            )
        )

        oracle = IUOBugDfaModelCheckingOracle(
            bug_dfa=prop_bug_dfa,
            mealy_input_to_dfa_input=dfa_input_from_mealy_input,
            mealy_output_to_dfa_output=dfa_output_from_mealy_output,
            is_dfa_input=is_dfa_input,
            dfa_letter_to_mealy_letter=mealy_letter_from_dfa_letter
        )

        with self.assertRaises(ValueError):
            oracle.find_cex(mealy)

        self.assertEqual(oracle.num_checks_performed, 1)
        self.assertEqual(oracle.num_counterexamples_found, 0)

    def test_counterexample_always_has_nonzero_length(self):
        mealy = MealyMachine.from_state_setup({
            'q': {'a': ('o', 'q')}
        })
        prop_bug_dfa = Dfa.from_state_setup({
            'p': (True, {
                dfa_input_from_mealy_input('a'): 'p',
                dfa_output_from_mealy_output('o'): 'p'
            }),
        })

        oracle = IUOBugDfaModelCheckingOracle(
            bug_dfa=prop_bug_dfa,
            mealy_input_to_dfa_input=dfa_input_from_mealy_input,
            mealy_output_to_dfa_output=dfa_output_from_mealy_output,
            is_dfa_input=is_dfa_input,
            dfa_letter_to_mealy_letter=mealy_letter_from_dfa_letter
        )

        cex = oracle.find_cex(mealy)
        self.assertEqual(cex, ('a',))
        self.assertEqual(oracle.num_checks_performed, 1)
        self.assertEqual(oracle.num_counterexamples_found, 1)

    def test_default_mealy_to_dfa(self):
        mealy = MealyMachine.from_state_setup({
            'q0': {
                'a': ('o0', 'q0'),
                'b': ('o1', 'q1')
            },
            'q1': {
                'a': ('o2', 'q1'),
                'b': ('o3', 'q1')
            }
        })

        dfa_a = dfa_input_from_mealy_input('a')
        dfa_b = dfa_input_from_mealy_input('b')
        dfa_o0 = dfa_output_from_mealy_output('o0')
        dfa_o1 = dfa_output_from_mealy_output('o1')
        dfa_o2 = dfa_output_from_mealy_output('o2')
        dfa_o3 = dfa_output_from_mealy_output('o3')
        mealy_dfa = Dfa.from_state_setup({
            'q0': (True, {
                dfa_a: 'aux_o0_q0',
                dfa_b: 'aux_o1_q1'
            }),
            'q1': (True, {
                dfa_a: 'aux_o2_q1',
                dfa_b: 'aux_o3_q1'
            }),
            'aux_o0_q0': (True, {
                dfa_o0: 'q0'
            }),
            'aux_o1_q1': (True, {
                dfa_o1: 'q1'
            }),
            'aux_o2_q1': (True, {
                dfa_o2: 'q1'
            }),
            'aux_o3_q1': (True, {
                dfa_o3: 'q1'
            })
        })

        prop_bug_dfa = Dfa.from_state_setup({
            'p': (True, {
                dfa_input_from_mealy_input('a'): 'p',
                dfa_output_from_mealy_output('o'): 'p'
            }),
        })

        oracle = IUOBugDfaModelCheckingOracle(
            bug_dfa=prop_bug_dfa,
            mealy_input_to_dfa_input=dfa_input_from_mealy_input,
            mealy_output_to_dfa_output=dfa_output_from_mealy_output,
            is_dfa_input=is_dfa_input,
            dfa_letter_to_mealy_letter=mealy_letter_from_dfa_letter
        )

        assert bisimilar(oracle.default_mealy_to_dfa(mealy), mealy_dfa, return_cex=False)

        self.assertEqual(oracle.num_checks_performed, 0)
        self.assertEqual(oracle.num_counterexamples_found, 0)

    def test_custom_mealy_to_dfa(self):
        mealy = MealyMachine.from_state_setup({
            'q0': {
                'a': ('o0', 'q0'),
                'b': ('o1', 'q1')
            },
            'q1': {
                'a': ('o2', 'q1'),
                'b': ('o3', 'q1')
            }
        })

        dfa_a = dfa_input_from_mealy_input('a')
        dfa_b = dfa_input_from_mealy_input('b')
        dfa = Dfa.from_state_setup({
            'q0': (True, {
                dfa_a: 'q0',
                dfa_b: 'q1'
            }),
            'q1': (False, {
                dfa_a: 'q1',
                dfa_b: 'q0'
            })
        })

        def custom_mealy_to_dfa(_: MealyMachine) -> Dfa:
            return dfa

        prop_bug_dfa = Dfa.from_state_setup({
            'p': (True, {
                dfa_input_from_mealy_input('a'): 'p',
                dfa_output_from_mealy_output('o'): 'p'
            }),
        })

        oracle = IUOBugDfaModelCheckingOracle(
            bug_dfa=prop_bug_dfa,
            mealy_input_to_dfa_input=dfa_input_from_mealy_input,
            mealy_output_to_dfa_output=dfa_output_from_mealy_output,
            is_dfa_input=is_dfa_input,
            dfa_letter_to_mealy_letter=mealy_letter_from_dfa_letter,
            mealy_to_dfa=custom_mealy_to_dfa
        )

        assert bisimilar(oracle.mealy_to_dfa(mealy), dfa, return_cex=False)

        self.assertEqual(oracle.num_checks_performed, 0)
        self.assertEqual(oracle.num_counterexamples_found, 0)

    def test_hyp_is_reset(self):
        mealy = MealyMachine.from_state_setup({
            'q0': {'a': ('o0', 'q1')},
            'q1': {'a': ('o1', 'q1')}
        })
        mealy.step('a')
        self.assertNotEqual(mealy.current_state, mealy.initial_state)

        dfa_alphabet = [dfa_input_from_mealy_input('a')] + [dfa_output_from_mealy_output(o) for o in ['o0', 'o1']]
        random_bug_dfa = generate_random_dfa(
            num_states=10,
            alphabet=dfa_alphabet,
            num_accepting_states=2,
            compute_prefixes=False,
            ensure_minimality=False)

        oracle = IUOBugDfaModelCheckingOracle(
            bug_dfa=random_bug_dfa,
            mealy_input_to_dfa_input=dfa_input_from_mealy_input,
            mealy_output_to_dfa_output=dfa_output_from_mealy_output,
            is_dfa_input=is_dfa_input,
            dfa_letter_to_mealy_letter=mealy_letter_from_dfa_letter
        )

        oracle.reset_hyp_and_sul(mealy)
        self.assertEqual(mealy.current_state, mealy.initial_state)

        self.assertEqual(oracle.num_checks_performed, 0)
        self.assertEqual(oracle.num_counterexamples_found, 0)

    def test_counts_are_accurate(self):
        satisfying_mealy = precise_word_mealy(
            word=('a', 'b', 'a'),
            alphabet=('a', 'b')
        )
        violating_mealy = precise_word_mealy(
            word=('a', 'b', 'a', 'b'),
            alphabet=('a', 'b')
        )
        prop_bug_dfa = precise_word_dfa(
            word=(
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('a'),
                dfa_output_from_mealy_output('o'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('x')
            ),
            alphabet=(
                dfa_input_from_mealy_input('a'),
                dfa_input_from_mealy_input('b'),
                dfa_output_from_mealy_output('o'),
                dfa_output_from_mealy_output('x')
            )
        )

        oracle = IUOBugDfaModelCheckingOracle(
            bug_dfa=prop_bug_dfa,
            mealy_input_to_dfa_input=dfa_input_from_mealy_input,
            mealy_output_to_dfa_output=dfa_output_from_mealy_output,
            is_dfa_input=is_dfa_input,
            dfa_letter_to_mealy_letter=mealy_letter_from_dfa_letter
        )

        self.assertEqual(oracle.num_checks_performed, 0)
        self.assertEqual(oracle.num_counterexamples_found, 0)

        self.assertEqual(oracle.find_cex(satisfying_mealy), None)
        self.assertEqual(oracle.num_checks_performed, 1)
        self.assertEqual(oracle.num_counterexamples_found, 0)

        self.assertEqual(oracle.find_cex(violating_mealy), ('a', 'b', 'a', 'b'))
        self.assertEqual(oracle.num_checks_performed, 2)
        self.assertEqual(oracle.num_counterexamples_found, 1)

        self.assertEqual(oracle.find_cex(violating_mealy), ('a', 'b', 'a', 'b'))
        self.assertEqual(oracle.num_checks_performed, 3)
        self.assertEqual(oracle.num_counterexamples_found, 2)

        self.assertEqual(oracle.find_cex(satisfying_mealy), None)
        self.assertEqual(oracle.num_checks_performed, 4)
        self.assertEqual(oracle.num_counterexamples_found, 2)


if __name__ == '__main__':
    unittest.main()
