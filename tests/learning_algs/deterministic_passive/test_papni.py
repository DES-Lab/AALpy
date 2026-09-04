import random
import unittest
from itertools import product

from aalpy.automata.Vpa import Vpa, VpaAlphabet, VpaState, VpaTransition
from aalpy.learning_algs.deterministic_passive.PAPNI import run_PAPNI
from aalpy.utils import is_balanced


def balanced_parens_vpa_with_internal():
    """1-state VPA accepting balanced '(' / ')' words, with an 'i' internal symbol self-loop."""
    q0 = VpaState('q0', is_accepting=True)
    q0.transitions['('].append(VpaTransition(q0, q0, '(', 'push', '('))
    q0.transitions[')'].append(VpaTransition(q0, q0, ')', 'pop', '('))
    q0.transitions['i'].append(VpaTransition(q0, q0, 'i', None, None))
    vpa = Vpa(q0, [q0])
    alphabet = VpaAlphabet(internal_alphabet=['i'], call_alphabet=['('], return_alphabet=[')'])
    return vpa, alphabet


def generate_data(vpa, alphabet, depth=4):
    merged_alphabet = alphabet.get_merged_alphabet()
    data = []
    for level in range(0, depth + 1):
        for seq in product(merged_alphabet, repeat=level):
            vpa.reset_to_initial()
            outputs = vpa.execute_sequence(vpa.initial_state, seq, [])
            label = outputs[-1] if outputs else vpa.initial_state.is_accepting
            data.append((seq, label))
    return data


class TestRunPapni(unittest.TestCase):
    def check_learned_model_matches_ground_truth(self, algorithm):
        vpa, alphabet = balanced_parens_vpa_with_internal()
        data = generate_data(vpa, alphabet, depth=4)

        learned_model = run_PAPNI(data, alphabet, algorithm=algorithm, print_info=False)
        self.assertIsNotNone(learned_model)

        random.seed(42)
        merged_alphabet = alphabet.get_merged_alphabet()
        for _ in range(200):
            length = random.randint(0, 6)
            seq = tuple(random.choice(merged_alphabet) for _ in range(length))

            expected = is_balanced(list(seq), alphabet)

            learned_model.reset_to_initial()
            outputs = learned_model.execute_sequence(learned_model.initial_state, seq, [])
            actual = outputs[-1] if outputs else learned_model.initial_state.is_accepting

            self.assertEqual(actual, expected, f'mismatch on sequence {seq} with algorithm {algorithm}')

    def test_edsm_algorithm(self):
        self.check_learned_model_matches_ground_truth('edsm')

    def test_gsm_algorithm(self):
        self.check_learned_model_matches_ground_truth('gsm')

    def test_classic_algorithm(self):
        self.check_learned_model_matches_ground_truth('classic')

    def test_unbalanced_sequences_are_filtered_out_of_data(self):
        vpa, alphabet = balanced_parens_vpa_with_internal()
        data = [((')',), False), (('(',), False), ((), True)]
        # even though the label for the unbalanced sequences is wrong (False for both, which is actually
        # correct here), run_PAPNI should still work since unbalanced sequences are dropped before learning.
        learned_model = run_PAPNI(data, alphabet, algorithm='classic', print_info=False)
        self.assertIsNotNone(learned_model)
        self.assertTrue(learned_model.initial_state.is_accepting)


def all_words_up_to_length(alphabet, max_length):
    for length in range(max_length + 1):
        yield from product(alphabet, repeat=length)


def accepts(model, seq):
    model.reset_to_initial()
    outputs = model.execute_sequence(model.initial_state, seq, [])
    return outputs[-1] if outputs else model.initial_state.is_accepting


class TestPapniLearnsFromCharacterizingSequences(unittest.TestCase):
    """
    get_characterizing_sequences builds a characteristic sample of the congruence characterizing visibly pushdown
    languages, without running any learner, so these are genuine checks of PAPNI against independently derived data.
    """

    def check_papni_learns(self, vpa, max_word_length, algorithms=('edsm', 'gsm')):
        from aalpy.utils import get_characterizing_sequences

        alphabet = vpa.get_input_alphabet()
        data = get_characterizing_sequences(vpa)

        for algorithm in algorithms:
            learned_model = run_PAPNI(data, alphabet, algorithm=algorithm, print_info=False)
            self.assertIsNotNone(learned_model)
            for seq in all_words_up_to_length(alphabet.get_merged_alphabet(), max_word_length):
                self.assertEqual(accepts(learned_model, seq), accepts(vpa, seq),
                                 f'mismatch on {seq} with algorithm {algorithm}')

    def test_learns_balanced_parens_vpa_with_internal(self):
        vpa, _ = balanced_parens_vpa_with_internal()
        self.check_papni_learns(vpa, max_word_length=8)

    def test_learns_benchmark_vpas(self):
        from aalpy.utils.BenchmarkVpaModels import (vpa_L1, vpa_L2, vpa_L3, vpa_L4, vpa_L9, vpa_L10, vpa_L12,
                                                    vpa_for_even_parentheses, vpa_for_odd_parentheses, vpa_json)

        # word length is chosen per model so that the exhaustive comparison stays cheap. Even parentheses is checked
        # with edsm only, see test_classic_rpni_merge_order_needs_more_than_a_characteristic_sample
        for model, max_word_length, algorithms in ((vpa_L1(), 12, ('edsm', 'gsm')), (vpa_L2(), 8, ('edsm', 'gsm')),
                                                   (vpa_L3(), 8, ('edsm', 'gsm')), (vpa_L4(), 8, ('edsm', 'gsm')),
                                                   (vpa_L9(), 8, ('edsm', 'gsm')), (vpa_L10(), 6, ('edsm', 'gsm')),
                                                   (vpa_L12(), 6, ('edsm', 'gsm')), (vpa_json(), 5, ('edsm', 'gsm')),
                                                   (vpa_for_odd_parentheses(), 12, ('edsm', 'gsm')),
                                                   (vpa_for_even_parentheses(), 12, ('edsm',))):
            with self.subTest(model=model.to_state_setup()):
                self.check_papni_learns(model, max_word_length, algorithms)

    def test_classic_rpni_merge_order_needs_more_than_a_characteristic_sample(self):
        """
        A characteristic sample of the language does not make every learner converge. PAPNI learns an automaton over
        the stack-aware alphabet, whose states are the classes of prefixes rather than of well-matched words, and two
        prefixes of different stack height are only separated by a word that is well-matched after one of them and
        not after the other. Such a word is not well-matched, so PAPNI discards it, and no sample of well-matched
        sequences can supply that evidence. Even parentheses is a language where the merge order of classic RPNI then
        does not recover the language, while the evidence driven order does.
        """
        from aalpy.automata.Vpa import find_vpa_counterexample
        from aalpy.utils import get_characterizing_sequences
        from aalpy.utils.BenchmarkVpaModels import vpa_for_even_parentheses

        vpa = vpa_for_even_parentheses()
        data = get_characterizing_sequences(vpa)

        learned_with_edsm = run_PAPNI(data, vpa.get_input_alphabet(), algorithm='edsm', print_info=False)
        self.assertIsNone(find_vpa_counterexample(vpa, learned_with_edsm))

        learned_with_gsm = run_PAPNI(data, vpa.get_input_alphabet(), algorithm='gsm', print_info=False)
        self.assertIsNotNone(find_vpa_counterexample(vpa, learned_with_gsm))

    def test_vpas_outside_of_the_model_class_of_papni_are_detectable(self):
        from aalpy.automata.Vpa import vpa_call_symbol_conflicts
        from aalpy.utils.BenchmarkVpaModels import vpa_L1, vpa_for_L16

        # L16 pushes '$' on the first 'a' and 'x' on every further one, while PAPNI identifies a stack symbol with
        # the call symbol that pushed it, so no model PAPNI can return is equivalent to this VPA
        self.assertEqual({'a': {'$', 'x'}}, vpa_call_symbol_conflicts(vpa_for_L16()))
        self.assertEqual({}, vpa_call_symbol_conflicts(vpa_L1()))


if __name__ == '__main__':
    unittest.main()
