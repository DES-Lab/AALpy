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


if __name__ == '__main__':
    unittest.main()
