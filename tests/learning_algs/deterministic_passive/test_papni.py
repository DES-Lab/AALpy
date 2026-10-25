import random
import unittest
from itertools import product

from aalpy.automata.Sevpa import Sevpa
from aalpy.automata.Vpa import Vpa, VpaAlphabet, VpaState, VpaTransition, vpa_from_sevpa
from aalpy.learning_algs.deterministic_passive.PAPNI import run_PAPNI
from aalpy.utils import generate_random_sevpa, is_balanced


def balanced_parens_vpa_with_internal(internal_alphabet=('i',)):
    """1-state VPA accepting balanced '(' / ')' words, with a self-loop on every internal symbol."""
    q0 = VpaState('q0', is_accepting=True)
    q0.transitions['('].append(VpaTransition(q0, q0, '(', 'push', '('))
    q0.transitions[')'].append(VpaTransition(q0, q0, ')', 'pop', '('))
    for symbol in internal_alphabet:
        q0.transitions[symbol].append(VpaTransition(q0, q0, symbol, None, None))
    vpa = Vpa(q0, [q0])
    alphabet = VpaAlphabet(internal_alphabet=list(internal_alphabet), call_alphabet=['('], return_alphabet=[')'])
    return vpa, alphabet


def generate_data(vpa, alphabet, depth=4):
    merged_alphabet = alphabet.get_merged_alphabet()
    data = []
    for level in range(0, depth + 1):
        for seq in product(merged_alphabet, repeat=level):
            data.append((seq, vpa.accepts(seq)))
    return data


def nested_blocks_vpa():
    """
    VPA accepting concatenations of perfectly nested blocks, ( (^n )^n )*. The call symbol '(' pushes '$' at the
    outermost level and '(' inside a block, so the pushed stack symbol is not determined by the call symbol.
    """
    alphabet = VpaAlphabet(internal_alphabet=[], call_alphabet=['('], return_alphabet=[')'])
    state_setup = {
        "q0": (True, {"(": [("q1", 'push', "$")]}),
        "q1": (False, {"(": [("q1", 'push', "(")],
                       ")": [("q0", 'pop', "$"), ("q2", 'pop', "(")]}),
        "q2": (False, {")": [("q2", 'pop', "("), ("q0", 'pop', "$")]}),
    }
    return Vpa.from_state_setup(state_setup, init_state_id="q0", input_alphabet=alphabet)


def exhaustive_balanced_data(vpa, budget=20000):
    """Every balanced word within the budget (see all_words_within_budget), labeled by the VPA."""
    alphabet = vpa.input_alphabet
    return [(seq, vpa.accepts(seq)) for seq in all_words_within_budget(alphabet.get_merged_alphabet(), budget)
            if is_balanced(seq, alphabet)]


def all_words_within_budget(alphabet, budget=5000):
    """All words over the alphabet, longest first up to the length at which their number exceeds the budget."""
    max_length, count = 0, 1
    while count * len(alphabet) <= budget:
        count *= len(alphabet)
        max_length += 1
    for length in range(max_length + 1):
        yield from product(alphabet, repeat=length)


def find_disagreement(model, other, budget=100000):
    """A word, within the budget, on which the two models disagree, or None if there is none."""
    for seq in all_words_within_budget(model.input_alphabet.get_merged_alphabet(), budget):
        if model.accepts(seq) != other.accepts(seq):
            return seq
    return None


class TestRunPapni(unittest.TestCase):
    def test_learned_model_matches_ground_truth(self):
        vpa, alphabet = balanced_parens_vpa_with_internal()
        data = generate_data(vpa, alphabet, depth=4)

        learned_model = run_PAPNI(data, alphabet, print_info=False)
        self.assertIsNotNone(learned_model)

        random.seed(42)
        merged_alphabet = alphabet.get_merged_alphabet()
        for _ in range(200):
            length = random.randint(0, 6)
            seq = tuple(random.choice(merged_alphabet) for _ in range(length))

            expected = is_balanced(list(seq), alphabet)

            self.assertEqual(learned_model.accepts(seq), expected, f'mismatch on sequence {seq}')

    def test_learns_vpas_with_call_symbol_conflicts(self):
        """
        Two VPAs whose pushed stack symbol is not determined by the call symbol. PAPNI learns the congruence of the
        canonical single entry VPA, whose stack symbols carry the state a call was read in, so these are within its
        model class.
        """
        from aalpy.utils.BenchmarkVpaModels import vpa_for_L16

        for vpa in (vpa_for_L16(), nested_blocks_vpa()):
            with self.subTest(model=vpa.to_state_setup()):
                learned_model = run_PAPNI(exhaustive_balanced_data(vpa), vpa.input_alphabet, print_info=False)
                self.assertIsNone(find_disagreement(vpa, learned_model))

    def test_sevpa_is_returned_when_requested(self):
        vpa, alphabet = balanced_parens_vpa_with_internal()
        data = generate_data(vpa, alphabet, depth=4)

        self.assertIsInstance(run_PAPNI(data, alphabet, print_info=False), Vpa)
        self.assertIsInstance(run_PAPNI(data, alphabet, automaton_type='sevpa', print_info=False), Sevpa)

    def test_unbalanced_sequences_are_filtered_out_of_data(self):
        vpa, alphabet = balanced_parens_vpa_with_internal()
        data = [((')',), False), (('(',), False), ((), True)]
        # even though the label for the unbalanced sequences is wrong (False for both, which is actually
        # correct here), run_PAPNI should still work since unbalanced sequences are dropped before learning.
        learned_model = run_PAPNI(data, alphabet, print_info=False)
        self.assertIsNotNone(learned_model)
        self.assertTrue(learned_model.initial_state.is_accepting)

    def test_non_deterministic_data_is_rejected(self):
        _, alphabet = balanced_parens_vpa_with_internal()
        data = [(('(', ')'), True), (('(', ')'), False)]
        self.assertIsNone(run_PAPNI(data, alphabet, print_info=False))

    def test_data_without_a_single_balanced_sequence(self):
        _, alphabet = balanced_parens_vpa_with_internal()
        # every sequence is dropped, so the model is the one state accepting nothing, and it keeps the alphabet it
        # was given rather than the empty one its transitions would imply
        for data in ([], [((')',), False), (('(',), False)]):
            with self.subTest(data=data):
                learned_model = run_PAPNI(data, alphabet, print_info=False)
                self.assertEqual(1, len(learned_model.states))
                self.assertFalse(learned_model.initial_state.is_accepting)
                self.assertEqual(alphabet.get_merged_alphabet(),
                                 learned_model.input_alphabet.get_merged_alphabet())

    def test_alphabet_mixing_symbol_types(self):
        """
        Symbols of different types are not mutually comparable, so the order the merging works in must not compare
        them against each other. DotModels/arithmetics.dot has such an alphabet, its internal symbols being 1 and +.
        """
        vpa, alphabet = balanced_parens_vpa_with_internal((1, '+'))
        data = generate_data(vpa, alphabet, depth=3)

        learned_model = run_PAPNI(data, alphabet, print_info=False)
        for seq, label in data:
            if is_balanced(list(seq), alphabet):
                self.assertEqual(label, learned_model.accepts(seq), f'mismatch on sequence {seq}')

    def test_learned_model_conforms_to_randomly_generated_data(self):
        """
        Conforming to the data is what a passive algorithm guarantees, whether or not the data happens to
        characterize the language, so this checks it on data that is merely sampled.
        """
        from aalpy.utils import generate_input_output_data_from_vpa
        from aalpy.utils.BenchmarkVpaModels import (vpa_L1, vpa_L3, vpa_L11, vpa_json, vpa_for_L16,
                                                    vpa_for_even_parentheses)

        for vpa in (vpa_L1(), vpa_L3(), vpa_L11(), vpa_json(), vpa_for_L16(), nested_blocks_vpa(),
                    vpa_for_even_parentheses()):
            with self.subTest(model=vpa.to_state_setup()):
                alphabet = vpa.input_alphabet
                random.seed(1)
                data = generate_input_output_data_from_vpa(vpa, num_sequences=1000, max_seq_len=14)

                learned_model = run_PAPNI(data, alphabet, print_info=False)
                for seq, label in data:
                    # unbalanced sequences are dropped, so the model makes no promise about them
                    if not is_balanced(list(seq), alphabet):
                        continue
                    self.assertEqual(label, learned_model.accepts(seq),
                                     f'learned model does not conform to the data on {seq}')


class TestPapniLearnsFromExhaustiveData(unittest.TestCase):
    """
    Data holding every balanced word up to a length is enough to identify these languages, so PAPNI has to learn
    them exactly. The learned models are compared to the ground truth on all words up to a (longer) length.
    """

    def test_learns_benchmark_vpas(self):
        from aalpy.utils.BenchmarkVpaModels import (vpa_L1, vpa_L10, vpa_L11, vpa_for_L16, vpa_for_even_parentheses,
                                                    vpa_for_odd_parentheses)

        for vpa in (vpa_L1(), vpa_L10(), vpa_L11(), vpa_for_L16(), vpa_for_odd_parentheses(),
                    vpa_for_even_parentheses()):
            data = exhaustive_balanced_data(vpa)
            # both merging strategies have to identify these languages from the exhaustive data
            for algorithm in ('gsm', 'edsm'):
                with self.subTest(model=vpa.to_state_setup(), algorithm=algorithm):
                    learned_model = run_PAPNI(data, vpa.input_alphabet, automaton_type='vpa', print_info=False,
                                              algorithm=algorithm)
                    self.assertIsNone(find_disagreement(vpa, learned_model))

    def test_vpa_cast_accepts_the_same_language_as_the_learned_sevpa(self):
        from aalpy.utils.BenchmarkVpaModels import vpa_L2, vpa_L12, vpa_for_even_parentheses

        for vpa in (vpa_L2(), vpa_L12(), vpa_for_even_parentheses(), nested_blocks_vpa()):
            with self.subTest(model=vpa.to_state_setup()):
                alphabet = vpa.input_alphabet
                data = exhaustive_balanced_data(vpa)

                learned_sevpa = run_PAPNI(data, alphabet, automaton_type='sevpa', print_info=False)
                learned_vpa = run_PAPNI(data, alphabet, automaton_type='vpa', print_info=False)
                self.assertIsInstance(learned_sevpa, Sevpa)
                self.assertIsInstance(learned_vpa, Vpa)

                for seq in all_words_within_budget(alphabet.get_merged_alphabet()):
                    self.assertEqual(learned_sevpa.accepts(seq), learned_vpa.accepts(seq),
                                     f'the cast to a VPA changed the language on {seq}')

    def test_vpa_cast_of_the_benchmark_sevpas_accepts_the_same_language(self):
        """The cast on hand written 1-SEVPAs, so that it is checked independently of what the learner produces."""
        import aalpy.utils.BenchmarkSevpaModels as benchmark_sevpas

        for name in sorted(n for n in dir(benchmark_sevpas) if n.startswith('sevpa_for')):
            sevpa = getattr(benchmark_sevpas, name)()
            with self.subTest(model=name):
                vpa = vpa_from_sevpa(sevpa)
                for seq in all_words_within_budget(sevpa.get_input_alphabet().get_merged_alphabet()):
                    self.assertEqual(sevpa.accepts(seq), vpa.accepts(seq),
                                     f'the cast to a VPA changed the language on {seq}')

    def test_learns_random_sevpas(self):
        """The 1-SEVPA is the model PAPNI learns, so random ones make a check independent of the benchmarks."""
        learned = 0
        for seed in range(25):
            random.seed(seed)
            sevpa = generate_random_sevpa(num_states=random.randint(2, 3),
                                          internal_alphabet_size=random.randint(0, 1),
                                          call_alphabet_size=1,
                                          return_alphabet_size=random.randint(1, 2),
                                          acceptance_prob=0.4, return_transition_prob=0.5)
            target = vpa_from_sevpa(sevpa)

            with self.subTest(seed=seed, model=sevpa.to_state_setup()):
                learned_model = run_PAPNI(exhaustive_balanced_data(target), target.input_alphabet, print_info=False)
                learned += find_disagreement(target, learned_model) is None

        self.assertGreater(learned, 20, 'too few random SEVPAs were learned exactly from exhaustive data')


if __name__ == '__main__':
    unittest.main()
