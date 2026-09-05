import random
import unittest
from itertools import product

from aalpy.automata.Sevpa import Sevpa
from aalpy.automata.Vpa import (Vpa, VpaAlphabet, VpaState, VpaTransition, find_vpa_counterexample,
                                vpa_from_sevpa)
from aalpy.learning_algs.deterministic_passive.PAPNI import run_PAPNI
from aalpy.utils import generate_random_sevpa, get_characterizing_sequences, is_balanced


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
            vpa.reset_to_initial()
            outputs = vpa.execute_sequence(vpa.initial_state, seq, [])
            label = outputs[-1] if outputs else vpa.initial_state.is_accepting
            data.append((seq, label))
    return data


def all_words_within_budget(alphabet, budget=5000):
    """All words over the alphabet, longest first up to the length at which their number exceeds the budget."""
    max_length, count = 0, 1
    while count * len(alphabet) <= budget:
        count *= len(alphabet)
        max_length += 1
    for length in range(max_length + 1):
        yield from product(alphabet, repeat=length)


def accepts(model, seq):
    """Whether the model accepts the sequence. Vpa and Sevpa share reset_to_initial and step, so both work."""
    model.reset_to_initial()
    outputs = [model.step(symbol) for symbol in seq]
    return outputs[-1] if outputs else model.initial_state.is_accepting


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

            self.assertEqual(accepts(learned_model, seq), expected, f'mismatch on sequence {seq}')

    def test_learns_vpas_with_call_symbol_conflicts(self):
        """
        Two VPAs whose pushed stack symbol is not determined by the call symbol. PAPNI learns the congruence of the
        canonical single entry VPA, whose stack symbols carry the state a call was read in, so these are within its
        model class.
        """
        from aalpy.utils.BenchmarkVpaModels import vpa_for_L16, vpa_for_nested_blocks

        for vpa in (vpa_for_L16(), vpa_for_nested_blocks()):
            with self.subTest(model=vpa.to_state_setup()):
                alphabet = vpa.get_input_alphabet()
                learned_model = run_PAPNI(get_characterizing_sequences(vpa), alphabet, print_info=False)
                self.assertIsNone(find_vpa_counterexample(vpa, learned_model))

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
                self.assertEqual(label, accepts(learned_model, seq), f'mismatch on sequence {seq}')

    def test_learned_model_conforms_to_randomly_generated_data(self):
        """
        Conforming to the data is what a passive algorithm guarantees, whether or not the data happens to
        characterize the language, so this checks it on data that is merely sampled.
        """
        from aalpy.utils import generate_input_output_data_from_vpa
        from aalpy.utils.BenchmarkVpaModels import (vpa_L1, vpa_L3, vpa_L11, vpa_json, vpa_for_L16,
                                                    vpa_for_nested_blocks, vpa_for_even_parentheses)

        for vpa in (vpa_L1(), vpa_L3(), vpa_L11(), vpa_json(), vpa_for_L16(), vpa_for_nested_blocks(),
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
                    self.assertEqual(label, accepts(learned_model, seq),
                                     f'learned model does not conform to the data on {seq}')


class TestPapniLearnsFromCharacterizingSequences(unittest.TestCase):
    """
    get_characterizing_sequences builds a characteristic sample of the congruence characterizing visibly pushdown
    languages, without running any learner, so these are genuine checks against independently derived data. It is
    the congruence of the canonical single entry VPA, which is exactly the model PAPNI learns.
    """

    def test_learns_benchmark_vpas(self):
        from aalpy.utils.BenchmarkVpaModels import (vpa_L1, vpa_L2, vpa_L3, vpa_L4, vpa_L9, vpa_L10, vpa_L11,
                                                    vpa_L12, vpa_for_even_parentheses, vpa_for_odd_parentheses,
                                                    vpa_json)

        for vpa in (vpa_L1(), vpa_L2(), vpa_L3(), vpa_L4(), vpa_L9(), vpa_L10(), vpa_L11(), vpa_L12(), vpa_json(),
                    vpa_for_odd_parentheses(), vpa_for_even_parentheses()):
            with self.subTest(model=vpa.to_state_setup()):
                data = get_characterizing_sequences(vpa)
                learned_model = run_PAPNI(data, vpa.input_alphabet, automaton_type='vpa', print_info=False)
                self.assertIsNone(find_vpa_counterexample(vpa, learned_model))

    def test_vpa_cast_accepts_the_same_language_as_the_learned_sevpa(self):
        from aalpy.utils.BenchmarkVpaModels import vpa_L2, vpa_L12, vpa_for_even_parentheses, vpa_for_nested_blocks

        for vpa in (vpa_L2(), vpa_L12(), vpa_for_even_parentheses(), vpa_for_nested_blocks()):
            with self.subTest(model=vpa.to_state_setup()):
                alphabet = vpa.input_alphabet
                data = get_characterizing_sequences(vpa)

                learned_sevpa = run_PAPNI(data, alphabet, automaton_type='sevpa', print_info=False)
                learned_vpa = run_PAPNI(data, alphabet, automaton_type='vpa', print_info=False)
                self.assertIsInstance(learned_sevpa, Sevpa)
                self.assertIsInstance(learned_vpa, Vpa)

                for seq in all_words_within_budget(alphabet.get_merged_alphabet()):
                    self.assertEqual(accepts(learned_sevpa, seq), accepts(learned_vpa, seq),
                                     f'the cast to a VPA changed the language on {seq}')

    def test_vpa_cast_of_the_benchmark_sevpas_accepts_the_same_language(self):
        """The cast on hand written 1-SEVPAs, so that it is checked independently of what the learner produces."""
        import aalpy.utils.BenchmarkSevpaModels as benchmark_sevpas

        for name in sorted(n for n in dir(benchmark_sevpas) if n.startswith('sevpa_for')):
            sevpa = getattr(benchmark_sevpas, name)()
            with self.subTest(model=name):
                vpa = vpa_from_sevpa(sevpa)
                for seq in all_words_within_budget(sevpa.get_input_alphabet().get_merged_alphabet()):
                    self.assertEqual(accepts(sevpa, seq), accepts(vpa, seq),
                                     f'the cast to a VPA changed the language on {seq}')

class TestPapniLearnsRandomSevpas(unittest.TestCase):
    """
    The 1-SEVPA is the model PAPNI learns, so a randomly generated one and the characteristic sample of its
    language together make an end to end check that does not depend on any hand written benchmark.
    """

    def test_learns_random_sevpas_from_their_characterizing_sequences(self):
        checked = 0
        for seed in range(25):
            random.seed(seed)
            sevpa = generate_random_sevpa(num_states=random.randint(2, 3),
                                          internal_alphabet_size=random.randint(0, 1),
                                          call_alphabet_size=1,
                                          return_alphabet_size=random.randint(1, 2),
                                          acceptance_prob=0.4, return_transition_prob=0.5)

            # get_characterizing_sequences works on a VPA, and the cast is the identity on the language
            target = vpa_from_sevpa(sevpa)
            data = get_characterizing_sequences(target)

            # the characteristic sample of the congruence is exponential in the size of the automaton, and a random
            # one occasionally lands far beyond what is worth running in a unit test
            if len(data) > 5000:
                continue

            with self.subTest(seed=seed, model=sevpa.to_state_setup()):
                learned_model = run_PAPNI(data, target.input_alphabet, automaton_type='vpa', print_info=False)
                self.assertIsNone(find_vpa_counterexample(target, learned_model))
                checked += 1

        self.assertGreater(checked, 10, 'too few random SEVPAs were small enough to be checked')


class TestVpaCallSymbolConflicts(unittest.TestCase):
    def test_vpas_pushing_more_than_one_stack_symbol_per_call_symbol_are_detectable(self):
        from aalpy.automata.Vpa import vpa_call_symbol_conflicts
        from aalpy.utils.BenchmarkVpaModels import vpa_L1, vpa_for_L16, vpa_for_nested_blocks

        # L16 pushes '$' on the first 'a' and 'x' on every further one
        self.assertEqual({'a': {'$', 'x'}}, vpa_call_symbol_conflicts(vpa_for_L16()))
        self.assertEqual({'(': {'$', '('}}, vpa_call_symbol_conflicts(vpa_for_nested_blocks()))
        self.assertEqual({}, vpa_call_symbol_conflicts(vpa_L1()))


if __name__ == '__main__':
    unittest.main()
