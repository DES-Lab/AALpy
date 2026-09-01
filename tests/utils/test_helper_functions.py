import random
import unittest

from aalpy.automata import (Dfa, DfaState, MooreMachine, MooreState, MealyMachine, MealyState, Mdp, MdpState,
                            MarkovChain, Vpa, VpaState, VpaAlphabet, VpaTransition)
from aalpy.utils.HelperFunctions import (
    extend_set,
    all_prefixes,
    all_suffixes,
    random_string_generator,
    is_suffix_of,
    get_cex_prefixes,
    make_input_complete,
    convert_i_o_traces_for_RPNI,
    is_balanced,
    product_with_possible_empty_iterable,
    dfa_from_moore,
    mc_from_mdp,
    mc_format_to_mdp,
    generate_input_output_data_from_automata,
    generate_input_output_data_from_vpa,
)


def parity_dfa():
    q0 = DfaState('q0', is_accepting=True)
    q1 = DfaState('q1', is_accepting=False)
    q0.transitions = {'a': q1, 'b': q0}
    q1.transitions = {'a': q0, 'b': q1}
    return Dfa(q0, [q0, q1])


def incomplete_dfa():
    q0 = DfaState('q0', is_accepting=True)
    q1 = DfaState('q1', is_accepting=False)
    q0.transitions = {'a': q1, 'b': q0}
    q1.transitions = {'a': q0}
    return Dfa(q0, [q0, q1])


def incomplete_moore():
    q0 = MooreState('q0', output='x')
    q1 = MooreState('q1', output='y')
    q0.transitions = {'a': q1, 'b': q0}
    q1.transitions = {'a': q0}
    return MooreMachine(q0, [q0, q1])


def incomplete_mealy():
    q0 = MealyState('q0')
    q1 = MealyState('q1')
    q0.transitions = {'a': q1, 'b': q0}
    q0.output_fun = {'a': 'o1', 'b': 'o2'}
    q1.transitions = {'a': q0}
    q1.output_fun = {'a': 'o2'}
    return MealyMachine(q0, [q0, q1])


class TestExtendSet(unittest.TestCase):
    def test_adds_only_new_elements(self):
        base = [1, 2, 3]
        added = extend_set(base, [2, 3, 4, 5])
        self.assertEqual(added, [4, 5])
        self.assertEqual(base, [1, 2, 3, 4, 5])

    def test_no_new_elements(self):
        base = [1, 2]
        added = extend_set(base, [1, 2])
        self.assertEqual(added, [])
        self.assertEqual(base, [1, 2])


class TestAllPrefixesSuffixes(unittest.TestCase):
    def test_all_prefixes(self):
        self.assertEqual(all_prefixes(['a', 'b', 'c']), [('a',), ('a', 'b'), ('a', 'b', 'c')])

    def test_all_prefixes_empty(self):
        self.assertEqual(all_prefixes([]), [])

    def test_all_suffixes(self):
        self.assertEqual(all_suffixes(['a', 'b', 'c']), [('c',), ('b', 'c'), ('a', 'b', 'c')])

    def test_all_suffixes_empty(self):
        self.assertEqual(all_suffixes([]), [])


class TestRandomStringGenerator(unittest.TestCase):
    def test_default_length(self):
        s = random_string_generator()
        self.assertEqual(len(s), 10)

    def test_custom_length_and_chars(self):
        s = random_string_generator(size=5, chars='x')
        self.assertEqual(s, 'xxxxx')

    def test_zero_length(self):
        self.assertEqual(random_string_generator(size=0), '')


class TestIsSuffixOf(unittest.TestCase):
    def test_true_case(self):
        self.assertTrue(is_suffix_of(('b', 'c'), ('a', 'b', 'c')))

    def test_false_case(self):
        self.assertFalse(is_suffix_of(('a', 'c'), ('a', 'b', 'c')))

    def test_suffix_longer_than_trace(self):
        self.assertFalse(is_suffix_of(('a', 'b', 'c'), ('b', 'c')))

    def test_empty_suffix_always_matches(self):
        # regression test: trace[-len(suffix):] with len(suffix) == 0 slices with `-0`, which Python
        # treats the same as `0` (the whole trace) rather than an empty slice - so an empty suffix
        # used to compare the entire trace against () instead of always matching.
        self.assertTrue(is_suffix_of((), ('a', 'b')))

    def test_empty_suffix_of_empty_trace(self):
        self.assertTrue(is_suffix_of((), ()))


class TestGetCexPrefixes(unittest.TestCase):
    def test_mdp_prefixes(self):
        cex = ('i1', 'o1', 'i2', 'o2')
        prefixes = get_cex_prefixes(cex, 'mdp')
        self.assertEqual(prefixes, [('i1',), ('i1', 'o1', 'i2')])

    def test_smm_prefixes(self):
        cex = ('i1', 'o1', 'i2', 'o2')
        prefixes = get_cex_prefixes(cex, 'smm')
        self.assertEqual(prefixes, [(), ('i1', 'o1'), ('i1', 'o1', 'i2', 'o2')])


class TestMakeInputComplete(unittest.TestCase):
    def test_already_complete_dfa_returned_unchanged(self):
        dfa = parity_dfa()
        result = make_input_complete(dfa)
        self.assertIs(result, dfa)

    def test_dfa_self_loop(self):
        dfa = incomplete_dfa()
        make_input_complete(dfa, missing_transition_go_to='self_loop')
        for state in dfa.states:
            self.assertEqual(set(state.transitions.keys()), {'a', 'b'})
        self.assertIs(dfa.states[1].transitions['b'], dfa.states[1])

    def test_dfa_sink_state(self):
        dfa = incomplete_dfa()
        make_input_complete(dfa, missing_transition_go_to='sink_state')
        sink_states = [s for s in dfa.states if s.state_id == 'sink']
        self.assertEqual(len(sink_states), 1)
        sink = sink_states[0]
        self.assertFalse(sink.is_accepting)
        q1 = next(s for s in dfa.states if s.state_id == 'q1')
        self.assertIs(q1.transitions['b'], sink)

    def test_moore_self_loop(self):
        moore = incomplete_moore()
        make_input_complete(moore, missing_transition_go_to='self_loop')
        for state in moore.states:
            self.assertEqual(set(state.transitions.keys()), {'a', 'b'})

    def test_mealy_self_loop_epsilon_output(self):
        mealy = incomplete_mealy()
        make_input_complete(mealy, missing_transition_go_to='self_loop')
        self.assertIs(mealy.states[1].transitions['b'], mealy.states[1])
        self.assertEqual(mealy.states[1].output_fun['b'], 'epsilon')

    def test_invalid_missing_transition_strategy_raises(self):
        dfa = incomplete_dfa()
        with self.assertRaises(AssertionError):
            make_input_complete(dfa, missing_transition_go_to='not_a_strategy')


class TestConvertIOTracesForRPNI(unittest.TestCase):
    def test_mealy_conversion(self):
        sequences = [[(1, 'a'), (2, 'b'), (3, 'c')], [(6, 'e'), (4, 'e'), (3, 'c')]]
        result = convert_i_o_traces_for_RPNI(sequences, automaton_type='mealy')
        self.assertEqual(result, [
            ((1,), 'a'), ((1, 2), 'b'), ((1, 2, 3), 'c'),
            ((6,), 'e'), ((6, 4), 'e'), ((6, 4, 3), 'c'),
        ])

    def test_dfa_conversion_includes_initial_output(self):
        sequences = [[True, (1, False)]]
        result = convert_i_o_traces_for_RPNI(sequences, automaton_type='dfa')
        self.assertEqual(result, [((), True), ((1,), False)])

    def test_invalid_automaton_type_raises(self):
        with self.assertRaises(ValueError):
            convert_i_o_traces_for_RPNI([[(1, 'a')]], automaton_type='bogus')

    def test_deduplicates_repeated_prefixes(self):
        sequences = [[(1, 'a')], [(1, 'a')]]
        result = convert_i_o_traces_for_RPNI(sequences, automaton_type='mealy')
        self.assertEqual(result, [((1,), 'a')])


class TestIsBalanced(unittest.TestCase):
    def alphabet(self):
        return VpaAlphabet(internal_alphabet=['i'], call_alphabet=['c'], return_alphabet=['r'])

    def test_balanced_sequence(self):
        self.assertTrue(is_balanced(['c', 'i', 'r'], self.alphabet()))

    def test_empty_sequence_is_balanced(self):
        self.assertTrue(is_balanced([], self.alphabet()))

    def test_unbalanced_more_returns_than_calls(self):
        self.assertFalse(is_balanced(['r'], self.alphabet()))

    def test_unbalanced_unclosed_call(self):
        self.assertFalse(is_balanced(['c', 'c', 'r'], self.alphabet()))


class TestProductWithPossibleEmptyIterable(unittest.TestCase):
    def test_all_nonempty_behaves_like_normal_product(self):
        result = list(product_with_possible_empty_iterable([1, 2], ['a']))
        self.assertEqual(result, [(1, 'a'), (2, 'a')])

    def test_one_empty_iterable_ignored(self):
        result = list(product_with_possible_empty_iterable([1, 2], []))
        self.assertEqual(result, [(1,), (2,)])

    def test_all_empty_iterables_gives_empty_tuple(self):
        result = list(product_with_possible_empty_iterable([], []))
        self.assertEqual(result, [()])


class TestDfaFromMoore(unittest.TestCase):
    def test_boolean_output_moore_converts(self):
        q0 = MooreState('q0', output=True)
        q1 = MooreState('q1', output=False)
        q0.transitions = {'a': q1}
        q1.transitions = {'a': q0}
        moore = MooreMachine(q0, [q0, q1])

        dfa = dfa_from_moore(moore)
        self.assertIsInstance(dfa, Dfa)
        self.assertTrue(dfa.initial_state.is_accepting)
        self.assertFalse(dfa.initial_state.transitions['a'].is_accepting)

    def test_none_output_treated_as_non_accepting(self):
        q0 = MooreState('q0', output=None)
        q0.transitions = {'a': q0}
        moore = MooreMachine(q0, [q0])
        dfa = dfa_from_moore(moore)
        self.assertFalse(dfa.initial_state.is_accepting)

    def test_non_boolean_output_raises(self):
        q0 = MooreState('q0', output='not_boolean')
        q0.transitions = {'a': q0}
        moore = MooreMachine(q0, [q0])
        with self.assertRaises(ValueError):
            dfa_from_moore(moore)


class TestMcFromMdp(unittest.TestCase):
    def make_mdp(self):
        s0 = MdpState('s0', output='o0')
        s1 = MdpState('s1', output='o1')
        s0.transitions['i'].append((s1, 1.0))
        s1.transitions['i'].append((s0, 1.0))
        return Mdp(s0, [s0, s1])

    def test_single_input_conversion(self):
        mdp = self.make_mdp()
        mc = mc_from_mdp(mdp)
        self.assertIsInstance(mc, MarkovChain)
        self.assertEqual(mc.initial_state.state_id, 's0')
        self.assertEqual(mc.initial_state.transitions, [(mc.states[1] if mc.states[1].state_id == 's1' else mc.states[0], 1.0)])

    def test_explicit_input_symbol(self):
        mdp = self.make_mdp()
        mc = mc_from_mdp(mdp, input_symbol='i')
        self.assertIsInstance(mc, MarkovChain)

    def test_multiple_inputs_without_symbol_raises(self):
        s0 = MdpState('s0', output='o0')
        s1 = MdpState('s1', output='o1')
        s0.transitions['i1'].append((s1, 1.0))
        s0.transitions['i2'].append((s1, 1.0))
        s1.transitions['i1'].append((s0, 1.0))
        s1.transitions['i2'].append((s0, 1.0))
        mdp = Mdp(s0, [s0, s1])
        with self.assertRaises(ValueError):
            mc_from_mdp(mdp)


class TestMcFormatToMdp(unittest.TestCase):
    def test_wraps_non_initial_elements_with_input_label(self):
        data = [['out0', 'a', 'b']]
        result = mc_format_to_mdp(data)
        self.assertEqual(result, [['out0', ('Input', 'a'), ('Input', 'b')]])

    def test_empty_data(self):
        self.assertEqual(mc_format_to_mdp([]), [])


class TestGenerateInputOutputDataFromAutomata(unittest.TestCase):
    def test_io_traces_format(self):
        random.seed(1)
        dfa = parity_dfa()
        data = generate_input_output_data_from_automata(dfa, num_sequences=5, min_seq_len=1, max_seq_len=3)
        self.assertEqual(len(data), 5)
        for trace in data:
            for i, o in trace:
                self.assertIn(i, {'a', 'b'})
                self.assertIn(o, {True, False})

    def test_labeled_sequences_format(self):
        random.seed(2)
        dfa = parity_dfa()
        data = generate_input_output_data_from_automata(dfa, num_sequences=5, min_seq_len=1, max_seq_len=3,
                                                        sequance_type='labeled_sequences')
        self.assertEqual(len(data), 5)
        for seq, label in data:
            self.assertIsInstance(seq, list)
            self.assertIn(label, {True, False})

    def test_invalid_sequence_type_raises(self):
        dfa = parity_dfa()
        with self.assertRaises(AssertionError):
            generate_input_output_data_from_automata(dfa, num_sequences=1, sequance_type='bogus')


class TestGenerateInputOutputDataFromVpa(unittest.TestCase):
    def make_simple_vpa(self):
        q0 = VpaState('q0', is_accepting=True)
        q1 = VpaState('q1', is_accepting=False)
        q0.transitions['i'].append(VpaTransition(q0, q0, 'i', None, None))
        q0.transitions['c'].append(VpaTransition(q0, q1, 'c', 'push', 'c'))
        q1.transitions['r'].append(VpaTransition(q1, q0, 'r', 'pop', 'c'))
        return Vpa(q0, [q0, q1])

    def test_generates_at_least_requested_number_of_sequences(self):
        # the generation loop only re-checks its stopping condition after a full inner sequence of
        # length max_seq_len is generated, so it can slightly overshoot num_sequences.
        random.seed(3)
        vpa = self.make_simple_vpa()
        data = generate_input_output_data_from_vpa(vpa, num_sequences=10, max_seq_len=4)
        self.assertGreaterEqual(len(data), 10)
        self.assertLessEqual(len(data), 10 + 4)
        for seq, output in data:
            self.assertIsInstance(seq, tuple)

    def test_respects_max_attempts(self):
        random.seed(4)
        vpa = self.make_simple_vpa()
        data = generate_input_output_data_from_vpa(vpa, num_sequences=1000, max_seq_len=2, max_attempts=5)
        self.assertLessEqual(len(data), 1000)


if __name__ == '__main__':
    unittest.main()
