import unittest

from aalpy.automata import Sevpa, SevpaAlphabet, SevpaState, SevpaTransition


def balanced_parens_sevpa():
    """
    Single-module 1-SEVPA recognizing the Dyck language, built via create_daisy_hypothesis:
    call='(' pushes (q0, '('), return ')' pops iff the stack guard matches.
    """
    alphabet = SevpaAlphabet(internal_alphabet=[], call_alphabet=['('], return_alphabet=[')'])
    q0 = SevpaState('q0', is_accepting=True)
    sevpa = Sevpa.create_daisy_hypothesis(q0, alphabet)
    return sevpa, q0, alphabet


class TestSevpaAlphabet(unittest.TestCase):
    def test_get_merged_alphabet(self):
        alphabet = SevpaAlphabet(internal_alphabet=['i'], call_alphabet=['c'], return_alphabet=['r'])
        self.assertEqual(alphabet.get_merged_alphabet(), ['i', 'c', 'r'])


class TestSevpaState(unittest.TestCase):
    def test_default_transitions_is_empty_defaultdict(self):
        state = SevpaState('s')
        self.assertEqual(state.transitions['unused_key'], [])

    def test_default_not_accepting(self):
        state = SevpaState('s')
        self.assertFalse(state.is_accepting)


class TestSevpaStep(unittest.TestCase):
    def test_reset_to_initial_returns_true_for_accepting_empty_stack(self):
        sevpa, q0, _ = balanced_parens_sevpa()
        self.assertTrue(sevpa.reset_to_initial())

    def test_accepts_empty_word(self):
        sevpa, *_ = balanced_parens_sevpa()
        sevpa.reset_to_initial()
        self.assertTrue(sevpa.step(None))

    def test_accepts_balanced_word(self):
        sevpa, *_ = balanced_parens_sevpa()
        sevpa.reset_to_initial()
        sevpa.step('(')
        result = sevpa.step(')')
        self.assertTrue(result)
        self.assertEqual(sevpa.stack, [Sevpa.empty])

    def test_rejects_incomplete_word(self):
        sevpa, *_ = balanced_parens_sevpa()
        sevpa.reset_to_initial()
        result = sevpa.step('(')
        self.assertFalse(result)
        self.assertEqual(len(sevpa.stack), 2)

    def test_unmatched_return_symbol_traps_in_error_state(self):
        sevpa, *_ = balanced_parens_sevpa()
        sevpa.reset_to_initial()
        result = sevpa.step(')')
        self.assertFalse(result)
        self.assertTrue(sevpa.error_state_reached)

        # once trapped, further steps stay False without raising
        self.assertFalse(sevpa.step('('))
        self.assertFalse(sevpa.step(None))

    def test_nested_balanced_word(self):
        sevpa, *_ = balanced_parens_sevpa()
        sevpa.reset_to_initial()
        outputs = [sevpa.step(c) for c in '(())']
        self.assertEqual(outputs, [False, False, False, True])

    def test_reset_to_initial_clears_error_state(self):
        sevpa, *_ = balanced_parens_sevpa()
        sevpa.reset_to_initial()
        sevpa.step(')')
        self.assertTrue(sevpa.error_state_reached)
        sevpa.reset_to_initial()
        self.assertFalse(sevpa.error_state_reached)


class TestSevpaStructural(unittest.TestCase):
    def test_get_input_alphabet(self):
        sevpa, *_ = balanced_parens_sevpa()
        alphabet = sevpa.get_input_alphabet()
        self.assertEqual(alphabet.call_alphabet, ['('])
        self.assertEqual(alphabet.return_alphabet, [')'])
        self.assertEqual(alphabet.internal_alphabet, [])

    def test_get_state_by_id(self):
        sevpa, q0, _ = balanced_parens_sevpa()
        self.assertIs(sevpa.get_state_by_id('q0'), q0)
        self.assertIsNone(sevpa.get_state_by_id('does_not_exist'))

    def test_get_error_state_none_for_single_state_automaton(self):
        sevpa, *_ = balanced_parens_sevpa()
        # only state present is initial & accepting, so there is no error state candidate
        self.assertIsNone(sevpa.get_error_state())

    def test_get_allowed_call_transitions(self):
        sevpa, q0, _ = balanced_parens_sevpa()
        allowed = sevpa.get_allowed_call_transitions()
        self.assertEqual(allowed['('], {'q0'})


class TestSevpaStateSetupRoundtrip(unittest.TestCase):
    def test_to_state_setup_from_state_setup_roundtrip(self):
        sevpa, q0, _ = balanced_parens_sevpa()
        setup = sevpa.to_state_setup()
        rebuilt = Sevpa.from_state_setup(setup, init_state_id='q0')

        for w in [[], ['('], ['(', ')'], ['(', '(', ')', ')']]:
            rebuilt.reset_to_initial()
            outputs = [rebuilt.step(letter) for letter in w]
            sevpa.reset_to_initial()
            expected = [sevpa.step(letter) for letter in w]
            self.assertEqual(outputs, expected)


class TestSevpaExecuteSequence(unittest.TestCase):
    def test_execute_sequence_from_initial_state(self):
        sevpa, q0, _ = balanced_parens_sevpa()
        result = sevpa.execute_sequence(q0, ['(', '(', ')', ')'])
        self.assertEqual(result, [False, False, False, True])

    def test_execute_sequence_from_state_with_different_prefix_raises(self):
        sevpa, q0, _ = balanced_parens_sevpa()
        foreign = SevpaState('foreign')
        foreign.prefix = ('unrelated',)
        with self.assertRaises(AssertionError):
            sevpa.execute_sequence(foreign, ['('])


class TestSevpaDeleteState(unittest.TestCase):
    def test_delete_state_removes_state_and_references(self):
        alphabet = SevpaAlphabet(internal_alphabet=['i'], call_alphabet=[], return_alphabet=[])
        q0 = SevpaState('q0', is_accepting=True)
        q1 = SevpaState('q1', is_accepting=False)
        q0.transitions['i'].append(SevpaTransition(q1, 'i', None))
        q1.transitions['i'].append(SevpaTransition(q1, 'i', None))
        sevpa = Sevpa(q0, [q0, q1])

        sevpa.delete_state(q1)

        self.assertNotIn(q1, sevpa.states)
        self.assertEqual(q0.transitions['i'], [])

    def test_delete_state_none_is_a_no_op(self):
        sevpa, q0, _ = balanced_parens_sevpa()
        num_states_before = len(sevpa.states)
        sevpa.delete_state(None)
        self.assertEqual(len(sevpa.states), num_states_before)


if __name__ == '__main__':
    unittest.main()
