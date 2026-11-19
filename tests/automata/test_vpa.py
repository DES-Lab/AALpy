import random
import unittest

from aalpy.automata import Vpa, VpaAlphabet, VpaState, VpaTransition


def balanced_parens_vpa():
    """
    Single-state VPA recognizing the Dyck language over '(' (push) / ')' (pop).
    q0 is initial and accepting; accepted iff the stack is empty.
    """
    q0 = VpaState('q0', is_accepting=True)
    q0.transitions['('].append(VpaTransition(q0, q0, '(', 'push', '('))
    q0.transitions[')'].append(VpaTransition(q0, q0, ')', 'pop', '('))
    return Vpa(q0, [q0]), q0


class TestVpaState(unittest.TestCase):
    def test_default_transitions_is_empty_defaultdict(self):
        state = VpaState('s')
        self.assertEqual(state.transitions['unused_key'], [])

    def test_default_not_accepting(self):
        state = VpaState('s')
        self.assertFalse(state.is_accepting)


class TestVpaStep(unittest.TestCase):
    def test_accepts_empty_word(self):
        vpa, q0 = balanced_parens_vpa()
        vpa.reset_to_initial()
        self.assertTrue(vpa.step(None))

    def test_accepts_balanced_word(self):
        vpa, q0 = balanced_parens_vpa()
        vpa.reset_to_initial()
        vpa.step('(')
        result = vpa.step(')')
        self.assertTrue(result)
        self.assertEqual(vpa.stack, [])

    def test_rejects_incomplete_word(self):
        vpa, q0 = balanced_parens_vpa()
        vpa.reset_to_initial()
        result = vpa.step('(')
        self.assertFalse(result)
        self.assertEqual(vpa.stack, ['('])

    def test_unmatched_return_symbol_traps_in_error_state(self):
        vpa, q0 = balanced_parens_vpa()
        vpa.reset_to_initial()
        result = vpa.step(')')
        self.assertFalse(result)
        self.assertIs(vpa.current_state, Vpa.error_state)

        # once trapped, further steps stay False without raising
        self.assertFalse(vpa.step('('))
        self.assertFalse(vpa.step(None))

    def test_nested_balanced_word(self):
        vpa, q0 = balanced_parens_vpa()
        vpa.reset_to_initial()
        outputs = [vpa.step(c) for c in '(())']
        self.assertEqual(outputs, [False, False, False, True])

    def test_top_of_empty_stack_is_empty_list(self):
        vpa, q0 = balanced_parens_vpa()
        vpa.reset_to_initial()
        self.assertEqual(vpa.top(), [])

    def test_top_reflects_last_pushed_symbol(self):
        vpa, q0 = balanced_parens_vpa()
        vpa.reset_to_initial()
        vpa.step('(')
        self.assertEqual(vpa.top(), '(')


class TestVpaExecuteSequence(unittest.TestCase):
    def test_execute_sequence_matches_stepwise(self):
        vpa, q0 = balanced_parens_vpa()
        result = vpa.execute_sequence(q0, ['(', '(', ')', ')'], stack=[])
        self.assertEqual(result, [False, False, False, True])
        self.assertEqual(vpa.stack, [])

    def test_execute_sequence_empty_returns_empty_list(self):
        vpa, q0 = balanced_parens_vpa()
        self.assertEqual(vpa.execute_sequence(q0, [], stack=[]), [])

    def test_execute_sequence_ignores_leftover_stack_from_prior_use(self):
        vpa, q0 = balanced_parens_vpa()
        vpa.reset_to_initial()
        vpa.step('(')  # leaves the stack non-empty: ['(']
        self.assertEqual(vpa.stack, ['('])

        result = vpa.execute_sequence(q0, [')'], stack=[])
        # a stale stack would make ')' match and incorrectly report acceptance;
        # execute_sequence must start from the given stack, not whatever was left over
        self.assertEqual(result, [False])

    def test_execute_sequence_resumes_from_explicit_stack(self):
        vpa, q0 = balanced_parens_vpa()
        result = vpa.execute_sequence(q0, [')'], stack=['('])
        # resuming as if one '(' had already been pushed makes the lone ')' balance out
        self.assertEqual(result, [True])
        self.assertEqual(vpa.stack, [])

    def test_execute_sequence_with_explicit_stack_matches_manual_stepping(self):
        vpa, q0 = balanced_parens_vpa()
        vpa.reset_to_initial()
        vpa.step('(')
        vpa.step('(')
        stack_after_two_pushes = list(vpa.stack)
        manual_result = [vpa.step(c) for c in '))']

        result = vpa.execute_sequence(q0, [')', ')'], stack=stack_after_two_pushes)
        self.assertEqual(result, manual_result)


class TestVpaStructural(unittest.TestCase):
    def test_get_input_alphabet(self):
        vpa, q0 = balanced_parens_vpa()
        alphabet = vpa.get_input_alphabet()
        self.assertEqual(alphabet.call_alphabet, ['('])
        self.assertEqual(alphabet.return_alphabet, [')'])
        self.assertEqual(alphabet.internal_alphabet, [])

    def test_get_merged_alphabet(self):
        alphabet = VpaAlphabet(internal_alphabet=['i'], call_alphabet=['c'], return_alphabet=['r'])
        self.assertEqual(alphabet.get_merged_alphabet(), ['i', 'c', 'r'])

    def test_is_input_complete_true(self):
        vpa, q0 = balanced_parens_vpa()
        self.assertTrue(vpa.is_input_complete())

    def test_is_input_complete_false(self):
        # the alphabet is inferred from ALL transitions present in the automaton, so we need a second
        # state that does use ')' for the alphabet to include it, exposing q0's missing transition
        q0 = VpaState('q0', is_accepting=False)
        q1 = VpaState('q1', is_accepting=True)
        q0.transitions['('].append(VpaTransition(q0, q1, '(', 'push', '('))
        q1.transitions[')'].append(VpaTransition(q1, q1, ')', 'pop', '('))
        # q0 has no ')' transition
        vpa = Vpa(q0, [q0, q1])
        self.assertFalse(vpa.is_input_complete())


class TestVpaStateSetupRoundtrip(unittest.TestCase):
    def test_to_state_setup_from_state_setup_roundtrip(self):
        vpa, q0 = balanced_parens_vpa()
        setup = vpa.to_state_setup()
        rebuilt = Vpa.from_state_setup(setup, init_state_id='q0')

        for w in [[], ['('], ['(', ')'], ['(', '(', ')', ')']]:
            rebuilt.reset_to_initial()
            outputs = [rebuilt.step(letter) for letter in w]
            vpa.reset_to_initial()
            expected = [vpa.step(letter) for letter in w]
            self.assertEqual(outputs, expected)


class TestVpaRandomAcceptingWord(unittest.TestCase):
    def test_generate_random_accepting_word_is_actually_accepting(self):
        vpa, q0 = balanced_parens_vpa()
        random.seed(0)
        # the walk is randomized and may run out of steps before balancing; a generous
        # max_steps budget keeps this reliable without pinning the RNG's exact trajectory
        word = vpa.generate_random_accepting_word(min_steps=2, max_steps=200)
        self.assertIsNotNone(word)

        vpa.reset_to_initial()
        outputs = [vpa.step(letter) for letter in word]
        self.assertTrue(outputs[-1])
        self.assertEqual(vpa.stack, [])


if __name__ == '__main__':
    unittest.main()
