import pickle
import unittest

from aalpy.automata import Dfa, DfaState


def parity_dfa():
    """2-state complete, minimal DFA accepting words with an even number of 'a's."""
    q0 = DfaState('q0', is_accepting=True)
    q1 = DfaState('q1', is_accepting=False)
    q0.transitions = {'a': q1, 'b': q0}
    q1.transitions = {'a': q0, 'b': q1}
    dfa = Dfa(q0, [q0, q1])
    dfa.compute_prefixes()
    return dfa, q0, q1


def contains_a_dfa_non_minimal():
    """3-state DFA accepting words containing at least one 'a'; q1 and q2 are equivalent."""
    q0 = DfaState('q0', is_accepting=False)
    q1 = DfaState('q1', is_accepting=True)
    q2 = DfaState('q2', is_accepting=True)
    q0.transitions = {'a': q1, 'b': q0}
    q1.transitions = {'a': q2, 'b': q2}
    q2.transitions = {'a': q2, 'b': q2}
    dfa = Dfa(q0, [q0, q1, q2])
    dfa.compute_prefixes()
    return dfa, q0, q1, q2


class TestDfaState(unittest.TestCase):
    def test_output_property_mirrors_is_accepting(self):
        state = DfaState('s', is_accepting=True)
        self.assertTrue(state.output)
        state.is_accepting = False
        self.assertFalse(state.output)

    def test_default_not_accepting(self):
        state = DfaState('s')
        self.assertFalse(state.is_accepting)


class TestDfaStep(unittest.TestCase):
    def test_step_transitions_and_reports_acceptance(self):
        dfa, q0, q1 = parity_dfa()
        self.assertTrue(dfa.step('a') is False)  # moved to q1 (non-accepting)
        self.assertIs(dfa.current_state, q1)
        self.assertTrue(dfa.step('a') is True)  # back to q0 (accepting)
        self.assertIs(dfa.current_state, q0)

    def test_step_none_does_not_move_but_reports_current_acceptance(self):
        dfa, q0, q1 = parity_dfa()
        dfa.step('a')
        self.assertIs(dfa.current_state, q1)
        result = dfa.step(None)
        self.assertFalse(result)
        self.assertIs(dfa.current_state, q1)

    def test_step_with_unknown_letter_raises(self):
        dfa, _, _ = parity_dfa()
        with self.assertRaises(KeyError):
            dfa.step('unknown_letter')

    def test_reset_to_initial(self):
        dfa, q0, _ = parity_dfa()
        dfa.step('a')
        dfa.step('a')
        dfa.step('a')
        self.assertIsNot(dfa.current_state, q0)
        dfa.reset_to_initial()
        self.assertIs(dfa.current_state, q0)


class TestDfaExecuteAndOutputSeq(unittest.TestCase):
    def test_execute_sequence_matches_stepwise(self):
        dfa, q0, q1 = parity_dfa()
        result = dfa.execute_sequence(q0, ['a', 'a', 'b', 'a'])
        self.assertEqual(result, [False, True, True, False])

    def test_execute_sequence_empty_returns_state_output(self):
        dfa, q0, q1 = parity_dfa()
        result = dfa.execute_sequence(q1, [])
        self.assertFalse(result)
        self.assertIs(dfa.current_state, q1)

    def test_compute_output_seq_empty(self):
        dfa, q0, q1 = parity_dfa()
        self.assertEqual(dfa.compute_output_seq(q1, []), [False])

    def test_compute_output_seq_does_not_mutate_current_state(self):
        dfa, q0, q1 = parity_dfa()
        dfa.reset_to_initial()
        dfa.compute_output_seq(q1, ['a', 'a', 'a'])
        self.assertIs(dfa.current_state, q0)


class TestDfaStructuralQueries(unittest.TestCase):
    def test_size(self):
        dfa, *_ = parity_dfa()
        self.assertEqual(dfa.size, 2)

    def test_get_input_alphabet(self):
        dfa, *_ = parity_dfa()
        self.assertEqual(set(dfa.get_input_alphabet()), {'a', 'b'})

    def test_get_state_by_id(self):
        dfa, q0, q1 = parity_dfa()
        self.assertIs(dfa.get_state_by_id('q1'), q1)
        self.assertIsNone(dfa.get_state_by_id('does_not_exist'))

    def test_is_input_complete_true(self):
        dfa, *_ = parity_dfa()
        self.assertTrue(dfa.is_input_complete())

    def test_is_input_complete_false(self):
        q0 = DfaState('q0')
        q1 = DfaState('q1')
        q0.transitions = {'a': q1}  # missing 'b'
        q1.transitions = {'a': q1, 'b': q1}
        dfa = Dfa(q0, [q0, q1])
        self.assertFalse(dfa.is_input_complete())

    def test_get_shortest_path_same_state(self):
        dfa, q0, _ = parity_dfa()
        self.assertEqual(dfa.get_shortest_path(q0, q0), ())

    def test_get_shortest_path_reachable(self):
        dfa, q0, q1 = parity_dfa()
        path = dfa.get_shortest_path(q0, q1)
        self.assertEqual(dfa.execute_sequence(q0, list(path)), [False] * (len(path) - 1) + [False])
        # following the path from q0 must land exactly on q1
        self.assertIs(dfa.current_state, q1)

    def test_get_shortest_path_unreachable_returns_none(self):
        dfa, q0, q1 = parity_dfa()
        unreachable = DfaState('isolated')
        unreachable.transitions = {'a': unreachable, 'b': unreachable}
        dfa.states.append(unreachable)
        self.assertIsNone(dfa.get_shortest_path(q0, unreachable))

    def test_get_shortest_path_state_not_in_automaton_warns_and_returns_none(self):
        dfa, q0, _ = parity_dfa()
        foreign = DfaState('foreign')
        with self.assertWarns(UserWarning):
            result = dfa.get_shortest_path(q0, foreign)
        self.assertIsNone(result)

    def test_is_strongly_connected_true(self):
        dfa, *_ = parity_dfa()
        self.assertTrue(dfa.is_strongly_connected())

    def test_is_strongly_connected_false(self):
        dfa, q0, q1, q2 = contains_a_dfa_non_minimal()
        # once we leave q0 we can never return to it
        self.assertFalse(dfa.is_strongly_connected())

    def test_is_strongly_connected_single_state(self):
        q0 = DfaState('q0', is_accepting=True)
        q0.transitions = {'a': q0}
        dfa = Dfa(q0, [q0])
        self.assertTrue(dfa.is_strongly_connected())


class TestDfaCharacterizationSet(unittest.TestCase):
    def test_is_minimal_true_for_minimal_dfa(self):
        dfa, *_ = parity_dfa()
        self.assertTrue(dfa.is_minimal())

    def test_is_minimal_false_for_redundant_states(self):
        dfa, *_ = contains_a_dfa_non_minimal()
        self.assertFalse(dfa.is_minimal())

    def test_compute_characterization_set_distinguishes_all_states(self):
        dfa, q0, q1 = parity_dfa()
        char_set = dfa.compute_characterization_set()
        self.assertIsNotNone(char_set)
        outputs = {tuple(tuple(dfa.compute_output_seq(s, list(seq))) for seq in char_set) for s in (q0, q1)}
        self.assertEqual(len(outputs), 2)

    def test_compute_characterization_set_return_same_states_for_non_minimal(self):
        dfa, q0, q1, q2 = contains_a_dfa_non_minimal()
        s1, s2 = dfa.compute_characterization_set(return_same_states=True)
        self.assertEqual({s1, s2}, {q1, q2})

    def test_compute_characterization_set_return_same_states_none_for_minimal(self):
        dfa, *_ = parity_dfa()
        s1, s2 = dfa.compute_characterization_set(return_same_states=True)
        self.assertIsNone(s1)
        self.assertIsNone(s2)


class TestDfaMinimize(unittest.TestCase):
    def test_minimize_reduces_redundant_states(self):
        dfa, *_ = contains_a_dfa_non_minimal()
        dfa.minimize()
        self.assertEqual(dfa.size, 2)

    def test_minimize_preserves_language(self):
        dfa, q0, q1, q2 = contains_a_dfa_non_minimal()
        words = [[], ['b'], ['a'], ['b', 'b'], ['a', 'b', 'a'], ['b', 'a', 'a']]
        expected = {tuple(w): dfa.execute_sequence(q0, w) for w in words}
        dfa.minimize()
        for w, exp in expected.items():
            self.assertEqual(dfa.execute_sequence(dfa.initial_state, list(w)), exp)

    def test_minimize_noop_on_already_minimal_dfa(self):
        dfa, *_ = parity_dfa()
        dfa.minimize()
        self.assertEqual(dfa.size, 2)

    def test_minimize_warns_on_incomplete_automaton(self):
        q0 = DfaState('q0')
        q1 = DfaState('q1')
        q0.transitions = {'a': q1}
        dfa = Dfa(q0, [q0, q1])
        with self.assertWarns(UserWarning):
            dfa.minimize()
        # nothing should have been merged
        self.assertEqual(dfa.size, 2)


class TestDfaStateSetupRoundtrip(unittest.TestCase):
    def test_to_state_setup_from_state_setup_roundtrip(self):
        dfa, q0, q1 = parity_dfa()
        setup = dfa.to_state_setup()
        rebuilt = Dfa.from_state_setup(setup)

        for w in [[], ['a'], ['b'], ['a', 'a'], ['a', 'b', 'a', 'a']]:
            self.assertEqual(rebuilt.execute_sequence(rebuilt.initial_state, w),
                              dfa.execute_sequence(dfa.initial_state, w))

    def test_from_state_setup_first_key_is_initial_state(self):
        setup = {
            'a': (True, {'x': 'b', 'y': 'a'}),
            'b': (False, {'x': 'a', 'y': 'b'}),
        }
        dfa = Dfa.from_state_setup(setup)
        self.assertEqual(dfa.initial_state.state_id, 'a')
        self.assertTrue(dfa.initial_state.is_accepting)

    def test_copy_produces_independent_deep_copy(self):
        dfa, q0, q1 = parity_dfa()
        dfa_copy = dfa.copy()
        self.assertEqual(dfa.size, dfa_copy.size)
        # mutating the copy must not affect the original
        dfa_copy.get_state_by_id('q0').is_accepting = False
        self.assertTrue(dfa.get_state_by_id('q0').is_accepting)

    def test_pickle_roundtrip(self):
        dfa, q0, q1 = parity_dfa()
        restored = pickle.loads(pickle.dumps(dfa))
        for w in [[], ['a'], ['b', 'a']]:
            self.assertEqual(restored.execute_sequence(restored.initial_state, w),
                              dfa.execute_sequence(dfa.initial_state, w))


class TestDfaEquality(unittest.TestCase):
    def test_eq_true_for_bisimilar_automata_with_different_structure(self):
        minimal, *_ = parity_dfa()
        non_minimal, *_ = contains_a_dfa_non_minimal()

        # build a differently-labeled but equivalent 2-state DFA for the 'contains a' language
        r0 = DfaState('r0', is_accepting=False)
        r1 = DfaState('r1', is_accepting=True)
        r0.transitions = {'a': r1, 'b': r0}
        r1.transitions = {'a': r1, 'b': r1}
        relabeled = Dfa(r0, [r0, r1])

        # non_minimal has states named differently but same language as `relabeled`
        self.assertEqual(non_minimal, relabeled)

    def test_eq_false_for_different_languages(self):
        parity, *_ = parity_dfa()
        contains_a, *_ = contains_a_dfa_non_minimal()
        self.assertNotEqual(parity, contains_a)


if __name__ == '__main__':
    unittest.main()
