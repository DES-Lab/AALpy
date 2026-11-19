import unittest

from aalpy.automata import Dfa, DfaState
from aalpy.base import AutomatonState


class TestAutomatonState(unittest.TestCase):
    def test_get_diff_and_same_state_transitions(self):
        s0 = AutomatonState('s0')
        s1 = AutomatonState('s1')
        s0.transitions = {'a': s1, 'b': s0, 'c': s0}

        self.assertEqual(sorted(s0.get_diff_state_transitions()), ['a'])
        self.assertEqual(sorted(s0.get_same_state_transitions()), ['b', 'c'])

    def test_all_self_loops(self):
        s0 = AutomatonState('s0')
        s0.transitions = {'a': s0, 'b': s0}

        self.assertEqual(s0.get_diff_state_transitions(), [])
        self.assertEqual(sorted(s0.get_same_state_transitions()), ['a', 'b'])

    def test_all_different_states(self):
        s0 = AutomatonState('s0')
        s1 = AutomatonState('s1')
        s2 = AutomatonState('s2')
        s0.transitions = {'a': s1, 'b': s2}

        self.assertEqual(sorted(s0.get_diff_state_transitions()), ['a', 'b'])
        self.assertEqual(s0.get_same_state_transitions(), [])

    def test_no_transitions(self):
        s0 = AutomatonState('s0')
        s0.transitions = {}

        self.assertEqual(s0.get_diff_state_transitions(), [])
        self.assertEqual(s0.get_same_state_transitions(), [])

    def test_prefix_defaults_to_none(self):
        s0 = AutomatonState('s0')
        self.assertIsNone(s0.prefix)


class TestAutomatonGeneric(unittest.TestCase):
    def test_size_property(self):
        q0 = DfaState('q0', True)
        q1 = DfaState('q1', False)
        q0.transitions = {'a': q1}
        q1.transitions = {'a': q0}
        dfa = Dfa(q0, [q0, q1])
        self.assertEqual(dfa.size, 2)

    def test_current_state_initialized_to_initial_state(self):
        q0 = DfaState('q0', True)
        q0.transitions = {'a': q0}
        dfa = Dfa(q0, [q0])
        self.assertIs(dfa.current_state, dfa.initial_state)

    def test_str_returns_string_representation(self):
        q0 = DfaState('q0', True)
        q0.transitions = {'a': q0}
        dfa = Dfa(q0, [q0])
        self.assertIsInstance(str(dfa), str)
        self.assertIn('q0', str(dfa))


if __name__ == '__main__':
    unittest.main()
