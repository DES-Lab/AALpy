import pickle
import unittest

from aalpy.automata import Dfa, MooreMachine, MooreState


def sample_moore():
    """
    3-state Moore machine over alphabet {x, y}, outputs 'A', 'B', 'C' per state.
    s0(A) --x--> s1(B)   s0 --y--> s0
    s1(B) --x--> s2(C)   s1 --y--> s0
    s2(C) --x--> s2      s2 --y--> s0
    """
    s0 = MooreState('s0', output='A')
    s1 = MooreState('s1', output='B')
    s2 = MooreState('s2', output='C')
    s0.transitions = {'x': s1, 'y': s0}
    s1.transitions = {'x': s2, 'y': s0}
    s2.transitions = {'x': s2, 'y': s0}
    mm = MooreMachine(s0, [s0, s1, s2])
    mm.compute_prefixes()
    return mm, s0, s1, s2


def boolean_moore():
    s0 = MooreState('s0', output=True)
    s1 = MooreState('s1', output=False)
    s0.transitions = {'a': s1, 'b': s0}
    s1.transitions = {'a': s0, 'b': s1}
    mm = MooreMachine(s0, [s0, s1])
    mm.compute_prefixes()
    return mm, s0, s1


class TestMooreStep(unittest.TestCase):
    def test_step_returns_output_of_reached_state(self):
        mm, s0, s1, s2 = sample_moore()
        self.assertEqual(mm.step('x'), 'B')
        self.assertIs(mm.current_state, s1)

    def test_step_none_returns_current_output_without_moving(self):
        mm, s0, s1, s2 = sample_moore()
        mm.step('x')
        self.assertEqual(mm.step(None), 'B')
        self.assertIs(mm.current_state, s1)

    def test_step_unknown_letter_raises(self):
        mm, *_ = sample_moore()
        with self.assertRaises(KeyError):
            mm.step('z')


class TestMooreExecuteAndOutputSeq(unittest.TestCase):
    def test_execute_sequence(self):
        mm, s0, s1, s2 = sample_moore()
        result = mm.execute_sequence(s0, ['x', 'x', 'y', 'x'])
        self.assertEqual(result, ['B', 'C', 'A', 'B'])

    def test_execute_sequence_empty_returns_state_output(self):
        mm, s0, s1, s2 = sample_moore()
        self.assertEqual(mm.execute_sequence(s1, []), 'B')
        self.assertIs(mm.current_state, s1)

    def test_compute_output_seq_empty(self):
        mm, s0, s1, s2 = sample_moore()
        self.assertEqual(mm.compute_output_seq(s2, []), ['C'])

    def test_compute_output_seq_does_not_mutate_current_state(self):
        mm, s0, s1, s2 = sample_moore()
        mm.reset_to_initial()
        mm.compute_output_seq(s2, ['x', 'y'])
        self.assertIs(mm.current_state, s0)


class TestMooreCharacterizationSet(unittest.TestCase):
    def test_is_minimal_true(self):
        mm, *_ = sample_moore()
        self.assertTrue(mm.is_minimal())

    def test_is_minimal_false_for_redundant_states(self):
        s0 = MooreState('s0', output='A')
        s1 = MooreState('s1', output='B')
        s2 = MooreState('s2', output='B')  # equivalent to s1
        s0.transitions = {'x': s1, 'y': s0}
        s1.transitions = {'x': s2, 'y': s2}
        s2.transitions = {'x': s2, 'y': s2}
        mm = MooreMachine(s0, [s0, s1, s2])
        self.assertFalse(mm.is_minimal())


class TestMooreStateSetupRoundtrip(unittest.TestCase):
    def test_to_state_setup_from_state_setup_roundtrip(self):
        mm, s0, s1, s2 = sample_moore()
        setup = mm.to_state_setup()
        rebuilt = MooreMachine.from_state_setup(setup)

        for w in [[], ['x'], ['x', 'x'], ['x', 'x', 'y', 'x']]:
            self.assertEqual(rebuilt.execute_sequence(rebuilt.initial_state, w),
                              mm.execute_sequence(mm.initial_state, w))

    def test_pickle_roundtrip(self):
        mm, *_ = sample_moore()
        restored = pickle.loads(pickle.dumps(mm))
        for w in [[], ['x'], ['y', 'x']]:
            self.assertEqual(restored.execute_sequence(restored.initial_state, w),
                              mm.execute_sequence(mm.initial_state, w))


class TestMooreToDfa(unittest.TestCase):
    def test_to_dfa_preserves_language(self):
        mm, s0, s1 = boolean_moore()
        dfa = MooreMachine.to_dfa(mm)
        self.assertIsInstance(dfa, Dfa)
        for w in [[], ['a'], ['b'], ['a', 'a'], ['a', 'b', 'a']]:
            moore_outputs = mm.execute_sequence(mm.initial_state, w)
            dfa_result = dfa.execute_sequence(dfa.initial_state, w)
            self.assertEqual(dfa_result, moore_outputs)

    def test_to_dfa_rejects_non_boolean_outputs(self):
        mm, *_ = sample_moore()  # outputs are strings, not bool
        with self.assertRaises(ValueError):
            MooreMachine.to_dfa(mm)

    def test_to_dfa_preserves_state_count(self):
        mm, *_ = boolean_moore()
        dfa = MooreMachine.to_dfa(mm)
        self.assertEqual(dfa.size, mm.size)


class TestMooreEquality(unittest.TestCase):
    def test_eq_true_for_relabeled_equivalent_machine(self):
        mm, *_ = boolean_moore()

        t0 = MooreState('t0', output=True)
        t1 = MooreState('t1', output=False)
        t0.transitions = {'a': t1, 'b': t0}
        t1.transitions = {'a': t0, 'b': t1}
        relabeled = MooreMachine(t0, [t0, t1])

        self.assertEqual(mm, relabeled)

    def test_eq_false_for_different_outputs(self):
        mm, *_ = boolean_moore()
        other = mm.copy()
        other.get_state_by_id('s0').output = False
        self.assertNotEqual(mm, other)


if __name__ == '__main__':
    unittest.main()
