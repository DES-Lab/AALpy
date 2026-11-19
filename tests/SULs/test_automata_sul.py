import unittest

from aalpy.automata import (Dfa, DfaState, MealyMachine, MealyState, Mdp, MdpState, MooreMachine, MooreState,
                            Onfsm, OnfsmState)
from aalpy.SULs import AutomatonSUL, DfaSUL, MealySUL, MdpSUL, MooreSUL, OnfsmSUL


def parity_dfa():
    q0 = DfaState('q0', is_accepting=True)
    q1 = DfaState('q1', is_accepting=False)
    q0.transitions = {'a': q1, 'b': q0}
    q1.transitions = {'a': q0, 'b': q1}
    return Dfa(q0, [q0, q1])


def sample_mealy():
    s0 = MealyState('s0')
    s1 = MealyState('s1')
    s0.transitions = {'x': s1, 'y': s0}
    s0.output_fun = {'x': 'o1', 'y': 'o2'}
    s1.transitions = {'x': s0, 'y': s1}
    s1.output_fun = {'x': 'o3', 'y': 'o1'}
    return MealyMachine(s0, [s0, s1])


def sample_moore():
    s0 = MooreState('s0', output='A')
    s1 = MooreState('s1', output='B')
    s0.transitions = {'x': s1, 'y': s0}
    s1.transitions = {'x': s0, 'y': s1}
    return MooreMachine(s0, [s0, s1])


def deterministic_mdp():
    s0 = MdpState('s0', output='A')
    s1 = MdpState('s1', output='B')
    s0.transitions['a'].append((s1, 1.0))
    s1.transitions['a'].append((s0, 1.0))
    return Mdp(s0, [s0, s1])


def sample_onfsm():
    s0 = OnfsmState('s0')
    s1 = OnfsmState('s1')
    s0.transitions['a'].append(('out1', s1))
    s1.transitions['a'].append(('out2', s0))
    return Onfsm(s0, [s0, s1])


class TestAutomatonSULAliases(unittest.TestCase):
    """DfaSUL, MealySUL, MooreSUL, MdpSUL, OnfsmSUL are all just AutomatonSUL under a different name."""

    def test_aliases_are_automaton_sul(self):
        self.assertIs(DfaSUL, AutomatonSUL)
        self.assertIs(MealySUL, AutomatonSUL)
        self.assertIs(MooreSUL, AutomatonSUL)
        self.assertIs(MdpSUL, AutomatonSUL)
        self.assertIs(OnfsmSUL, AutomatonSUL)


class TestAutomatonSULConstruction(unittest.TestCase):
    def test_wraps_given_automaton(self):
        dfa = parity_dfa()
        sul = AutomatonSUL(dfa)
        self.assertIs(sul.automaton, dfa)

    def test_counters_start_at_zero(self):
        sul = AutomatonSUL(parity_dfa())
        self.assertEqual(sul.num_queries, 0)
        self.assertEqual(sul.num_steps, 0)
        self.assertEqual(sul.num_cached_queries, 0)


class TestAutomatonSULWithDfa(unittest.TestCase):
    def test_step_delegates_to_automaton_and_moves_state(self):
        dfa = parity_dfa()
        sul = AutomatonSUL(dfa)
        self.assertFalse(sul.step('a'))
        self.assertIs(dfa.current_state, dfa.get_state_by_id('q1'))
        self.assertTrue(sul.step('a'))
        self.assertIs(dfa.current_state, dfa.initial_state)

    def test_pre_resets_wrapped_automaton_to_initial_state(self):
        dfa = parity_dfa()
        sul = AutomatonSUL(dfa)
        sul.step('a')
        self.assertIsNot(dfa.current_state, dfa.initial_state)
        sul.pre()
        self.assertIs(dfa.current_state, dfa.initial_state)

    def test_post_is_a_no_op(self):
        dfa = parity_dfa()
        sul = AutomatonSUL(dfa)
        sul.step('a')
        state_before = dfa.current_state
        sul.post()
        self.assertIs(dfa.current_state, state_before)

    def test_query_matches_manual_stepping(self):
        dfa = parity_dfa()
        sul = AutomatonSUL(dfa)
        self.assertEqual(sul.query(('a', 'b', 'a')), [False, False, True])

    def test_query_empty_word_returns_initial_state_output(self):
        sul = AutomatonSUL(parity_dfa())
        self.assertEqual(sul.query(()), [True])

    def test_query_resets_before_running(self):
        dfa = parity_dfa()
        sul = AutomatonSUL(dfa)
        sul.step('a')  # leave the automaton in a non-initial state
        self.assertEqual(sul.query(('a',)), [False])  # query() calls pre() first, so this is from q0 again

    def test_io_query_pairs_inputs_with_outputs(self):
        sul = AutomatonSUL(parity_dfa())
        self.assertEqual(sul.io_query(('a', 'a')), [('a', False), ('a', True)])

    def test_num_queries_and_steps_accumulate_across_calls(self):
        sul = AutomatonSUL(parity_dfa())
        sul.query(('a', 'b'))
        sul.query(('a',))
        sul.query(())
        self.assertEqual(sul.num_queries, 3)
        self.assertEqual(sul.num_steps, 3)  # the empty query contributes 0 steps

    def test_multiple_suls_wrapping_the_same_automaton_share_state(self):
        dfa = parity_dfa()
        sul1 = AutomatonSUL(dfa)
        sul2 = AutomatonSUL(dfa)
        sul1.step('a')
        # both wrap the very same automaton object, so state is not per-SUL
        self.assertIs(sul1.automaton.current_state, sul2.automaton.current_state)


class TestAutomatonSULWithMealy(unittest.TestCase):
    def test_step_returns_output_and_moves_state(self):
        mm = sample_mealy()
        sul = AutomatonSUL(mm)
        self.assertEqual(sul.step('x'), 'o1')
        self.assertIs(mm.current_state, mm.get_state_by_id('s1'))

    def test_query_matches_manual_stepping(self):
        sul = AutomatonSUL(sample_mealy())
        self.assertEqual(sul.query(('x', 'x', 'y')), ['o1', 'o3', 'o2'])

    def test_query_empty_word_raises_key_error(self):
        # Mealy machines have no state-based output, so there is no well-defined answer for the
        # empty word; SUL.query(()) calls step(None), and Mealy's output_fun has no None entry.
        sul = AutomatonSUL(sample_mealy())
        with self.assertRaises(KeyError):
            sul.query(())

    def test_io_query_pairs_inputs_with_outputs(self):
        sul = AutomatonSUL(sample_mealy())
        self.assertEqual(sul.io_query(('x', 'y')), [('x', 'o1'), ('y', 'o1')])


class TestAutomatonSULWithMoore(unittest.TestCase):
    def test_step_returns_output_of_reached_state(self):
        mm = sample_moore()
        sul = AutomatonSUL(mm)
        self.assertEqual(sul.step('x'), 'B')
        self.assertIs(mm.current_state, mm.get_state_by_id('s1'))

    def test_query_matches_manual_stepping(self):
        sul = AutomatonSUL(sample_moore())
        self.assertEqual(sul.query(('x', 'y', 'x')), ['B', 'B', 'A'])

    def test_query_empty_word_returns_initial_state_output(self):
        sul = AutomatonSUL(sample_moore())
        self.assertEqual(sul.query(()), ['A'])

    def test_io_query_pairs_inputs_with_outputs(self):
        sul = AutomatonSUL(sample_moore())
        self.assertEqual(sul.io_query(('x',)), [('x', 'B')])


class TestAutomatonSULWithMdp(unittest.TestCase):
    def test_step_delegates_and_moves_state(self):
        mdp = deterministic_mdp()
        sul = AutomatonSUL(mdp)
        self.assertEqual(sul.step('a'), 'B')
        self.assertIs(mdp.current_state, mdp.get_state_by_id('s1'))

    def test_query_matches_manual_stepping(self):
        sul = AutomatonSUL(deterministic_mdp())
        self.assertEqual(sul.query(('a', 'a', 'a')), ['B', 'A', 'B'])

    def test_query_empty_word_returns_initial_state_output_without_moving(self):
        mdp = deterministic_mdp()
        sul = AutomatonSUL(mdp)
        self.assertEqual(sul.query(()), ['A'])
        self.assertIs(mdp.current_state, mdp.initial_state)


class TestAutomatonSULWithOnfsm(unittest.TestCase):
    def test_step_returns_one_of_the_possible_outputs(self):
        onfsm = sample_onfsm()
        sul = AutomatonSUL(onfsm)
        output = sul.step('a')
        self.assertIn(output, ('out1',))
        self.assertIs(onfsm.current_state, onfsm.get_state_by_id('s1'))

    def test_query_matches_manual_stepping(self):
        sul = AutomatonSUL(sample_onfsm())
        self.assertEqual(sul.query(('a', 'a')), ['out1', 'out2'])


if __name__ == '__main__':
    unittest.main()
