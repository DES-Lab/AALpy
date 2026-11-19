import random
import unittest

from aalpy.automata import MarkovChain, McState


def deterministic_chain():
    """s0(A) -[1.0]-> s1(B) -[1.0]-> s2(C), s2 is terminal (no outgoing transitions)."""
    s0 = McState('s0', output='A')
    s1 = McState('s1', output='B')
    s2 = McState('s2', output='C')
    s0.transitions.append((s1, 1.0))
    s1.transitions.append((s2, 1.0))
    return MarkovChain(s0, [s0, s1, s2]), s0, s1, s2


def branching_chain():
    s0 = McState('s0', output='A')
    s1 = McState('s1', output='B')
    s2 = McState('s2', output='C')
    s0.transitions.append((s1, 0.5))
    s0.transitions.append((s2, 0.5))
    return MarkovChain(s0, [s0, s1, s2]), s0, s1, s2


class TestMcState(unittest.TestCase):
    def test_default_transitions_is_empty_list(self):
        state = McState('s', output='A')
        self.assertEqual(state.transitions, [])


class TestMarkovChainStep(unittest.TestCase):
    def test_step_moves_and_returns_output(self):
        mc, s0, s1, s2 = deterministic_chain()
        self.assertEqual(mc.step(), 'B')
        self.assertIs(mc.current_state, s1)

    def test_step_on_terminal_state_returns_output_without_moving(self):
        mc, s0, s1, s2 = deterministic_chain()
        mc.step()
        mc.step()
        self.assertIs(mc.current_state, s2)
        # s2 has no outgoing transitions, so stepping again is a no-op
        self.assertEqual(mc.step(), 'C')
        self.assertIs(mc.current_state, s2)

    def test_reset_to_initial(self):
        mc, s0, s1, s2 = deterministic_chain()
        mc.step()
        self.assertIsNot(mc.current_state, s0)
        mc.reset_to_initial()
        self.assertIs(mc.current_state, s0)

    def test_step_respects_branching_distribution(self):
        mc, s0, s1, s2 = branching_chain()
        for seed in range(20):
            mc.reset_to_initial()
            random.seed(seed)
            output = mc.step()
            self.assertIn(output, ('B', 'C'))
            self.assertIn(mc.current_state, (s1, s2))


class TestMarkovChainExecuteSequence(unittest.TestCase):
    def test_execute_sequence_matches_stepwise(self):
        mc, s0, s1, s2 = deterministic_chain()
        # MarkovChain.step() ignores its argument; only the sequence's length matters
        result = mc.execute_sequence(s0, [None, None])
        self.assertEqual(result, ['B', 'C'])
        self.assertIs(mc.current_state, s2)

    def test_execute_sequence_empty_returns_empty_list(self):
        mc, s0, s1, s2 = deterministic_chain()
        self.assertEqual(mc.execute_sequence(s0, []), [])

    def test_execute_sequence_resets_to_origin_state_first(self):
        mc, s0, s1, s2 = deterministic_chain()
        mc.reset_to_initial()
        mc.step()  # move to s1
        result = mc.execute_sequence(s0, [None])
        self.assertEqual(result, ['B'])


class TestMarkovChainStepTo(unittest.TestCase):
    def test_step_to_moves_to_matching_output_state(self):
        mc, s0, s1, s2 = branching_chain()
        result = mc.step_to('C')
        self.assertEqual(result, 'C')
        self.assertIs(mc.current_state, s2)

    def test_step_to_returns_none_for_unreachable_output(self):
        mc, s0, s1, s2 = branching_chain()
        result = mc.step_to('does_not_exist')
        self.assertIsNone(result)
        self.assertIs(mc.current_state, s0)  # state unchanged on failed step_to


class TestMarkovChainUnimplemented(unittest.TestCase):
    def test_to_state_setup_not_implemented(self):
        mc, *_ = deterministic_chain()
        with self.assertRaises(NotImplementedError):
            mc.to_state_setup()

    def test_from_state_setup_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            MarkovChain.from_state_setup({})

    def test_copy_not_implemented(self):
        mc, *_ = deterministic_chain()
        with self.assertRaises(NotImplementedError):
            mc.copy()

    def test_get_input_alphabet_not_supported(self):
        # McState.transitions is a plain list (not a dict), so the generic Automaton.get_input_alphabet
        # (which does state.transitions.keys()) cannot work on a MarkovChain. Documents current behaviour.
        mc, *_ = deterministic_chain()
        with self.assertRaises(AttributeError):
            mc.get_input_alphabet()


class TestMarkovChainStructural(unittest.TestCase):
    def test_size(self):
        mc, *_ = deterministic_chain()
        self.assertEqual(mc.size, 3)


if __name__ == '__main__':
    unittest.main()
