import random
import unittest

from aalpy.automata import Mdp, MdpState


def deterministic_mdp():
    """
    2-state MDP over alphabet {a, b} with only probability-1.0 transitions, so behaviour is deterministic.
    s0(A) --a[1.0]--> s1   s0 --b[1.0]--> s0
    s1(B) --a[1.0]--> s0   s1 --b[1.0]--> s1
    """
    s0 = MdpState('s0', output='A')
    s1 = MdpState('s1', output='B')
    s0.transitions['a'].append((s1, 1.0))
    s0.transitions['b'].append((s0, 1.0))
    s1.transitions['a'].append((s0, 1.0))
    s1.transitions['b'].append((s1, 1.0))
    return Mdp(s0, [s0, s1]), s0, s1


def branching_mdp():
    """s0 --a--> s1 (0.5) or s2 (0.5)."""
    s0 = MdpState('s0', output='A')
    s1 = MdpState('s1', output='B')
    s2 = MdpState('s2', output='C')
    s0.transitions['a'].append((s1, 0.5))
    s0.transitions['a'].append((s2, 0.5))
    return Mdp(s0, [s0, s1, s2]), s0, s1, s2


class TestMdpState(unittest.TestCase):
    def test_default_transitions_is_empty_defaultdict(self):
        state = MdpState('s')
        self.assertEqual(state.transitions['unused_key'], [])

    def test_default_output_is_none(self):
        state = MdpState('s')
        self.assertIsNone(state.output)


class TestMdpStep(unittest.TestCase):
    def test_step_moves_and_returns_output(self):
        mdp, s0, s1 = deterministic_mdp()
        self.assertEqual(mdp.step('a'), 'B')
        self.assertIs(mdp.current_state, s1)

    def test_step_none_returns_current_output_without_moving(self):
        mdp, s0, s1 = deterministic_mdp()
        mdp.step('a')
        self.assertEqual(mdp.step(None), 'B')
        self.assertIs(mdp.current_state, s1)

    def test_step_unknown_letter_raises_on_empty_distribution(self):
        mdp, *_ = deterministic_mdp()
        with self.assertRaises(IndexError):
            mdp.step('unknown_letter')

    def test_reset_to_initial(self):
        mdp, s0, s1 = deterministic_mdp()
        mdp.step('a')
        self.assertIsNot(mdp.current_state, s0)
        mdp.reset_to_initial()
        self.assertIs(mdp.current_state, s0)

    def test_step_respects_branching_distribution(self):
        mdp, s0, s1, s2 = branching_mdp()
        for seed in range(20):
            mdp.reset_to_initial()
            random.seed(seed)
            output = mdp.step('a')
            self.assertIn(output, ('B', 'C'))
            self.assertIn(mdp.current_state, (s1, s2))


class TestMdpExecuteSequence(unittest.TestCase):
    def test_execute_sequence_matches_stepwise(self):
        mdp, s0, s1 = deterministic_mdp()
        result = mdp.execute_sequence(s0, ['a', 'b', 'a', 'a'])
        self.assertEqual(result, ['B', 'B', 'A', 'B'])
        self.assertIs(mdp.current_state, s1)

    def test_execute_sequence_empty_returns_empty_list(self):
        mdp, s0, s1 = deterministic_mdp()
        self.assertEqual(mdp.execute_sequence(s0, []), [])

    def test_execute_sequence_resets_to_origin_state_first(self):
        mdp, s0, s1 = deterministic_mdp()
        mdp.reset_to_initial()
        mdp.step('a')  # move to s1
        result = mdp.execute_sequence(s0, ['a'])
        # regardless of where the mdp was, execute_sequence starts fresh from origin_state
        self.assertEqual(result, ['B'])


class TestMdpStepTo(unittest.TestCase):
    def test_step_to_moves_to_matching_output_state(self):
        mdp, s0, s1, s2 = branching_mdp()
        result = mdp.step_to('a', 'C')
        self.assertEqual(result, 'C')
        self.assertIs(mdp.current_state, s2)

    def test_step_to_returns_none_for_unreachable_output(self):
        mdp, s0, s1, s2 = branching_mdp()
        result = mdp.step_to('a', 'does_not_exist')
        self.assertIsNone(result)
        self.assertIs(mdp.current_state, s0)  # state unchanged on failed step_to


class TestMdpStateSetupRoundtrip(unittest.TestCase):
    def test_to_state_setup_from_state_setup_roundtrip(self):
        mdp, s0, s1 = deterministic_mdp()
        setup = mdp.to_state_setup()
        rebuilt = Mdp.from_state_setup(setup)

        for w in [[], ['a'], ['b'], ['a', 'a', 'b']]:
            rebuilt.reset_to_initial()
            outputs = [rebuilt.step(letter) for letter in w]
            mdp.reset_to_initial()
            expected = [mdp.step(letter) for letter in w]
            self.assertEqual(outputs, expected)

    def test_from_state_setup_first_key_is_initial_state(self):
        setup = {
            's0': ('A', {'a': [('s1', 1.0)]}),
            's1': ('B', {'a': [('s0', 1.0)]}),
        }
        mdp = Mdp.from_state_setup(setup)
        self.assertEqual(mdp.initial_state.state_id, 's0')
        self.assertEqual(mdp.initial_state.output, 'A')

    def test_to_state_setup_puts_initial_state_first(self):
        mdp, s0, s1 = deterministic_mdp()
        # reorder states so initial_state is not first in the list
        mdp.states = [s1, s0]
        mdp.to_state_setup()
        self.assertIs(mdp.states[0], s0)


class TestMdpStructural(unittest.TestCase):
    def test_get_input_alphabet(self):
        mdp, *_ = deterministic_mdp()
        self.assertEqual(set(mdp.get_input_alphabet()), {'a', 'b'})

    def test_size(self):
        mdp, *_ = deterministic_mdp()
        self.assertEqual(mdp.size, 2)


if __name__ == '__main__':
    unittest.main()
