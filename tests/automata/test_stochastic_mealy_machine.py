import random
import unittest

from aalpy.automata import Mdp, StochasticMealyMachine, StochasticMealyState


def deterministic_smm():
    """
    2-state SMM over alphabet {a, b}, only probability-1.0 transitions.
    s0 --a/o1[1.0]--> s1   s0 --b/o2[1.0]--> s0
    s1 --a/o3[1.0]--> s0   s1 --b/o1[1.0]--> s1
    """
    s0 = StochasticMealyState('s0')
    s1 = StochasticMealyState('s1')
    s0.transitions['a'].append((s1, 'o1', 1.0))
    s0.transitions['b'].append((s0, 'o2', 1.0))
    s1.transitions['a'].append((s0, 'o3', 1.0))
    s1.transitions['b'].append((s1, 'o1', 1.0))
    return StochasticMealyMachine(s0, [s0, s1]), s0, s1


def branching_smm():
    """s0 --a--> (s1, 'o1', 0.5) or (s2, 'o2', 0.5)."""
    s0 = StochasticMealyState('s0')
    s1 = StochasticMealyState('s1')
    s2 = StochasticMealyState('s2')
    s0.transitions['a'].append((s1, 'o1', 0.5))
    s0.transitions['a'].append((s2, 'o2', 0.5))
    return StochasticMealyMachine(s0, [s0, s1, s2]), s0, s1, s2


class TestStochasticMealyState(unittest.TestCase):
    def test_default_transitions_is_empty_defaultdict(self):
        state = StochasticMealyState('s')
        self.assertEqual(state.transitions['unused_key'], [])


class TestStochasticMealyStep(unittest.TestCase):
    def test_step_returns_output_and_moves_state(self):
        smm, s0, s1 = deterministic_smm()
        output = smm.step('a')
        self.assertEqual(output, 'o1')
        self.assertIs(smm.current_state, s1)

    def test_step_sequence(self):
        smm, s0, s1 = deterministic_smm()
        outputs = [smm.step(i) for i in ['a', 'a', 'b']]
        self.assertEqual(outputs, ['o1', 'o3', 'o2'])

    def test_step_respects_branching_distribution(self):
        smm, s0, s1, s2 = branching_smm()
        for seed in range(20):
            smm.reset_to_initial()
            random.seed(seed)
            output = smm.step('a')
            self.assertIn(output, ('o1', 'o2'))
            self.assertIn(smm.current_state, (s1, s2))

    def test_reset_to_initial(self):
        smm, s0, s1 = deterministic_smm()
        smm.step('a')
        self.assertIsNot(smm.current_state, s0)
        smm.reset_to_initial()
        self.assertIs(smm.current_state, s0)


class TestStochasticMealyExecuteSequence(unittest.TestCase):
    def test_execute_sequence_matches_stepwise(self):
        smm, s0, s1 = deterministic_smm()
        result = smm.execute_sequence(s0, ['a', 'a', 'b'])
        self.assertEqual(result, ['o1', 'o3', 'o2'])
        self.assertIs(smm.current_state, s0)

    def test_execute_sequence_empty_returns_empty_list(self):
        smm, s0, s1 = deterministic_smm()
        self.assertEqual(smm.execute_sequence(s0, []), [])

    def test_execute_sequence_resets_to_origin_state_first(self):
        smm, s0, s1 = deterministic_smm()
        smm.reset_to_initial()
        smm.step('a')  # move to s1
        result = smm.execute_sequence(s0, ['a'])
        self.assertEqual(result, ['o1'])


class TestStochasticMealyStepTo(unittest.TestCase):
    def test_step_to_moves_to_matching_output_state(self):
        smm, s0, s1, s2 = branching_smm()
        result = smm.step_to('a', 'o2')
        self.assertEqual(result, 'o2')
        self.assertIs(smm.current_state, s2)

    def test_step_to_returns_none_for_unreachable_output(self):
        smm, s0, s1, s2 = branching_smm()
        result = smm.step_to('a', 'does_not_exist')
        self.assertIsNone(result)
        self.assertIs(smm.current_state, s0)


class TestStochasticMealyStateSetupRoundtrip(unittest.TestCase):
    def test_to_state_setup_from_state_setup_roundtrip(self):
        smm, s0, s1 = deterministic_smm()
        setup = smm.to_state_setup()
        rebuilt = StochasticMealyMachine.from_state_setup(setup)

        for w in [['a'], ['b'], ['a', 'a', 'b'], ['b', 'a']]:
            rebuilt.reset_to_initial()
            outputs = [rebuilt.step(letter) for letter in w]
            smm.reset_to_initial()
            expected = [smm.step(letter) for letter in w]
            self.assertEqual(outputs, expected)

    def test_from_state_setup_first_key_is_initial_state(self):
        setup = {
            's0': {'a': [('s1', 'o1', 1.0)]},
            's1': {'a': [('s0', 'o2', 1.0)]},
        }
        smm = StochasticMealyMachine.from_state_setup(setup)
        self.assertEqual(smm.initial_state.state_id, 's0')

    def test_to_state_setup_puts_initial_state_first(self):
        smm, s0, s1 = deterministic_smm()
        smm.states = [s1, s0]
        smm.to_state_setup()
        self.assertIs(smm.states[0], s0)


class TestStochasticMealyToMdp(unittest.TestCase):
    def test_to_mdp_returns_mdp(self):
        smm, *_ = deterministic_smm()
        mdp = smm.to_mdp()
        self.assertIsInstance(mdp, Mdp)

    def test_to_mdp_preserves_behavior(self):
        smm, s0, s1 = deterministic_smm()
        mdp = smm.to_mdp()

        # walk the SMM deterministically and confirm the MDP can reproduce the same output trace
        # by following matching outputs via step_to
        for w in [['a'], ['b'], ['a', 'a', 'b']]:
            smm.reset_to_initial()
            smm_outputs = [smm.step(letter) for letter in w]

            mdp.reset_to_initial()
            mdp_outputs = [mdp.step_to(letter, out) for letter, out in zip(w, smm_outputs)]
            self.assertEqual(mdp_outputs, smm_outputs)


class TestStochasticMealyStructural(unittest.TestCase):
    def test_get_input_alphabet(self):
        smm, *_ = deterministic_smm()
        self.assertEqual(set(smm.get_input_alphabet()), {'a', 'b'})

    def test_size(self):
        smm, *_ = deterministic_smm()
        self.assertEqual(smm.size, 2)


if __name__ == '__main__':
    unittest.main()
