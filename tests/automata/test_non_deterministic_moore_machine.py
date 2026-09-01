import random
import unittest

from aalpy.automata import NDMooreMachine, NDMooreState


def deterministic_ndmoore():
    """
    2-state NDMoore machine over alphabet {a, b} where every input has a single successor.
    s0(A) --a--> s1   s0 --b--> s0
    s1(B) --a--> s0   s1 --b--> s1
    """
    s0 = NDMooreState('s0', output='A')
    s1 = NDMooreState('s1', output='B')
    s0.transitions['a'].append(s1)
    s0.transitions['b'].append(s0)
    s1.transitions['a'].append(s0)
    s1.transitions['b'].append(s1)
    return NDMooreMachine(s0, [s0, s1]), s0, s1


def branching_ndmoore():
    """s0 --a--> s1 or s2 (non-deterministic choice)."""
    s0 = NDMooreState('s0', output='A')
    s1 = NDMooreState('s1', output='B')
    s2 = NDMooreState('s2', output='C')
    s0.transitions['a'].append(s1)
    s0.transitions['a'].append(s2)
    return NDMooreMachine(s0, [s0, s1, s2]), s0, s1, s2


class TestNDMooreState(unittest.TestCase):
    def test_default_transitions_is_empty_list(self):
        state = NDMooreState('s')
        self.assertEqual(state.transitions['unused_key'], [])

    def test_default_output_is_none(self):
        state = NDMooreState('s')
        self.assertIsNone(state.output)


class TestNDMooreStep(unittest.TestCase):
    def test_step_moves_and_returns_output(self):
        mm, s0, s1 = deterministic_ndmoore()
        self.assertEqual(mm.step('a'), 'B')
        self.assertIs(mm.current_state, s1)

    def test_step_sequence(self):
        mm, s0, s1 = deterministic_ndmoore()
        outputs = [mm.step(i) for i in ['a', 'b', 'a']]
        self.assertEqual(outputs, ['B', 'B', 'A'])

    def test_step_picks_among_non_deterministic_choices(self):
        mm, s0, s1, s2 = branching_ndmoore()
        seen = set()
        for seed in range(30):
            mm.reset_to_initial()
            random.seed(seed)
            output = mm.step('a')
            self.assertIn(output, ('B', 'C'))
            self.assertIn(mm.current_state, (s1, s2))
            seen.add(output)
        # with enough seeds both branches should be exercised at least once
        self.assertEqual(seen, {'B', 'C'})

    def test_step_unknown_letter_raises_on_empty_options(self):
        mm, *_ = deterministic_ndmoore()
        with self.assertRaises(IndexError):
            mm.step('unknown_letter')

    def test_reset_to_initial(self):
        mm, s0, s1 = deterministic_ndmoore()
        mm.step('a')
        self.assertIsNot(mm.current_state, s0)
        mm.reset_to_initial()
        self.assertIs(mm.current_state, s0)


class TestNDMooreExecuteSequence(unittest.TestCase):
    def test_execute_sequence_matches_stepwise(self):
        mm, s0, s1 = deterministic_ndmoore()
        result = mm.execute_sequence(s0, ['a', 'b', 'a'])
        self.assertEqual(result, ['B', 'B', 'A'])
        self.assertIs(mm.current_state, s0)

    def test_execute_sequence_empty_returns_empty_list(self):
        mm, s0, s1 = deterministic_ndmoore()
        self.assertEqual(mm.execute_sequence(s0, []), [])

    def test_execute_sequence_resets_to_origin_state_first(self):
        mm, s0, s1 = deterministic_ndmoore()
        mm.reset_to_initial()
        mm.step('a')  # move to s1
        result = mm.execute_sequence(s0, ['a'])
        self.assertEqual(result, ['B'])


class TestNDMooreStateSetup(unittest.TestCase):
    def test_to_state_setup_returns_dict(self):
        mm, s0, s1 = deterministic_ndmoore()
        setup = mm.to_state_setup()
        self.assertIsInstance(setup, dict)
        self.assertEqual(set(setup.keys()), {'s0', 's1'})

    def test_to_state_setup_first_key_is_initial_state(self):
        mm, s0, s1 = deterministic_ndmoore()
        setup = mm.to_state_setup()
        self.assertEqual(next(iter(setup)), 's0')

    def test_to_state_setup_from_state_setup_roundtrip(self):
        mm, s0, s1 = deterministic_ndmoore()
        setup = mm.to_state_setup()
        rebuilt = NDMooreMachine.from_state_setup(setup)

        for w in [['a'], ['b'], ['a', 'a', 'b']]:
            rebuilt.reset_to_initial()
            outputs = [rebuilt.step(letter) for letter in w]
            mm.reset_to_initial()
            expected = [mm.step(letter) for letter in w]
            self.assertEqual(outputs, expected)

    def test_from_state_setup_first_key_is_initial_state(self):
        setup = {
            's0': ('A', {'a': ['s1']}),
            's1': ('B', {'a': ['s0']}),
        }
        mm = NDMooreMachine.from_state_setup(setup)
        self.assertEqual(mm.initial_state.state_id, 's0')
        self.assertEqual(mm.initial_state.output, 'A')

    def test_copy_produces_independent_deep_copy(self):
        mm, s0, s1 = deterministic_ndmoore()
        mm_copy = mm.copy()
        self.assertEqual(mm.size, mm_copy.size)
        mm_copy.get_state_by_id('s0').output = 'CHANGED'
        self.assertEqual(mm.get_state_by_id('s0').output, 'A')


class TestNDMooreStructural(unittest.TestCase):
    def test_get_input_alphabet(self):
        mm, *_ = deterministic_ndmoore()
        self.assertEqual(set(mm.get_input_alphabet()), {'a', 'b'})

    def test_size(self):
        mm, *_ = deterministic_ndmoore()
        self.assertEqual(mm.size, 2)


if __name__ == '__main__':
    unittest.main()
