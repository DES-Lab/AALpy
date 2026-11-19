import pickle
import unittest

from aalpy.automata import MealyMachine, MealyState


def sample_mealy():
    """
    2-state Mealy machine over alphabet {x, y}.
    s0 --x/o1--> s1   s0 --y/o2--> s0
    s1 --x/o3--> s0   s1 --y/o1--> s1
    """
    s0 = MealyState('s0')
    s1 = MealyState('s1')
    s0.transitions = {'x': s1, 'y': s0}
    s0.output_fun = {'x': 'o1', 'y': 'o2'}
    s1.transitions = {'x': s0, 'y': s1}
    s1.output_fun = {'x': 'o3', 'y': 'o1'}
    mm = MealyMachine(s0, [s0, s1])
    mm.compute_prefixes()
    return mm, s0, s1


class TestMealyState(unittest.TestCase):
    def test_defaults_are_empty(self):
        state = MealyState('s')
        self.assertEqual(state.transitions, {})
        self.assertEqual(state.output_fun, {})


class TestMealyStep(unittest.TestCase):
    def test_step_returns_output_and_moves_state(self):
        mm, s0, s1 = sample_mealy()
        output = mm.step('x')
        self.assertEqual(output, 'o1')
        self.assertIs(mm.current_state, s1)

    def test_step_sequence(self):
        mm, s0, s1 = sample_mealy()
        outputs = [mm.step(i) for i in ['x', 'x', 'y', 'x']]
        self.assertEqual(outputs, ['o1', 'o3', 'o2', 'o1'])

    def test_step_unknown_letter_raises(self):
        mm, *_ = sample_mealy()
        with self.assertRaises(KeyError):
            mm.step('z')


class TestMealyExecuteAndOutputSeq(unittest.TestCase):
    def test_execute_sequence(self):
        mm, s0, s1 = sample_mealy()
        result = mm.execute_sequence(s0, ['x', 'y', 'x'])
        self.assertEqual(result, ['o1', 'o1', 'o3'])

    def test_execute_sequence_empty_returns_empty_list(self):
        mm, s0, _ = sample_mealy()
        self.assertEqual(mm.execute_sequence(s0, []), [])

    def test_compute_output_seq_does_not_mutate_current_state(self):
        mm, s0, s1 = sample_mealy()
        mm.reset_to_initial()
        mm.compute_output_seq(s1, ['x', 'x'])
        self.assertIs(mm.current_state, s0)


class TestMealyStructural(unittest.TestCase):
    def test_get_input_alphabet(self):
        mm, *_ = sample_mealy()
        self.assertEqual(set(mm.get_input_alphabet()), {'x', 'y'})

    def test_is_input_complete_true(self):
        mm, *_ = sample_mealy()
        self.assertTrue(mm.is_input_complete())

    def test_is_input_complete_false(self):
        s0 = MealyState('s0')
        s1 = MealyState('s1')
        s0.transitions = {'x': s1}
        s0.output_fun = {'x': 'o1'}
        s1.transitions = {'x': s1, 'y': s1}
        s1.output_fun = {'x': 'o1', 'y': 'o1'}
        mm = MealyMachine(s0, [s0, s1])
        self.assertFalse(mm.is_input_complete())

    def test_find_distinguishing_seq(self):
        mm, s0, s1 = sample_mealy()
        seq = mm.find_distinguishing_seq(s0, s1, mm.get_input_alphabet())
        self.assertIsNotNone(seq)
        self.assertNotEqual(mm.compute_output_seq(s0, seq), mm.compute_output_seq(s1, seq))

    def test_find_distinguishing_seq_same_state_is_none(self):
        mm, s0, _ = sample_mealy()
        self.assertIsNone(mm.find_distinguishing_seq(s0, s0, mm.get_input_alphabet()))

    def test_is_minimal(self):
        mm, *_ = sample_mealy()
        self.assertTrue(mm.is_minimal())


class TestMealyStateSetupRoundtrip(unittest.TestCase):
    def test_to_state_setup_from_state_setup_roundtrip(self):
        mm, s0, s1 = sample_mealy()
        setup = mm.to_state_setup()
        rebuilt = MealyMachine.from_state_setup(setup)

        for w in [[], ['x'], ['y'], ['x', 'x', 'y'], ['y', 'x', 'x', 'y']]:
            self.assertEqual(rebuilt.execute_sequence(rebuilt.initial_state, w),
                              mm.execute_sequence(mm.initial_state, w))

    def test_copy_is_independent(self):
        mm, s0, s1 = sample_mealy()
        mm_copy = mm.copy()
        mm_copy.get_state_by_id('s0').output_fun['x'] = 'CHANGED'
        self.assertEqual(mm.get_state_by_id('s0').output_fun['x'], 'o1')

    def test_pickle_roundtrip(self):
        mm, *_ = sample_mealy()
        restored = pickle.loads(pickle.dumps(mm))
        for w in [[], ['x'], ['y', 'x']]:
            self.assertEqual(restored.execute_sequence(restored.initial_state, w),
                              mm.execute_sequence(mm.initial_state, w))


class TestMealyEquality(unittest.TestCase):
    def test_eq_true_for_relabeled_equivalent_machine(self):
        mm, *_ = sample_mealy()

        t0 = MealyState('t0')
        t1 = MealyState('t1')
        t0.transitions = {'x': t1, 'y': t0}
        t0.output_fun = {'x': 'o1', 'y': 'o2'}
        t1.transitions = {'x': t0, 'y': t1}
        t1.output_fun = {'x': 'o3', 'y': 'o1'}
        relabeled = MealyMachine(t0, [t0, t1])

        self.assertEqual(mm, relabeled)

    def test_eq_false_for_different_outputs(self):
        mm, *_ = sample_mealy()
        other = mm.copy()
        other.get_state_by_id('s0').output_fun['x'] = 'different_output'
        self.assertNotEqual(mm, other)


if __name__ == '__main__':
    unittest.main()
