import importlib
import unittest
from unittest.mock import patch

from aalpy.automata import MealyMachine, MealyState
from aalpy.oracles.UserInputEqOracle import UserInputEqOracle
from aalpy.SULs import AutomatonSUL


def sample_mealy():
    """2-state Mealy machine over {x, y}."""
    s0 = MealyState('s0')
    s1 = MealyState('s1')
    s0.transitions = {'x': s1, 'y': s0}
    s0.output_fun = {'x': 'o1', 'y': 'o2'}
    s1.transitions = {'x': s0, 'y': s1}
    s1.output_fun = {'x': 'o3', 'y': 'o1'}
    mm = MealyMachine(s0, [s0, s1])
    mm.compute_prefixes()
    return mm


class UserInputEqOracleTests(unittest.TestCase):

    def setUp(self):
        module = importlib.import_module("aalpy.oracles.UserInputEqOracle")
        self.visualize_patcher = patch.object(module, "visualize_automaton")
        self.visualize_patcher.start()

    def tearDown(self):
        self.visualize_patcher.stop()

    def test_user_enters_inputs_then_requests_cex(self):
        mm = sample_mealy()
        oracle = UserInputEqOracle(['x', 'y'], AutomatonSUL(mm))

        with patch('builtins.input', side_effect=['x', 'y', 'cex']):
            cex = oracle.find_cex(mm)

        self.assertEqual(cex, ['x', 'y'])

    def test_user_ends_session_without_a_counterexample(self):
        mm = sample_mealy()
        oracle = UserInputEqOracle(['x', 'y'], AutomatonSUL(mm))

        with patch('builtins.input', side_effect=['x', 'end']):
            cex = oracle.find_cex(mm)

        self.assertIsNone(cex)

    def test_cex_command_with_no_inputs_yet_is_ignored_and_prompts_again(self):
        mm = sample_mealy()
        oracle = UserInputEqOracle(['x', 'y'], AutomatonSUL(mm))

        # 'cex' before any input is entered is a no-op (inputs is empty and falsy), so the loop must continue
        with patch('builtins.input', side_effect=['cex', 'x', 'cex']):
            cex = oracle.find_cex(mm)

        self.assertEqual(cex, ['x'])

    def test_reset_clears_inputs_entered_so_far(self):
        mm = sample_mealy()
        oracle = UserInputEqOracle(['x', 'y'], AutomatonSUL(mm))

        with patch('builtins.input', side_effect=['x', 'x', 'reset', 'y', 'cex']):
            cex = oracle.find_cex(mm)

        self.assertEqual(cex, ['y'])

    def test_unknown_command_and_letter_not_in_alphabet_are_rejected(self):
        mm = sample_mealy()
        oracle = UserInputEqOracle(['x', 'y'], AutomatonSUL(mm))

        with patch('builtins.input', side_effect=['help', 'print alphabet', 'current inputs',
                                                   'not_a_valid_letter', 'x', 'cex']) as mocked_input:
            cex = oracle.find_cex(mm)

        self.assertEqual(cex, ['x'])
        self.assertEqual(mocked_input.call_count, 6)


if __name__ == '__main__':
    unittest.main()
