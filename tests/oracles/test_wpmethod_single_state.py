import unittest

from aalpy.automata import MealyMachine, MealyState
from aalpy.learning_algs import run_Lsharp, run_Lstar
from aalpy.oracles.WpMethodEqOracle import WpMethodEqOracle
from aalpy.SULs import AutomatonSUL
from aalpy.utils import bisimilar


def two_state_mealy() -> MealyMachine:
    """'b' outputs 'e' until 'a' is input once, then 'k': a single-state hypothesis is wrong only after 'a', 'b'."""
    s0, s1 = MealyState("s0"), MealyState("s1")
    s0.transitions["a"], s0.output_fun["a"] = s1, "o"
    s0.transitions["b"], s0.output_fun["b"] = s0, "e"
    s1.transitions["a"], s1.output_fun["a"] = s1, "o"
    s1.transitions["b"], s1.output_fun["b"] = s1, "k"
    return MealyMachine(s0, [s0, s1])


class TestWpMethodSingleStateHypothesis(unittest.TestCase):
    """The Wp-method with a bound m finds every SUL with at most m states, also from a hypothesis with one state."""

    def test_bound_equal_to_the_states_of_the_sul(self):
        model = two_state_mealy()
        alphabet = model.get_input_alphabet()
        for algorithm in (run_Lsharp, run_Lstar):
            with self.subTest(algorithm=algorithm.__name__):
                sul = AutomatonSUL(model)
                learned = algorithm(alphabet, sul, WpMethodEqOracle(alphabet, sul, 2), "mealy", print_level=0)
                self.assertTrue(bisimilar(learned, model))


if __name__ == "__main__":
    unittest.main()
