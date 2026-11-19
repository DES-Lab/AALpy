import unittest

from aalpy.automata import MealyMachine, MealyState
from aalpy.base import SUL
from aalpy.learning_algs import run_Lsharp
from aalpy.oracles.WpMethodEqOracle import WpMethodEqOracle


class Letter:
    """An input with a given hash, so that the same alphabet can be learned with different hashes."""

    def __init__(self, name, hash_value):
        self.name = name
        self.hash_value = hash_value

    def __hash__(self):
        return self.hash_value

    def __repr__(self):
        return self.name


class LoggingSUL(SUL):
    """A Mealy machine that records the names of the inputs of each query."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.queries = []

    def pre(self):
        self.model.reset_to_initial()
        self.queries.append([])

    def post(self):
        pass

    def step(self, letter):
        self.queries[-1].append(letter.name)
        return self.model.step(letter)


def counter(a, b, c):
    """A counter modulo 4 of the input a, which b resets and c reads."""
    states = [MealyState(f"s{i}") for i in range(4)]
    for i, state in enumerate(states):
        state.transitions[a], state.output_fun[a] = states[(i + 1) % 4], "o"
        state.transitions[b], state.output_fun[b] = states[0], "o"
        state.transitions[c], state.output_fun[c] = state, str(i)
    return MealyMachine(states[0], states)


def queries(hash_values):
    alphabet = [Letter(name, hash_value) for name, hash_value in zip("abc", hash_values)]
    sul = LoggingSUL(counter(*alphabet))
    oracle = WpMethodEqOracle(alphabet, sul, 6)
    run_Lsharp(alphabet, sul, oracle, "mealy", print_level=0, cache_and_non_det_check=False)
    return sul.queries


class TestWpMethodOrder(unittest.TestCase):
    """The Wp-method asks the same queries in the same order, whatever the hashes of the inputs."""

    def test_same_queries_with_different_hashes(self):
        self.assertEqual(queries([1, 2, 3]), queries([3, 2, 1]))
        self.assertEqual(queries([1, 2, 3]), queries([104729, 7919, 13]))


if __name__ == "__main__":
    unittest.main()
