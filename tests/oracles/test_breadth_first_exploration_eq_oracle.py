import unittest

from aalpy.automata import MealyMachine, MealyState
from aalpy.oracles import BreadthFirstExplorationEqOracle
from aalpy.SULs import AutomatonSUL


def chain_mealy(length, alphabet=('a',)):
    """
    Chain of states over `alphabet`, deterministic on the first letter of `alphabet`. Repeatedly feeding the
    first letter walks s0 -> s1 -> ... -> s_length.
    """
    states = [MealyState(f's{i}') for i in range(length + 1)]
    first = alphabet[0]
    for i in range(length):
        states[i].transitions = {a: states[i] for a in alphabet}
        states[i].transitions[first] = states[i + 1]
        states[i].output_fun = {a: 'o' for a in alphabet}
    states[length].transitions = {a: states[length] for a in alphabet}
    states[length].output_fun = {a: 'o' for a in alphabet}
    mm = MealyMachine(states[0], states)
    mm.compute_prefixes()
    return mm


def with_diverging_transition(mm, diverge_at, alphabet=('a',)):
    first = alphabet[0]
    mm.states[diverge_at - 1].output_fun[first] = 'x'
    return mm


class BreadthFirstExplorationEqOracleTests(unittest.TestCase):

    def test_finds_cex_within_depth(self):
        reference = chain_mealy(5)
        hypothesis = chain_mealy(5)
        with_diverging_transition(hypothesis, 3)

        oracle = BreadthFirstExplorationEqOracle(['a'], AutomatonSUL(reference), depth=5)
        cex = oracle.find_cex(hypothesis)

        self.assertIsNotNone(cex)
        reference.reset_to_initial()
        hypothesis.reset_to_initial()
        sul_out = [reference.step(i) for i in cex]
        hyp_out = [hypothesis.step(i) for i in cex]
        self.assertNotEqual(sul_out[-1], hyp_out[-1])

    def test_returns_shortest_cex_not_full_depth(self):
        reference = chain_mealy(5)
        hypothesis = chain_mealy(5)
        with_diverging_transition(hypothesis, 2)

        oracle = BreadthFirstExplorationEqOracle(['a'], AutomatonSUL(reference), depth=5)
        cex = oracle.find_cex(hypothesis)

        self.assertEqual(tuple(cex), ('a', 'a'))

    def test_difference_beyond_depth_bound_is_not_found(self):
        reference = chain_mealy(5)
        hypothesis = chain_mealy(5)
        with_diverging_transition(hypothesis, 5)

        oracle = BreadthFirstExplorationEqOracle(['a'], AutomatonSUL(reference), depth=3)
        self.assertIsNone(oracle.find_cex(hypothesis))

    def test_no_cex_for_equivalent_hypothesis(self):
        reference = chain_mealy(4)
        hypothesis = chain_mealy(4)

        oracle = BreadthFirstExplorationEqOracle(['a'], AutomatonSUL(reference), depth=4)
        self.assertIsNone(oracle.find_cex(hypothesis))


if __name__ == '__main__':
    unittest.main()
