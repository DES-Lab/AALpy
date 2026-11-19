import random
import unittest

from aalpy.automata import MealyMachine, MealyState
from aalpy.oracles import RandomWordEqOracle
from aalpy.SULs import AutomatonSUL


def chain_mealy(length, alphabet=('a',)):
    """Chain of states, s0 -> s1 -> ... -> s_length, driven by repeating the first letter of `alphabet`."""
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
    mm.states[diverge_at - 1].output_fun[alphabet[0]] = 'x'
    return mm


class RandomWordEqOracleTests(unittest.TestCase):

    def test_finds_cex_over_several_seeds(self):
        successes = 0
        for seed in range(10):
            random.seed(seed)
            reference = chain_mealy(4, alphabet=('a', 'b'))
            hypothesis = chain_mealy(4, alphabet=('a', 'b'))
            with_diverging_transition(hypothesis, 4, alphabet=('a', 'b'))

            oracle = RandomWordEqOracle(['a', 'b'], AutomatonSUL(reference), num_walks=200,
                                        min_walk_len=5, max_walk_len=10)
            cex = oracle.find_cex(hypothesis)
            if cex is not None:
                successes += 1
                reference.reset_to_initial()
                hypothesis.reset_to_initial()
                sul_out = [reference.step(i) for i in cex]
                hyp_out = [hypothesis.step(i) for i in cex]
                self.assertNotEqual(sul_out[-1], hyp_out[-1])

        self.assertGreaterEqual(successes, 9)

    def test_no_cex_for_equivalent_hypothesis(self):
        random.seed(0)
        reference = chain_mealy(4, alphabet=('a', 'b'))
        hypothesis = chain_mealy(4, alphabet=('a', 'b'))

        oracle = RandomWordEqOracle(['a', 'b'], AutomatonSUL(reference), num_walks=200)
        self.assertIsNone(oracle.find_cex(hypothesis))

    def test_fixed_walk_length_bounds_reachable_difference(self):
        # min_walk_len == max_walk_len == 5, alphabet of size 1, so every walk is exactly 'aaaaa'
        reference = chain_mealy(6)
        hypothesis = chain_mealy(6)
        with_diverging_transition(hypothesis, 6)

        oracle = RandomWordEqOracle(['a'], AutomatonSUL(reference), num_walks=5, min_walk_len=5, max_walk_len=5)
        self.assertIsNone(oracle.find_cex(hypothesis), "walk length is fixed below the divergence depth")

    def test_fixed_walk_length_finds_reachable_difference(self):
        reference = chain_mealy(5)
        hypothesis = chain_mealy(5)
        with_diverging_transition(hypothesis, 5)

        oracle = RandomWordEqOracle(['a'], AutomatonSUL(reference), num_walks=5, min_walk_len=5, max_walk_len=5)
        cex = oracle.find_cex(hypothesis)
        self.assertEqual(tuple(cex), ('a',) * 5)

    def test_reset_after_cex_false_does_not_replenish_walk_budget(self):
        reference = chain_mealy(5)
        hypothesis = chain_mealy(5)
        with_diverging_transition(hypothesis, 5)

        oracle = RandomWordEqOracle(['a'], AutomatonSUL(reference), num_walks=1, min_walk_len=5, max_walk_len=5,
                                    reset_after_cex=False)

        first_cex = oracle.find_cex(hypothesis)
        self.assertIsNotNone(first_cex)
        self.assertEqual(oracle.num_walks_done, oracle.num_walks)

        second_cex = oracle.find_cex(hypothesis)
        self.assertIsNone(second_cex)


if __name__ == '__main__':
    unittest.main()
