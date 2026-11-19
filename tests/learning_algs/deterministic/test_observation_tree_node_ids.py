import random
import unittest

from aalpy.SULs import AutomatonSUL
from aalpy.base import SUL
from aalpy.learning_algs import run_Lsharp
from aalpy.learning_algs.deterministic.ObservationTree import ObservationTree
from aalpy.oracles import WpMethodEqOracle
from aalpy.utils import generate_random_deterministic_automata, get_Angluin_dfa


class LoggingSUL(SUL):
    """A Mealy machine that records the inputs of each query."""

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
        self.queries[-1].append(letter)
        return self.model.step(letter)


def lsharp_queries(model):
    alphabet = model.get_input_alphabet()
    sul = LoggingSUL(model)
    oracle = WpMethodEqOracle(alphabet, sul, len(model.states))
    run_Lsharp(alphabet, sul, oracle, "mealy", print_level=0, cache_and_non_det_check=False)
    return sul.queries


class TestObservationTreeNodeIds(unittest.TestCase):
    """The nodes of an observation tree are numbered from 1, whatever trees were created before it."""

    def test_each_tree_numbers_its_nodes_from_one(self):
        for automaton_type in ("mealy", "dfa"):
            with self.subTest(automaton_type=automaton_type):
                first = ObservationTree(["a", "b"], AutomatonSUL(get_Angluin_dfa()), automaton_type, None, "SepSeq")
                first.insert_observation(("a", "b", "a"), (True, False, True))
                second = ObservationTree(["a", "b"], AutomatonSUL(get_Angluin_dfa()), automaton_type, None, "SepSeq")
                self.assertEqual(second.root.id, 1)
                self.assertEqual(second.get_size(), 1)

    def test_same_queries_in_the_same_process(self):
        random.seed(3)
        model = generate_random_deterministic_automata("mealy", num_states=30, input_alphabet_size=8, output_alphabet_size=4)
        self.assertEqual(lsharp_queries(model), lsharp_queries(model))


if __name__ == "__main__":
    unittest.main()
