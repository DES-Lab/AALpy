# Equivalence oracle that exhaustively explores all input combinations up to a fixed depth.
from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL
from aalpy.base.Automaton import Automaton

from itertools import product
from random import shuffle


class BreadthFirstExplorationEqOracle(Oracle):
    """
    Breadth-First Exploration of all possible input combinations up to a certain depth.
    Extremely inefficient equivalence oracle and should only be used for demonstrations.
    """

    def __init__(self, alphabet: list, sul: SUL, depth: int = 5) -> None:
        """
        Constructs the oracle and pre-generates all test cases of the given depth.

        :param list alphabet: Input alphabet.
        :param SUL sul: System under learning.
        :param int depth: Depth of the tree, i.e. length of each generated test case.
        """

        super().__init__(alphabet, sul)
        self.depth = depth
        self.queue = []

        # generate all test-cases
        for seq in product(self.alphabet, repeat=self.depth):
            self.queue.append(seq)

        shuffle(self.queue)

    def find_cex(self, hypothesis: Automaton) -> tuple | None:
        """
        Executes queued test cases against the SUL and hypothesis until a counterexample is found.

        :param Automaton hypothesis: Current hypothesis.
        :return tuple | None: Counterexample inputs, None if no counterexample is found.
        """
        while self.queue:
            test_case = self.queue.pop()
            self.reset_hyp_and_sul(hypothesis)

            for ind, letter in enumerate(test_case):
                out_hyp = hypothesis.step(letter)
                out_sul = self.sul.step(letter)
                self.num_steps += 1

                if out_hyp != out_sul:
                    self.sul.post()
                    return test_case[:ind + 1]

            # cleanup after the test case
            self.sul.post()

        return None
