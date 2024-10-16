from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL

from itertools import product
from random import shuffle


class BreadthFirstExplorationEqOracle(Oracle):
    """
    Breadth-First Exploration of all possible input combinations up to a certain depth.
    Extremely inefficient equivalence oracle and should only be used for demonstrations.
    """

    def __init__(self, alphabet, sul: SUL, depth=5):
        """
        Args:

            alphabet: input alphabet

            sul: system under learning

            depth: depth of the tree
        """

        super().__init__(alphabet, sul)
        self.depth = depth
        self.queue = []

        # generate all test-cases
        for seq in product(self.alphabet, repeat=self.depth):
            self.queue.append(seq)

        shuffle(self.queue)

    def find_cex(self, hypothesis):

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

        return None
