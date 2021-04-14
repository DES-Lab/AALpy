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
        self.queue = list(self.alphabet)
        self.cache = set()

    def find_cex(self, hypothesis):

        for i in range(self.depth):
            tmp = []
            for seq in product(self.alphabet, self.queue):
                input_seq = tuple([i for sub in seq for i in sub])
                tmp.append(input_seq)
            self.queue = tmp

            shuffle(self.queue)

            for seq in self.queue:
                self.reset_hyp_and_sul(hypothesis)

                for ind, letter in enumerate(seq):
                    out_hyp = hypothesis.step(letter)
                    out_sul = self.sul.step(letter)
                    self.num_steps += 1

                    if out_hyp != out_sul:
                        return seq[:ind + 1]

        return None
