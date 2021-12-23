from math import ceil, log
from random import choice, randint

from aalpy.base import Oracle, SUL


class PacOracle(Oracle):
    """
    Probably approximately correct oracle. Number of queries is defined by the following equation:
    1 / self.epsilon * (log(1 / self.delta) + self.round * log(2)), where epsilon is the generalization error and delta
    the confidence. Thus, returned hypothesis is the epsilon-approximation of the correct hypothesis with the probability
    1 - delta (Mohri, M et al.: Foundations of Machine Learning).
    Queries are of random length in a predefined range.
    """

    def __init__(self, alphabet: list, sul: SUL, epsilon=0.01, delta=0.01, min_walk_len=10, max_walk_len=25):

        super().__init__(alphabet, sul)
        self.min_walk_len = min_walk_len
        self.max_walk_len = max_walk_len
        self.epsilon = epsilon
        self.delta = delta
        self.round = 0

    def find_cex(self, hypothesis):
        self.round += 1
        num_test_cases = 1 / self.epsilon * (log(1 / self.delta) + self.round * log(2))

        for i in range(ceil(num_test_cases)):
            inputs = []
            self.reset_hyp_and_sul(hypothesis)

            num_steps = randint(self.min_walk_len, self.max_walk_len)

            for _ in range(num_steps):
                inputs.append(choice(self.alphabet))

                out_sul = self.sul.step(inputs[-1])
                out_hyp = hypothesis.step(inputs[-1])
                self.num_steps += 1

                if out_sul != out_hyp:
                    self.sul.post()
                    return inputs

        return None
