# Probably approximately correct (PAC) equivalence oracle.
from math import ceil, log
from random import choice, randint

from aalpy.base import Oracle, SUL
from aalpy.base.Automaton import Automaton


class PacOracle(Oracle):
    """
    Probably approximately correct oracle. Number of queries is defined by the following equation:
    1 / self.epsilon * (log(1 / self.delta) + self.round * log(2)), where epsilon is the generalization error and delta
    the confidence. Thus, returned hypothesis is the epsilon-approximation of the correct hypothesis with the probability
    1 - delta (Mohri, M et al.: Foundations of Machine Learning).
    Queries are of random length in a predefined range.
    """

    def __init__(self, alphabet: list, sul: SUL, epsilon: float = 0.01, delta: float = 0.01,
                 min_walk_len: int = 10, max_walk_len: int = 25) -> None:
        """
        Constructs the oracle.

        :param list alphabet: Input alphabet.
        :param SUL sul: System under learning.
        :param float epsilon: Generalization error.
        :param float delta: Confidence.
        :param int min_walk_len: Minimum length of each random query.
        :param int max_walk_len: Maximum length of each random query.
        """
        super().__init__(alphabet, sul)
        self.min_walk_len = min_walk_len
        self.max_walk_len = max_walk_len
        self.epsilon = epsilon
        self.delta = delta
        self.round = 0

    def find_cex(self, hypothesis: Automaton) -> list | None:
        """
        Performs a number of random-length queries, growing per round, until a counterexample is found.

        :param Automaton hypothesis: Current hypothesis.
        :return list | None: Counterexample inputs, None if no counterexample is found.
        """
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

            # cleanup after the test case
            self.sul.post()

        return None
