from random import choices, shuffle

from aalpy.base import Oracle, SUL
from itertools import combinations, permutations


class KWayStateCoverageEqOracle(Oracle):
    """
    A test case will be computed for every k-combination or k-permutation of states with additional
    random walk at the end.
    """

    def __init__(self, alphabet: list, sul: SUL, k=2, random_walk_len=100, method='combinations'):
        """

        Args:

            alphabet: input alphabet

            sul: system under learning

            k: k value used for k-wise combinations/permutations of states

            random_walk_len: length of random walk performed at the end of each combination/permutation

            method: either 'combinations' or 'permutations'
        """
        super().__init__(alphabet, sul)
        assert k > 1 and method in ['combinations', 'permutations']
        self.k = k
        self.cache = set()
        self.fun = combinations if method == 'combinations' else permutations
        self.random_walk_len = random_walk_len

    def find_cex(self, hypothesis):

        if len(hypothesis.states) == 1:
            for _ in range(self.random_walk_len):
                path = choices(self.alphabet, k=self.random_walk_len)
                hypothesis.reset_to_initial()
                self.sul.post()
                self.sul.pre()
                for i, p in enumerate(path):
                    out_sul = self.sul.step(p)
                    out_hyp = hypothesis.step(p)
                    self.num_steps += 1

                    if out_sul != out_hyp:
                        return path[:i + 1]

        states = hypothesis.states
        shuffle(states)

        for comb in self.fun(hypothesis.states, self.k):
            prefixes = frozenset([c.prefix for c in comb])
            if prefixes in self.cache:
                continue
            else:
                self.cache.add(prefixes)

            index = 0
            path = comb[0].prefix
            while index < len(comb) - 1:
                path += hypothesis.get_shortest_path(comb[index], comb[index + 1])
                index += 1

            path += tuple(choices(self.alphabet, k=self.random_walk_len))

            self.reset_hyp_and_sul(hypothesis)

            for i, p in enumerate(path):
                out_sul = self.sul.step(p)
                out_hyp = hypothesis.step(p)
                self.num_steps += 1

                if out_sul != out_hyp:
                    return path[:i + 1]

        return None
