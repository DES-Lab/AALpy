from random import choices, shuffle

from aalpy.base import Oracle, SUL
from itertools import combinations, permutations


class KWayStateCoverageEqOracle(Oracle):
    """
    A test case will be computed for every k-combination or k-permutation of states with additional
    random walk at the end.
    """

    def __init__(self, alphabet: list, sul: SUL, k=2, random_walk_len=20,
                 method='permutations',
                 num_test_lower_bound=None,
                 num_test_upper_bound=None):
        """

        Args:

            alphabet: input alphabet
            sul: system under learning
            k: k value used for k-wise combinations/permutations of states
            random_walk_len: length of random walk performed at the end of each combination/permutation
            method: either 'combinations' or 'permutations'
            num_test_lower_bound= either None or number a minimum number of test-cases to be performed in each testing round
            num_test_upper_bound= either None or number a maximum number of test-cases to be performed in each testing round

        """
        super().__init__(alphabet, sul)
        assert k > 1 and method in ['combinations', 'permutations']
        self.k = k
        self.cache = set()
        self.fun = combinations if method == 'combinations' else permutations
        self.random_walk_len = random_walk_len

        self.num_test_lower_bound = num_test_lower_bound
        self.num_test_upper_bound = num_test_upper_bound

    def find_cex(self, hypothesis):

        shuffle(hypothesis.states)

        test_cases = []
        for comb in self.fun(hypothesis.states, self.k):
            prefixes = frozenset([c.prefix for c in comb])
            if prefixes in self.cache:
                continue
            self.cache.add(prefixes)

            index = 0
            path = comb[0].prefix
            possible_test_case = True

            while index < len(comb) - 1:
                path_between_states = hypothesis.get_shortest_path(comb[index], comb[index + 1])
                index += 1

                if not path_between_states:
                    possible_test_case = False
                    break

                path += path_between_states

            if possible_test_case is None:
                continue

            path += tuple(choices(self.alphabet, k=self.random_walk_len))
            test_cases.append(path)

        # lower bound (also accounts for single state hypothesis when a lower bound is not defined)
        lower_bound = self.num_test_lower_bound
        if len(hypothesis.states) == 1 and lower_bound is None:
            lower_bound = 50

        while lower_bound is not None and len(test_cases) < lower_bound:
            path = tuple(choices(self.alphabet, k=self.random_walk_len))
            test_cases.append(path)

        # upper bound
        if self.num_test_upper_bound is not None:
            test_cases = test_cases[:self.num_test_upper_bound]

        for path in test_cases:
            self.reset_hyp_and_sul(hypothesis)
            for i, p in enumerate(path):
                out_sul = self.sul.step(p)
                out_hyp = hypothesis.step(p)
                self.num_steps += 1

                if out_sul != out_hyp:
                    self.sul.post()
                    return path[:i + 1]

        return None
