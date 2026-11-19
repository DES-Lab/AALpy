# Equivalence oracle that covers k-wise combinations/permutations of states with a trailing random walk.
from random import choices, shuffle

from aalpy.base import Oracle, SUL
from aalpy.base.Automaton import Automaton
from itertools import combinations, permutations


class KWayStateCoverageEqOracle(Oracle):
    """
    A test case will be computed for every k-combination or k-permutation of states with additional
    random walk at the end.
    """

    def __init__(self, alphabet: list, sul: SUL, k: int = 2, random_walk_len: int = 20,
                 method: str = 'permutations',
                 num_test_lower_bound: int | None = None,
                 num_test_upper_bound: int | None = None) -> None:
        """
        Constructs the oracle.

        :param list alphabet: Input alphabet.
        :param SUL sul: System under learning.
        :param int k: k value used for k-wise combinations/permutations of states.
        :param int random_walk_len: Length of random walk performed at the end of each combination/permutation.
        :param str method: Either 'combinations' or 'permutations'.
        :param int | None num_test_lower_bound: Either None or a minimum number of test-cases to be performed in
            each testing round.
        :param int | None num_test_upper_bound: Either None or a maximum number of test-cases to be performed in
            each testing round.
        """
        super().__init__(alphabet, sul)
        assert k > 1 and method in ['combinations', 'permutations']
        self.k = k
        self.cache = set()
        self.fun = combinations if method == 'combinations' else permutations
        self.random_walk_len = random_walk_len

        self.num_test_lower_bound = num_test_lower_bound
        self.num_test_upper_bound = num_test_upper_bound

    def find_cex(self, hypothesis: Automaton) -> tuple | None:
        """
        Generates and executes test cases covering k-wise state combinations/permutations until a counterexample
        is found.

        :param Automaton hypothesis: Current hypothesis.
        :return tuple | None: Counterexample inputs, None if no counterexample is found.
        """
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

        # lower bound (also accounts for hypotheses with fewer states than k, where no k-wise
        # combination/permutation exists at all, so test_cases would otherwise stay empty)
        lower_bound = self.num_test_lower_bound
        if len(hypothesis.states) < self.k and lower_bound is None:
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

            # cleanup after the test case
            self.sul.post()

        return None
