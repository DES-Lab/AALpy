# Equivalence oracle that reuses the trace cache built during learning to guide test case selection.
from typing import Any

from aalpy.base import Oracle, SUL
from aalpy.base.SUL import CacheSUL
from aalpy.base.Automaton import Automaton

from random import choice


class CacheBasedEqOracle(Oracle):
    """
    Equivalence oracle where test case selection is based on the multiset of all traces observed during learning and
    conformance checking. Firstly all leaves of the tree are gathered and then random leaves are extended with a suffix
    of length (max_tree_depth + 'depth_increase') - len(prefix), where prefix is a path to the leaf.
    """

    def __init__(self, alphabet: list, sul: SUL, num_walks: int = 100, depth_increase: int = 5,
                 reset_after_cex: bool = True) -> None:
        """
        Constructs the oracle.

        :param list alphabet: Input alphabet.
        :param SUL sul: System under learning. Must wrap or be a CacheSUL.
        :param int num_walks: Number of random walks to perform.
        :param int depth_increase: Length of random walk that exceeds the maximum depth of the tree.
        :param bool reset_after_cex: If False, total number of queries will equal num_walks, if True, in each
            execution of find_cex method at most num_walks will be executed.
        """

        super().__init__(alphabet, sul)
        self.cache_tree = None
        self.num_walks = num_walks
        self.depth_increase = depth_increase
        self.reset_after_cex = reset_after_cex
        self.num_walks_done = 0

    def find_cex(self, hypothesis: Automaton) -> list | None:
        """
        Performs random walks starting from cached prefixes until a counterexample is found.

        :param Automaton hypothesis: Current hypothesis.
        :return list | None: Counterexample inputs, None if no counterexample is found.
        """
        assert isinstance(self.sul, CacheSUL)
        self.cache_tree = self.sul.cache

        paths_to_leaves = self.get_paths(self.cache_tree.root_node)
        max_tree_depth = len(max(paths_to_leaves, key=len))

        while self.num_walks_done < self.num_walks:
            self.num_walks_done += 1
            self.reset_hyp_and_sul(hypothesis)

            prefix = choice(paths_to_leaves)
            walk_len = (max_tree_depth + self.depth_increase) - len(prefix)
            inputs = []
            inputs.extend(prefix)

            for p in prefix:
                hypothesis.step(p)
                self.sul.step(p)
                self.num_steps += 1

            for _ in range(walk_len):
                inputs.append(choice(self.alphabet))

                out_sul = self.sul.step(inputs[-1])
                out_hyp = hypothesis.step(inputs[-1])
                self.num_steps += 1

                if out_sul != out_hyp:
                    if self.reset_after_cex:
                        self.num_walks_done = 0
                    self.sul.post()
                    return inputs

            # cleanup after the test case
            self.sul.post()

        return None

    def get_paths(self, t: Any, paths: list | None = None, current_path: list | None = None) -> list:
        """
        Recursively collects the paths (sequences of inputs) from the root of a cache tree node to all its leaves.

        :param Any t: Cache tree node to collect paths from.
        :param list | None paths: Accumulator of completed paths, created if None.
        :param list | None current_path: Path accumulated so far to reach node t, created if None.
        :return list: List of paths (each a list of inputs) from t to its leaves.
        """
        if paths is None:
            paths = []
        if current_path is None:
            current_path = []

        if len(t.children) == 0:
            paths.append(current_path)
        else:
            for inp, child in t.children.items():
                current_path.append(inp)
                self.get_paths(child, paths, list(current_path))
        return paths
