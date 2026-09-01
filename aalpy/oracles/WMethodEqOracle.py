# Equivalence oracles based on the W-method and its randomized variant.
from collections.abc import Iterator
from random import shuffle, choice, randint

from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL
from aalpy.base.Automaton import Automaton
from itertools import product


class WMethodEqOracle(Oracle):
    """
    Equivalence oracle based on characterization set/ W-set. From 'Tsun S. Chow.   Testing software design modeled by
    finite-state machines'.
    """

    def __init__(self, alphabet: list, sul: SUL, max_number_of_states: int) -> None:
        """
        Constructs the oracle.

        :param list alphabet: Input alphabet.
        :param SUL sul: System under learning.
        :param int max_number_of_states: Maximum number of states in the automaton.
        """

        super().__init__(alphabet, sul)
        self.m = max_number_of_states
        self.cache = set()

    def test_suite(self, cover: list, depth: int, char_set: list) -> Iterator[tuple]:
        """
        Constructs the test suite for the W Method using the provided state cover and characterization set,
        exploring up to a given depth.

        :param list cover: List of states to cover.
        :param int depth: Maximum length of middle part.
        :param list char_set: Characterization set.
        :return Iterator[tuple]: Iterator of generated test sequences.
        """
        # fix the length of the middle part per loop
        # to avoid generating large sequences early on
        char_set = char_set or [()]
        for d in range(depth):
            middle = product(self.alphabet, repeat=d)
            for m in middle:
                for (s, c) in product(cover, char_set):
                    yield s + m + c

    def find_cex(self, hypothesis: Automaton) -> tuple | None:
        """
        Runs the W-method test suite against the SUL until a counterexample is found.

        :param Automaton hypothesis: Current hypothesis.
        :return tuple | None: Counterexample inputs, None if no counterexample is found.
        """
        if not hypothesis.characterization_set:
            hypothesis.characterization_set = hypothesis.compute_characterization_set()

        # covers every transition of the specification at least once.
        transition_cover = [
            state.prefix + (letter,)
            for state in hypothesis.states
            for letter in self.alphabet
        ]

        depth = self.m + 1 - len(hypothesis.states)
        for seq in self.test_suite(transition_cover, depth, hypothesis.characterization_set):
            if seq not in self.cache:
                self.reset_hyp_and_sul(hypothesis)
                outputs = []

                for ind, letter in enumerate(seq):
                    out_hyp = hypothesis.step(letter)
                    out_sul = self.sul.step(letter)
                    self.num_steps += 1

                    outputs.append(out_sul)
                    if out_hyp != out_sul:
                        self.sul.post()
                        return seq[:ind + 1]

                self.cache.add(seq)
                self.sul.post()

        return None


class RandomWMethodEqOracle(Oracle):
    """
    Randomized version of the W-Method equivalence oracle.
    Random walks stem from fixed prefix (path to the state). At the end of the random
    walk an element from the characterization set is added to the test case.
    """

    def __init__(self, alphabet: list, sul: SUL, walks_per_state: int = 25, walk_len: int = 12) -> None:
        """
        Constructs the oracle.

        :param list alphabet: Input alphabet.
        :param SUL sul: System under learning.
        :param int walks_per_state: Number of random walks that should start from each state.
        :param int walk_len: Length of random walk.
        """

        super().__init__(alphabet, sul)
        self.walks_per_state = walks_per_state
        self.random_walk_len = walk_len
        self.freq_dict = dict()

    def find_cex(self, hypothesis: Automaton) -> tuple | None:
        """
        Performs random walks from each state, ending with a characterizing suffix, until a counterexample is
        found.

        :param Automaton hypothesis: Current hypothesis.
        :return tuple | None: Counterexample inputs, None if no counterexample is found.
        """
        if not hypothesis.characterization_set:
            hypothesis.characterization_set = hypothesis.compute_characterization_set()
            # fix for non-minimal intermediate hypothesis that can occur in KV
            if not hypothesis.characterization_set:
                hypothesis.characterization_set = [(a,) for a in hypothesis.get_input_alphabet()]

        states_to_cover = []
        for state in hypothesis.states:
            if state.prefix is None:
                state.prefix = hypothesis.get_shortest_path(hypothesis.initial_state, state)
            if state.prefix not in self.freq_dict.keys():
                self.freq_dict[state.prefix] = 0

            states_to_cover.extend([state] * (self.walks_per_state - self.freq_dict[state.prefix]))

        shuffle(states_to_cover)

        for state in states_to_cover:
            self.freq_dict[state.prefix] = self.freq_dict[state.prefix] + 1

            self.reset_hyp_and_sul(hypothesis)

            prefix = state.prefix
            random_walk = tuple(choice(self.alphabet) for _ in range(randint(1, self.random_walk_len)))

            test_case = prefix + random_walk + choice(hypothesis.characterization_set)

            for ind, i in enumerate(test_case):
                output_hyp = hypothesis.step(i)
                output_sul = self.sul.step(i)
                self.num_steps += 1

                if output_sul != output_hyp:
                    self.sul.post()
                    return test_case[:ind + 1]

            self.sul.post()

        return None
