from random import shuffle, choice, randint, Random
import random

from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL
from aalpy.utils.HelperFunctions import product_with_possible_empty_iterable
from aalpy.learning_algs.deterministic.ObservationTree import ObservationTree


class WMethodEqOracle(Oracle):
    """
    Equivalence oracle based on characterization set/ W-set. From 'Tsun S. Chow.   Testing software design modeled by
    finite-state machines'.
    """

    def __init__(self, alphabet: list, sul: SUL, max_number_of_states, lookahead=None, shuffle_test_set=False):
        """
        Args:

            alphabet: input alphabet
            sul: system under learning
            max_number_of_states: maximum number of states in the automaton
            shuffle_test_set: if True, test cases will be shuffled
        """

        super().__init__(alphabet, sul)
        self.m = max_number_of_states
        self.shuffle = shuffle_test_set
        self.cache = set()
        self.lookahead = lookahead

    def find_cex(self, hypothesis, ob_tree=None):
        if not hypothesis.characterization_set:
            hypothesis.characterization_set = hypothesis.compute_characterization_set()

        # covers every transition of the specification at least once.
        transition_cover = [state.prefix + (letter,) for state in hypothesis.states for letter in self.alphabet]
        middle = []

        # Check for the number of expected states
        k_extra_states = self.m + 1 - len(hypothesis.states)
        # Check for k additional states
        if self.lookahead:
            k_extra_states = self.lookahead

        for i in range(k_extra_states):
            middle.extend(list(product_with_possible_empty_iterable(self.alphabet, repeat=i)))

        test_suite = product_with_possible_empty_iterable(transition_cover, middle, hypothesis.characterization_set) 
        if self.shuffle:
            test_suite = list(test_suite)
            Random(51).shuffle(test_suite)

        for seq in test_suite:
            inp_seq = tuple([i for sub in seq for i in sub])
            if inp_seq not in self.cache:
                self.reset_hyp_and_sul(hypothesis)
                outputs = []

                for ind, letter in enumerate(inp_seq):
                    out_hyp = hypothesis.step(letter)
                    out_sul = self.sul.step(letter)
                    self.num_steps += 1

                    outputs.append(out_sul)
                    if out_hyp != out_sul:
                        self.sul.post()
                        return inp_seq[:ind + 1]
                self.cache.add(inp_seq)
                # If an observation tree is given, we add the test queries to the observation tree
                if ob_tree:
                    ob_tree.insert_observation(inp_seq, outputs)

        return None


class RandomWMethodEqOracle(Oracle):
    """
    Randomized version of the W-Method equivalence oracle.
    Random walks stem from fixed prefix (path to the state). At the end of the random
    walk an element from the characterization set is added to the test case.
    """

    def __init__(self, alphabet: list, sul: SUL, walks_per_state=12, walk_len=12):
        """
        Args:

            alphabet: input alphabet

            sul: system under learning

            walks_per_state: number of random walks that should start from each state

            walk_len: length of random walk
        """

        super().__init__(alphabet, sul)
        self.walks_per_state = walks_per_state
        self.random_walk_len = walk_len
        self.freq_dict = dict()

    def find_cex(self, hypothesis):

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

        return None
