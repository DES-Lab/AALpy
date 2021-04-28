from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL
from itertools import combinations, product
from random import shuffle, choice, randint


class WMethodEqOracle(Oracle):
    """
    Equivalence oracle based on characterization set/ W-set. From 'Tsun S. Chow.   Testing software design modeled by
    finite-state machines'.
    """
    def __init__(self, alphabet: list, sul: SUL, max_number_of_states, shuffle_test_set=True):
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

    def find_cex(self, hypothesis):

        assert hypothesis.characterization_set is not None

        # covers every transition of the specification at least once.
        transition_cover = {state.prefix + (letter,) for state in hypothesis.states for letter in self.alphabet}

        middle = []
        for i in range(self.m - len(hypothesis.states)):
            middle.extend(combinations(self.alphabet, i + 1))

        test_set = []
        for seq in product(transition_cover, middle, hypothesis.characterization_set):
            inp_seq = tuple([i for sub in seq for i in sub])
            if inp_seq not in self.cache:
                test_set.append(inp_seq)

        if self.shuffle:
            shuffle(test_set)
        else:
            test_set.sort(key=len, reverse=True)

        for seq in test_set:
            self.reset_hyp_and_sul(hypothesis)
            outputs = []

            for ind, letter in enumerate(seq):
                out_hyp = hypothesis.step(letter)
                out_sul = self.sul.step(letter)
                self.num_steps += 1

                outputs.append(out_sul)
                if out_hyp != out_sul:
                    return seq[:ind + 1]
            self.cache.add(seq)

        return None


class RandomWMethodEqOracle(Oracle):
    """
    Randomized version of the W-Method equivalence oracle.
    Random walks stem from fixed prefix (path to the state). At the end of the random
    walk an element from the characterization set is added to the test case.
    """
    def __init__(self, alphabet: list, sul: SUL, walks_per_state=10, walk_len=20):
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

        states_to_cover = []
        for state in hypothesis.states:
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
                    return test_case[:ind + 1]

        return None
