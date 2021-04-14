import random

from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL


class StatePrefixEqOracle(Oracle):
    """
    Equivalence oracle that achieves guided exploration by starting random walks from each state a walk_per_state
    times. Starting the random walk ensures that all states are reached at least walk_per_state times and that their
    surrounding is randomly explored. Note that each state serves as a root of random exploration of maximum length
    rand_walk_len exactly walk_per_state times during learning. Therefore excessive testing of initial states is
    avoided.
    """
    def __init__(self, alphabet: list, sul: SUL, walks_per_state=10, walk_len=30, depth_first=False):
        """
        Args:

            alphabet: input alphabet

            sul: system under learning

            walks_per_state:individual walks per state of the automaton over the whole learning process

            walk_len:length of random walk

            depth_first:first explore newest states
        """

        super().__init__(alphabet, sul)
        self.walks_per_state = walks_per_state
        self.steps_per_walk = walk_len
        self.depth_first = depth_first

        self.freq_dict = dict()

    def find_cex(self, hypothesis):

        states_to_cover = []
        for state in hypothesis.states:
            if state.prefix not in self.freq_dict.keys():
                self.freq_dict[state.prefix] = 0

            states_to_cover.extend([state] * (self.walks_per_state - self.freq_dict[state.prefix]))

        if self.depth_first:
            # reverse sort the states by length of their access sequences
            # first do the random walk on the state with longest access sequence
            states_to_cover.sort(key=lambda x: len(x.prefix), reverse=True)
        else:
            random.shuffle(states_to_cover)

        for state in states_to_cover:
            self.freq_dict[state.prefix] = self.freq_dict[state.prefix] + 1

            self.reset_hyp_and_sul(hypothesis)

            prefix = state.prefix
            for p in prefix:
                hypothesis.step(p)
                self.sul.step(p)
                self.num_steps += 1

            suffix = ()
            for _ in range(self.steps_per_walk):
                suffix += (random.choice(self.alphabet),)

                out_sul = self.sul.step(suffix[-1])
                out_hyp = hypothesis.step(suffix[-1])
                self.num_steps += 1

                if out_sul != out_hyp:
                    return prefix + suffix

        return None
