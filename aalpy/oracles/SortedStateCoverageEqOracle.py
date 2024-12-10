import random

from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL

modes = ['random', 'newest', 'oldest']

class SortedStateCoverageEqOracle(Oracle):
    def __init__(self, alphabet: list, sul: SUL, walks_per_state=10, walk_len=12, mode='random'):
        """
        Args:

            alphabet: input alphabet

            sul: system under learning

            walks_per_state:individual walks per state of the automaton over the whole learning process

            walk_len:length of random walk

            mode: 'random', 'newest' or 'oldest'
        """

        assert mode in modes, f"Mode must be one of {modes}"

        super().__init__(alphabet, sul)
        self.walks_per_state = walks_per_state
        self.steps_per_walk = walk_len
        self.mode = mode
        # add a dict that keeps track of the 'age' of the state
        # the age is incremented with every new hypothesis
        self.age_dict = dict()

    def find_cex(self, hypothesis):
        # update the age of the states
        for state in hypothesis.states:
            if state.state_id not in self.age_dict.keys():
                self.age_dict[state.state_id] = 1
            else:
                self.age_dict[state.state_id] += 1

        for state in hypothesis.states:
            if state.prefix is None:
                state.prefix = hypothesis.get_shortest_path(hypothesis.initial_state, state)

        states_to_cover = [s for s in hypothesis.states for _ in range(self.walks_per_state)]
        
        if self.mode == 'random':
            random.shuffle(states_to_cover)
        elif self.mode == 'newest':
            states_to_cover.sort(key=lambda x: self.age_dict[x.state_id])
        else:
            states_to_cover.sort(key=lambda x: self.age_dict[x.state_id], reverse=True)

        for state in states_to_cover:
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
                    self.sul.post()
                    return prefix + suffix

        return None
