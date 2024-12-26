import random

from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL


class StatePrefixEqOracleTest(Oracle):
    """
    Equivalence oracle that achieves guided exploration by starting random walks from each state a walk_per_state
    times. Starting the random walk ensures that all states are reached at least walk_per_state times and that their
    surrounding is randomly explored. Note that each state serves as a root of random exploration of maximum length
    rand_walk_len exactly walk_per_state times during learning. Therefore excessive testing of initial states is
    avoided.
    """
    def __init__(self, alphabet: list, sul: SUL,  walks_per_state=10, walk_len=12, mode='sample'):
        """
        Args:

            alphabet: input alphabet

            sul: system under learning

            walks_per_state:individual walks per state of the automaton over the whole learning process

            walk_len:length of random walk

            depth_first:first explore newest states
        """

        assert mode in ['sample', 'shuffle']
        super().__init__(alphabet, sul)
        self.walks_per_state = walks_per_state
        self.steps_per_walk = walk_len
        self.mode = mode

        self.freq_dict = dict()

    def find_cex(self, hypothesis):
        
        if self.mode == 'shuffle':
            sampled_states = []
            for state in hypothesis.states:
                if state.prefix is None:
                    state.prefix = hypothesis.get_shortest_path(hypothesis.initial_state, state)
                if state.prefix not in self.freq_dict.keys():
                    self.freq_dict[state.prefix] = 0

                sampled_states.extend([state] * (self.walks_per_state - self.freq_dict[state.prefix]))
            random.shuffle(sampled_states)
        else:
            states = []
            weights = []
            for state in hypothesis.states:
                if state.prefix is None:
                    state.prefix = hypothesis.get_shortest_path(hypothesis.initial_state, state)
                if state.prefix not in self.freq_dict.keys():
                    self.freq_dict[state.prefix] = 0
                weight = self.walks_per_state - self.freq_dict[state.prefix]
                if weight > 0:  # Only consider states with a positive weight
                    states.append(state)
                    weights.append(weight)
            total_iterations = sum(weights)
            normalized = [w / total_iterations for w in weights]
            sampled_states = random.choices(states, weights=normalized, k=total_iterations)

        for state in sampled_states:
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
                    self.sul.post()
                    return prefix + suffix

        return None
