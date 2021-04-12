from random import choices, shuffle

from aalpy.base import Automaton, Oracle, SUL

class KWayTransitionCoverageEqOracle(Oracle):
    """
    WIP
    """
    def __init__(self, alphabet: list, sul: SUL, k=2, random_walk_len=100):
        super().__init__(alphabet, sul)
        assert k == 2

        self.k = k
        self.random_walk_len = random_walk_len

    def find_cex(self, hypothesis: Automaton):                

        states = hypothesis.states
        shuffle(states)

        for target_state in states:
            self.num_queries += 1

            for prev_state, prev_transition in hypothesis.get_prev_states(target_state):
                for next_transiton in target_state.get_transitions():
                    path = prev_state.prefix + (prev_transition, next_transiton)

                    path += tuple(choices(self.alphabet, k=self.random_walk_len))
                    counter_example = self.check_path(hypothesis, path)

                    if counter_example is not None:
                        return counter_example        
        return None

    def check_path(self, hypothesis, path):
        hypothesis.reset_to_initial()
        self.sul.post()
        self.sul.pre()

        for i, p in enumerate(path):
            out_sul = self.sul.step(p)
            out_hyp = hypothesis.step(p)

            self.num_steps += 1

            if out_sul != out_hyp:
                return path[:i + 1]
        
        return None