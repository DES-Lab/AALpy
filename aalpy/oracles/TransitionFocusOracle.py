import random

from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL


class TransitionFocusOracle(Oracle):
    """
    This equivalence oracle focuses either on the same state transitions or transitions that lead to the different
    states. This equivalence oracle should be used on grammars like balanced parentheses. In such grammars,
    all interesting behavior occurs on the transitions between states and potential bugs can be found only by
    focusing on transitions.
    """
    def __init__(self, alphabet, sul: SUL, num_random_walks=1000, walk_len=20, same_state_prob=0.2):
        """
        Args:
            alphabet: input alphabet
            sul: system under learning
            num_random_walks: number of walks
            walk_len: length of each walk
            same_state_prob: probability that the next input will lead to same state transition
        """

        super().__init__(alphabet, sul)
        self.num_walks = num_random_walks
        self.steps_per_walk = walk_len
        self.same_state_prob = same_state_prob

    def find_cex(self, hypothesis):

        for _ in range(self.num_walks):
            self.reset_hyp_and_sul(hypothesis)

            curr_state = hypothesis.initial_state
            inputs = []
            for _ in range(self.steps_per_walk):
                if random.random() <= self.same_state_prob:
                    possible_inputs = curr_state.get_same_state_transitions()
                else:
                    possible_inputs = curr_state.get_diff_state_transitions()

                act = random.choice(possible_inputs) if possible_inputs else random.choice(self.alphabet)
                inputs.append(act)

                out_sul = self.sul.step(inputs[-1])
                out_hyp = hypothesis.step(inputs[-1])
                self.num_steps += 1

                if out_sul != out_hyp:
                    return inputs

        return None
