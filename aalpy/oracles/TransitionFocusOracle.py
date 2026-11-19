# Equivalence oracle that biases random walks towards same-state or different-state transitions.
import random

from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL
from aalpy.base.Automaton import Automaton


class TransitionFocusOracle(Oracle):
    """
    This equivalence oracle focuses either on the same state transitions or transitions that lead to the different
    states. This equivalence oracle should be used on grammars like balanced parentheses. In such grammars,
    all interesting behavior occurs on the transitions between states and potential bugs can be found only by
    focusing on transitions.
    """
    def __init__(self, alphabet: list, sul: SUL, num_random_walks: int = 500, walk_len: int = 20,
                 same_state_prob: float = 0.2) -> None:
        """
        Constructs the oracle.

        :param list alphabet: Input alphabet.
        :param SUL sul: System under learning.
        :param int num_random_walks: Number of walks.
        :param int walk_len: Length of each walk.
        :param float same_state_prob: Probability that the next input will lead to a same-state transition.
        """

        super().__init__(alphabet, sul)
        self.num_walks = num_random_walks
        self.steps_per_walk = walk_len
        self.same_state_prob = same_state_prob

    def find_cex(self, hypothesis: Automaton) -> list | None:
        """
        Performs random walks biased towards same-state or different-state transitions until a counterexample is
        found.

        :param Automaton hypothesis: Current hypothesis.
        :return list | None: Counterexample inputs, None if no counterexample is found.
        """
        for _ in range(self.num_walks):
            self.reset_hyp_and_sul(hypothesis)

            curr_state = hypothesis.current_state
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

                curr_state = hypothesis.current_state

                if out_sul != out_hyp:
                    self.sul.post()
                    return inputs

            # cleanup after the test case
            self.sul.post()

        return None
