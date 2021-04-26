import random

from aalpy.automata import Onfsm, Mdp
from aalpy.base import Automaton, Oracle, SUL


class RandomWalkEqOracle(Oracle):
    """
    Equivalence oracle where queries contain random inputs. After every step, 'reset_prob' determines the probability
    that the system will reset and a new query asked.
    """
    def __init__(self, alphabet: list, sul: SUL, num_steps=5000, reset_after_cex=True, reset_prob=0.09):
        """

        Args:
            alphabet: input alphabet

            sul: system under learning

            num_steps: number of steps to be preformed

            reset_after_cex: if true, num_steps will be preformed after every counter example, else the total number
                or steps will equal to num_steps

            reset_prob: probability that the new query will be asked
        """

        super().__init__(alphabet, sul)
        self.step_limit = num_steps
        self.reset_after_cex = reset_after_cex
        self.reset_prob = reset_prob
        self.random_steps_done = 0

    def find_cex(self, hypothesis: Automaton):

        inputs = []
        self.reset_hyp_and_sul(hypothesis)

        while self.random_steps_done < self.step_limit:
            self.num_steps += 1
            self.random_steps_done += 1

            if random.random() <= self.reset_prob:
                self.reset_hyp_and_sul(hypothesis)
                inputs.clear()

            inputs.append(random.choice(self.alphabet))

            out_sul = self.sul.step(inputs[-1])
            out_hyp = hypothesis.step(inputs[-1])

            if out_sul != out_hyp:
                if self.reset_after_cex:
                    self.random_steps_done = 0

                return inputs

        return None


class UnseenOutputRandomWalkEqOracle(Oracle):
    """
    This variation of random walk equivalence oracle can be used when learning stochastic or non-deterministic systems.
    It uses step_to method of the hypothesis. With this method walk trough the hypothesis is performed, based on the
    outputs of the SUL. If our hypothesis cannot step to the new state based on random input and SULs output,
    counterexample is returned.
    """

    def __init__(self, alphabet: list, sul: SUL, num_steps, reset_after_cex=True, reset_prob=0.09):
        """

        Args:
            alphabet: input alphabet

            sul: system under learning

            num_steps: number of steps to be preformed

            reset_after_cex: if true, num_steps will be preformed after every counter example, else the total number
                or steps will equal to num_steps

            reset_prob: probability that the new query will be asked
        """
        super().__init__(alphabet, sul)
        self.step_limit = num_steps
        self.reset_after_cex = reset_after_cex
        self.reset_prob = reset_prob
        self.random_steps_done = 0

    def find_cex(self, hypothesis):

        inputs = []
        outputs = []
        self.reset_hyp_and_sul(hypothesis)

        while self.random_steps_done < self.step_limit:
            self.random_steps_done += 1
            self.num_steps += 1

            if random.random() <= self.reset_prob:
                self.reset_hyp_and_sul(hypothesis)
                inputs.clear()
                outputs = []

            inputs.append(random.choice(self.alphabet))

            out_sul = self.sul.step(inputs[-1])

            outputs.append(out_sul)
            out_hyp = hypothesis.step_to(inputs[-1], out_sul)

            if out_hyp is None:
                if self.reset_after_cex:
                    self.random_steps_done = 0

                if isinstance(hypothesis, Onfsm):
                    return inputs, outputs
                else:
                    # hypothesis is MDP or SMM
                    cex = [hypothesis.initial_state.output] if isinstance(hypothesis, Mdp) else []
                    for i, o in zip(inputs, outputs):
                        cex.extend([i, o])
                    return cex

        return None
