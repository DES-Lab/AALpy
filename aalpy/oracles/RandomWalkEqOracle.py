# Equivalence oracle that performs random walks with a per-step reset probability.
import random

from aalpy.automata import Onfsm, Mdp, StochasticMealyMachine
from aalpy.base import Oracle, SUL
from aalpy.base.Automaton import Automaton

automaton_dict = {Onfsm: 'onfsm', Mdp: 'mdp', StochasticMealyMachine: 'smm'}


class RandomWalkEqOracle(Oracle):
    """
    Equivalence oracle where queries contain random inputs. After every step, 'reset_prob' determines the probability
    that the system will reset and a new query asked.
    """

    def __init__(self, alphabet: list, sul: SUL, num_steps: int = 5000, reset_after_cex: bool = True,
                 reset_prob: float = 0.09) -> None:
        """
        Constructs the oracle.

        :param list alphabet: Input alphabet.
        :param SUL sul: System under learning.
        :param int num_steps: Number of steps to be performed.
        :param bool reset_after_cex: If true, num_steps will be performed after every counterexample, else the
            total number of steps will equal num_steps.
        :param float reset_prob: Probability that a new query will be asked after each step.
        """

        super().__init__(alphabet, sul)
        self.step_limit = num_steps
        self.reset_after_cex = reset_after_cex
        self.reset_prob = reset_prob
        self.random_steps_done = 0
        self.automata_type = None

    def find_cex(self, hypothesis: Automaton) -> tuple | list | None:
        """
        Performs a random walk, resetting probabilistically, until a counterexample is found or the step limit
        is reached.

        :param Automaton hypothesis: Current hypothesis.
        :return tuple | list | None: Counterexample inputs, None if no counterexample is found.
        """
        if not self.automata_type:
            self.automata_type = automaton_dict.get(type(hypothesis), 'det')

        inputs = []
        outputs = []
        self.reset_hyp_and_sul(hypothesis)

        while self.random_steps_done < self.step_limit:
            self.num_steps += 1
            self.random_steps_done += 1

            if random.random() <= self.reset_prob:
                self.sul.post()
                self.reset_hyp_and_sul(hypothesis)
                inputs.clear()
                outputs.clear()

            inputs.append(random.choice(self.alphabet))

            out_sul = self.sul.step(inputs[-1])
            outputs.append(out_sul)

            if self.automata_type == 'det':
                out_hyp = hypothesis.step(inputs[-1])
            else:
                out_hyp = hypothesis.step_to(inputs[-1], out_sul)

            if self.automata_type == 'det' and out_sul != out_hyp:
                if self.reset_after_cex:
                    self.random_steps_done = 0

                self.sul.post()
                return inputs
            elif out_hyp is None and self.automata_type != 'det':
                if self.reset_after_cex:
                    self.random_steps_done = 0
                self.sul.post()

                if self.automata_type == 'onfsm':
                    return inputs, outputs
                else:
                    # hypothesis is MDP or SMM
                    cex = [hypothesis.initial_state.output] if self.automata_type == 'mdp' else []
                    for i, o in zip(inputs, outputs):
                        cex.extend([i, o])
                    return cex

        return None

    def reset_counter(self) -> None:
        """
        Resets the count of random steps performed since the last reset/counterexample.
        """
        if self.reset_after_cex:
            self.random_steps_done = 0