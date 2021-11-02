import random

from aalpy.automata import Onfsm, Mdp, StochasticMealyMachine
from aalpy.base import Oracle, SUL

automaton_dict = {Onfsm: 'onfsm', Mdp: 'mdp', StochasticMealyMachine: 'smm'}


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
        self.automata_type = None

    def find_cex(self, hypothesis):
        if not self.automata_type:
            self.automata_type = automaton_dict.get(type(hypothesis), 'det')

        inputs = []
        outputs = []
        self.reset_hyp_and_sul(hypothesis)

        while self.random_steps_done < self.step_limit:
            self.num_steps += 1
            self.random_steps_done += 1

            if random.random() <= self.reset_prob:
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
            elif out_hyp is None:
                if self.reset_after_cex:
                    self.random_steps_done = 0

                if self.automata_type == 'onfsm':
                    self.sul.post()
                    return inputs, outputs
                else:
                    # hypothesis is MDP or SMM
                    cex = [hypothesis.initial_state.output] if self.automata_type == 'mdp' else []
                    for i, o in zip(inputs, outputs):
                        cex.extend([i, o])
                    self.sul.post()
                    return cex

        return None

    def reset_counter(self):
        if self.reset_after_cex:
            self.random_steps_done = 0