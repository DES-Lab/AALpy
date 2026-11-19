# Equivalence oracle that performs full-reset random walks of random length in a predefined range.
from statistics import mean

from aalpy.automata import Onfsm, Mdp, StochasticMealyMachine
from aalpy.base import Oracle, SUL
from aalpy.base.Automaton import Automaton
from random import randint, choice

automaton_dict = {Onfsm: 'onfsm', Mdp: 'mdp', StochasticMealyMachine: 'smm'}


class RandomWordEqOracle(Oracle):
    """
    Equivalence oracle where queries are of random length in a predefined range.
    """

    def __init__(self, alphabet: list, sul: SUL, num_walks: int = 500, min_walk_len: int = 10,
                 max_walk_len: int = 30, reset_after_cex: bool = True) -> None:
        """
        Constructs the oracle.

        :param list alphabet: Input alphabet.
        :param SUL sul: System under learning.
        :param int num_walks: Number of walks to perform during search for a counterexample.
        :param int min_walk_len: Minimum length of each walk.
        :param int max_walk_len: Maximum length of each walk.
        :param bool reset_after_cex: If True, num_walks will be performed after every counterexample, else the
            total number of walks will equal num_walks.
        """

        super().__init__(alphabet, sul)
        self.num_walks = num_walks
        self.min_walk_len = min_walk_len
        self.max_walk_len = max_walk_len
        self.reset_after_cex = reset_after_cex
        self.num_walks_done = 0
        self.automata_type = None

        self.walk_lengths = [randint(min_walk_len, max_walk_len) for _ in range(num_walks)]

    def find_cex(self, hypothesis: Automaton) -> tuple | list | None:
        """
        Performs random-length walks from the initial state until a counterexample is found or num_walks is
        reached.

        :param Automaton hypothesis: Current hypothesis.
        :return tuple | list | None: Counterexample inputs, None if no counterexample is found.
        """
        if not self.automata_type:
            self.automata_type = automaton_dict.get(type(hypothesis), 'det')

        while self.num_walks_done < self.num_walks:
            inputs = []
            outputs = []
            self.reset_hyp_and_sul(hypothesis)
            self.num_walks_done += 1

            num_steps = self.walk_lengths.pop(0)

            for _ in range(num_steps):
                inputs.append(choice(self.alphabet))

                out_sul = self.sul.step(inputs[-1])
                if self.automata_type == 'det':
                    out_hyp = hypothesis.step(inputs[-1])
                else:
                    out_hyp = hypothesis.step_to(inputs[-1], out_sul)
                    outputs.append(out_sul)

                self.num_steps += 1

                if self.automata_type == 'det' and out_sul != out_hyp:
                    self.sul.post()

                    if self.reset_after_cex:
                        self.walk_lengths = [randint(self.min_walk_len, self.max_walk_len) for _ in range(self.num_walks)]
                        self.num_walks_done = 0

                    return inputs

                elif out_hyp is None and self.automata_type != 'det':
                    self.sul.post()

                    if self.reset_after_cex:
                        self.walk_lengths = [randint(self.min_walk_len, self.max_walk_len) for _ in range(self.num_walks)]
                        self.num_walks_done = 0

                    if self.automata_type == 'onfsm':
                        return inputs, outputs
                    else:
                        # hypothesis is MDP or SMM
                        cex = [hypothesis.initial_state.output] if self.automata_type == 'mdp' else []
                        for i, o in zip(inputs, outputs):
                            cex.extend([i, o])
                        return cex

            # cleanup after the test case
            self.sul.post()

        return None

    def reset_counter(self) -> None:
        """
        Resets the count of walks performed since the last reset/counterexample.
        """
        if self.reset_after_cex:
            self.num_walks_done = 0
