from aalpy.automata import Onfsm, Mdp
from aalpy.base import Oracle, SUL
from random import randint, choice


class RandomWordEqOracle(Oracle):
    """
    Equivalence oracle where queries are of random length in a predefined range.
    """

    def __init__(self, alphabet: list, sul: SUL, num_walks=100, min_walk_len=10, max_walk_len=100,
                 reset_after_cex=True):
        """
        Args:
            alphabet: input alphabet

            sul: system under learning

            num_walks: number of walks to perform during search for cex

            min_walk_len: minimum length of each walk

            max_walk_len: maximum length of each walk

            reset_after_cex: if True, num_walks will be preformed after every counter example, else the total number
                or walks will equal to num_walks
        """

        super().__init__(alphabet, sul)
        self.num_walks = num_walks
        self.min_walk_len = min_walk_len
        self.max_walk_len = max_walk_len
        self.reset_after_cex = reset_after_cex
        self.num_walks_done = 0

    def find_cex(self, hypothesis):

        while self.num_walks_done < self.num_walks:
            inputs = []
            self.reset_hyp_and_sul(hypothesis)
            self.num_walks_done += 1

            num_steps = randint(self.min_walk_len, self.max_walk_len)

            for _ in range(num_steps):
                inputs.append(choice(self.alphabet))

                out_sul = self.sul.step(inputs[-1])
                out_hyp = hypothesis.step(inputs[-1])
                self.num_steps += 1

                if out_sul != out_hyp:
                    if self.reset_after_cex:
                        self.num_walks_done = 0

                    return inputs

        return None


class UnseenOutputRandomWordEqOracle(Oracle):
    """
    This variation of random word equivalence oracle can be used when learning stochastic or non-deterministic systems.
    It uses step_to method of the hypothesis. With this method walk trough, the hypothesis is performed, based on the
    outputs of the SUL. If our hypothesis cannot step to the new state-based on random input and SULs output,
    a counterexample is returned.
    """

    def __init__(self, alphabet: list, sul: SUL, num_walks=100, min_walk_len=10, max_walk_len=100,
                 reset_after_cex=True):
        """
        Args:
            alphabet: input alphabet

            sul: system under learning

            num_walks: number of walks to perform during search for cex

            min_walk_len: minimum length of each walk

            max_walk_len: maximum length of each walk

            reset_after_cex: if True, num_walks will be preformed after every counter example, else the total number
                or walks will equal to num_walks
        """
        super().__init__(alphabet, sul)
        self.num_walks = num_walks
        self.min_walk_len = min_walk_len
        self.max_walk_len = max_walk_len
        self.reset_after_cex = reset_after_cex
        self.num_walks_done = 0

    def find_cex(self, hypothesis):

        while self.num_walks_done < self.num_walks:
            inputs = []
            outputs = []
            self.reset_hyp_and_sul(hypothesis)
            self.num_walks_done += 1

            num_steps = randint(self.min_walk_len, self.max_walk_len)

            for _ in range(num_steps):
                inputs.append(choice(self.alphabet))

                out_sul = self.sul.step(inputs[-1])
                outputs.append(out_sul)
                self.num_steps += 1

                out_hyp = hypothesis.step_to(inputs[-1], out_sul)

                if out_hyp is None:
                    if self.reset_after_cex:
                        self.num_walks_done = 0

                    if isinstance(hypothesis, Onfsm):
                        return inputs, outputs
                    else:
                        # hypothesis is MDP or SMM
                        cex = [hypothesis.initial_state.output] if isinstance(hypothesis, Mdp) else []
                        for i, o in zip(inputs, outputs):
                            cex.extend([i, o])
                        return cex

        return None
