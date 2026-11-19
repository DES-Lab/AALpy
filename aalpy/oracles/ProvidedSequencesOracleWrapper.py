# Equivalence oracle wrapper that first replays user-provided candidate counterexamples before delegating.
from aalpy.base import Oracle, SUL
from aalpy.base.Automaton import Automaton


class ProvidedSequencesOracleWrapper(Oracle):
    """
    Oracle wrapper which first executes provided sequences (possible counterexamples) and then switches to another
    oracle instance.
    """

    def __init__(self, alphabet: list, sul: SUL, oracle: Oracle, provided_counterexamples: list) -> None:
        """
        Constructs the oracle wrapper.

        :param list alphabet: Input alphabet.
        :param SUL sul: System under learning.
        :param Oracle oracle: Oracle which will be used once all provided counterexamples are used.
        :param list provided_counterexamples: List of input sequence lists, e.g. [[1,2,3], [2,3,1], ...] where
            1,2,3 are elements of the input alphabet.
        """
        super().__init__(alphabet, sul)
        self.provided_counterexamples = provided_counterexamples
        self.oracle = oracle

    def find_cex(self, hypothesis: Automaton) -> tuple | list | None:
        """
        Replays the remaining provided counterexamples, then delegates to the wrapped oracle.

        :param Automaton hypothesis: Current hypothesis.
        :return tuple | list | None: Counterexample inputs, None if no counterexample is found.
        """
        for provided_cex in self.provided_counterexamples.copy():
            inputs = []
            self.reset_hyp_and_sul(hypothesis)

            for i in provided_cex:
                inputs.append(i)
                out_sul = self.sul.step(i)
                out_hyp = hypothesis.step(i)
                self.num_steps += 1

                if out_sul != out_hyp:
                    self.sul.post()
                    return tuple(inputs)

            # cleanup after the test case
            self.sul.post()

            self.provided_counterexamples.remove(provided_cex)

        cex = self.oracle.find_cex(hypothesis)

        # to account for steps statistics from actual oracle
        if cex is None:
            self.num_queries += self.oracle.num_queries
            self.num_steps += self.oracle.num_steps

        return cex

