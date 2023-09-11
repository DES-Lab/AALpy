from aalpy.base import Oracle, SUL


class ProvidedSequencesOracleWrapper(Oracle):
    def __init__(self, alphabet: list, sul: SUL, oracle: Oracle, provided_counterexamples: list):
        """
        Oracle wrapper which first executes provided sequences (possible counterexamples) and then switches to another
        oracle instance.

        Args:
            alphabet: input alphabet
            sul: system under learning
            oracle: oracle which will be used once all provided counterexamples are used
            provided_counterexamples: list of input sequance lists. eg [[1,2,3], [2,3,1], ...] where 1,2,3 are elements
            of input alphabet
        """
        super().__init__(alphabet, sul)
        self.provided_counterexamples = provided_counterexamples
        self.oracle = oracle

    def find_cex(self, hypothesis):
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

            self.provided_counterexamples.remove(provided_cex)

        cex = self.oracle.find_cex(hypothesis)

        # to account for steps statistics from actual oracle
        if cex is None:
            self.num_queries += self.oracle.num_queries
            self.num_steps += self.oracle.num_steps

        return cex

