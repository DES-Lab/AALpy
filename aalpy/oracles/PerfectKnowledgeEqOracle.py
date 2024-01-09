from aalpy.base import Oracle, SUL, DeterministicAutomaton
from aalpy.utils import bisimilar


class PerfectKnowledgeEqOracle(Oracle):
    """
    Oracle that can be used when developing and testing deterministic learning algorithms,
    so that the focus is put off equivalence query.
    """
    def __init__(self, alphabet: list, sul: SUL, model_under_learning: DeterministicAutomaton):
        super().__init__(alphabet, sul, )
        self.model_under_learning = model_under_learning

    def find_cex(self, hypothesis):
        return bisimilar(hypothesis, self.model_under_learning, return_cex=True)
