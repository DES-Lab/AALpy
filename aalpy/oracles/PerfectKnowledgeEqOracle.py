# Equivalence oracle that computes exact counterexamples via bisimilarity checking against the true model.
from aalpy.base import Oracle, SUL, DeterministicAutomaton
from aalpy.utils import bisimilar


class PerfectKnowledgeEqOracle(Oracle):
    """
    Oracle that can be used when developing and testing deterministic learning algorithms,
    so that the focus is put off equivalence query.
    """
    def __init__(self, alphabet: list, sul: SUL, model_under_learning: DeterministicAutomaton) -> None:
        """
        Constructs the oracle.

        :param list alphabet: Input alphabet.
        :param SUL sul: System under learning.
        :param DeterministicAutomaton model_under_learning: The ground-truth model to compare hypotheses against.
        """
        super().__init__(alphabet, sul, )
        self.model_under_learning = model_under_learning

    def find_cex(self, hypothesis: DeterministicAutomaton) -> tuple | None:
        """
        Checks bisimilarity between the hypothesis and the ground-truth model.

        :param DeterministicAutomaton hypothesis: Current hypothesis.
        :return tuple | None: Counterexample inputs, None if no counterexample is found.
        """
        return bisimilar(hypothesis, self.model_under_learning, return_cex=True)
