# Compatibility checkers used by Alergia/IOAlergia to decide whether two FPTA states may be merged.
from abc import ABC, abstractmethod
from math import sqrt, log

from aalpy.learning_algs.stochastic_passive.FPTA import AlergiaPtaNode


class CompatibilityChecker(ABC):
    """
    Abstract class implemented by all compatibility checkers used to decide whether two states of the FPTA are
    statistically different (and thus should not be merged).
    """

    @abstractmethod
    def are_states_different(self, a: AlergiaPtaNode, b: AlergiaPtaNode, **kwargs) -> bool:
        """
        Checks whether two FPTA nodes are statistically different.

        :param AlergiaPtaNode a: First node.
        :param AlergiaPtaNode b: Second node.
        :param kwargs: Additional implementation-specific arguments.
        :return bool: True if the nodes are statistically different, False otherwise.
        """
        pass


class HoeffdingCompatibility(CompatibilityChecker):
    """
    Compatibility checker based on the Hoeffding bound, comparing observed output frequency distributions of two
    FPTA nodes.
    """

    def __init__(self, eps: float) -> None:
        """
        Creates a Hoeffding-bound-based compatibility checker.

        :param float eps: Epsilon value controlling the strictness of the Hoeffding bound.
        """
        self.eps = eps
        self.log_term = sqrt(0.5 * log(2 / self.eps))

    def hoeffding_bound(self, a: dict, b: dict) -> bool:
        """
        Checks whether two output frequency distributions differ by more than the Hoeffding bound.

        :param dict a: Frequency distribution of the first node.
        :param dict b: Frequency distribution of the second node.
        :return bool: True if the distributions differ by more than the Hoeffding bound, False otherwise.
        """
        n1 = sum(a.values())
        n2 = sum(b.values())

        if n1 * n2 == 0:
            return False

        bound = (sqrt(1 / n1) + sqrt(1 / n2)) * self.log_term

        for o in set(a.keys()).union(b.keys()):
            a_freq = a[o] if o in a else 0
            b_freq = b[o] if o in b else 0

            if abs(a_freq / n1 - b_freq / n2) > bound:
                return True
        return False

    def are_states_different(self, a: AlergiaPtaNode, b: AlergiaPtaNode, **kwargs) -> bool:
        """
        Checks whether two FPTA nodes are statistically different based on the Hoeffding bound, conditioned on
        inputs in the IOAlergia case.

        :param AlergiaPtaNode a: First node.
        :param AlergiaPtaNode b: Second node.
        :param kwargs: Unused, present for interface compatibility.
        :return bool: True if the nodes are statistically different, False otherwise.
        """

        # no data available for any node
        if len(a.original_input_frequency) * len(b.original_children) == 0:
            return False

        # assuming tuples are used for IOAlergia and not as Alergia outputs
        if not isinstance(list(a.original_input_frequency.keys())[0], tuple):
            return self.hoeffding_bound(a.original_input_frequency, b.original_input_frequency)

        # IOAlergia: check hoeffding bound conditioned on inputs
        for i in a.get_immutable_inputs().intersection(b.get_immutable_inputs()):
            if self.hoeffding_bound(a.get_original_output_frequencies(i), b.get_original_output_frequencies(i)):
                return True
        return False
