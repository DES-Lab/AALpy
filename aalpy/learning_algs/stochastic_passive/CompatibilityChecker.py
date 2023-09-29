from abc import ABC, abstractmethod
from math import sqrt, log

from aalpy.learning_algs.stochastic_passive.FPTA import AlergiaPtaNode


class CompatibilityChecker(ABC):

    @abstractmethod
    def are_states_different(self, a: AlergiaPtaNode, b: AlergiaPtaNode, **kwargs) -> bool:
        pass


class HoeffdingCompatibility(CompatibilityChecker):
    def __init__(self, eps):
        self.eps = eps
        self.log_term = sqrt(0.5 * log(2 / self.eps))

    def hoeffding_bound(self, a: dict, b: dict):
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

    def are_states_different(self, a: AlergiaPtaNode, b: AlergiaPtaNode, **kwargs):

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
