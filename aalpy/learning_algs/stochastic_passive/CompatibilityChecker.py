from abc import ABC, abstractmethod
from math import sqrt, log

from aalpy.learning_algs.stochastic_passive.FPTA import AlergiaPtaNode


class CompatibilityChecker(ABC):

    @abstractmethod
    def check_difference(self, a: AlergiaPtaNode, b: AlergiaPtaNode, **kwargs) -> bool:
        pass


class HoeffdingCompatibility(CompatibilityChecker):
    def __init__(self, eps):
        self.eps = eps

    def check_difference(self, a: AlergiaPtaNode, b: AlergiaPtaNode, **kwargs):
        n1 = sum(a.input_frequency.values())
        n2 = sum(b.input_frequency.values())

        # for non existing keys set freq to 0
        outs = set(a.children.keys()).union(b.children.keys())

        if n1 > 0 and n2 > 0:
            for o in outs:
                a_freq = a.input_frequency[o] if o in a.children.keys() else 0
                b_freq = b.input_frequency[o] if o in b.children.keys() else 0

                if abs(a_freq / n1 - b_freq / n2) > ((sqrt(1 / n1) + sqrt(1 / n2)) * sqrt(0.5 * log(2 / self.eps))):
                    return False
        return True
