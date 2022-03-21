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

        if n1 > 0 and n2 > 0:
            a_children = a.children.keys()
            b_children = b.children.keys()
            outs = set(a_children).union(b_children)

            for o in outs:
                # for non existing keys set freq to 0
                a_freq = a.input_frequency[o] if o in a_children else 0
                b_freq = b.input_frequency[o] if o in b_children else 0

                if abs(a_freq / n1 - b_freq / n2) > ((sqrt(1 / n1) + sqrt(1 / n2)) * sqrt(0.5 * log(2 / self.eps))):
                    return False
        return True
