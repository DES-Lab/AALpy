from abc import ABC, abstractmethod
from math import sqrt, log


class DifferenceChecker(ABC):

    @abstractmethod
    def check_difference(self, c1: dict, c2: dict) -> bool:
        pass


class HoeffdingChecker(DifferenceChecker):

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def check_difference(self, c1: dict, c2: dict) -> bool:
        if c1.keys() != c2.keys():
            return True

        n1 = sum(c1.values())
        n2 = sum(c2.values())

        if n1 > 0 and n2 > 0:
            for o in c1.keys():
                if abs(c1[o] / n1 - c2[o] / n2) > \
                        ((sqrt(1 / n1) + sqrt(1 / n2)) * sqrt(0.5 * log(2 / self.alpha))):
                    return True


class AdvancedHoeffdingChecker(DifferenceChecker):
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def check_difference(self, c1: dict, c2: dict) -> bool:
        n1 = sum(c1.values())
        n2 = sum(c2.values())

        if n1 > 0 and n2 > 0:
            for o in set(c1.keys()).union(c2.keys()):
                c1o = c1[o] if o in c1.keys() else 0
                c2o = c2[o] if o in c2.keys() else 0
                alpha1 = 0.05
                alpha2 = 0.05
                epsilon1 = sqrt((1. / (2 * n1)) * log(2. / alpha1))
                epsilon2 = sqrt((1. / (2 * n2)) * log(2. / alpha2))

                if abs(c1o / n1 - c2o / n2) > epsilon1 + epsilon2:
                    return True
            return False
