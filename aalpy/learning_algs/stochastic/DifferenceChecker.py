from abc import ABC, abstractmethod
from math import sqrt, log
from scipy.stats import chi2


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


class ChisquareChecker(DifferenceChecker):

    def __init__(self, alpha=0.001):
        self.alpha = alpha
        self.chi2_cache = dict()

    def check_difference(self, c1_out_freq: dict, c2_out_freq: dict) -> bool:
        # chi square test for homogeneity (see, for instance: https://online.stat.psu.edu/stat415/lesson/17/17.1)
        keys = list(set(c1_out_freq.keys()).union(c2_out_freq.keys()))
        n_1 = sum(c1_out_freq.values())
        n_2 = sum(c2_out_freq.values())
        if n_1 == 0 or n_2 == 0:
            return False
        dof = len(keys) - 1
        if dof == 0:
            hoeffdingChecker = HoeffdingChecker()
            return hoeffdingChecker.check_difference(c1_out_freq, c2_out_freq)

        default_val = 0

        Q = 0
        for k in keys:
            p_hat_k = float(c1_out_freq.get(k, default_val) + c2_out_freq.get(k, default_val)) / (n_1 + n_2)
            q_1_k = float((c1_out_freq.get(k, default_val) - n_1 * p_hat_k) ** 2) / (n_1 * p_hat_k)
            q_2_k = float((c2_out_freq.get(k, default_val) - n_2 * p_hat_k) ** 2) / (n_2 * p_hat_k)
            Q = Q + q_1_k + q_2_k
        if dof in self.chi2_cache.keys():
            chi2_val = self.chi2_cache[dof]
        else:
            chi2_val = chi2.ppf(1 - self.alpha, dof)
            self.chi2_cache[dof] = chi2_val

        return Q >= chi2_val
