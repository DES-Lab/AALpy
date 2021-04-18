from abc import ABC, abstractmethod
from math import sqrt, log

chi2_table = dict()

chi2_table[0.95] = \
    dict([(1, 3.841458820694124), (2, 5.991464547107979), (3, 7.814727903251179), (4, 9.487729036781154),
          (5, 11.070497693516351), (6, 12.591587243743977), (7, 14.067140449340169), (8, 15.50731305586545),
          (9, 16.918977604620448), (10, 18.307038053275146), (11, 19.67513757268249), (12, 21.02606981748307),
          (13, 22.362032494826934), (14, 23.684791304840576), (15, 24.995790139728616), (16, 26.29622760486423),
          (17, 27.58711163827534), (18, 28.869299430392623), (19, 30.14352720564616), (20, 31.410432844230918)])
chi2_table[0.99] = \
    dict([(1, 6.6348966010212145), (2, 9.21034037197618), (3, 11.344866730144373), (4, 13.276704135987622),
          (5, 15.08627246938899), (6, 16.811893829770927), (7, 18.475306906582357), (8, 20.090235029663233),
          (9, 21.665994333461924), (10, 23.209251158954356), (11, 24.724970311318277), (12, 26.216967305535853),
          (13, 27.68824961045705), (14, 29.141237740672796), (15, 30.57791416689249), (16, 31.999926908815176),
          (17, 33.40866360500461), (18, 34.805305734705065), (19, 36.19086912927004), (20, 37.56623478662507)])

chi2_table[0.999] = \
    dict([(1, 10.827566170662733), (2, 13.815510557964274), (3, 16.26623619623813), (4, 18.46682695290317),
          (5, 20.515005652432873), (6, 22.457744484825323), (7, 24.321886347856854), (8, 26.12448155837614),
          (9, 27.877164871256568), (10, 29.58829844507442), (11, 31.264133620239985), (12, 32.90949040736021),
          (13, 34.52817897487089), (14, 36.12327368039813), (15, 37.69729821835383), (16, 39.252354790768464),
          (17, 40.79021670690253), (18, 42.31239633167996), (19, 43.82019596451753), (20, 45.31474661812586)])


class DifferenceChecker(ABC):

    @abstractmethod
    def check_difference(self, c1: dict, c2: dict) -> bool:
        pass

    def difference_value(self, c1: dict, c2: dict):
        return None

    def use_diff_value(self):
        return False


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
        return False


def compute_epsilon(alpha1, n1):
    epsilon1 = sqrt((1. / (2 * n1)) * log(2. / alpha1))
    return epsilon1


class AdvancedHoeffdingChecker(DifferenceChecker):
    def __init__(self, alpha=0.05, use_diff=False):
        self.alpha = alpha
        self.use_diff = use_diff

    def check_difference(self, c1: dict, c2: dict) -> bool:
        n1 = sum(c1.values())
        n2 = sum(c2.values())

        if n1 > 0 and n2 > 0:
            for o in set(c1.keys()).union(c2.keys()):
                c1o = c1[o] if o in c1.keys() else 0
                c2o = c2[o] if o in c2.keys() else 0
                alpha1 = self.alpha
                alpha2 = self.alpha
                epsilon1 = compute_epsilon(alpha1, n1)
                epsilon2 = compute_epsilon(alpha2, n2)

                if abs(c1o / n1 - c2o / n2) > epsilon1 + epsilon2:
                    return True
        return False

    def use_diff_value(self):
        return self.use_diff

    def difference_value(self, c1_out_freq: dict, c2_out_freq: dict):
        n1 = 0 if not c1_out_freq else sum(c1_out_freq.values())
        n2 = 0 if not c2_out_freq else sum(c2_out_freq.values())

        if n1 > 0 and n2 > 0:
            dist = 0
            for o in set(c1_out_freq.keys()).union(c2_out_freq.keys()):
                c1o = c1_out_freq[o] if o in c1_out_freq.keys() else 0
                c2o = c2_out_freq[o] if o in c2_out_freq.keys() else 0
                dist += abs(c1o / n1 - c2o / n2)
            return dist
        elif n1 > 0 or n2 > 0:
            alpha1 = self.alpha
            alpha2 = self.alpha
            epsilon1 = compute_epsilon(alpha1, max(n1, n2))
            epsilon2 = compute_epsilon(alpha2, max(n1, n2))
            return epsilon1 + epsilon2
        else:
            return 0


class ChiSquareChecker(DifferenceChecker):

    def __init__(self, alpha=0.001, use_diff_value=False):
        self.alpha = alpha
        self.chi2_cache = dict()
        if 1 - self.alpha not in chi2_table.keys():
            raise ValueError("alpha must be in [0.01,0.001,0.05]")
        self.chi2_values = chi2_table[1 - self.alpha]
        self.use_diff = use_diff_value

    def check_difference(self, c1_out_freq: dict, c2_out_freq: dict) -> bool:
        # chi square test for homogeneity (see, for instance: https://online.stat.psu.edu/stat415/lesson/17/17.1)
        if not c1_out_freq or not c2_out_freq:
            return False
        keys = list(set(c1_out_freq.keys()).union(c2_out_freq.keys()))
        dof = len(keys) - 1
        if dof == 0:
            return False
        shared_keys = set(c1_out_freq.keys()).intersection(c2_out_freq.keys())
        if len(shared_keys) == 0:
            # if the supports of the tested frequencies are completely then chi2 makes no sense, use the Hoeffding test
            # to determine if there are enough observations for a difference
            hoeffding_checker = AdvancedHoeffdingChecker()
            return hoeffding_checker.check_difference(c1_out_freq, c2_out_freq)

        Q = self.compute_Q(c1_out_freq, c2_out_freq, keys)
        if dof not in self.chi2_values.keys():
            raise ValueError("Too many possible outputs, chi2 table needs to be extended.")
        else:
            chi2_val = self.chi2_values[dof]

        return Q >= chi2_val

    def use_diff_value(self):
        return self.use_diff

    def difference_value(self, c1_out_freq: dict, c2_out_freq: dict):
        if not c1_out_freq or not c2_out_freq:
            # return a value on the threshold if we don't have information
            c1_outs = set(c1_out_freq.keys()) if c1_out_freq else set()
            c2_outs = set(c2_out_freq.keys()) if c2_out_freq else set()
            nr_outs = len(c1_outs.union(c2_outs))
            return self.chi2_values[max(1, nr_outs)]
        keys = list(set(c1_out_freq.keys()).union(c2_out_freq.keys()))
        shared_keys = set(c1_out_freq.keys()).intersection(c2_out_freq.keys())
        dof = len(keys) - 1
        if dof == 0:
            return 0
        Q = self.compute_Q(c1_out_freq, c2_out_freq, keys)
        return Q

    def compute_Q(self, c1_out_freq, c2_out_freq, keys):
        n_1 = sum(c1_out_freq.values())
        n_2 = sum(c2_out_freq.values())

        Q = 0
        default_val = 0
        yates_correction = -0.5 if len(keys) == 2 and \
                                   any(c1_out_freq.get(k, 0) < 5 or c2_out_freq.get(k, 0) < 5 for k in keys) else 0
        for k in keys:
            p_hat_k = float(c1_out_freq.get(k, default_val) + c2_out_freq.get(k, default_val)) / (n_1 + n_2)
            q_1_k = float(((abs(c1_out_freq.get(k, default_val) - n_1 * p_hat_k)) + yates_correction) ** 2) / (
                    n_1 * p_hat_k)
            q_2_k = float(((abs(c2_out_freq.get(k, default_val) - n_2 * p_hat_k)) + yates_correction) ** 2) / (
                    n_2 * p_hat_k)
            Q = Q + q_1_k + q_2_k
        return Q
