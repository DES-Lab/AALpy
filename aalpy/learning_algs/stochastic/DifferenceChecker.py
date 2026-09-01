# Statistical checkers used to decide whether two output-frequency distributions differ.
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
    """
    Abstract class implemented by all checkers that decide whether two observed output-frequency
    distributions (cells) are statistically different.
    """

    @abstractmethod
    def are_cells_different(self, c1: dict, c2: dict, **kwargs) -> bool:
        """
        Determine whether two cells (output frequency dictionaries) are different.

        :param dict c1: Output frequencies of the first cell.
        :param dict c2: Output frequencies of the second cell.
        :param kwargs: Additional checker-specific arguments.
        :return bool: True if the cells are considered different, False otherwise.
        """
        pass

    def difference_value(self, c1: dict, c2: dict) -> float | None:
        """
        Compute a numeric difference value between two cells, if supported by the checker.

        :param dict c1: Output frequencies of the first cell.
        :param dict c2: Output frequencies of the second cell.
        :return float | None: Difference value, or None if not supported.
        """
        return None

    def use_diff_value(self) -> bool:
        """
        Whether this checker supports computing a numeric difference value.

        :return bool: True if difference_value can be used, False otherwise.
        """
        return False


class HoeffdingChecker(DifferenceChecker):
    """
    Difference checker based on the Hoeffding bound.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        """
        Create a Hoeffding-bound based difference checker.

        :param float alpha: Significance level used in the Hoeffding bound.
        """
        self.alpha = alpha

    def are_cells_different(self, c1: dict, c2: dict, **kwargs) -> bool:
        """
        Determine whether two cells are different using the Hoeffding bound.

        :param dict c1: Output frequencies of the first cell.
        :param dict c2: Output frequencies of the second cell.
        :param kwargs: Unused, present for interface compatibility.
        :return bool: True if the cells are considered different, False otherwise.
        """
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


def compute_epsilon(alpha1: float, n1: int) -> float:
    """
    Compute the Hoeffding-bound epsilon value for a given significance level and sample size.

    :param float alpha1: Significance level.
    :param int n1: Sample size.
    :return float: Computed epsilon value.
    """
    epsilon1 = sqrt((1. / (2 * n1)) * log(2. / alpha1))
    return epsilon1


class AdvancedHoeffdingChecker(DifferenceChecker):
    """
    Difference checker based on per-output Hoeffding bounds, optionally exposing a numeric
    difference value.
    """

    def __init__(self, alpha: float = 0.05, use_diff: bool = False) -> None:
        """
        Create an advanced Hoeffding-bound based difference checker.

        :param float alpha: Significance level used in the Hoeffding bound.
        :param bool use_diff: Whether difference_value should be usable.
        """
        self.alpha = alpha
        self.use_diff = use_diff

    def are_cells_different(self, c1: dict, c2: dict, **kwargs) -> bool:
        """
        Determine whether two cells are different using per-output Hoeffding bounds.

        :param dict c1: Output frequencies of the first cell.
        :param dict c2: Output frequencies of the second cell.
        :param kwargs: Unused, present for interface compatibility.
        :return bool: True if the cells are considered different, False otherwise.
        """
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

    def use_diff_value(self) -> bool:
        """
        Whether this checker supports computing a numeric difference value.

        :return bool: True if use_diff was set on construction, False otherwise.
        """
        return self.use_diff

    def difference_value(self, c1_out_freq: dict, c2_out_freq: dict) -> float:
        """
        Compute a numeric difference value between two cells.

        :param dict c1_out_freq: Output frequencies of the first cell.
        :param dict c2_out_freq: Output frequencies of the second cell.
        :return float: Sum of absolute output frequency differences, a combined epsilon bound if
            only one cell has observations, or 0 if neither has observations.
        """
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
    """
    Difference checker based on the chi-square test for homogeneity.
    """

    def __init__(self, alpha: float = 0.001, use_diff_value: bool = False) -> None:
        """
        Create a chi-square test based difference checker.

        :param float alpha: Significance level, must have a precomputed chi2 table entry.
        :param bool use_diff_value: Whether difference_value should be usable.
        """
        self.alpha = alpha
        self.chi2_cache = dict()
        if 1 - self.alpha not in chi2_table.keys():
            raise ValueError("alpha must be in [0.01,0.001,0.05]")
        self.chi2_values = chi2_table[1 - self.alpha]
        self.use_diff = use_diff_value

    def are_cells_different(self, c1_out_freq: dict, c2_out_freq: dict, **kwargs) -> bool:
        """
        Determine whether two cells are different using a chi-square test for homogeneity
        (see, for instance: https://online.stat.psu.edu/stat415/lesson/17/17.1).

        :param dict c1_out_freq: Output frequencies of the first cell.
        :param dict c2_out_freq: Output frequencies of the second cell.
        :param kwargs: Unused, present for interface compatibility.
        :return bool: True if the cells are considered different, False otherwise.
        """
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
            return hoeffding_checker.are_cells_different(c1_out_freq, c2_out_freq)

        Q = self.compute_Q(c1_out_freq, c2_out_freq, keys)
        if dof not in self.chi2_values.keys():
            raise ValueError("Too many possible outputs, chi2 table needs to be extended.")
        else:
            chi2_val = self.chi2_values[dof]

        return Q >= chi2_val

    def use_diff_value(self) -> bool:
        """
        Whether this checker supports computing a numeric difference value.

        :return bool: True if use_diff_value was set on construction, False otherwise.
        """
        return self.use_diff

    def difference_value(self, c1_out_freq: dict, c2_out_freq: dict) -> float:
        """
        Compute a numeric difference value between two cells based on the chi-square statistic.

        :param dict c1_out_freq: Output frequencies of the first cell.
        :param dict c2_out_freq: Output frequencies of the second cell.
        :return float: Chi-square statistic Q, a threshold value if one cell has no observations,
            or 0 if there is a single degree of freedom.
        """
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

    def compute_Q(self, c1_out_freq: dict, c2_out_freq: dict, keys: list) -> float:
        """
        Compute the chi-square test statistic Q for two output-frequency distributions.

        :param dict c1_out_freq: Output frequencies of the first cell.
        :param dict c2_out_freq: Output frequencies of the second cell.
        :param list keys: Union of the output keys present in both cells.
        :return float: Chi-square test statistic Q.
        """
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
