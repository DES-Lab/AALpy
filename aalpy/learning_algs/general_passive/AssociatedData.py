import math
from abc import abstractmethod, ABC
from collections import defaultdict
from typing import Any


ProbabilityDict = dict[Any, dict[Any, float]]

class StochasticData(ABC):
    """
    Interface class for data used with `transition_behavior` set to "stochastic".
    """
    @abstractmethod
    def get_probabilities(self) -> ProbabilityDict:
        """
        Method for extracting transition probabilities when converting to automaton models.

        :return ProbabilityDict: Nested dictionary of transition probabilities.
        """
        pass

CountDict = dict[Any, dict[Any, int]]

def int_dict_increment(c_dict, out_sym, cnt):
    c_dict[out_sym] = c_dict.get(out_sym, 0) + cnt

class CountData(StochasticData):
    def __init__(self):
        # TODO get rid of this indirection
        self.transition_count: CountDict = defaultdict(dict)

    def local_log_likelihood_contribution(self):
        llc = 0
        for in_sym, trans in self.transition_count.items():
            total_count = 0
            for out_sym, count in trans.items():
                total_count += count
                llc += count * math.log(count)
            if total_count != 0:
                llc -= total_count * math.log(total_count)
        return llc

    def count(self):
        return sum(sum(trans.values()) for trans in self.transition_count.values())

    @staticmethod
    def merge(x: CountDict, y: CountDict) -> CountDict:
        for in_sym, y_o_dict in y.items():
            x_o_dict = x.get(in_sym, None)
            if x_o_dict is None:
                x[in_sym] = y_o_dict
                continue
            for out_sym, count in y_o_dict.items():
                int_dict_increment(x_o_dict, out_sym, count)
        return x

    def get_probabilities(self) -> ProbabilityDict:
        ret = dict()
        for in_sym, trans in self.transition_count.items():
            total_count = sum(trans.values())
            ret[in_sym] = {out_sym: count / total_count for out_sym, count in trans.items()}
        return ret


ShadowPTA = dict[Any, dict[Any, 'GsmNode']]
class ShadowPTAData:
    def __init__(self):
        self.shadow_pta: ShadowPTA = defaultdict(dict)

class CountOnPTAData(ShadowPTAData, CountData):
    def __init__(self):
        ShadowPTAData.__init__(self)
        CountData.__init__(self)
        self.pta_count: CountDict = defaultdict(dict)