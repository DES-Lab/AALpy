import math
from abc import abstractmethod, ABC
from collections import defaultdict
from copy import copy
from typing import Generic, TypeVar, Any

T = TypeVar("T")

class IOHandler(Generic[T]):
    @abstractmethod
    def init(self, data, output_format, data_format):
        ...

    def init_merge(self, red: 'GsmNode[T]', blue: 'GsmNode[T]', first_pass: bool):
        pass

    @abstractmethod
    def abstract(self, in_val, out_val):
        ...

    @abstractmethod
    def init_data(self) -> T:
        ...

    @abstractmethod
    def aggregate_data(self, src_node: 'GsmNode[T]', in_value, out_value, dst_node: 'GsmNode[T]'):
        ...

    @abstractmethod
    def merge(self, x: T, y: T) -> T:
        ...

    @abstractmethod
    def copy(self, x: T) -> T:
        ...

class NoAbstractionIOHandler(IOHandler[T], ABC):
    def init(self, data, output_format, data_format):
        pass

    def abstract(self, in_val, out_val):
        return in_val, out_val

class NoIOHandler(NoAbstractionIOHandler[T]):
    def init_data(self) -> T:
        return None

    def aggregate_data(self, src_node: 'GsmNode[T]', in_sym, out_value, dst_node: 'GsmNode[T]'):
        pass

    def merge(self, x: T, y: T) -> T:
        return None

    def copy(self, x: T) -> T:
        return None

class CopyOnWriteIOHandler(IOHandler[T], ABC):
    def __init__(self):
        self.copied_on_write = set()

    def init_merge(self, red: 'GsmNode[T]', blue: 'GsmNode[T]', first_pass: bool):
        self.first_pass = first_pass
        if first_pass:
            self.copied_on_write.clear()

    def merge(self, x: T, y: T) -> T:
        if self.first_pass and id(x) not in self.copied_on_write:
            x = self.copy_on_write(x)
            self.copied_on_write.add(id(x))
        self.merge_into_x(x, y)
        return x

    @abstractmethod
    def merge_into_x(self, x: T, y: T):
        pass

    @abstractmethod
    def copy_on_write(self, x: T) -> T:
        pass

    def copy(self, x: T) -> T:
        return x

ProbabilityDict = dict[Any, dict[Any, float]]

class StochasticData(ABC):
    @abstractmethod
    def get_probabilities(self) -> ProbabilityDict: pass

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

class CountHandler(NoAbstractionIOHandler[CountData], CopyOnWriteIOHandler):
    def init_data(self) -> CountData:
        return CountData()

    def aggregate_data(self, src_node: 'GsmNode[CountData]', in_value, out_value, dst_node: 'GsmNode[CountData]'):
        if src_node is not None:
            int_dict_increment(src_node.data.transition_count[in_value], out_value, 1)

    def merge_into_x(self, x: CountData, y: CountData):
        CountData.merge(x.transition_count, y.transition_count)

    def copy_on_write(self, x: CountData) -> CountData:
        new_x = copy(x)
        new_x.transition_count = defaultdict(dict)
        for in_sym, trans in x.transition_count.items():
            new_x.transition_count[in_sym] = trans.copy()
        return x

class ShadowPTAData:
    def __init__(self):
        self.shadow_pta: ShadowPTA = defaultdict(dict)

ShadowPTA = dict[Any, dict[Any, 'GsmNode']]
class CountOnPTAData(ShadowPTAData, CountData):
    def __init__(self):
        ShadowPTAData.__init__(self)
        CountData.__init__(self)
        self.pta_count: CountDict = defaultdict(dict)

class CountOnPTAHandler(CountHandler):
    def init_data(self) -> CountOnPTAData:
        return CountOnPTAData()

    def aggregate_data(self, src_node: 'GsmNode[CountOnPTAData]', in_value, out_value, dst_node: 'GsmNode[CountOnPTAData]'):
        if src_node is None:
            return
        int_dict_increment(src_node.data.transition_count[in_value], out_value, 1)
        int_dict_increment(src_node.data.pta_count[in_value], out_value, 1)
        src_node.data.shadow_pta[in_value][out_value] = dst_node
