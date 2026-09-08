import math
from abc import abstractmethod, ABC
from collections import defaultdict
from typing import Generic, TypeVar, Any

T = TypeVar("T")

class DataHandler(Generic[T], ABC):
    # TODO: consider merging `init` with `GsmNode.createPTA`. Could then eliminate PTA post-processing.
    @abstractmethod
    def init(self, data: Any, output_behavior: 'OutputBehavior', data_format: 'DataFormat'):
        """
        Initializes the data handler on the data from which the PTA is constructed.

        :param data: Learning data, in one of the supported data formats (or already a GsmNode tree).
        :param OutputBehavior output_behavior: Either "moore" or "mealy".
        :param DataFormat data_format: Indicates the format of the provided data. Options are:
            - "io_traces": describes prefix-closed data. either
              - Moore traces [[o, (i,o), (i,o), ...], ...]
              - Mealy traces [[(i,o), (i,o), ...], ...]
            - "labeled_sequences": [([i, i, ...], o), ...]
            - "traces": [[o, o, ...], ...]
            - "tree": a tree-shaped automaton provided as a GsmNode
        """
        ...

    @abstractmethod
    def init_merge(self, red: 'GsmNode[T]', blue: 'GsmNode[T]', first_pass: bool):
        """
        Callback triggered just before the partitioning for a merge candidate is constructed.

        :param GsmNode red: The red node of the merge candidate.
        :param GsmNode blue: The blue node of the merge candidate.
        :param bool first_pass: Indicates whether this is the first partitioning pass for score calculation, or the
          second pass for finalizing the partitioning.
        """
        ...

    @abstractmethod
    def abstract(self, in_val: Any, out_val: Any) -> tuple[Any, Any]:
        """
        Method used during PTA construction to abstract from potentially continuous input data.

        :param Any in_val: The input value.
        :param Any out_val: The output value.
        :return tuple[Any, Any]: The abstract output value and the input symbols.
        """
        ...

    @abstractmethod
    def init_data(self) -> T:
        """
        Provides a fresh data object for a new GsmNode instance.

        :return T: The 'empty' data object.
        """
        ...

    @abstractmethod
    def aggregate_data(self, src_node: 'GsmNode[T]', in_value, out_value, dst_node: 'GsmNode[T]'):
        """
        Method for aggregating data in GsmNodes during PTA construction for an individual transition observed in the data.

        :param GsmNode src_node: The source GsmNode instance of the transition.
        :param Any in_value: The (concrete) input value of the transition.
        :param Any out_value: The (concrete) output value of the transition.
        :param GsmNode dst_node: The destination GsmNode instance of the transition.
        """
        ...

    @abstractmethod
    def merge(self, x: T, y: T) -> T:
        """
        Method for merging the data of two GsmNodes during partition construction.

        :param T x: The data of the first GsmNode instance, corresponding to the partition representative.
        :param T y: The data of the second GsmNode instance, corresponding to the GsmNode to be merged into the partition.
        :return T: The updated data for the partition.
        """
        ...

    @abstractmethod
    def copy(self, x: T) -> T:
        """
        Copy operation for the data attached to GsmNodes used when computing a reversible partitioning.

        :param T x: The data to be copied.
        :return T: The copy.
        """
        ...

class NoAbstractionDataHandler(DataHandler[T], ABC):
    """
    DataHandler using input and output symbols "as is". Might still keep track of other information.
    """

    def init(self, data, output_behavior, data_format):
        pass

    def abstract(self, in_val, out_val):
        return in_val, out_val

class NoOpDataHandler(NoAbstractionDataHandler[None]):
    """
    DataHandler that does nothing.
    """

    def init_data(self) -> None:
        return None

    def aggregate_data(self, src_node: 'GsmNode[None]', in_sym, out_value, dst_node: 'GsmNode[None]'):
        pass

    def init_merge(self, red: 'GsmNode[None]', blue: 'GsmNode[None]', first_pass: bool):
        return None

    def merge(self, x: None, y: None) -> None:
        return None

    def copy(self, x: None) -> None:
        return None

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

class CountDataHandler(NoAbstractionDataHandler[CountData]):
    def init_merge(self, red: 'GsmNode[CountData]', blue: 'GsmNode[CountData]', first_pass: bool):
        pass

    def merge(self, x: CountData, y: CountData) -> CountData:
        for in_sym, y_count in y.transition_count.items():
            x_count = x.transition_count[in_sym]
            for out_sym, count in y_count.items():
                int_dict_increment(x_count, out_sym, count)
        return x

    def copy(self, x: CountData) -> CountData:
        ret = CountData()
        ret.transition_count = {k: v.copy() for k, v in x.transition_count.items()}
        return ret

    def init_data(self) -> CountData:
        return CountData()

    def aggregate_data(self, src_node: 'GsmNode[CountData]', in_value, out_value, dst_node: 'GsmNode[CountData]'):
        if src_node is not None:
            int_dict_increment(src_node.data.transition_count[in_value], out_value, 1)


ShadowPTA = dict[Any, dict[Any, 'GsmNode']]
class ShadowPTAData:
    def __init__(self):
        self.shadow_pta: ShadowPTA = defaultdict(dict)

class CountOnPTAData(ShadowPTAData, CountData):
    def __init__(self):
        ShadowPTAData.__init__(self)
        CountData.__init__(self)
        self.pta_count: CountDict = defaultdict(dict)

class CountOnPTADataHandler(CountDataHandler, DataHandler[CountOnPTAData]):
    def init_data(self) -> CountOnPTAData:
        return CountOnPTAData()

    def aggregate_data(self, src_node: 'GsmNode[CountOnPTAData]', in_value, out_value, dst_node: 'GsmNode[CountOnPTAData]'):
        if src_node is None:
            return
        int_dict_increment(src_node.data.transition_count[in_value], out_value, 1)
        int_dict_increment(src_node.data.pta_count[in_value], out_value, 1)
        src_node.data.shadow_pta[in_value][out_value] = dst_node
