from abc import abstractmethod, ABC
from typing import Generic, TypeVar, Any

from aalpy.learning_algs.general_passive.AssociatedData import CountData, CountOnPTAData, int_dict_increment
from aalpy.learning_algs.general_passive.GsmNode import GsmNode, IOTrace, IOExample, unknown_output, no_op_input, OutputBehavior


T = TypeVar("T")

DataFormat = str
DataFormatRange = ["io_traces", "labeled_sequences", "traces", "tree"]

# TODO reuse in RPNI
def detect_data_format(data: Any, check_consistency: bool = False, guess: bool = False) -> DataFormat:
    """
    Guess the data format of the provided learning data.

    :param Any data: Input data: a GsmNode (tree), or a sequence of traces/examples.
    :param bool check_consistency: Whether to check all data points instead of returning as soon as a unique format is found.
    :param bool guess: Whether to allow guessing a single format when multiple formats remain ambiguous.
    :return DataFormat: The detected data format string (see DataFormatRange).
    """
    # The different data formats are
    # - "tree": a tree-shaped automaton provided as a GsmNode
    # - "io_traces": either
    #   - Moore traces [[o, (i,o), (i,o), ...], ...]
    #   - Mealy traces [[(i,o), (i,o), ...], ...]
    # - "labeled_sequences": [([i, i, ...], o), ...]
    # - "traces": [[o, o, ...], ...]

    if isinstance(data, GsmNode):
        return "tree"

    accepted_types = (tuple, list)

    # mapping data formats to compatibility criteria
    check_dict = dict(
        io_traces=lambda obj: len(obj) <= 1 or all(isinstance(o, accepted_types) and len(o) == 2 for o in obj[1:]),
        labeled_sequences=lambda obj: len(obj) == 2 and isinstance(obj[0], accepted_types),
    )
    accept_dict = {k: True for k in check_dict}

    if not isinstance(data, accepted_types):
        raise ValueError("wrong input format. expected tuple or list.")
    if len(data) == 0:
        return "io_traces"

    accepted_formats = list(accept_dict.keys())
    for data_point in data:
        if not isinstance(data_point, accepted_types):
            raise ValueError("wrong input format. expected tuple or list.")
        for k, check in check_dict.items():
            accept_dict[k] &= check(data_point)
        accepted_formats = [k for k, v in accept_dict.items() if v]
        if len(accepted_formats) == 1 and not check_consistency:
            return accepted_formats[0]
        if len(accepted_formats) == 0:
            return "traces" # default to traces
            #raise ValueError("invalid or inconsistent data. no options left")
    if len(accepted_formats) != 1 and not guess:
        raise ValueError("ambiguous data format. data format needs to be specified explicitly.")
    return accepted_formats[0]

class DataHandler(Generic[T], ABC):
    def add_trace(self, root_node: GsmNode[T], trace: IOTrace):
        """
        Add an IO trace to a given root node, extending it with new nodes as necessary.

        :param GsmNode root_node: GsmNode to which the trace should be added.
        :param IOTrace trace: Sequence of (input, output) pairs to add.
        """
        curr_node: GsmNode[T] = root_node
        for in_value, out_value in trace:
            prefix_access_pair = self.abstract(in_value, out_value)
            in_sym, out_sym = prefix_access_pair
            transitions = curr_node.transitions[in_sym]
            node = transitions.get(out_sym)
            if node is None:
                node = GsmNode(prefix_access_pair, curr_node, self.init_data())
                transitions[out_sym] = node
            self.aggregate_data(curr_node, in_value, out_value, node)
            curr_node = node

    def add_labeled_sequence(self, root_node: GsmNode[T], example: IOExample):
        """
        Add a labeled input sequence (inputs with a single label attached at the end) to the tree.

        :param IOExample example: (inputs, output) pair, where output labels the state reached by inputs.
        :param DataHandler[T] self: IOHandler used for abstraction and aggregation of trace data
        """
        inputs, output = example
        curr_node: GsmNode = root_node
        in_sym = None

        if len(inputs) == 0:
            self.aggregate_data(None, no_op_input, output, root_node)
            in_sym, out_sym = self.abstract(no_op_input, output)

        # step through inputs and add transitions
        for idx, in_value in enumerate(inputs):
            out_value = output if idx == len(inputs) - 1 else unknown_output
            in_sym, out_sym = self.abstract(in_value, out_value)
            transitions = curr_node.transitions[in_sym]
            if len(transitions) == 0:
                node = GsmNode((in_sym, out_sym), curr_node)
                transitions[out_sym] = node
            elif len(transitions) == 1:
                node = next(iter(transitions.values()))
            else:
                raise ValueError("Nondeterminism encountered for GSM with labeled_sequences. not supported")
            self.aggregate_data(curr_node, in_value, out_value, node)
            curr_node = node

        # fix prefix / predecessor
        curr_node.resolve_unknown_prefix_output(out_sym)
        pred = curr_node.predecessor
        if pred:
            transitions = pred.transitions[in_sym]
            if unknown_output in transitions:
                transitions[out_sym] = transitions.pop(unknown_output)
            if out_sym not in transitions:
                raise ValueError("nondeterminism encountered for GSM with labeled_sequences. not supported")

    def createPTA(self, data: Any, output_behavior: OutputBehavior, data_format: DataFormat = None) -> 'GsmNode[T]':
        """
        Build a prefix tree acceptor (PTA) from the given data.

        :param Any data: Learning data, in one of the supported data formats (or already a GsmNode tree).
        :param OutputBehavior output_behavior: Either "moore" or "mealy".
        :param DataFormat | None data_format: Explicit data format, or None to auto-detect.
        :param DataHandler[T] self: IOHandler used for abstraction and aggregation of trace data
        :return GsmNode: The root node of the constructed (or passed-through) PTA.
        """
        if data_format is None:
            data_format = detect_data_format(data)
        if data_format not in DataFormatRange:
            raise ValueError(f"invalid data format {data_format}. should be in {DataFormatRange}")

        if data_format == "tree":
            if not data.is_tree():
                raise ValueError("provided automaton is not a tree")
            return data
        # TODO extract method for replaying data on dot model
        root_node = GsmNode((no_op_input, unknown_output), None, self.init_data())
        if data_format == "labeled_sequences":
            for example in data:
                self.add_labeled_sequence(root_node, example)
        if data_format == "io_traces" or data_format == "traces":
            if output_behavior == "moore":
                root_node.prefix_access_pair = self.abstract(no_op_input, data[0][0])
                initial_output_symbol = root_node.prefix_access_pair[1]

                for trace in data:
                    initial_output = trace[0]
                    _, ios = self.abstract(no_op_input, initial_output)
                    if ios != initial_output_symbol:
                        raise ValueError("expect unique initial output symbol for Moore behavior")
                    self.aggregate_data(None, no_op_input, initial_output, root_node)

                data = (d[1:] for d in data)
            for trace in data:
                if data_format == "traces":
                    trace = (("step", t) for t in trace)
                self.add_trace(root_node, trace)
        return root_node

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

    def abstract(self, in_val: Any, out_val: Any) -> tuple[Any, Any]:
        """
        Method used during PTA construction to abstract from potentially continuous input data. By default, no abstraction is performed.

        :param Any in_val: The input value.
        :param Any out_val: The output value.
        :return tuple[Any, Any]: The abstract output value and the input symbols.
        """
        return in_val, out_val

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

class NoOpDataHandler(DataHandler[None]):
    """
    DataHandler that neither abstracts the traces, nor tracks any other values during PTA construction.
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


class CountDataHandler(DataHandler[CountData]):
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


class CountOnPTADataHandler(CountDataHandler, DataHandler[CountOnPTAData]):
    def init_data(self) -> CountOnPTAData:
        return CountOnPTAData()

    def aggregate_data(self, src_node: 'GsmNode[CountOnPTAData]', in_value, out_value, dst_node: 'GsmNode[CountOnPTAData]'):
        if src_node is None:
            return
        int_dict_increment(src_node.data.transition_count[in_value], out_value, 1)
        int_dict_increment(src_node.data.pta_count[in_value], out_value, 1)
        src_node.data.shadow_pta[in_value][out_value] = dst_node
