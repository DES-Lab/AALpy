# Core implementation of the Generalized State Merging (GSM) algorithm: a red-blue
# state-merging framework used to passively learn deterministic, nondeterministic and
# stochastic automata from data.
import functools
import warnings
from collections import deque
from copy import copy
from typing import Callable, Any

from aalpy import Automaton
from aalpy.learning_algs.general_passive.GsmNode import GsmNode, OutputBehavior, TransitionBehavior, OutputBehaviorRange, \
    TransitionBehaviorRange, unknown_output, detect_data_format, IOHandler, NoIOHandler, DataFormat
from aalpy.learning_algs.general_passive.IOHandler import CountOnPTAHandler
from aalpy.learning_algs.general_passive.ScoreFunctionsGSM import ScoreCalculation, hoeffding_compatibility, \
    CheckFutureScore


# TODO add option for making checking of futures and partition non mutual exclusive?
#  Easiest done by adding a new method / field to ScoreCalculation

class Partitioning:
    """Represents the tentative result of merging a blue node into a red node, plus the resulting node mapping."""

    def __init__(self, red: GsmNode, blue: GsmNode) -> None:
        """
        Create a (not yet scored) partitioning for the merge of blue into red.

        :param GsmNode red: Red (already accepted) node the merge targets.
        :param GsmNode blue: Blue (candidate) node being merged.
        """
        self.red: GsmNode = red
        self.blue: GsmNode = blue
        self.score = None
        self.red_mapping: dict[GsmNode, GsmNode] = dict()
        self.full_mapping: dict[GsmNode, GsmNode] = dict()
        self.new_blue = []
        self.remaining_merges = None
        self.nr_merged_states = 0

class Instrumentation:
    """Base class for hooks that observe/report on the progress of GeneralizedStateMerging.run."""

    def __init__(self) -> None:
        """
        Create an instrumentation instance. No state by default.
        """
        pass

    def reset(self, gsm: 'GeneralizedStateMerging') -> None:
        """
        Called once at the start of a learning run.

        :param GeneralizedStateMerging gsm: The GSM instance being run.
        """
        pass

    def pta_construction_done(self, root: GsmNode) -> None:
        """
        Called after the initial PTA has been constructed.

        :param GsmNode root: Root node of the constructed PTA.
        """
        pass

    def log_promote(self, node: GsmNode) -> None:
        """
        Called whenever a blue node is promoted to red.

        :param GsmNode node: The promoted node.
        """
        pass

    def log_merge(self, part: Partitioning) -> None:
        """
        Called whenever a merge is performed.

        :param Partitioning part: The partitioning describing the performed merge.
        """
        pass

    def learning_done(self, root: GsmNode) -> None:
        """
        Called once learning has finished.

        :param GsmNode root: Root node of the learned model.
        """
        pass

class GeneralizedStateMerging:
    """Implements the red-blue state-merging framework used to passively learn automata from data."""

    def __init__(self, *,
                 output_behavior: OutputBehavior = "moore",
                 transition_behavior: TransitionBehavior = "deterministic",
                 score_calc: ScoreCalculation = None,
                 pta_preprocessing: Callable[[GsmNode], GsmNode] = None,
                 postprocessing: Callable[[GsmNode], GsmNode] = None,
                 data_handler: IOHandler = None,
                 node_order: Callable[[GsmNode], Any] = None,
                 consider_only_min_blue = False,
                 depth_first = False,
                 ):
        """
        Configure a GeneralizedStateMerging instance.

        :param OutputBehavior output_behavior: Either "moore" or "mealy".
        :param TransitionBehavior transition_behavior: Either "deterministic", "nondeterministic" or "stochastic".
        :param ScoreCalculation score_calc: Local compatibility / global score calculation to use.
        :param Callable[[GsmNode], GsmNode] pta_preprocessing: Pre-processing function applied to the constructed PTA.
        :param Callable[[GsmNode], GsmNode] postprocessing: Post-processing function applied to the learned model.
        :param Callable[[GsmNode], Any] node_order: Comparison key to determine the order in which merge candidates are considered.
        :param bool consider_only_min_blue: Whether to only consider the minimal blue node in each round.
        :param bool depth_first: Whether compatibility is checked depth-first instead of breadth-first.
        """

        if output_behavior not in OutputBehaviorRange:
            raise ValueError(f"invalid output behavior {output_behavior}. should be in {OutputBehaviorRange}")
        self.output_behavior: OutputBehavior = output_behavior
        if transition_behavior not in TransitionBehaviorRange:
            raise ValueError(f"invalid transition behavior {transition_behavior}. should be in {TransitionBehaviorRange}")
        self.transition_behavior: TransitionBehavior = transition_behavior

        if score_calc is None:
            if transition_behavior == "deterministic":
                score_calc = ScoreCalculation(GsmNode.deterministic_compatible)
            elif transition_behavior == "nondeterministic" :
                raise ValueError("Missing score_calc for nondeterministic transition behavior. No default available.")
            elif transition_behavior == "stochastic" :
                score_calc = CheckFutureScore(hoeffding_compatibility(0.005, True), compatibility_on_pta=True)
                if data_handler is not None:
                    raise ValueError("Using default algorithm for stochastic systems but a data_handler was provided.")
                data_handler = CountOnPTAHandler()
        self.score_calc: ScoreCalculation = score_calc

        if isinstance(node_order, str) and node_order == "short-lex":
            node_order = functools.cmp_to_key(lambda a, b: -1 if GsmNode.short_lex_order(a, b) else 1)
        self.node_order = node_order

        self.pta_preprocessing = pta_preprocessing or (lambda x: x)
        self.postprocessing = postprocessing or (lambda x: x)

        self.data_handler = data_handler or NoIOHandler()

        self.consider_only_min_blue = consider_only_min_blue
        self.depth_first = depth_first

    # TODO: make more generic by adding the option to use a different algorithm than red blue
    #  for selecting potential merge candidates. Maybe using inheritance with abstract `run`.
    def run(self, data: Any, convert: bool = True, instrumentation: Instrumentation | None = None,
            data_format: DataFormat | None = None) -> Automaton | GsmNode:
        """
        Run the state-merging algorithm on the provided data.

        :param Any data: Learning data, in one of the supported data formats (or already a GsmNode tree).
        :param bool convert: Whether to convert the resulting GsmNode tree into a concrete AALpy automaton.
        :param Instrumentation | None instrumentation: Instrumentation object used to report progress, defaults to a no-op instance.
        :param DataFormat | None data_format: Explicit data format of `data`, or None to auto-detect.
        :return Automaton | GsmNode: The learned automaton (if convert is True) or the raw GsmNode tree.
        """
        if instrumentation is None:
            instrumentation = Instrumentation()
        instrumentation.reset(self)

        if data_format is None:
            data_format = detect_data_format(data)
        if data_format == "labeled_sequences" and self.transition_behavior != "deterministic":
            raise ValueError("learning from labeled_sequences is not possible for nondeterministic systems")
        if data_format == "traces" and self.transition_behavior == "deterministic":
            print("learning deterministic systems from (output) traces only. this rarely makes sense. is `data_format` set correctly?")
        root = GsmNode.createPTA(data, self.output_behavior, data_format, self.data_handler)

        root = self.pta_preprocessing(root)
        instrumentation.pta_construction_done(root)
        instrumentation.log_promote(root)

        if self.transition_behavior == "deterministic":
            if not root.is_deterministic():
                warnings.warn("required deterministic automaton but input data is nondeterministic")

        # sorted list of states already considered as distinct
        red_states = [root]
        red_states_backing_set = {root}
        blue_states = list(root.child_iterator())

        partition_candidates: dict[tuple[GsmNode, GsmNode], Partitioning] = dict()
        while len(blue_states) != 0:
            blue_states_to_consider = blue_states
            if self.consider_only_min_blue: # does it make sense to check the score function here?
                if self.node_order is None:
                    blue_states_to_consider = [blue_states[0]]
                else:
                    blue_states_to_consider = [min(blue_states, key=self.node_order)]

            # could make this sort unconditional, but i think this is closer to the original in any case?
            if self.node_order is not None:
                # TODO: this could be done using insort as long as the order is static?
                blue_states_to_consider.sort(key=self.node_order)
                red_states.sort(key=self.node_order)

            # loop over blue states
            best_promotion_candidate = None
            best_promotion_score = None
            for blue_state in blue_states_to_consider:
                # FUTURE: Parallelize
                # FUTURE: Save partitions?

                # calculate partitions resulting from merges with red states if necessary
                current_candidates: dict[GsmNode, Partitioning] = dict()
                perfect_partitioning = None
                red_state = None
                for red_state in red_states:
                    partitioning = partition_candidates.get((red_state, blue_state))
                    if partitioning is None:
                        partitioning = Partitioning(red_state, blue_state)
                        self._partition_from_merge(partitioning, red_states_backing_set, True)
                    if partitioning.score is True:
                        perfect_partitioning = partitioning
                        break
                    current_candidates[red_state] = partitioning

                # partition with perfect score found: don't consider anything else
                if perfect_partitioning:
                    partition_candidates = {(red_state, blue_state): perfect_partitioning}
                    break

                # update tracking dict with new candidates
                new_candidates = (((red, blue_state), part) for red, part in current_candidates.items())
                partition_candidates.update(new_candidates)

                # no merge candidates for this blue state -> promotion candidate
                if all(part.score is False for part in current_candidates.values()):
                    score = self.score_calc.promotion_score(blue_state)
                    if best_promotion_candidate is None or score is True or best_promotion_score < score:
                        best_promotion_candidate = blue_state
                        best_promotion_score = score
                    if score is True:
                        break

            # check for state promotion
            if best_promotion_candidate is not None:
                # a state was promoted -> only forget scores for this blue node
                for red in red_states:
                    del partition_candidates[(red, best_promotion_candidate)]
                    
                # promote best candidate
                red_states.append(best_promotion_candidate)
                red_states_backing_set.add(best_promotion_candidate)
                blue_states.remove(best_promotion_candidate)
                blue_states.extend(best_promotion_candidate.child_iterator())
                instrumentation.log_promote(best_promotion_candidate)
            else:
                # find best partitioning and apply
                best_candidate = max(partition_candidates.values(), key=lambda part: part.score)
                for real_node, partition_node in best_candidate.red_mapping.items():
                    real_node.transitions = partition_node.transitions
                    real_node.predecessor = partition_node.predecessor
                    real_node.data = partition_node.data
                    real_node.prefix_access_pair = partition_node.prefix_access_pair
                self._partition_from_merge(best_candidate, red_states_backing_set, False)
                blue_states.extend(best_candidate.new_blue)
                blue_states.remove(best_candidate.blue)
                instrumentation.log_merge(best_candidate)

                # a merge was performed -> merge scores are invalidated
                # FUTURE: optimizations for compatibility tests where merges can be orthogonal
                # FUTURE: caching for aggregating compatibility tests
                partition_candidates.clear()

        instrumentation.learning_done(root)

        root = self.postprocessing(root)
        if convert:
            root = root.to_automaton(self.output_behavior, self.transition_behavior)
        return root

    def _partition_from_merge(self, partitioning: Partitioning, red_nodes: set[GsmNode], first_pass):
        """
        Compute the partitioning resulting from merging blue into red, including its score.

        Assumes that blue is a tree and red is not reachable from blue.
        It works in two passes:
        - first pass: create partial partitioning sufficient for score calculation
        - second pass: merge has been accepted, partitioning needs to be completed

        :param Partitioning partitioning: Partitioning object indicating which states to merge.
        :param first_pass: Which pass to perform.
        """

        red = partitioning.red
        blue = partitioning.blue

        if first_pass:
            # for Moore machines the outputs have to match. for prefix-closed data (io-traces) this check is sufficient
            # since Moore-ness is preserved for implied merges.
            if self.output_behavior == "moore" and not GsmNode.moore_compatible(red, blue):
                partitioning.score = False
                return

            # check whether there is an early verdict and adapt helper functions accordingly
            # TODO maybe split init from early verdict and also call init (maybe with first_pass as an argument) in both cases
            partitioning.score = self.score_calc.initialize_merge(red, blue)
            if partitioning.score is not None:
                return
            partitioning.remaining_merges = []

            # uncertain -> need to construct partitioning
            red_partitions: set[GsmNode] = set()
            def update_partition(red_node: GsmNode, blue_node: GsmNode | None) -> GsmNode:
                p = partitioning.full_mapping.get(red_node) # could check smaller .red_mapping?
                if p is None:
                    # there is no partition yet for the 'red' node -> lazily copy
                    p = copy(red_node)
                    p.data = self.data_handler.copy(red_node.data)
                    p.transitions = red_node.transitions.copy()

                    # add to partition table
                    partitioning.full_mapping[red_node] = p
                    partitioning.red_mapping[red_node] = p

                    # check whether the partition is (proper) red
                    if red_node in red_nodes:
                        red_partitions.add(p)
                assert red_node not in red_nodes or p in red_partitions
                if blue_node is not None:
                    partitioning.full_mapping[blue_node] = p
                return p

            cow_set = set()
            def get_partition_trans(part: GsmNode, in_symbol):
                trans = part.transitions[in_symbol]
                if id(trans) not in cow_set:
                    trans = trans.copy()
                    part.transitions[in_symbol] = trans
                    cow_set.add(id(trans))
                return trans
        elif partitioning.remaining_merges is None or len(partitioning.remaining_merges) != 0:
            # best scoring merge candidate -> can manipulate nodes directly
            red_partitions = red_nodes
            def update_partition(red_node: GsmNode, blue_node: GsmNode | None) -> GsmNode:
                return red_node

            def get_partition_trans(part: GsmNode, in_symbol):
                return part.transitions[in_symbol]
        else:
            # first pass already did all the work
            return

        self.data_handler.init_merge(red, blue, first_pass)
        q: deque[tuple[GsmNode, GsmNode]] = deque()

        if first_pass or partitioning.remaining_merges is None:
            # initialize the merge. this should happen only once:
            # - in the first pass if there is no early verdict
            # - in the second pass if there is an early verdict
            assert (first_pass and partitioning.score is None) or (not first_pass and partitioning.score is not None)

            # rewire the blue node's parent
            blue_parent = update_partition(blue.predecessor, None)
            blue_in_sym, blue_out_sym = blue.prefix_access_pair
            get_partition_trans(blue_parent, blue_in_sym)[blue_out_sym] = red

            # create a partition for the red node and check, whether the new output data is available
            partition = update_partition(red, None)
            if self.output_behavior == "moore":
                partition.resolve_unknown_prefix_output(blue_out_sym)

            # initialize the work queue to the initial merge pair
            q.append((red, blue))
        else:
            # work on the remaining merges
            q.extend(partitioning.remaining_merges)

        # loop over implied merges
        pop = q.pop if self.depth_first else q.popleft
        while len(q) != 0:
            red, blue = pop()
            partition = update_partition(red, blue)
            partitioning.nr_merged_states += 1

            if first_pass:
                local_compat = self.score_calc.local_compatibility(partition, blue)
                moore_check = self.output_behavior == "moore" and self.transition_behavior == "deterministic" and not GsmNode.moore_compatible(red, blue)
                if local_compat is False or moore_check:
                    partitioning.score = False
                    return
                if local_compat is None:
                    partitioning.remaining_merges.append((red, blue))
                    continue

            partition.data = self.data_handler.merge(partition.data, blue.data)

            # create implied merges for all common successors
            for in_sym, blue_transitions in blue.transitions.items():
                partition_transitions = get_partition_trans(partition, in_sym)
                for out_sym, blue_successor in blue_transitions.items():
                    partition_successor = partition_transitions.get(out_sym)
                    # handle unknown output
                    if partition_successor is None and len(partition_transitions) != 0:
                        if out_sym is unknown_output:
                            # option A: the output is unknown in the added node
                            assert len(partition_transitions) == 1
                            partition_successor = list(partition_transitions.values())[0]
                        if unknown_output in partition_transitions:
                            # option B: the output is unknown in the partition
                            assert len(partition_transitions) == 1
                            partition_successor = partition_transitions.pop(unknown_output)
                            partition_transitions[out_sym] = partition_successor
                            # re-hook access pair
                            succ_part = update_partition(partition_successor, None)
                            if self.output_behavior == "moore" or succ_part.predecessor is red:
                                succ_part.resolve_unknown_prefix_output(out_sym)
                    # add pairs
                    if partition_successor is not None:
                        q.append((partition_successor, blue_successor))
                    else:
                        # blue_successor is blue after merging if the partition is red
                        if partition in red_partitions:
                            partitioning.new_blue.append(blue_successor)
                        # add new transition to partition
                        partition_transitions[out_sym] = blue_successor
                        # update predecessor of blue child
                        blue_target_partition = update_partition(blue_successor, None)
                        blue_target_partition.predecessor = red

        if first_pass:
            partitioning.score = self.score_calc.score_function(partitioning.full_mapping)


def run_GSM(data: list, *,
            output_behavior: OutputBehavior = "moore",
            transition_behavior: TransitionBehavior = "deterministic",
            score_calc: ScoreCalculation = None,
            pta_preprocessing: Callable[[GsmNode], GsmNode] = None,
            postprocessing: Callable[[GsmNode], GsmNode] = None,
            data_handler: IOHandler = None,
            node_order: Callable[[GsmNode], Any] = None,
            consider_only_min_blue=False,
            depth_first=False,
            instrumentation=None,
            convert=True,
            data_format=None,
            ):
    """
    Performs a state merging algorithm in the red-blue framework on provided data.

    :param list data: Data used for learning. Recorded behavior of the system.
    :param OutputBehavior output_behavior: Specifies whether outputs are emitted by states ("moore") or transitions ("mealy").
    :param TransitionBehavior transition_behavior: Either "deterministic", "nondeterministic" or "stochastic".
    :param ScoreCalculation score_calc: A ScoreCalculation object which determines how compatibility and merge scores are calculated.
    :param Callable[[GsmNode], GsmNode] pta_preprocessing: A pre-processing function applied to the PTA.
    :param Callable[[GsmNode], GsmNode] postprocessing: A postprocessing function applied to the learned automaton.
    :param IOHandler data_handler: IOHandler object governing abstraction and aggregation of data
    :param Callable[[GsmNode], Any] node_order: Sorting key which determines the order in which merge candidates are considered. Defaults to insertion order
    :param bool consider_only_min_blue: Whether to consider merge candidates from all blue nodes or just a single.
    :param bool depth_first: Whether compatibility is checked depth- or breadth-first.
    :param Instrumentation | None instrumentation: Instrumentation object for reporting progress or debugging.
    :param bool convert: Whether to return a normal AALpy automaton type or a `GsmNode` object (internal representation).
    :param DataFormat | None data_format: Whether the input is given in the form of input-output traces or labeled input traces.
    :return Automaton | GsmNode: The learned automaton.
    """
    # instantiate gsm
    gsm = GeneralizedStateMerging(
        output_behavior=output_behavior,
        transition_behavior=transition_behavior,
        score_calc=score_calc,
        pta_preprocessing=pta_preprocessing,
        postprocessing=postprocessing,
        data_handler=data_handler,
        node_order=node_order,
        consider_only_min_blue=consider_only_min_blue,
        depth_first=depth_first,
    )

    # run the algorithm
    return gsm.run(data=data, instrumentation=instrumentation, convert=convert, data_format=data_format)
