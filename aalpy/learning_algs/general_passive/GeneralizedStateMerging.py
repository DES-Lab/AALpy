import functools
from collections import deque
from copy import copy
from typing import Dict, Tuple, Callable, List, Optional

from aalpy.learning_algs.general_passive.GsmNode import GsmNode, OutputBehavior, TransitionBehavior, \
    OutputBehaviorRange, TransitionBehaviorRange, intersection_iterator, unknown_output, detect_data_format, IOHandler, \
    NoIOHandler
from aalpy.learning_algs.general_passive.ScoreFunctionsGSM import ScoreCalculation, hoeffding_compatibility


# TODO add option for making checking of futures and partition non mutual exclusive?
#  Easiest done by adding a new method / field to ScoreCalculation

class Partitioning:
    def __init__(self, red: GsmNode, blue: GsmNode):
        self.red: GsmNode = red
        self.blue: GsmNode = blue
        self.score = False
        self.red_mapping: Dict[GsmNode, GsmNode] = dict()
        self.full_mapping: Dict[GsmNode, GsmNode] = dict()
        self.new_blue = []


class Instrumentation:
    def __init__(self):
        pass

    def reset(self, gsm: 'GeneralizedStateMerging'):
        pass

    def pta_construction_done(self, root: GsmNode):
        pass

    def log_promote(self, node: GsmNode):
        pass

    def log_merge(self, part: Partitioning):
        pass

    def learning_done(self, root: GsmNode):
        pass


class GeneralizedStateMerging:
    def __init__(self, *,
                 output_behavior: OutputBehavior = "moore",
                 transition_behavior: TransitionBehavior = "deterministic",
                 score_calc: ScoreCalculation = None,
                 pta_preprocessing: Callable[[GsmNode], GsmNode] = None,
                 postprocessing: Callable[[GsmNode], GsmNode] = None,
                 data_handler: IOHandler = None,
                 use_early_verdicts: bool = False,
                 node_order: Callable[[GsmNode, GsmNode], bool] = None,
                 consider_only_min_blue=False,
                 depth_first=False,
                 ):

        if output_behavior not in OutputBehaviorRange:
            raise ValueError(f"invalid output behavior {output_behavior}. should be in {OutputBehaviorRange}")
        self.output_behavior: OutputBehavior = output_behavior
        if transition_behavior not in TransitionBehaviorRange:
            raise ValueError(f"invalid transition behavior {transition_behavior}. should be in {TransitionBehaviorRange}")
        self.transition_behavior: TransitionBehavior = transition_behavior

        if score_calc is None:
            if transition_behavior == "deterministic":
                score_calc = ScoreCalculation()
            elif transition_behavior == "nondeterministic" :
                raise ValueError("Missing score_calc for nondeterministic transition behavior. No default available.")
            elif transition_behavior == "stochastic" :
                score_calc = ScoreCalculation(hoeffding_compatibility(0.005, True))
        self.score_calc: ScoreCalculation = score_calc

        if node_order is None:
            self.node_order = GsmNode.default_order
        else:
            self.node_order = functools.cmp_to_key(lambda a, b: -1 if node_order(a, b) else 1)

        self.pta_preprocessing = pta_preprocessing or (lambda x: x)
        self.postprocessing = postprocessing or (lambda x: x)

        self.data_handler = data_handler or NoIOHandler()

        self.use_early_verdicts = use_early_verdicts

        self.consider_only_min_blue = consider_only_min_blue
        self.depth_first = depth_first

    def compute_local_compatibility(self, a: GsmNode, b: GsmNode):
        if self.output_behavior == "moore" and not GsmNode.moore_compatible(a, b):
            return False
        if self.transition_behavior == "deterministic" and not GsmNode.deterministic_compatible(a, b):
            return False
        return self.score_calc.local_compatibility(a, b)

    # TODO: make more generic by adding the option to use a different algorithm than red blue
    #  for selecting potential merge candidates. Maybe using inheritance with abstract `run`.
    def run(self, data, convert=True, instrumentation: Instrumentation=None, data_format=None):
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
                raise ValueError("required deterministic automaton but input data is nondeterministic")

        # sorted list of states already considered
        red_states = [root]
        red_states_backing_set = {root}
        blue_states = list(root.child_iterator())

        partition_candidates: Dict[Tuple[GsmNode, GsmNode], Partitioning] = dict()
        while True:
            # no blue states left -> done
            if len(blue_states) == 0:
                break

            blue_states_to_consider = blue_states
            if self.consider_only_min_blue: # does it make sense to check the score function here?
                blue_states_to_consider = [min(blue_states, key=self.node_order)]

            # could make this sort unconditional, but i think this is closer to the original in any case?
            if self.node_order is not GsmNode.default_order:
                blue_states_to_consider.sort(key=self.node_order)

            # sort red states. states are always sorted using default order on original prefix
            if self.node_order is not GsmNode.default_order:
                red_states.sort(key=self.node_order)

            # loop over blue states
            promotion = False
            for blue_state in blue_states_to_consider:
                # FUTURE: Parallelize
                # FUTURE: Save partitions?

                # calculate partitions resulting from merges with red states if necessary
                current_candidates: Dict[GsmNode, Partitioning] = dict()
                perfect_partitioning = None
                red_state = None
                for red_state in red_states:
                    partition = partition_candidates.get((red_state, blue_state))
                    if partition is None:
                        partition = self._partition_from_merge(red_state, blue_state, red_states_backing_set)
                    if partition.score is True:
                        perfect_partitioning = partition
                        break
                    current_candidates[red_state] = partition
                assert red_state is not None

                # partition with perfect score found: don't consider anything else
                if perfect_partitioning:
                    partition_candidates = {(red_state, blue_state): perfect_partitioning}
                    break

                # no merge candidates for this blue state -> promote
                if all(part.score is False for part in current_candidates.values()):
                    red_states.append(blue_state)
                    red_states_backing_set.add(blue_state)
                    blue_states.remove(blue_state)
                    blue_states.extend(blue_state.child_iterator())
                    instrumentation.log_promote(blue_state)
                    promotion = True
                    break

                # update tracking dict with new candidates
                new_candidates = (((red, blue_state), part) for red, part in current_candidates.items() if
                                  part.score is not False)
                partition_candidates.update(new_candidates)

            # a state was promoted -> don't clear candidates
            if promotion:
                continue

            # find best partitioning and clear candidates
            best_candidate = max(partition_candidates.values(), key=lambda part: part.score)
            for real_node, partition_node in best_candidate.red_mapping.items():
                real_node.transitions = partition_node.transitions
                real_node.predecessor = partition_node.predecessor
                real_node.data = partition_node.data
                real_node.prefix_access_pair = partition_node.prefix_access_pair
            blue_states.extend(best_candidate.new_blue)
            blue_states.remove(best_candidate.blue)
            instrumentation.log_merge(best_candidate)
            # FUTURE: optimizations for compatibility tests where merges can be orthogonal
            # FUTURE: caching for aggregating compatibility tests
            partition_candidates.clear()

        instrumentation.learning_done(root)

        root = self.postprocessing(root)
        if convert:
            root = root.to_automaton(self.output_behavior, self.transition_behavior)
        return root

    def _partition_from_merge(self, red: GsmNode, blue: GsmNode, red_nodes: set[GsmNode]) -> Partitioning:
        # Compatibility check based on partitions.
        # assumes that blue is a tree and red is not reachable from blue

        partitioning = Partitioning(red, blue)

        early_verdict = self.score_calc.initialize_merge(red, blue) if self.use_early_verdicts else None
        if early_verdict is False:
            # early reject -> return failing partitioning
            return partitioning
        elif early_verdict is True:
            # early accept -> can manipulate nodes directly
            red_partitions = red_nodes
            def update_partition(red_node: GsmNode, blue_node: Optional[GsmNode]) -> GsmNode:
                return red_node

            def get_partition_trans(part: GsmNode, in_symbol):
                return part.transitions[in_symbol]
        else:
            # uncertain -> need to construct partitioning
            red_partitions: set[GsmNode] = set()
            def update_partition(red_node: GsmNode, blue_node: Optional[GsmNode]) -> GsmNode:
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

        self.data_handler.init_merge(red, blue)

        # rewire the blue node's parent
        blue_parent = update_partition(blue.predecessor, None)
        blue_in_sym, blue_out_sym = blue.prefix_access_pair
        get_partition_trans(blue_parent, blue_in_sym)[blue_out_sym] = red

        partition = update_partition(red, None)
        if self.output_behavior == "moore":
            partition.resolve_unknown_prefix_output(blue_out_sym)

        # loop over implied merges
        q: deque[Tuple[GsmNode, GsmNode]] = deque([(red, blue)])
        pop = q.pop if self.depth_first else q.popleft
        while len(q) != 0:
            red, blue = pop()
            partition = update_partition(red, blue)

            if early_verdict is None and self.compute_local_compatibility(partition, blue) is False:
                return partitioning

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

        partitioning.score = self.score_calc.score_function(partitioning.full_mapping)
        return partitioning


def run_GSM(data: list, *,
            output_behavior: OutputBehavior = "moore",
            transition_behavior: TransitionBehavior = "deterministic",
            score_calc: ScoreCalculation = None,
            pta_preprocessing: Callable[[GsmNode], GsmNode] = None,
            postprocessing: Callable[[GsmNode], GsmNode] = None,
            data_handler: IOHandler = None,
            use_early_verdicts: bool = False,
            node_order: Callable[[GsmNode, GsmNode], bool] = None,
            consider_only_min_blue=False,
            depth_first=False,
            instrumentation=None,
            convert=True,
            data_format=None,
            ):
    """
    Performs a state merging algorithm in the red-blue framework on provided data.

    Args:
        data: Data used for learning. Recorded behavior of the system.

        output_behavior: Specifies whether outputs are emitted by states ("moore") or transitions ("mealy").

        transition_behavior: Either "deterministic", "nondeterministic" or "stochastic".

        score_calc: A ScoreCalculation object which determines how compatibility and merge scores are calculated.

        pta_preprocessing: A pre-processing function applied to the PTA.

        postprocessing: A postprocessing function applied to the learned automaton.

        use_early_verdicts: Whether to use potential early verdicts from the score object. Defaults to False.

        node_order: Order in which merge candidates are considered. Defaults to short-lex.

        consider_only_min_blue: Whether to consider merge candidates from all blue nodes or just a single.

        depth_first: Whether compatibility is checked depth- or breadth-first.

        instrumentation: Instrumentation object for reporting progress or debugging.

        convert: Whether to return a normal AALpy automaton type or a `GsmNode` object (internal representation).

        data_format: Whether the input is given in the form of input-output traces or labeled input traces.

    Returns: The learned automaton.
    """
    # instantiate gsm
    gsm = GeneralizedStateMerging(
        output_behavior=output_behavior,
        transition_behavior=transition_behavior,
        score_calc=score_calc,
        pta_preprocessing=pta_preprocessing,
        postprocessing=postprocessing,
        data_handler=data_handler,
        use_early_verdicts=use_early_verdicts,
        node_order=node_order,
        consider_only_min_blue=consider_only_min_blue,
        depth_first=depth_first,
    )

    # run the algorithm
    return gsm.run(data=data, instrumentation=instrumentation, convert=convert, data_format=data_format)
