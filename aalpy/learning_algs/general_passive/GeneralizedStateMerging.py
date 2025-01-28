import functools
from typing import Dict, Tuple, Callable, List
from collections import deque

from aalpy.learning_algs.general_passive.helpers import Node, OutputBehavior, TransitionBehavior, TransitionInfo, \
    OutputBehaviorRange, TransitionBehaviorRange, intersection_iterator
from aalpy.learning_algs.general_passive.ScoreFunctionsGSM import ScoreCalculation, NoRareEventNonDetScore, \
    hoeffding_compatibility, Score

# TODO add option for making checking of futures and partition non mutual exclusive?
#  Easiest done by adding a new method / field to ScoreCalculation

class Partitioning:
    def __init__(self, red : Node, blue : Node):
        self.red : Node = red
        self.blue : Node = blue
        self.score : Score = False
        self.red_mapping : Dict[Node, Node] = dict()
        self.full_mapping : Dict[Node, Node] = dict()

class Instrumentation:
  def __init__(self):
      pass

  def reset(self, gsm: 'GeneralizedStateMerging'):
      pass

  def pta_construction_done(self, root: Node):
      pass

  def log_promote(self, node: Node):
      pass

  def log_merge(self, part: Partitioning):
      pass

  def learning_done(self, root: Node, red_states: List[Node]):
      pass

class GeneralizedStateMerging:
    def __init__(self, *,
                 output_behavior : OutputBehavior = "moore",
                 transition_behavior : TransitionBehavior = "deterministic",
                 score_calc : ScoreCalculation = None,
                 pta_preprocessing : Callable[[Node], Node] = None,
                 postprocessing : Callable[[Node], Node] = None,
                 eval_compat_on_pta : bool = False,
                 eval_compat_on_futures : bool = False,
                 node_order : Callable[[Node, Node], bool] = None,
                 consider_only_min_blue = False,
                 depth_first = False):

        if output_behavior not in OutputBehaviorRange:
            raise ValueError(f"invalid output behavior {output_behavior}")
        self.output_behavior : OutputBehavior = output_behavior
        if transition_behavior not in TransitionBehaviorRange:
            raise ValueError(f"invalid transition behavior {transition_behavior}")
        self.transition_behavior : TransitionBehavior = transition_behavior

        if score_calc is None:
            if transition_behavior == "deterministic" :
                score_calc = ScoreCalculation()
            elif transition_behavior == "nondeterministic" :
                score_calc = NoRareEventNonDetScore(0.5, 0.001)
            elif transition_behavior == "stochastic" :
                score_calc = ScoreCalculation(hoeffding_compatibility(0.005, eval_compat_on_pta))
        self.score_calc : ScoreCalculation = score_calc

        if node_order is None:
            node_order = Node.__lt__
        self.node_order = functools.cmp_to_key(lambda a, b: -1 if node_order(a, b) else 1)

        self.pta_preprocessing = pta_preprocessing or (lambda x: x)
        self.postprocessing = postprocessing or (lambda x: x)

        self.eval_compat_on_pta = eval_compat_on_pta
        self.eval_compat_on_futures = eval_compat_on_futures

        self.consider_only_min_blue = consider_only_min_blue
        self.depth_first = depth_first

    def compute_local_compatibility(self, a : Node, b : Node):
        if self.output_behavior == "moore" and not Node.moore_compatible(a,b):
            return False
        if self.transition_behavior == "deterministic" and not Node.deterministic_compatible(a,b):
            return False
        return self.score_calc.local_compatibility(a, b)

    # TODO: make more generic by adding the option to use a different algorithm than red blue
    #  for selecting potential merge candidates. Maybe using inheritance with abstract `run`.
    def run(self, data, convert = True, instrumentation: Instrumentation = None):
        if instrumentation is None:
            instrumentation = Instrumentation()
        instrumentation.reset(self)

        if isinstance(data, Node):
            root = data
        else:
            root = Node.createPTA(data, self.output_behavior)
        root = self.pta_preprocessing(root)
        instrumentation.pta_construction_done(root)
        instrumentation.log_promote(root)

        # This was removed because it is also checked during extraction
        # if self.transition_behavior == "deterministic":
        #     if not root.is_deterministic():
        #         raise ValueError("required deterministic automaton but input data is nondeterministic")

        # sorted list of states already considered
        red_states = [root]

        partition_candidates : Dict[Tuple[Node, Node], Partitioning] = dict()
        while True:
            # get blue states
            blue_states = []
            for r in red_states:
                for _, t in r.transition_iterator():
                    c = t.target
                    if c in red_states:
                        continue
                    blue_states.append(c)
                    if self.consider_only_min_blue:
                        blue_states = [min(blue_states, key=self.node_order)]

            # no blue states left -> done
            if len(blue_states) == 0:
                break
            blue_states.sort(key=self.node_order)

            # loop over blue states
            promotion = False
            for blue_state in blue_states:
                # FUTURE: Parallelize
                # FUTURE: Save partitions?

                # calculate partitions resulting from merges with red states if necessary
                current_candidates : Dict[Node, Partitioning] = dict()
                perfect_partitioning = None
                for red_state in red_states:
                    partition = partition_candidates.get((red_state, blue_state))
                    if partition is None:
                        partition = self._partition_from_merge(red_state, blue_state)
                    if partition.score is True:
                        perfect_partitioning = partition
                        break
                    current_candidates[red_state] = partition

                # partition with perfect score found: don't consider anything else
                if perfect_partitioning:
                    partition_candidates = {(red_state, blue_state) : perfect_partitioning}
                    break

                # no merge candidates for this blue state -> promote
                if all(part.score is False for part in current_candidates.values()):
                    red_states.append(blue_state)
                    instrumentation.log_promote(blue_state)
                    promotion = True
                    break

                # update tracking dict with new candidates
                new_candidates = (((red, blue_state), part) for red, part in current_candidates.items() if part.score is not False)
                partition_candidates.update(new_candidates)

            # a state was promoted -> don't clear candidates
            if promotion:
                continue

            # find best partitioning and clear candidates
            best_candidate = max(partition_candidates.values(), key = lambda part : part.score)
            instrumentation.log_merge(best_candidate)
            # FUTURE: optimizations for compatibility tests where merges can be orthogonal
            # FUTURE: caching for aggregating compatibility tests
            partition_candidates.clear()
            for real_node, partition_node in best_candidate.red_mapping.items():
                real_node.transitions = partition_node.transitions
                for _, t_info in real_node.transition_iterator():
                    if t_info.target not in red_states:
                        t_info.target.predecessor = real_node

        instrumentation.learning_done(root, red_states)

        root = self.postprocessing(root)
        if convert:
            root = root.to_automaton(self.output_behavior, self.transition_behavior)
        return root

    def _check_futures(self, red: Node, blue: Node) -> bool:
        q : deque[Tuple[Node, Node]] = deque([(red, blue)])
        pop = q.pop if self.depth_first else q.popleft

        while len(q) != 0:
            red, blue = pop()

            if self.compute_local_compatibility(red, blue) is False:
                return False

            for in_sym, red_trans, blue_trans in intersection_iterator(red.transitions, blue.transitions):
                for out_sym, red_child, blue_child in intersection_iterator(red_trans, blue_trans):
                    if self.eval_compat_on_pta:
                        if blue_child.original_count == 0 or red_child.original_count == 0:
                            continue
                        q.append((red_child.original_target, blue_child.original_target))
                    else:
                        q.append((red_child.target, blue_child.target))

        return True

    def _partition_from_merge(self, red: Node, blue: Node) -> Partitioning :
        # Compatibility check based on partitions.
        # assumes that blue is a tree and red is not reachable from blue

        partitioning = Partitioning(red, blue)

        self.score_calc.reset()

        if self.eval_compat_on_futures:
            if self._check_futures(red, blue) is False:
                return partitioning

        # when compatibility is determined only by future and scores are disabled, we need not create partitions.
        if self.eval_compat_on_futures and not self.score_calc.has_score_function():
            def update_partition(red_node: Node, blue_node: Node) -> Node:
                return red_node
        else:
            def update_partition(red_node: Node, blue_node: Node) -> Node:
                if red_node not in partitioning.full_mapping:
                    p = red_node.shallow_copy()
                    partitioning.full_mapping[red_node] = p
                    partitioning.red_mapping[red_node] = p
                else:
                    p = partitioning.full_mapping[red_node]
                if blue_node is not None:
                    partitioning.full_mapping[blue_node] = p
                return p

        # rewire the blue node's parent
        blue_parent = update_partition(blue.predecessor, None)
        blue_in_sym, blue_out_sym = blue.prefix_access_pair
        blue_parent.transitions[blue_in_sym][blue_out_sym].target = red

        q : deque[Tuple[Node, Node]] = deque([(red, blue)])
        pop = q.pop if self.depth_first else q.popleft

        while len(q) != 0:
            red, blue = pop()
            partition = update_partition(red, blue)

            if not self.eval_compat_on_futures:
                if self.compute_local_compatibility(partition, blue) is False:
                    return partitioning

            for in_sym, blue_transitions in blue.transitions.items():
                partition_transitions = partition.get_or_create_transitions(in_sym)
                for out_sym, blue_transition in blue_transitions.items():
                    partition_transition = partition_transitions.get(out_sym)
                    if partition_transition is not None:
                        q.append((partition_transition.target, blue_transition.target))
                        partition_transition.count += blue_transition.count
                    else:
                        # blue_child is blue after merging if there is a red state in blue's partition
                        partition_transition = TransitionInfo(blue_transition.target, blue_transition.count, None, 0)
                        partition_transitions[out_sym] = partition_transition

        partitioning.score = self.score_calc.score_function(partitioning.full_mapping)
        return partitioning


# TODO nicer interface?
def run_GSM(data, *,
            output_behavior : OutputBehavior = "moore",
            transition_behavior : TransitionBehavior = "deterministic",
            score_calc : ScoreCalculation = None,
            pta_preprocessing : Callable[[Node], Node] = None,
            postprocessing : Callable[[Node], Node] = None,
            eval_compat_on_pta : bool = False,
            eval_compat_on_futures : bool = False,
            node_order : Callable[[Node, Node], bool] = None,
            consider_only_min_blue = False,
            depth_first = False,
            instrumentation = None,
            convert = True,
            ):
    all_params = locals()
    run_param_names = ["data", "instrumentation", "convert"]
    run_params = {key: all_params[key] for key in run_param_names}
    ctor_params = {key: val for key, val in all_params.items() if key not in run_params}
    return GeneralizedStateMerging(**ctor_params).run(**run_params)