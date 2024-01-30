import functools
import time
from typing import Dict, Tuple, Callable
from collections import deque

from aalpy.learning_algs.general_passive.helpers import Node, OutputBehavior, TransitionBehavior, TransitionInfo, \
    OutputBehaviorRange, TransitionBehaviorRange
from aalpy.learning_algs.general_passive.ScoreFunctionsGSM import ScoreCalculation, NoRareEventNonDetScore, \
    hoeffding_compatibility, Score

# TODO make non-mutual exclusive? Easiest done by adding a new method / field to ScoreCalculation
# future: Only compare futures of states
# partition: Check compatibility while partition is created
CompatibilityBehavior = str
CompatibilityBehaviorRange = ["future", "partition"]

class Partitioning:
    def __init__(self, red : Node, blue : Node):
        self.red : Node = red
        self.blue : Node = blue
        self.score : Score = False
        self.red_mapping : Dict[Node, Node] = dict()
        self.full_mapping : Dict[Node, Node] = dict()

class DebugInfo:
    def __init__(self, lvl):
        self.lvl = lvl

    @staticmethod
    def min_lvl(lvl):
        def decorator(fn):
            from functools import wraps
            @wraps(fn)
            def wrapper(*args, **kw):
                if args[0].lvl < lvl:
                    return
                fn(*args, **kw)
            return wrapper
        return decorator

class DebugInfoGSM(DebugInfo):
    min_lvl = DebugInfo.min_lvl

    def __init__(self, lvl, instance : 'GeneralizedStateMerging'):
        super().__init__(lvl)
        if lvl < 1:
            return
        self.instance = instance
        self.log = []
        self.pta_size = None
        self.nr_merged_states_total = 0
        self.nr_merged_states = 0
        self.nr_red_states = 1

    @min_lvl(1)
    def pta_construction_done(self, start_time):
        print(f'PTA Construction Time: {round(time.time() - start_time, 2)}')
        if self.lvl != 1:
            states = self.instance.root.get_all_nodes()
            leafs = [state for state in states if len(state.transitions.keys()) == 0]
            depth = [len(state.prefix) for state in leafs]
            self.pta_size = len(states)
            print(f'PTA has {len(states)} states leading to {len(leafs)} leafs')
            print(f'min / avg / max depth : {min(depth)} / {sum(depth) / len(depth)} / {max(depth)}')

    def print_status(self):
        print_str = f'\rCurrent automaton size: {self.nr_red_states}'
        if self.lvl != 1:
            print_str += f' Merged: {self.nr_merged_states_total} Remaining: {self.pta_size - self.nr_red_states - self.nr_merged_states_total}'
        print(print_str, end="")

    @min_lvl(1)
    def log_promote(self, node : Node, red_states):
        self.log.append(["promote", (node.prefix,)])
        self.nr_red_states = len(red_states) # could be done incrementally, here for historic reasons
        self.print_status()

    @min_lvl(1)
    def log_merge(self, part : Partitioning):
        self.log.append(["merge", (part.red.prefix, part.blue.prefix)])
        self.nr_merged_states_total += len(part.full_mapping) - len(part.red_mapping)
        self.nr_merged_states += 1
        self.print_status()

    @min_lvl(1)
    def learning_done(self, red_states, start_time):
        print(f'\nLearning Time: {round(time.time() - start_time, 2)}')
        print(f'Learned {len(red_states)} state automaton via {self.nr_merged_states} merges.')
        if 2 < self.lvl:
            self.instance.root.visualize("model",self.instance.output_behavior)

class GeneralizedStateMerging:
    def __init__(self, data, *,
                 output_behavior : OutputBehavior = "moore",
                 transition_behavior : TransitionBehavior = "deterministic",
                 compatibility_behavior : CompatibilityBehavior = "partition",
                 score_calc : ScoreCalculation = None,
                 eval_compat_on_pta : bool = False,
                 node_order : Callable[[Node, Node], bool] = None,
                 consider_all_blue_states = False,
                 depth_first = False,
                 debug_lvl = 0):
        self.eval_compat_on_pta = eval_compat_on_pta
        self.data = data
        self.debug = DebugInfoGSM(debug_lvl, self)

        if output_behavior not in OutputBehaviorRange:
            raise ValueError(f"invalid output behavior {output_behavior}")
        self.output_behavior : OutputBehavior = output_behavior
        if transition_behavior not in TransitionBehaviorRange:
            raise ValueError(f"invalid transition behavior {transition_behavior}")
        self.transition_behavior : TransitionBehavior = transition_behavior
        if compatibility_behavior not in CompatibilityBehaviorRange:
            raise ValueError(f"invalid compatibility behavior {compatibility_behavior}")
        self.compatibility_behavior : CompatibilityBehavior = compatibility_behavior

        if score_calc is None:
            if transition_behavior == "deterministic" :
                score_calc = ScoreCalculation()
            elif transition_behavior == "nondeterministic" :
                score_calc = NoRareEventNonDetScore(0.5, 0.001)
            elif transition_behavior == "stochastic" :
                score_calc = ScoreCalculation(hoeffding_compatibility(0.005, self.eval_compat_on_pta))
        self.score_calc : ScoreCalculation = score_calc

        if node_order is None:
            node_order = Node.__lt__
        self.node_order = functools.cmp_to_key(lambda a, b: -1 if node_order(a,b) else 1)

        self.consider_all_blue_states = consider_all_blue_states
        self.depth_first = depth_first

        pta_construction_start = time.time()
        self.root: Node
        if isinstance(data, Node):
            self.root = data
        elif output_behavior == "moore":
            self.root = Node.createPTA((d[1:] for d in data), data[0][0])
        else :
            self.root = Node.createPTA(data)

        self.debug.pta_construction_done(pta_construction_start)

        if transition_behavior == "deterministic":
            if not self.root.is_deterministic():
                raise ValueError("required deterministic automaton but input data is nondeterministic")

    def compute_local_compatibility(self, a : Node, b : Node):
        if self.output_behavior == "moore" and not Node.moore_compatible(a,b):
            return False
        if self.transition_behavior == "deterministic" and not Node.deterministic_compatible(a,b):
            return False
        return self.score_calc.local_compatibility(a, b)

    def run(self):
        start_time = time.time()

        # sorted list of states already considered
        red_states = [self.root]

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
                    if not self.consider_all_blue_states:
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
                    self.debug.log_promote(blue_state, red_states)
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
            # FUTURE: optimizations for compatibility tests where merges can be orthogonal
            # FUTURE: caching for aggregating compatibility tests
            partition_candidates.clear()
            for real_node, partition_node in best_candidate.red_mapping.items():
                real_node.transitions = partition_node.transitions
            self.debug.log_merge(best_candidate)

        self.debug.learning_done(red_states, start_time)

        return self.root.to_automaton(self.output_behavior, self.transition_behavior)

    def _check_futures(self, red: Node, blue: Node) -> bool:
        q : deque[Tuple[Node, Node]] = deque([(red, blue)])
        pop = q.pop if self.depth_first else q.popleft

        while len(q) != 0:
            red, blue = pop()

            if self.compute_local_compatibility(red, blue) is False:
                return False

            for in_sym, blue_transitions in blue.transitions.items():
                red_transitions = red.get_transitions_safe(in_sym)
                for out_sym, blue_child in blue_transitions.items():
                    red_child = red_transitions.get(out_sym)
                    if red_child is None:
                        continue
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

        if self.compatibility_behavior == "future":
            if self._check_futures(red, blue) is False:
                return partitioning

        # when compatibility is determined only by future and scores are disabled, we need not create partitions.
        if self.compatibility_behavior == "future" and not self.score_calc.has_score_function():
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
        blue_parent = update_partition(self.root.get_by_prefix(blue.prefix[:-1]), None)
        blue_parent.transitions[blue.prefix[-1][0]][blue.prefix[-1][1]].target = red

        q : deque[Tuple[Node, Node]] = deque([(red, blue)])
        pop = q.pop if self.depth_first else q.popleft

        while len(q) != 0:
            red, blue = pop()
            partition = update_partition(red, blue)

            if self.compatibility_behavior == "partition":
                if self.compute_local_compatibility(partition, blue) is False:
                    return partitioning

            for in_sym, blue_transitions in blue.transitions.items():
                partition_transitions = partition.get_transitions_safe(in_sym)
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
            compatibility_behavior : CompatibilityBehavior = "partition",
            score_calc : ScoreCalculation = None,
            eval_compat_on_pta : bool = False,
            node_order : Callable[[Node, Node], bool] = None,
            consider_all_blue_states = False,
            depth_first = False,
            debug_lvl = 0):
    return GeneralizedStateMerging(**locals()).run()


def run_alergia(data, output_behavior : OutputBehavior = "moore",
                epsilon : float = 0.005,
                compatibility_behavior : CompatibilityBehavior = "future",
                eval_compat_on_pta=True,
                global_score=None,
                **kwargs) :
    return GeneralizedStateMerging(
        data,
        output_behavior=output_behavior,
        transition_behavior="stochastic",
        compatibility_behavior=compatibility_behavior,
        score_calc=ScoreCalculation(hoeffding_compatibility(epsilon), global_score),
        eval_compat_on_pta=eval_compat_on_pta,
        **kwargs
    ).run()

def run_RPNI(data, output_behavior : OutputBehavior = "moore", *args, **kwargs):
    return GeneralizedStateMerging(
        data,
        output_behavior=output_behavior,
        transition_behavior="deterministic",
        compatibility_behavior="partition",
        *args, **kwargs
    ).run()