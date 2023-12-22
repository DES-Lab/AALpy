import functools
import typing
from dataclasses import dataclass
from math import sqrt, log
import time
from typing import Dict, Tuple, Callable, Any, Literal, List

from aalpy.learning_algs.stochastic_passive.helpers import Node, OutputBehavior, TransitionBehavior, TransitionInfo

# TODO make non-mutual exclusive
# future: Only compare futures of states
# partition: Check compatibility while partition is created
# merge: Check compatibility after partition is created
CompatibilityBehavior = Literal["future", "partition", "merge"]

Score = bool | float
ScoreFunction = Callable[[Node,Node,Any,bool], Score]

@dataclass
class Partitioning:
    score : Score
    partitions : dict[Node, Node]

def hoeffding_compatibility(eps) -> ScoreFunction:
    def similar(a: Node, b: Node, _: Any, compare_original):
        for in_sym in filter(lambda x : x in a.transitions.keys(), b.transitions.keys()):
            # could create appropriate dict here
            a_trans, b_trans = (x.transitions[in_sym] for x in [a,b])
            if compare_original:
                a_total, b_total = (sum(x.original_count for x in x.values()) for x in (a_trans, b_trans))
            else:
                a_total, b_total = (sum(x.count for x in x.values()) for x in (a_trans, b_trans))
            if a_total == 0 or b_total == 0:
                continue
            threshold = ((sqrt(1 / a_total) + sqrt(1 / b_total)) * sqrt(0.5 * log(2 / eps)))
            for out_sym in set(a_trans.keys()).union(b_trans.keys()):
                if compare_original:
                    ac, bc = (x[out_sym].original_count if out_sym in x else 0 for x in (a_trans, b_trans))
                else:
                    ac, bc = (x[out_sym].count if out_sym in x else 0 for x in (a_trans, b_trans))
                if abs(ac / a_total - bc / b_total) > threshold:
                    return False
        return True
    return similar

def non_det_compatibility(eps) -> ScoreFunction:
    print("Warning: using experimental compatibility criterion for nondeterministic automata")
    def similar(a: Node, b: Node, _: Any, compare_original : bool):
        if compare_original:
            raise NotImplementedError()
        for in_sym in filter(lambda x : x in a.transitions.keys(), b.transitions.keys()):
            a_trans, b_trans = (x.transitions[in_sym] for x in [a,b])
            a_total, b_total = (sum(x.count for x in x.values()) for x in (a_trans, b_trans))
            if a_total < eps or b_total < eps:
                continue
            if set(a_trans.keys()) != set(b_trans.keys()):
                return False
        return True
    return similar

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

    @min_lvl(1)
    def pta_construction_done(self, start_time):
        print(f'PTA Construction Time: {round(time.time() - start_time, 2)}')
        if self.lvl != 1:
            states = self.instance.root.get_all_nodes()
            leafs = [state for state in states if len(state.transitions.keys()) == 0]
            depth = [len(state.prefix) for state in leafs]
            print(f'PTA has {len(states)} states leading to {len(leafs)} leafs')
            print(f'min / avg / max depth : {min(depth)} / {sum(depth) / len(depth)} / {max(depth)}')

    @min_lvl(1)
    def log_promote(self, node : Node, red_states):
        self.log.append(["promote", (node.prefix,)])
        print(f'\rCurrent automaton size: {len(red_states)}', end="")

    @min_lvl(1)
    def log_merge(self, a : Node, b : Node):
        self.log.append(["merge", (a.prefix, b.prefix)])

    @min_lvl(1)
    def learning_done(self, red_states, start_time):
        print(f'\nLearning Time: {round(time.time() - start_time, 2)}')
        print(f'Learned {len(red_states)} state automaton.')
        if 1 < self.lvl:
            self.instance.root.visualize("model",self.instance.output_behavior)

class GeneralizedStateMerging:
    def __init__(self, data, output_behavior : OutputBehavior = "moore",
                 transition_behavior : TransitionBehavior = "deterministic",
                 compatibility_behavior : CompatibilityBehavior = "partition",
                 local_score : ScoreFunction = None, info_update : Callable[[Node, Node, Any],Any] = None,
                 eval_compat_on_pta : bool = False,
                 node_order : Callable[[Node,Node],bool] = None,
                 debug_lvl=0):
        self.eval_compat_on_pta = eval_compat_on_pta
        self.data = data
        self.debug = DebugInfoGSM(debug_lvl, self)

        if output_behavior not in typing.get_args(OutputBehavior):
            raise ValueError(f"invalid output behavior {output_behavior}")
        self.output_behavior : OutputBehavior = output_behavior
        if transition_behavior not in typing.get_args(TransitionBehavior):
            raise ValueError(f"invalid transition behavior {transition_behavior}")
        self.transition_behavior : TransitionBehavior = transition_behavior
        if compatibility_behavior not in typing.get_args(CompatibilityBehavior):
            raise ValueError(f"invalid compatibility behavior {compatibility_behavior}")
        self.compatibility_behavior : CompatibilityBehavior = compatibility_behavior

        if info_update is None:
            info_update = lambda a, b, c : c
        self.info_update = info_update

        if local_score is None:
            match transition_behavior:
                case "deterministic" : local_score = lambda x,y,_ : True
                case "nondeterministic" : local_score = non_det_compatibility(20)
                case "stochastic" : local_score = hoeffding_compatibility(0.005)
        self.local_score : ScoreFunction = local_score

        if node_order is None:
            node_order = Node.__lt__
        self.node_order = functools.cmp_to_key(lambda a, b: -1 if node_order(a,b) else 1)

        pta_construction_start = time.time()
        self.root: Node
        if isinstance(data, Node):
            self.root = data
        elif output_behavior == "moore":
            self.root = Node.createPTA([d[1:] for d in data], data[0][0])
        else :
            self.root = Node.createPTA(data)

        self.debug.pta_construction_done(pta_construction_start)

        if transition_behavior == "deterministic":
            if not self.root.is_deterministic():
                raise ValueError("required deterministic automaton but input data is nondeterministic")

    def compute_local_score(self, a : Node, b : Node, info : Any):
        if self.output_behavior == "moore" and not Node.moore_compatible(a,b):
            return False
        if self.transition_behavior == "deterministic" and not Node.deterministic_compatible(a,b):
            return False
        return self.local_score(a, b, info, self.eval_compat_on_pta)

    def run(self):
        start_time = time.time()

        # sorted list of states already considered
        red_states = [self.root]

        partition_candidates : dict[tuple[Node, Node], Partitioning] = dict()
        while True:
            # get blue states
            blue_states = []
            for r in red_states:
                for _, t in r.transition_iterator():
                    c = t.target
                    if c in red_states:
                        continue
                    blue_states.append(c)
                    if True: # FUTURE support for loop over blue states?
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
                current_candidates : dict[Node, Partitioning]= dict()
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
                if all(part.score is not False for part in current_candidates.values()) == 0:
                    red_states.append(blue_state)
                    self.debug.log_promote(blue_state, red_states)
                    promotion = True
                    break

                # update tracking dict with new candidates
                new_candidates = (((red, blue_state), part) for red, part in current_candidates.items())
                partition_candidates.update(new_candidates)

            # a state was promoted -> don't clear candidates
            if promotion:
                continue

            # find best partitioning and clear candidates
            (red, blue), best_candidate = max(partition_candidates.items(), key = lambda part : part[1].score)
            # FUTURE: optimizations for compatibility tests where merges can be orthogonal
            # FUTURE: caching for aggregating compatibility tests
            partition_candidates.clear()
            for real_node, partition_node in best_candidate.partitions.items():
                    real_node.transitions = partition_node.transitions
            self.debug.log_merge(red, blue)

        self.debug.learning_done(red_states, start_time)

        return self.root.to_automaton(self.output_behavior, self.transition_behavior)

    def _check_futures(self, red: Node, blue: Node) -> bool:
        q : List[Tuple[Node, Node, Any]] = [(red, blue, None)]

        while len(q) != 0:
            red, blue, info = q.pop(0)

            info = self.info_update(red, blue, info)
            if not self.compute_local_score(red, blue, info):
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
                        q.append((red_child.original_target, blue_child.original_target, info))
                    else:
                        q.append((red_child.target, blue_child.target, info))

        return True

    def _partition_from_merge(self, red: Node, blue: Node) -> Partitioning :
        """
        Compatibility check based on partitions.
        assumes that blue is a tree and red is not in blue
        """

        partitions = dict()
        partitioning = Partitioning(False, dict())

        if self.compatibility_behavior == "future":
            score = self._check_futures(red, blue)
            if score is False:
                return partitioning

        def update_partition(red_node: Node, blue_node: Node) -> Node:
            if self.compatibility_behavior == "future":
                return red_node
            if red_node not in partitions:
                p = red_node.shallow_copy()
                partitions[red_node] = p
                partitioning.partitions[red_node] = p
            else:
                p = partitions[red_node]
            if blue_node is not None:
                partitions[blue_node] = p
            return p

        # rewire the blue node's parent
        blue_parent = update_partition(self.root.get_by_prefix(blue.prefix[:-1]), None)
        blue_parent.transitions[blue.prefix[-1][0]][blue.prefix[-1][1]].target = red

        q : List[Tuple[Node,Node,Any]] = [(red, blue, None)]

        while len(q) != 0:
            red, blue, info = q.pop(0)
            partition = update_partition(red, blue)

            if self.compatibility_behavior == "partition":
                info = self.info_update(partition, blue, info)
                if not self.compute_local_score(partition, blue, info) :
                    return partitioning

            for in_sym, blue_transitions in blue.transitions.items():
                partition_transitions = partition.get_transitions_safe(in_sym)
                for out_sym, blue_transition in blue_transitions.items():
                    partition_transition = partition_transitions.get(out_sym)
                    if partition_transition is not None:
                        q.append((partition_transition.target, blue_transition.target, info))
                        partition_transition.count += blue_transition.count
                    else:
                        # blue_child is blue after merging if there is a red state in blue's partition
                        partition_transition = TransitionInfo(blue_transition.target, blue_transition.count, None, 0)
                        partition_transitions[out_sym] = partition_transition

        if self.compatibility_behavior == "merge":
            for new_node, old_node in partitions.items():
                info = self.info_update(new_node, old_node, None)
                if not self.compute_local_score(new_node, old_node, info):
                    return partitioning
        partitioning.score = True
        return partitioning


# TODO nicer interface?
def runGSM(
        data, output_behavior : OutputBehavior = "moore",
        transition_behavior : TransitionBehavior = "deterministic",
        compatibility_behavior : CompatibilityBehavior = "partition",
        local_score : ScoreFunction = None, info_update : Callable[[Node, Node, Any],Any] = None,
        eval_compat_on_pta : bool = False, node_order : Callable[[Node, Node], bool] = None, debug_lvl=0
    ) :
    return GeneralizedStateMerging(data, output_behavior, transition_behavior, compatibility_behavior, local_score, info_update, eval_compat_on_pta, node_order, debug_lvl).run()


def runAlergia(data, output_behavior : OutputBehavior = "moore", epsilon : float = 0.005) :
    return GeneralizedStateMerging(
        data, output_behavior, "stochastic", "future",
        hoeffding_compatibility(epsilon), eval_compat_on_pta=True, debug_lvl=1
    ).run()