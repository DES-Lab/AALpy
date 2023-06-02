import copy
from math import sqrt, log
from queue import Queue
import time
from typing import Dict, Tuple, Callable

from aalpy.learning_algs.stochastic_passive.rpni_helper_functions import Node, OutputBehavior, TransitionBehavior

Score = bool
ScoreFunction = Callable[[Node,Node], Score]
def hoeffding_compatibility(eps) -> ScoreFunction:
    def similar(a: Node, b: Node):
        for in_sym in filter(lambda x : x in a.transitions.keys(), b.transitions.keys()):
            a_count, b_count = (x.transition_count[in_sym] for x in [a,b])
            a_total, b_total = (sum(x.values()) for x in [a_count, b_count])
            if a_total == 0 or b_total == 0:
                continue
            for out_sym in set(a.transitions[in_sym].keys()).union(b.transitions[in_sym].keys()):
                if abs(a_count[out_sym] / a_total - b_count[out_sym] / b_total) > ((sqrt(1 / a_total) + sqrt(1 / b_total)) * sqrt(0.5 * log(2 / eps))):
                    return False
        return True
    return similar

class DebugInfo:
    def __init__(self, lvl):
        self.lvl = lvl

    @staticmethod
    def level_required(lvl):
        def decorator(fn):
            from functools import wraps
            @wraps(fn)
            def wrapper(*args, **kw):
                if args[0].lvl < lvl:
                    return
                fn(*args, **kw)
            return wrapper
        return decorator

class GeneralizedStateMerging:
    class DebugInfo(DebugInfo):
        lvl_required = DebugInfo.level_required

        def __init__(self, lvl, instance):
            super().__init__(lvl)
            if lvl < 1:
                return
            self.instance = instance
            self.log = []
            self.pta = None

        @lvl_required(1)
        def pta_construction_done(self, start_time):
            print(f'PTA Construction Time: {round(time.time() - start_time, 2)}')
            self.pta = copy.deepcopy(self.instance.root)
            states = self.instance.root.get_all_nodes()
            leafs = [state for state in states if len(state.transitions.keys()) == 0]
            depth = [len(state.prefix) for state in leafs]
            print(f'PTA has {len(states)} states leading to {len(leafs)} leafs')
            print(f'min / avg / max depth : {min(depth)} / {sum(depth) / len(depth)} / {max(depth)}')

        @lvl_required(1)
        def log_promote(self, node : Node, red_states):
            self.log.append(["promote", (node.prefix,)])
            print(f'\rCurrent automaton size: {len(red_states)}', end="")

        @lvl_required(1)
        def log_merge(self, a : Node, b : Node):
            self.log.append(["merge", (a.prefix, b.prefix)])

        @lvl_required(1)
        def learning_done(self, red_states, start_time):
            print(f'\nLearning Time: {round(time.time() - start_time, 2)}')
            print(f'Learned {len(red_states)} state automaton.')
            self.instance.root.visualize("model.pdf")
            #self.pta.visualize("pta")

    def __init__(self, data, output_behavior : OutputBehavior = "moore",
                 transition_behavior : TransitionBehavior = "deterministic",
                 local_score : ScoreFunction = None, update_count : bool = False, debug_lvl=0):
        self.data = data
        self.debug = GeneralizedStateMerging.DebugInfo(debug_lvl, self)
        self.output_behavior : OutputBehavior = output_behavior
        self.transition_behavior : TransitionBehavior = transition_behavior

        if local_score is None:
            if output_behavior == "deterministic" :
                local_score = lambda x,y : True
            else :
                local_score = hoeffding_compatibility(0.005)
        self.local_score : ScoreFunction = local_score
        self.update_transition_count = update_count

        pta_construction_start = time.time()
        self.root: Node
        if isinstance(data, Node):
            self.root = copy.deepcopy(data)
        elif output_behavior == "moore":
            self.root = Node.createPTA([d[1:] for d in data], data[0][0])
        else :
            self.root = Node.createPTA(data)
        self.debug.pta_construction_done(pta_construction_start)

        if transition_behavior == "deterministic":
            if not self.root.is_deterministic():
                raise ValueError("required deterministic automaton but input data is nondeterministic")

    def local_merge_score(self, a : Node, b : Node):
        if self.output_behavior == "moore" and not Node.moore_compatible(a,b):
            return False
        if self.transition_behavior == "deterministic" and not Node.deterministic_compatible(a,b):
            return False
        return self.local_score(a,b)

    def run(self):
        start_time = time.time()

        # sorted list of states already considered
        red_states = [self.root]
        # used to get the minimal non-red state
        blue_states = list(child for _, child in self.root.transition_iterator())

        while blue_states:
            blue_state = min(blue_states, key=lambda x: len(x.prefix))
            partition = None
            for red_state in red_states:
                score, partition = self._partition_from_merge(red_state, blue_state)
                if score:
                    break

            if not score:
                red_states.append(blue_state)
                self.debug.log_promote(blue_state, red_states)
            else:
                self.debug.log_merge(red_state, blue_state)

                # use the partition for merging
                for node in partition.keys():
                    block = partition[node]
                    node.transitions = block.transitions
                    if self.update_transition_count:
                        node.transition_count = block.transition_count

            blue_states.clear()
            for r in red_states:
                for _, c in r.transition_iterator():
                    if c not in red_states:
                        blue_states.append(c)

        self.debug.learning_done(red_states, start_time)

        return self.root.to_automaton(self.output_behavior, self.transition_behavior)

    def _partition_from_merge(self, red: Node, blue: Node) -> Tuple[bool,Dict[Node, Node]] :
        """
        Compatibility check based on partitions
        """

        partitions = dict()
        remaining_nodes = dict()

        def get_partition(node: Node) -> Node:
            if node not in partitions:
                p = node.shallow_copy()
                partitions[node] = p
                remaining_nodes[node] = p
            else:
                p = partitions[node]
            return p

        node = get_partition(self.root.get_by_prefix(blue.prefix[:-1]))
        node.transitions[blue.prefix[-1][0]][blue.prefix[-1][1]] = red

        q = Queue[Tuple[Node,Node]]()
        q.put((red, blue))

        while not q.empty():
            red, blue = q.get()
            partition = get_partition(red)

            if not self.local_merge_score(partition, blue) :
                return False, dict()

            partitions[blue] = partition

            for in_sym, blue_transitions in blue.transitions.items():
                partition_transitions = partition.transitions[in_sym]
                for out_sym, blue_child in blue_transitions.items():
                    if out_sym in partition_transitions:
                        q.put((partition_transitions[out_sym], blue_child))
                    else:
                        # blue_child is blue after merging if there is a red state in the partition
                        partition_transitions[out_sym] = blue_child
                    partition.transition_count[in_sym][out_sym] += blue.transition_count[in_sym][out_sym]
        return True, remaining_nodes
