import copy
from queue import Queue
import time
from typing import Dict, Optional, Tuple

from aalpy.learning_algs.non_deterministic_passive.rpni_helper_functions import Node


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
            print(f'\nRPNI-GSM Learning Time: {round(time.time() - start_time, 2)}')
            print(f'RPNI-GSM Learned {len(red_states)} state automaton.')
            self.instance.root.visualize("model", self.instance.data, self.pta)
            #self.pta.visualize("pta")

    def __init__(self, data, debug_lvl=0):
        self.data = data
        self.debug = GeneralizedStateMerging.DebugInfo(debug_lvl, self)

        pta_construction_start = time.time()
        self.root : Node = copy.deepcopy(data) if isinstance(data, Node) else Node.createPTA(data)
        self.root.prune(20)
        self.debug.pta_construction_done(pta_construction_start)

    def run_rpni(self):
        start_time = time.time()

        # sorted list of states already considered
        red_states = [self.root]
        # used to get the minimal non-red state
        blue_states = list(self.root.transitions.values())

        while blue_states:
            blue_state = min(list(blue_states), key=lambda x: len(x.prefix))
            partition = None
            for red_state in red_states:
                partition = self._partition_from_merge(red_state, blue_state)
                if partition is not None:
                    break

            if partition is None:
                red_states.append(blue_state)
                self.debug.log_promote(blue_state, red_states)
            else:
                self.debug.log_merge(red_state, blue_state)

                # use the partition for merging
                for node in partition.keys():
                    block = partition[node]
                    node.transitions = block.transitions

            blue_states.clear()
            for r in red_states:
                for c in r.transitions.values():
                    if c not in red_states:
                        blue_states.append(c)

        self.debug.learning_done(red_states, start_time)

        return self.root.to_automaton()

    def _partition_from_merge(self, red: Node, blue: Node) -> Optional[Dict[Node, Node]] :
        """
        Compatibility check based on partitions
        """

        partitions = dict()

        def get_partition(node: Node) -> Node:
            if node not in partitions:
                p = node.shallow_copy()
                partitions[node] = p
            else:
                p = partitions[node]
            return p

        node = get_partition(self.root.get_by_prefix(blue.prefix[:-1]))
        node.transitions[blue.prefix[-1]] = red

        q = Queue[Tuple[Node,Node]]()
        q.put((red, blue))

        while not q.empty():
            red, blue = q.get()
            partition = get_partition(red)

            if not Node.compatible_outputs(partition, blue) or (0 < Node.nondeterministic_additions(partition, blue) and 0 < Node.nondeterministic_additions(blue, partition)) :
                return None

            partitions[blue] = partition

            for sym_pair, blue_child in blue.transitions.items():
                if sym_pair in partition.transitions.keys():
                    q.put((partition.transitions[sym_pair], blue_child))
                else:
                    # blue_child is blue after merging if there is a red state in the partition
                    partition.transitions[sym_pair] = blue_child
        return partitions
