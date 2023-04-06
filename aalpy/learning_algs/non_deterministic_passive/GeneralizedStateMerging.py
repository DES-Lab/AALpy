from queue import Queue
import time
from typing import Dict, Optional, Tuple

from aalpy.learning_algs.non_deterministic_passive.rpni_helper_functions import Node, createPTA

class GeneralizedStateMerging:
    def __init__(self, data, print_info=True):
        self.data = data
        self.print_info = print_info

        pta_construction_start = time.time()
        self.root = createPTA(data)
        self.log = []

        if self.print_info:
            print(f'PTA Construction Time: {round(time.time() - pta_construction_start, 2)}')

    def run_rpni(self):
        start_time = time.time()

        # sorted list of states already considered
        red_states = [self.root]
        # used to get the minimal non-red state
        blue_states = self.root.get_children()

        while blue_states:
            blue_state = min(list(blue_states), key=lambda x: len(x.prefix))
            partition = None
            for red_state in red_states:
                partition = self._partition_from_merge(red_state, blue_state)
                if partition is not None:
                    break

            if partition is None:
                self.log.append(["promote", (blue_state.prefix,)])
                red_states.append(blue_state)
                if self.print_info:
                    print(f'\rCurrent automaton size: {len(red_states)}', end="")
            else:
                self.log.append(["merge", (red_state.prefix, blue_state.prefix)])

                # use the partition for merging
                for node in partition.keys():
                    block = partition[node]
                    node.transitions = block.transitions

                node = self.root.get_by_prefix(blue_state.prefix[:-1])
                in_sym, out_sym = blue_state.prefix[-1]
                node.transitions[in_sym][out_sym] = red_state

            blue_states.clear()
            for r in red_states:
                for c in r.get_children():
                    if c not in red_states:
                        blue_states.append(c)

        if self.print_info:
            print(f'\nRPNI-GSM Learning Time: {round(time.time() - start_time, 2)}')
            print(f'RPNI-GSM Learned {len(red_states)} state automaton.')

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

        q = Queue[Tuple[Node,Node]]()
        q.put((red, blue))

        while not q.empty():
            red, blue = q.get()
            partition = get_partition(red)

            if not Node.compatible_outputs(partition, blue) or 0 < Node.nondeterministic_additions(partition, blue) :
                return None

            partitions[blue] = partition

            for in_sym, blue_opts in blue.transitions.items():
                for out_sym, blue_child in blue_opts.items():
                    if in_sym in partition.transitions.keys() and out_sym in partition.transitions[in_sym]:
                        q.put((partition.transitions[in_sym][out_sym], blue_child))
                    else:
                        # blue_child is blue after merging if there is a red state in the partition
                        partition.transitions[in_sym][out_sym] = blue_child
        return partitions
