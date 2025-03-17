import time
from collections import deque

from aalpy.learning_algs.deterministic_passive.rpni_helper_functions import to_automaton, RpniNode, createPTA


class GsmRPNI:
    def __init__(self, data, automaton_type, print_info=True):
        self.data = data
        self.final_automaton_type = automaton_type
        self.automaton_type = automaton_type if automaton_type != 'dfa' else 'moore'
        self.print_info = print_info

        pta_construction_start = time.time()
        self.root_node = createPTA(data, self.automaton_type)
        self.log = []

        if self.print_info:
            print(f'PTA Construction Time: {round(time.time() - pta_construction_start, 2)}')

    def run_rpni(self):
        start_time = time.time()

        # sorted list of states already considered
        red_states = [self.root_node]
        # used to get the minimal non-red state
        blue_states = list(red_states[0].children.values())

        while blue_states:
            blue_state = min(list(blue_states))

            partition = None
            red_state = None
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
                    # assert RpniNode.compatible(node, block)
                    node.output = block.output
                    node.children = block.children

                node = self.root_node.get_child_by_prefix(blue_state.prefix[:-1])
                node.children[blue_state.prefix[-1]] = red_state

            blue_states.clear()
            for r in red_states:
                for c in r.children.values():
                    if c not in red_states:
                        blue_states.append(c)

        if self.print_info:
            print(f'\nRPNI-GSM Learning Time: {round(time.time() - start_time, 2)}')
            print(f'RPNI-GSM Learned {len(red_states)} state automaton.')

        return to_automaton(red_states, self.final_automaton_type)

    def _partition_from_merge(self, red: RpniNode, blue: RpniNode):
        """
        Compatibility check based on partitions
        """

        partitions = dict()
        q = deque()
        q.append((red, blue))

        while len(q) != 0:
            red, blue = q.popleft()

            def get_partition(node: RpniNode):
                if node not in partitions:
                    p = node.shallow_copy()
                    partitions[node] = p
                else:
                    p = partitions[node]
                return p

            partition = get_partition(red)

            if not RpniNode.compatible_outputs(partition, blue):
                return None
            if self.automaton_type == 'moore' and partition.output is None:
                partition.output = blue.output
            if self.automaton_type == 'mealy':
                for key in filter(lambda k: k not in partition.output or partition.output[k] is None, blue.output):
                    partition.output[key] = blue.output[key]

            partitions[blue] = partition

            for symbol, blue_child in blue.children.items():
                if symbol in partition.children.keys():
                    q.append((partition.children[symbol], blue_child))
                else:
                    # blue_child is blue after merging if there is a red state in the partition
                    partition.children[symbol] = blue_child
        return partitions
