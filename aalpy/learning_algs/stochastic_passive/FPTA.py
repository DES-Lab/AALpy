from collections import defaultdict
from copy import deepcopy


class AlergiaPtaNode:

    def __init__(self, output):
        self.output = output
        self.input_frequency = defaultdict(int)
        self.children = dict()
        self.prefix = []
        # for visualization
        self.state_id = None
        self.children_prob = dict()

    def succs(self):
        return list(self.children.values())

    def __lt__(self, other):
        return len(self.prefix) < len(other.prefix)

    def __le__(self, other):
        return len(self.prefix) <= len(other.prefix)

    def __eq__(self, other):
        return self.prefix == other.prefix

    def copy(self):
        return deepcopy(self)


def create_fpta(data, automaton_type):
    is_iofpta = True if automaton_type != 'mc' else False
    root_node = AlergiaPtaNode(data[0][0])
    for seq in data:
        if seq[0] != root_node.output:
            print('All strings should have the same initial output')
            assert False
        curr_node = root_node

        for el in seq[1:]:
            inp_out = el if not is_iofpta else (el[0], el[1])

            if inp_out not in curr_node.children.keys():
                node = AlergiaPtaNode(el if not is_iofpta else el[1])
                node.prefix = list(curr_node.prefix)
                node.prefix.append(inp_out)
                curr_node.children[inp_out] = node

            curr_node.input_frequency[inp_out] += 1
            curr_node = curr_node.children[inp_out]

    return root_node, deepcopy(root_node)