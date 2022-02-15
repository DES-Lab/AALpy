from collections import defaultdict


class AlergiaPtaNode:
    __slots__ = ['output', 'input_frequency', 'children', 'prefix', 'state_id', 'children_prob', 'children_keys']

    def __init__(self, output):
        self.output = output
        self.input_frequency = defaultdict(int)
        self.children = dict()
        self.prefix = ()
        # # for visualization
        self.state_id = None
        self.children_prob = None

    def succs(self):
        return list(self.children.values())

    def __lt__(self, other):
        return len(self.prefix) < len(other.prefix)

    def __le__(self, other):
        return len(self.prefix) <= len(other.prefix)

    def __eq__(self, other):
        return self.prefix == other.prefix


def create_fpta(data, automaton_type):
    is_iofpta = True if automaton_type != 'mc' else False
    # NOTE: This approach with _copy is not optimal, but a big time save from doing deep copy at the end
    root_node, root_copy = AlergiaPtaNode(data[0][0]), AlergiaPtaNode(data[0][0])
    for seq in data:
        if seq[0] != root_node.output:
            print('All strings should have the same initial output')
            assert False
        curr_node, curr_copy = root_node, root_copy

        for el in seq[1:]:
            inp_out = el if not is_iofpta else (el[0], el[1])
            out = el if not is_iofpta else el[1]
            if inp_out not in curr_node.children.keys():
                node, node_copy = AlergiaPtaNode(out), AlergiaPtaNode(out)

                node.prefix = tuple(curr_node.prefix)
                node.prefix += (inp_out,)
                node_copy.prefix = node.prefix

                curr_node.children[inp_out] = node
                curr_copy.children[inp_out] = node_copy

            curr_node.input_frequency[inp_out] += 1
            curr_node = curr_node.children[inp_out]

            curr_copy.input_frequency[inp_out] += 1
            curr_copy = curr_copy.children[inp_out]

    return root_node, root_copy
