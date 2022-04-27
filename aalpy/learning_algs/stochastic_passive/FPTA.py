from collections import defaultdict


class AlergiaPtaNode:
    __slots__ = ['output', 'input_frequency', 'children', 'parent_io', 'state_id', 'children_prob']

    def __init__(self, output):
        self.output = output
        self.input_frequency = defaultdict(int)
        self.children = dict()
        self.parent_io = None
        # # for visualization
        self.state_id = None
        self.children_prob = None

    def getPrefix(self):
        prefix = ()
        curr_node = self
        while curr_node.parent_io is not None:
            prefix = (curr_node.parent_io[1],) + prefix
            curr_node = curr_node.parent_io[0]
        return prefix

    def successors(self):
        return list(self.children.values())

    def __lt__(self, other):
        return len(self.getPrefix()) < len(other.getPrefix())

    def __le__(self, other):
        return len(self.getPrefix()) <= len(other.getPrefix())

    def __eq__(self, other):
        return self.getPrefix() == other.getPrefix()


def create_fpta(data, automaton_type, optimize_for='accuracy'):
    # in case of single tree
    if optimize_for == 'memory':
        return create_single_fpta(data, automaton_type)

    is_iofpta = True if automaton_type != 'mc' else False
    seq_iter_index = 0 if automaton_type == 'smm' else 1
    not_smm = automaton_type != 'smm'
    # NOTE: This approach with _copy is not optimal, but a big time save from doing deep copy at the end
    if automaton_type != 'smm':
        root_node, root_copy = AlergiaPtaNode(data[0][0]), AlergiaPtaNode(data[0][0])
    else:
        root_node, root_copy = AlergiaPtaNode(None), AlergiaPtaNode(None)

    root_node.parent_io, root_copy.parent_io = None, None
    for seq in data:
        if not_smm and seq[0] != root_node.output:
            print('All strings should have the same initial output')
            assert False
        curr_node, curr_copy = root_node, root_copy

        for el in seq[seq_iter_index:]:
            if el not in curr_node.children.keys():
                if not_smm:
                    out = el if not is_iofpta else el[1]
                    node, node_copy = AlergiaPtaNode(out), AlergiaPtaNode(out)
                else:
                    node, node_copy = AlergiaPtaNode(None), AlergiaPtaNode(None)

                node.parent_io, node_copy.parent_io = (curr_node, el), (curr_copy, el)

                curr_node.children[el] = node
                curr_copy.children[el] = node_copy

            curr_node.input_frequency[el] += 1
            curr_node = curr_node.children[el]

            curr_copy.input_frequency[el] += 1
            curr_copy = curr_copy.children[el]

    return root_node, root_copy


def create_single_fpta(data, automaton_type):
    is_iofpta = True if automaton_type != 'mc' else False
    seq_iter_index = 0 if automaton_type == 'smm' else 1
    not_smm = automaton_type != 'smm'
    # NOTE: This approach with _copy is not optimal, but a big time save from doing deep copy at the end
    if automaton_type != 'smm':
        root_node = AlergiaPtaNode(data[0][0])
    else:
        root_node = AlergiaPtaNode(None)

    root_node.parent_io = None

    for seq in data:
        if not_smm and seq[0] != root_node.output:
            print('All strings should have the same initial output')
            assert False
        curr_node = root_node

        for el in seq[seq_iter_index:]:
            if el not in curr_node.children.keys():
                if not_smm:
                    out = el if not is_iofpta else el[1]
                    node = AlergiaPtaNode(out)
                else:
                    node = AlergiaPtaNode(None)

                node.parent_io = (curr_node, el)

                curr_node.children[el] = node

            curr_node.input_frequency[el] += 1
            curr_node = curr_node.children[el]

    return None, root_node
