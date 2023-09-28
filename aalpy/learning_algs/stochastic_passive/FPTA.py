from collections import namedtuple
from functools import total_ordering

TreeNode = namedtuple("TreeNode", ["value", "left", "right"])

@total_ordering
class AlergiaPtaNode:
    __slots__ = ['prefix', 'output', 'input_frequency', 'children', 'state_id', 'children_prob']

    def __init__(self, output, prefix):
        self.prefix = prefix
        self.output = output
        self.input_frequency = dict()
        self.children = dict()
        # # for visualization
        self.state_id = None
        self.children_prob = None

    def successors(self):
        return list(self.children.values())

    def get_input_frequency(self, target_input):
        return sum(freq for (i, o), freq in self.input_frequency.items() if i == target_input)

    def __lt__(self, other):
        s_prefix, o_prefix = self.prefix, other.prefix
        return (len(s_prefix), s_prefix) < (len(o_prefix), o_prefix)

    def __le__(self, other):
        return self < other or self == other

    def __eq__(self, other):
        return self.prefix == other.prefix


def create_fpta(data, automaton_type, optimize_for='accuracy'):
    # in case of single tree
    if optimize_for == 'memory':
        return create_single_fpta(data, automaton_type)

    # in case of SMM, there is no initial input
    seq_iter_index = 0 if automaton_type == 'smm' else 1

    if automaton_type != 'smm':
        # NOTE: This approach with _copy is not optimal, but a big time save from doing deep copy at the end
        root_node, root_copy = AlergiaPtaNode(data[0][0], ()), AlergiaPtaNode(data[0][0], ())
    else:
        root_node, root_copy = AlergiaPtaNode(None, ()), AlergiaPtaNode(None, ())

    for seq in data:
        if automaton_type != 'smm' and seq[0] != root_node.output:
            print('All sequances passed to Alergia should have the same initial output!')
            assert False

        curr_node, curr_copy = root_node, root_copy

        for el in seq[seq_iter_index:]:
            if el not in curr_node.children:
                out = None
                if automaton_type == 'mc':
                    out = el
                if automaton_type == 'mdp':
                    out = el[1]

                reached_node, reached_node_copy = AlergiaPtaNode(out, curr_node.prefix + (el,)), \
                                                  AlergiaPtaNode(out, curr_copy.prefix + (el,))

                curr_node.children[el] = reached_node
                curr_copy.children[el] = reached_node_copy

                curr_node.input_frequency[el] = 0
                curr_copy.input_frequency[el] = 0

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
        root_node = AlergiaPtaNode(data[0][0], ())
    else:
        root_node = AlergiaPtaNode(None, ())

    root_node.prefix = ()

    for seq in data:
        if not_smm and seq[0] != root_node.output:
            print('All sequances passed to Alergia should have the same initial output!')
            assert False
        curr_node = root_node

        for el in seq[seq_iter_index:]:
            if el not in curr_node.children.keys():
                out = None
                if automaton_type == 'mc':
                    out = el
                if automaton_type == 'mdp':
                    out = el[1]

                reached_node = AlergiaPtaNode(out, curr_node.prefix + (el,))

                curr_node.children[el] = reached_node

                reached_node.prefix = curr_node.prefix + (el,)

                curr_node.input_frequency[el] = 0

            curr_node.input_frequency[el] += 1
            curr_node = curr_node.children[el]

    return None, root_node
