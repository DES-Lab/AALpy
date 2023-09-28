from functools import total_ordering


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

    def get_inputs(self):
        return {i for i, _ in self.input_frequency.keys()}

    def get_input_frequency(self, target_input):
        return sum(freq for (i, _), freq in self.input_frequency.items() if i == target_input)

    def get_output_frequencies(self, target_input):
        return {o: freq for (i, o), freq in self.input_frequency.items() if i == target_input}

    def __lt__(self, other):
        return (len(self.prefix), self.prefix) < (len(other.prefix), other.prefix)

    def __le__(self, other):
        return self < other or self == other

    def __eq__(self, other):
        return self.prefix == other.prefix


def create_fpta(data, automaton_type, optimize_for='accuracy'):
    # in case of single tree

    # in case of SMM, there is no initial input
    seq_iter_index = 0 if automaton_type == 'smm' else 1

    initial_output = None if automaton_type == 'smm' else data[0][0]

    # NOTE: This approach with _copy is not optimal, but a big time save from doing deep copy at the end
    root_node = AlergiaPtaNode(initial_output, ())
    root_copy = AlergiaPtaNode(initial_output, ()) if optimize_for == 'accuracy' else None

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

                reached_node = AlergiaPtaNode(out, curr_node.prefix + (el,))
                curr_node.children[el] = reached_node
                curr_node.input_frequency[el] = 0

                if curr_copy:
                    reached_node_copy = AlergiaPtaNode(out, curr_copy.prefix + (el,))
                    curr_copy.children[el] = reached_node_copy
                    curr_copy.input_frequency[el] = 0

            curr_node.input_frequency[el] += 1
            curr_node = curr_node.children[el]

            if curr_copy:
                curr_copy.input_frequency[el] += 1
                curr_copy = curr_copy.children[el]

    return root_node, root_copy
