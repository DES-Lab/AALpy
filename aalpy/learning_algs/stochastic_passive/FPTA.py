from functools import total_ordering


@total_ordering
class AlergiaPtaNode:
    __slots__ = ['prefix', 'output', 'input_frequency', 'children', 'original_input_frequency',
                 'original_children', 'state_id', 'children_prob']

    def __init__(self, output, prefix):
        self.prefix = prefix
        self.output = output
        # mutable values
        self.input_frequency = dict()
        self.children = dict()
        # immutable values used for statistical computability check
        self.original_input_frequency = dict()
        self.original_children = dict()
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

    def get_immutable_inputs(self):
        return {i for i, _ in self.original_children.keys()}

    def get_immutable_input_frequency(self, target_input):
        return sum(freq for (i, _), freq in self.original_input_frequency.items() if i == target_input)

    def get_original_output_frequencies(self, target_input):
        return {o: freq for (i, o), freq in self.original_input_frequency.items() if i == target_input}

    def __lt__(self, other):
        return (len(self.prefix), self.prefix) < (len(other.prefix), other.prefix)

    def __le__(self, other):
        return self < other or self == other

    def __eq__(self, other):
        return self.prefix == other.prefix


def create_fpta(data, automaton_type):
    # in case of SMM, there is no initial input
    seq_iter_index = 0 if automaton_type == 'smm' else 1

    initial_output = None if automaton_type == 'smm' else data[0][0]

    root_node = AlergiaPtaNode(initial_output, ())

    for seq in data:
        if automaton_type != 'smm' and seq[0] != root_node.output:
            print('All sequances passed to Alergia should have the same initial output!')
            assert False

        curr_node = root_node

        for el in seq[seq_iter_index:]:
            if el not in curr_node.children:
                out = None
                if automaton_type == 'mc':
                    out = el
                elif automaton_type == 'mdp':
                    out = el[1]

                reached_node = AlergiaPtaNode(out, curr_node.prefix + (el,))
                curr_node.children[el] = reached_node
                curr_node.original_children[el] = reached_node

                curr_node.input_frequency[el] = 0
                curr_node.original_input_frequency[el] = 0

            curr_node.input_frequency[el] += 1
            curr_node.original_input_frequency[el] += 1

            curr_node = curr_node.children[el]

    return root_node
