import pickle


class RpniNode:
    __slots__ = ['output', 'children', 'prefix']

    def __init__(self, output):
        self.output = output
        self.children = dict()
        self.prefix = ()

    def copy(self):
        return pickle.loads(pickle.dumps(self, -1))

    def __lt__(self, other):
        return len(self.prefix) < len(other.prefix)

    def __le__(self, other):
        return len(self.prefix) <= len(other.prefix)

    def __eq__(self, other):
        return self.prefix == other.prefix


def check_sequence(root_node, seq, automaton_type):
    """
    Checks whether each sequence in the dataset is valid in the current automaton.
    """
    curr_node = root_node
    for i, o in seq:
        if automaton_type == 'mealy':
            input_outputs = {i: o for i, o in curr_node.children.keys()}
            if i[0] not in input_outputs.keys() or o is not None and input_outputs[i[0]] != o:
                return False
            curr_node = curr_node.children[(i[0], input_outputs[i[0]])]
        else:
            # For dfa and moore, check if outputs are the same, iff output in test data is concrete (not None)
            curr_node = curr_node.children[i]
            if o is not None and curr_node.output != o:
                return False
    return True


def createPTA(data, automaton_type):
    data.sort(key=lambda x: len(x[0]))

    root_node = RpniNode(None)
    for seq, label in data:
        curr_node = root_node
        for i in seq:
            if i not in curr_node.children.keys():
                node = RpniNode(None)
                node.prefix = curr_node.prefix + (i,)
                curr_node.children[i] = node

            curr_node = curr_node.children[i]
        if curr_node.output is None:
            curr_node.output = label
        else:
            # check for non-determinism
            if curr_node.output != label:
                return None

    # Breath first traversal over the automaton to update the edges for RPNI-Mealy
    if automaton_type == 'mealy':
        queue = [root_node]
        root_node.prefix = ()
        while queue:
            node_in_processing = queue.pop(0)
            input_output_edge_keys = dict()
            for input_value, child in node_in_processing.children.items():
                input_output_edge_keys[(input_value, child.output)] = child
                child.prefix = node_in_processing.prefix + ((input_value, child.output),)
                queue.append(child)
            node_in_processing.children = input_output_edge_keys

    return root_node


def extract_unique_sequences(root_node):
    def get_leaf_nodes(root):
        leaves = []

        def _get_leaf_nodes(node):
            if node is not None:
                if len(node.children.keys()) == 0:
                    leaves.append(node)
                for n in node.children.values():
                    _get_leaf_nodes(n)

        _get_leaf_nodes(root)
        return leaves

    leaf_nodes = get_leaf_nodes(root_node)
    paths = []
    for node in leaf_nodes:
        seq = []
        curr_node = root_node
        for i in node.prefix:
            curr_node = curr_node.children[i]
            seq.append((i, curr_node.output))
        paths.append(seq)

    return paths


def to_automaton(red, automaton_type):
    from aalpy.automata import DfaState, Dfa, MooreMachine, MooreState, MealyMachine, MealyState

    if automaton_type == 'dfa':
        state, automaton = DfaState, Dfa
    elif automaton_type == 'moore':
        state, automaton = MooreState, MooreMachine
    else:
        state, automaton = MealyState, MealyMachine

    initial_state = None
    prefix_state_map = {}
    for i, r in enumerate(red):
        if automaton_type != 'mealy':
            prefix_state_map[r.prefix] = state(f's{i}', r.output)
        else:
            prefix_state_map[r.prefix] = state(f's{i}')
        if i == 0:
            initial_state = prefix_state_map[r.prefix]

    for r in red:
        for i, c in r.children.items():
            if automaton_type != 'mealy':
                prefix_state_map[r.prefix].transitions[i] = prefix_state_map[c.prefix]
            else:
                prefix_state_map[r.prefix].transitions[i[0]] = prefix_state_map[c.prefix]
                prefix_state_map[r.prefix].output_fun[i[0]] = i[1]

    return automaton(initial_state, list(prefix_state_map.values()))


def visualize_pta(root_node, path='pta.pdf'):
    from pydot import Dot, Node, Edge
    graph = Dot('fpta', graph_type='digraph')

    graph.add_node(Node(str(root_node.prefix), label=f'{root_node.output}'))

    queue = [root_node]
    visited = set()
    visited.add(root_node.prefix)
    while queue:
        curr = queue.pop(0)
        for i, c in curr.children.items():
            if c.prefix not in visited:
                graph.add_node(Node(str(c.prefix), label=f'{c.output}'))
            graph.add_edge(Edge(str(curr.prefix), str(c.prefix), label=f'{i}'))
            if c.prefix not in visited:
                queue.append(c)
            visited.add(c.prefix)

    graph.add_node(Node('__start0', shape='none', label=''))
    graph.add_edge(Edge('__start0', str(root_node.prefix), label=''))

    graph.write(path=path, format='pdf')
