from collections import defaultdict

from aalpy.base import SUL


class SULWrapper(SUL):
    """
    Wrapper for non-deterministic SUL. After every step, input/output pair is added to the tree containing all traces.
    """

    def __init__(self, sul: SUL):
        super().__init__()
        self.sul = sul
        self.pta = TraceTree()

    def pre(self):
        self.pta.reset()
        self.sul.pre()

    def post(self):
        self.sul.post()

    def step(self, letter):
        out = self.sul.step(letter)
        self.pta.add_to_tree(letter, out)
        return out


class Node:
    __slots__ = ['output', 'children', 'parent']

    def __init__(self, output):
        self.output = output
        self.children = defaultdict(list)
        self.parent = None

    def get_child(self, inp, out):
        """
        Args:
          inp:
          out:

        Returns:

        """
        return next((child for child in self.children[inp] if child.output == out), None)

    def get_prefix(self):
        prefix = ()
        curr_node = self
        while curr_node.parent is not None:
            prefix = (curr_node.output,) + prefix
            curr_node = curr_node.parent
        return prefix


class TraceTree:
    """
    Tree used for keeping track of seen observations.
    """

    def __init__(self):
        self.root_node = Node(None)
        self.curr_node = None

    def reset(self):
        self.curr_node = self.root_node

    def add_to_tree(self, inp, out):
        """
        Adds new element to tree and makes it the current node

        Args:

          inp: Input
          out: Output

        """
        if inp not in self.curr_node.children.keys() or \
                out not in {child.output for child in self.curr_node.children[inp]}:
            node = Node(out)
            self.curr_node.children[inp].append(node)
            node.parent = self.curr_node
        self.curr_node = self.curr_node.get_child(inp, out)

    def get_to_node(self, inputs, outputs):
        """
        Follows the path described by inp and out and returns the node which is reached

        Args:
          inputs: Inputs
          outputs: Outputs

        Returns:

          Node that is reached when following the given input and output through the tree
        """
        curr_node = self.root_node
        for i, o in zip(inputs, outputs):
            node = curr_node.get_child(i, o)
            if node is None:
                assert False
            curr_node = node

        return curr_node

    def get_all_traces(self, curr_node=None, e=None):
        """

        Args:

          curr_node: current node
          e: List of inputs

        Returns:

          Traces of outputs corresponding to the input-sequence given by e
        """

        if not curr_node or not e:
            return []

        queue = [(curr_node, 0)]
        reached_nodes = []
        while queue:
            node, depth = queue.pop(0)
            if depth == len(e):
                reached_nodes.append(node)
            else:
                children_with_same_input = node.children[e[depth]]
                for c in children_with_same_input:
                    queue.append((c, depth + 1))

        cell = [node.get_prefix()[-len(e):] for node in reached_nodes]
        return cell

    def get_table(self, s, e):
        """
        Generates a table from the tree

        Args:
          s: rows from S, S_dot_A, or both which should be presented in the table.
          e: E

        Returns:
          a table in a format that can be used for printing.
        """
        result = {}
        for pair in s:
            curr_node = self.get_to_node(pair[0], pair[1])
            result[pair] = {}

            for inp in e:
                result[pair][inp] = self.get_all_traces(curr_node, inp)

        return result
