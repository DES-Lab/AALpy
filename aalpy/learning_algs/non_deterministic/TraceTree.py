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
        """ """
        self.pta.reset()
        self.sul.pre()

    def post(self):
        """ """
        self.sul.post()

    def step(self, letter):
        """

        Args:
          letter:

        Returns:

        """
        out = self.sul.step(letter)
        self.pta.add_to_tree(letter, out)
        return out


class Node:
    """ """

    def __init__(self, output):
        self.output = output
        self.children = defaultdict(list)

    def get_child(self, inp, out):
        """
        Args:
          inp:
          out:

        Returns:

        """
        return next((child for child in self.children[inp] if child.output == out), None)


class TraceTree:
    """ """

    def __init__(self):
        self.root_node = Node(None)
        self.curr_node = None

    def reset(self):
        """ """
        self.curr_node = self.root_node

    def add_to_tree(self, inp, out):
        """
        Adds new element to tree and makes it the current node

        Args:
          inp: Inputs
          out: Outputs

        Returns:
        """
        if inp not in self.curr_node.children.keys() or out not in [child.output for child in
                                                                    self.curr_node.children[inp]]:
            node = Node(out)
            self.curr_node.children[inp].append(node)
        # This was in an else statement. But that seems wrong.
        self.curr_node = self.curr_node.get_child(inp, out)

    def get_to_node(self, inp, out):
        """
        Follows the path described by inp and out and returns the node which is reached

        Args:
          inp: Inputs
          out: Outputs

        Returns:
          Node that is reached when following the given input and output through the tree
        """
        curr_node = self.root_node
        for i, o in zip(inp, out):
            node = curr_node.get_child(i, o)
            if node is None:
                return None
            curr_node = node

        return curr_node

    def get_single_trace(self, curr_node=None, e=None):
        """
        Args:
          curr_node: current node
          e: List of inputs

        Returns:
          Traces of outputs corresponding to the input-sequence given by e
        """

        if not curr_node or not e:
            return []

        e = list(e)
        size_e = len(e)
        single_input = e.pop(0)
        children_with_same_input = curr_node.children[single_input]
        number_of_children = len(children_with_same_input)
        result = []

        if number_of_children == 0:
            return []

        for child_index, child in enumerate(children_with_same_input):
            result.append([child.output])

        for child_index, child in enumerate(children_with_same_input):
            following_traces = self.get_single_trace(child, e)
            number_of_following_traces = len(following_traces)

            for trace_index, trace in enumerate(following_traces):
                if trace_index < number_of_following_traces - 1:
                    result.append(result[child_index] + list(trace))
                else:
                    result[child_index].extend(trace)

        for i in range(0, len(result)):
            if len(result[i]) != size_e:
                return []
            result[i] = tuple(result[i])

        return result

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
                result[pair][inp] = self.get_single_trace(curr_node, inp)

        return result

    def print_trace_tree(self, curr=None, depth=0, curr_str=""):
        """
        Prints trace tree

        Args:
          curr: current node. normally root of tree, but can be any node
          depth: needed for recursion. should not be changed.
          curr_str: needed for recursion. should not be changed.

        Returns:
        """

        if curr is None and depth == 0:
            curr = self.root_node
            print("()")

        curr_str = curr_str + " ├─ "

        # go through all inputs
        for i, node in enumerate(list(curr.children.keys())):

            # go through all outputs of a single input
            for c in range(0, len(curr.children[node])):

                # if it is the last output of the last input on the current level
                if i == len(list(curr.children.keys())) - 1 and c == len(curr.children[node]) - 1:
                    curr_str = list(curr_str)[:-4]
                    curr_str = ''.join(curr_str) + " └─ "
                elif c <= len(curr.children[node]) - 1:
                    curr_str = list(curr_str)[:-4]
                    curr_str = ''.join(curr_str) + " ├─ "

                print(curr_str + node, curr.children[node][c].output)

                # if it is the last output of the last input on the current level
                if i == len(list(curr.children.keys())) - 1 and c == len(curr.children[node]) - 1:
                    curr_str = list(curr_str)[:-4]
                    curr_str = ''.join(curr_str) + "    "

                else:
                    curr_str = list(curr_str)[:-4]
                    curr_str = ''.join(curr_str) + " |  "

                self.print_trace_tree(curr.children[node][c], depth + 1, curr_str)
