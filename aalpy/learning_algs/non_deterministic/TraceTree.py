from collections import defaultdict

from aalpy.base import SUL


class SULWrapper(SUL):
    """ """
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

        Args:
          inp: 
          out: 

        Returns:

        """
        if inp not in self.curr_node.children.keys() or out not in [child.output for child in self.curr_node.children[inp]]:
            node = Node(out)
            self.curr_node.children[inp].append(node)
        else:
            self.curr_node = self.curr_node.get_child(inp, out)

    def get_all_outputs(self, inputs, outputs, e):
        """

        Args:
          inputs: 
          outputs: 
          e: 

        Returns:

        """
        e = list(e)
        curr_node = self.root_node
        for i, o in zip(inputs, outputs):
            node = curr_node.get_child(i, o)
            if node is None:
                return None
            curr_node = node

        # from this point all record all possible outputs
        nodes_to_process = [curr_node]
        # TODO RETURN WHOLE TRACES NOT JUST SINGLE OUTPUTS
        while True:
            if not e:
                return tuple(node.output for node in nodes_to_process if node.output)
            i = e.pop(0)
            children = []
            for node in nodes_to_process:
                if node.children[i]:
                    children.extend(node.children[i])
            nodes_to_process = children
