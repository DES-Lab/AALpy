from aalpy.automata import DfaState, Dfa
from aalpy.base import SUL
from aalpy.learning_algs.deterministic.KV_helpers import state_name_gen


class CTNode:
    def __init__(self, parent):
        self.parent = parent

    @property
    def root(self):
        p = self.parent
        while p:
            p = p.parent
        return p

    @property
    def path_to_node(self):
        return self.parent.children[True] == self if self.parent else None


class CTInternalNode(CTNode):
    def __init__(self, distinguishing_string: tuple, parent: 'CTInternalNode'):
        super().__init__(parent)
        self.distinguishing_string = distinguishing_string
        self.children = {True: None, False: None}

    def __repr__(self):
        return f"{self.__class__.__name__} '{self.distinguishing_string}'"


class CTLeafNode(CTNode):
    def __init__(self, access_string: tuple, parent: 'CTInternalNode', tree: 'ClassificationTree'):
        super().__init__(parent)
        self.access_string = access_string
        self.tree = tree
        assert access_string not in tree.leaf_nodes, f"a leaf node with {access_string=} already exists!"
        tree.leaf_nodes[access_string] = self

    def __repr__(self):
        return f"{self.__class__.__name__} '{self.access_string}'"

    @property
    def in_right_side(self):
        c = self
        p = self.parent
        while p.parent:
            c = p
            p = p.parent
        return p.children[True] == c

class ClassificationTree:
    def __init__(self, alphabet: list, sul: SUL, cex: tuple, empty_is_true: bool):
        self.root = CTInternalNode(distinguishing_string=(None,), parent=None)
        self.leaf_nodes = {}
        self.root.children[empty_is_true] = CTLeafNode(access_string=(None,),
                                                       parent=self.root,
                                                       tree=self)
        self.root.children[not empty_is_true] = CTLeafNode(access_string=cex,
                                                           parent=self.root,
                                                           tree=self)
        self.alphabet = alphabet
        self.sul = sul

    def sift(self, word):
        '''
        Sifting a word into the classification tree.
        Starting at the root, at every inner node (a CTInternalNode),
        we branch into the "true" or "false" child, depending on the result of the
        membership query (word * node.distinguishing_string). Repeated until a leaf
        (a CTLeafNode) is reached, which is the result of the sifting.

        Args:

            word: the word to sift into the discrimination tree (a tuple of all letters)

        Returns:

            the CTLeafNode that is reached by the sifting operation.
        '''
        for letter in word:
            assert letter is None or letter in self.alphabet

        node = self.root

        while isinstance(node, CTInternalNode):
            mq_result = self.sul.query((*word, *node.distinguishing_string))[-1]
            node = node.children[mq_result]

        assert isinstance(node, CTLeafNode)
        return node.access_string

    def gen_hypothesis(self):

        # for each CTLeafNode of this CT,
        # create a state in the hypothesis that is labeled by that
        # node's access string. The start state is the empty word
        states = {}
        initial_state = None
        for node in self.leaf_nodes.values():
            new_state = DfaState(state_id=node.access_string,
                                 is_accepting=node.in_right_side)
            if new_state.state_id == (None,):
                initial_state = new_state
            states[new_state.state_id] = new_state
        assert initial_state is not None

        # For each access state s of the hypothesis and each letter b in the
        # alphabet, compute the b-transition out of state s by sifting s.state_id*b
        for state in states.values():
            for letter in self.alphabet:
                transition_target_id = self.sift((*state.state_id, letter))
                state.transitions[letter] = states[transition_target_id]

        return Dfa(initial_state=initial_state,
                   states=list(states.values()))

    # def least_common_ancestor(self, node_1_id, node_2_id):
    #     def _lca(root, n1, n2):
    #         if isinstance(root, CTInternalNode):
    #             if root



        # return _lca(self.root, node_1_id, node_2_id).distinguishing_string

    def least_common_ancestor(self, node_1_id, node_2_id):
        '''
        Find the distinguishing string of the least common ancestor
        of the leaf nodes node_1 and node_2. Both nodes have to exist.
        Adapted from https://www.geeksforgeeks.org/lowest-common-ancestor-binary-tree-set-1/

        Args:

            node_1_id: first leaf node's id
            node_2_id: second leaf node's id

        Returns:

            the distinguishing string of the lca

        '''

        def findLCA(root, n1, n2):

            # Base Case
            if root is None:
                return None

            # If either n1 or n2 matches with root's key, report
            #  the presence by returning root (Note that if a key is
            #  ancestor of other, then the ancestor key becomes LCA
            if isinstance(root, CTLeafNode) and (root.access_string == n1 or root.access_string == n2):
                return root.parent

            # Look for keys in left and right subtrees
            left_lca = findLCA(root.children[False], n1, n2) if isinstance(root, CTInternalNode) else None
            right_lca = findLCA(root.children[True], n1, n2) if isinstance(root, CTInternalNode) else None

            # If both of the above calls return Non-NULL, then one key
            # is present in once subtree and other is present in other,
            # So this node is the LCA
            if left_lca and right_lca:
                return root

            # Otherwise check if left subtree or right subtree is LCA
            return left_lca if left_lca is not None else right_lca

        return findLCA(self.root, node_1_id, node_2_id).distinguishing_string

