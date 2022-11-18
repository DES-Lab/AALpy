from aalpy.automata import DfaState, Dfa, MealyState, MealyMachine
from aalpy.base import SUL

from collections import defaultdict

automaton_class = {'dfa': Dfa, 'mealy': MealyMachine}

class CTNode:
    def __init__(self, parent, path_to_node):
        self.parent = parent
        self.path_to_node = path_to_node

    @property
    def root(self):
        p = self.parent
        while p:
            p = p.parent
        return p



class CTInternalNode(CTNode):
    def __init__(self, distinguishing_string: tuple, parent: 'CTInternalNode', path_to_node):
        super().__init__(parent, path_to_node)
        self.distinguishing_string = distinguishing_string
        self.children = defaultdict(lambda: None) #{True: None, False: None}
        self.query_cache = dict()

    def __repr__(self):
        return f"{self.__class__.__name__} '{self.distinguishing_string}'"


class CTLeafNode(CTNode):
    def __init__(self, access_string: tuple, parent: 'CTInternalNode', tree: 'ClassificationTree', path_to_node):
        super().__init__(parent, path_to_node)
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

# all nodes same class, then differentiate with is_leaf, ...

class ClassificationTree:
    def __init__(self, alphabet: list, sul: SUL, automaton_type: str, cex: tuple):
       
        self.leaf_nodes = {}
        self.automaton_type = automaton_type
        if self.automaton_type == "dfa":
            empty_is_true = sul.query(())[-1]
            self.root = CTInternalNode(distinguishing_string=tuple(), parent=None, path_to_node=None)
            self.root.children[empty_is_true] = CTLeafNode(access_string=tuple(), parent=self.root, tree=self, path_to_node=empty_is_true)
            self.root.children[not empty_is_true] = CTLeafNode(access_string=cex, parent=self.root, tree=self, path_to_node=not empty_is_true)
        elif self.automaton_type == "mealy":
            self.root = CTInternalNode(distinguishing_string=(cex[-1],), parent=None, path_to_node=None)
            hypothesis_output = sul.query((cex[-1],))[-1]
            cex_output = sul.query(cex)[-1]
            self.root.children[hypothesis_output] = CTLeafNode(access_string=tuple(), parent=self.root, tree=self, path_to_node=hypothesis_output)
            self.root.children[cex_output] = CTLeafNode(access_string=cex[:-1], parent=self.root, tree=self, path_to_node=cex_output)

        self.alphabet = alphabet
        self.sul = sul

    def sift(self, word):
        """
        Sifting a word into the classification tree.
        Starting at the root, at every inner node (a CTInternalNode),
        we branch into the "true" or "false" child, depending on the result of the
        membership query (word * node.distinguishing_string). Repeated until a leaf
        (a CTLeafNode) is reached, which is the result of the sifting.

        Args:

            word: the word to sift into the discrimination tree (a tuple of all letters)

        Returns:

            the CTLeafNode that is reached by the sifting operation.
        """
        for letter in word:
            assert letter is None or letter in self.alphabet

        node = self.root

        while isinstance(node, CTInternalNode):
            query = (*word, *node.distinguishing_string)
            if query not in node.query_cache.keys():
                mq_result = self.sul.query(query)[-1]
                node.query_cache[query] = mq_result
            else:
                mq_result = node.query_cache[query]
            # TODO: check if mq_result is part of the children
            node = node.children[mq_result]

        assert isinstance(node, CTLeafNode)
        return node.access_string

    def gen_hypothesis(self):
        # for each CTLeafNode of this CT,
        # create a state in the hypothesis that is labeled by that
        # node's access string. The start state is the empty word
        states = {}
        initial_state = None
        state_counter = 0
        for node in self.leaf_nodes.values():
            state_counter += 1
            if self.automaton_type == "dfa":
                new_state = DfaState(state_id=f's{state_counter}',
                                    is_accepting=node.in_right_side)
            elif self.automaton_type == "mealy":
                new_state = MealyState(state_id=f's{state_counter}')
            new_state.prefix = node.access_string
            if new_state.prefix == ():
                initial_state = new_state
            states[new_state.prefix] = new_state
        assert initial_state is not None

        # For each access state s of the hypothesis and each letter b in the
        # alphabet, compute the b-transition out of state s by sifting s.state_id*b
        for state in states.values():
            for letter in self.alphabet:
                transition_target_id = self.sift((*state.prefix, letter))
                state.transitions[letter] = states[transition_target_id]
                if self.automaton_type == "mealy":
                    output = self.sul.query((*state.prefix, letter))[-1]
                    state.output_fun[letter] = output


        return automaton_class[self.automaton_type](initial_state=initial_state,
                   states=list(states.values()))

    def least_common_ancestor(self, node_1_id, node_2_id):
        """
        Find the distinguishing string of the least common ancestor
        of the leaf nodes node_1 and node_2. Both nodes have to exist.
        Adapted from https://www.geeksforgeeks.org/lowest-common-ancestor-binary-tree-set-1/

        Args:

            node_1_id: first leaf node's id
            node_2_id: second leaf node's id

        Returns:

            the distinguishing string of the lca

        """

        def findLCAOld(root, n1, n2):
            if root is None:
                return None

            if isinstance(root, CTLeafNode) and (root.access_string == n1 or root.access_string == n2):
                return root.parent

            # TODO: adapt this method to be more generic, idea iterate through all children

            left_lca = findLCAOld(root.children[False], n1, n2) if isinstance(root, CTInternalNode) else None
            right_lca = findLCAOld(root.children[True], n1, n2) if isinstance(root, CTInternalNode) else None

            if left_lca and right_lca:
                return root

            return left_lca if left_lca is not None else right_lca
        
        def ancestor(parent, node):
            for child in parent.children.values():
                if isinstance(child, CTLeafNode):
                    if child.access_string == node:
                        return True
                else: 
                    next_ancestor = ancestor(child, node)
                    if next_ancestor:
                        return True
            return False
        
        def findLCA(node_1_id, node_2_id):
            node = self.leaf_nodes[node_1_id]
            parent = node.parent
            while parent:
                if ancestor(parent, node_2_id):
                    return parent
                if parent.parent:
                    parent = parent.parent
                else:
                    return parent
            return None

        return findLCA(node_1_id, node_2_id).distinguishing_string  #findLCAOld(self.root, node_1_id, node_2_id).distinguishing_string 
    def update(self, cex: tuple, hypothesis):
        """
        Updates the classification tree based on a counterexample.
        - For each prefix cex[:i] of the counterexample, get
              s_i      = self.sift(cex[:i])    and
              s_star_i = id of the state with the access sequence cex[:i]
                         in the hypothesis
          and let j be the least i such that s_i != s_star_i.
        - Replace the CTLeafNode labeled with the access string of the state
          that is reached by the sequence cex[:j-1] in the hypothesis
          with an CTInternalNode with two CTLeafNodes: one keeps the old
          access string, and one gets the new access string cex[:j-1].
          The internal node is labeled with the distinguishing string (cex[j-1],*d),
          where d is the distinguishing string of the LCA of s_i and s_star_i.

        Args:
            cex: the counterexample used to update the tree
            hypothesis: the former (wrong) hypothesis

        """
        j = d = None
        for i in range(1, len(cex) + 1):
            s_i = self.sift(cex[:i])
            hypothesis.execute_sequence(hypothesis.initial_state, cex[:i])
            s_star_i = hypothesis.current_state.prefix
            if s_i != s_star_i:
                j = i
                d = self.least_common_ancestor(s_i, s_star_i)
                break
        if j is None and d is None:
            j = len(cex)
            d = []
        assert j is not None and d is not None

        hypothesis.execute_sequence(hypothesis.initial_state, cex[:j - 1] or tuple())

        self.insert_new_leaf(discriminator=(cex[j - 1], *d),
                             old_leaf_access_string=hypothesis.current_state.prefix,
                             new_leaf_access_string=tuple(cex[:j - 1]) or tuple(),
                             new_leaf_position=self.sul.query((*cex[:j - 1], *(cex[j - 1], *d)))[-1])

    def update_rs(self, cex: tuple, hypothesis: Dfa):
        """
        Updates the classification tree based on a counterexample,
        using Rivest & Schapire's counterexample processing
        - Replace the CTLeafNode labeled with the access string of the state
          that is reached by the sequence cex[:j-1] in the hypothesis
          with an CTInternalNode with two CTLeafNodes: one keeps the old
          access string, and one gets the new access string cex[:j-1].
          The internal node is labeled with the distinguishing string (cex[j-1],*d),
          where d is the distinguishing string of the LCA of s_i and s_star_i.

        Args:
            cex: the counterexample used to update the tree
            hypothesis: the former (wrong) hypothesis

        """
        from aalpy.learning_algs.deterministic.CounterExampleProcessing import rs_cex_processing
        v = max(rs_cex_processing(self.sul, cex, hypothesis, suffix_closedness=True), key=len)
        a = cex[len(cex) - len(v) - 1]
        u = cex[:len(cex) - len(v) - 1]
        assert (*u, a, *v) == cex

        hypothesis.execute_sequence(hypothesis.initial_state, u)
        # TODO change to prefix field
        u_state = hypothesis.current_state.prefix
        hypothesis.step(a)
        ua_state = hypothesis.current_state.prefix

        self.insert_new_leaf(discriminator=v,
                             old_leaf_access_string=ua_state,
                             new_leaf_access_string=(*u_state, a),
                             new_leaf_position=self.sul.query((*u, a, *v))[
                                 -1])  # TODO we could probably sub this with not hyp.get_result(cex)

    def insert_new_leaf(self, discriminator, old_leaf_access_string, new_leaf_access_string, new_leaf_position):
        if self.automaton_type == "dfa":
            other_leaf_position = not new_leaf_position
        else:
            other_leaf_position = self.sul.query((*old_leaf_access_string, *discriminator))[-1]
            

        old_leaf = self.leaf_nodes[old_leaf_access_string]

        discriminator_node = CTInternalNode(distinguishing_string=discriminator,
                                            parent=old_leaf.parent, path_to_node=old_leaf.path_to_node)

        new_leaf = CTLeafNode(access_string=new_leaf_access_string,
                              parent=discriminator_node,
                              tree=self,path_to_node=new_leaf_position)

        old_leaf.parent.children[old_leaf.path_to_node] = discriminator_node

        old_leaf.parent = discriminator_node
        old_leaf.query_cache = dict()

        discriminator_node.children[new_leaf_position] = new_leaf
        discriminator_node.children[other_leaf_position] = old_leaf

