from collections import defaultdict
from typing import Union

from aalpy.automata import DfaState, Dfa, MealyState, MealyMachine, MooreState, MooreMachine, \
    SevpaAlphabet, SevpaState, SevpaTransition, Sevpa
from aalpy.base import SUL
from aalpy.learning_algs.deterministic.CounterExampleProcessing import rs_cex_processing, linear_cex_processing, \
    exponential_cex_processing

automaton_class = {'dfa': Dfa, 'mealy': MealyMachine, 'moore': MooreMachine}


class CTNode:
    __slots__ = ['parent', 'path_to_node']

    def __init__(self, parent, path_to_node):
        self.parent = parent
        self.path_to_node = path_to_node

    def is_leaf(self):
        pass


class CTInternalNode(CTNode):
    __slots__ = ['distinguishing_string', 'children']

    def __init__(self, distinguishing_string: tuple, parent, path_to_node):
        super().__init__(parent, path_to_node)
        self.distinguishing_string = distinguishing_string
        self.children = defaultdict(None)  # {True: None, False: None}

    def is_leaf(self):
        return False


class CTLeafNode(CTNode):
    __slots__ = ['access_string']

    def __init__(self, access_string: tuple, parent, path_to_node):
        super().__init__(parent, path_to_node)
        self.access_string = access_string

    def __repr__(self):
        return f"{self.__class__.__name__} '{self.access_string}'"

    @property
    def output(self):
        c, p = self, self.parent
        while p.parent:
            c = p
            p = p.parent
        for output, child in p.children.items():
            if child == c:
                return output
        assert False

    def is_leaf(self):
        return True


class ClassificationTree:
    def __init__(self, alphabet: Union[list, SevpaAlphabet], sul: SUL, automaton_type: str, cex: tuple):
        self.sul = sul
        self.alphabet = alphabet
        self.automaton_type = automaton_type

        self.leaf_nodes = {}
        self.query_cache = dict()

        self.sifting_cache = {}

        # prefix of identified error state in VPDA learning
        self.error_state_prefix = None

        if self.automaton_type != 'mealy':
            initial_output = sul.query(())[-1]
            cex_output = sul.query(cex)[-1]

            self.query_cache[()] = initial_output

            root_distinguishing_string = () if automaton_type != 'vpa' else ([(), ()])

            self.root = CTInternalNode(distinguishing_string=root_distinguishing_string, parent=None, path_to_node=None)

            initial_output_node = CTLeafNode(access_string=tuple(), parent=self.root, path_to_node=initial_output)
            cex_output_node = CTLeafNode(access_string=cex, parent=self.root, path_to_node=cex_output)

            self.root.children[initial_output] = initial_output_node
            self.root.children[cex_output] = cex_output_node

            self.leaf_nodes[tuple()] = initial_output_node
            self.leaf_nodes[cex] = cex_output_node

        else:
            self.root = CTInternalNode(distinguishing_string=(cex[-1],), parent=None, path_to_node=None)

            hypothesis_output = sul.query((cex[-1],))[-1]
            cex_output = sul.query(cex)[-1]

            hypothesis_output_node = CTLeafNode(access_string=tuple(), parent=self.root, path_to_node=hypothesis_output)
            cex_output_node = CTLeafNode(access_string=cex[:-1], parent=self.root, path_to_node=cex_output)

            self.root.children[hypothesis_output] = hypothesis_output_node
            self.root.children[cex_output] = cex_output_node

            self.leaf_nodes[tuple()] = self.root.children[hypothesis_output]
            self.leaf_nodes[cex[:-1]] = self.root.children[cex_output]

    def _sift(self, word):
        """
        Sifting a word into the classification tree.
        Starting at the root, at every inner node (a CTInternalNode),
        we branch into the child, depending on the result of the
        membership query (word * node.distinguishing_string). Repeated until a leaf
        (a CTLeafNode) is reached, which is the result of the sifting.

        Args:

            word: the word to sift into the discrimination tree (a tuple of all letters)

        Returns:

            the CTLeafNode that is reached by the sifting operation.
        """

        if word in self.sifting_cache:
            return self.sifting_cache[word]

        node = self.root
        while not node.is_leaf():

            if self.automaton_type != 'vpa':
                query = word + node.distinguishing_string
            else:
                query = node.distinguishing_string[0] + word + node.distinguishing_string[1]

            if query not in self.query_cache.keys():
                mq_result = self.sul.query(query)
                # keep track of transitions (this might miss some due to other caching, but rest can be obtained from
                # cache in gen hyp)
                if self.automaton_type == 'mealy' and word not in self.query_cache.keys():
                    self.query_cache[word] = mq_result[len(word) - 1]

                mq_result = mq_result[-1]
                self.query_cache[query] = mq_result
            else:
                mq_result = self.query_cache[query]

            if mq_result not in node.children.keys():
                new_leaf = CTLeafNode(access_string=word, parent=node, path_to_node=mq_result)
                self.leaf_nodes[word] = new_leaf
                node.children[mq_result] = new_leaf

            node = node.children[mq_result]

        self.sifting_cache[word] = node
        assert node.is_leaf()
        return node

    def gen_hypothesis(self):
        # for each CTLeafNode of this CT,
        # create a state in the hypothesis that is labeled by that
        # node's access string. The start state is the empty word
        states = {}
        initial_state = None
        state_counter = 0
        for node in self.leaf_nodes.values():

            if self.automaton_type == 'dfa':
                new_state = DfaState(state_id=f's{state_counter}', is_accepting=node.output)
            elif self.automaton_type == 'moore':
                new_state = MooreState(state_id=f's{state_counter}', output=node.output)
            elif self.automaton_type == 'vpa':
                new_state = SevpaState(state_id=f'q{state_counter}', is_accepting=node.output)
            else:
                new_state = MealyState(state_id=f's{state_counter}')

            new_state.prefix = node.access_string
            if new_state.prefix == ():
                initial_state = new_state
            states[new_state.prefix] = new_state
            state_counter += 1
        assert initial_state is not None

        # For each access state s of the hypothesis and each letter b in the
        # alphabet, compute the b-transition out of state s by sifting s.state_id*b
        states_for_transitions = list(states.values())
        for state in states_for_transitions:
            if self.automaton_type != 'vpa':
                for letter in self.alphabet:
                    transition_target_node = self._sift(state.prefix + (letter,))
                    transition_target_access_string = transition_target_node.access_string

                    if self.automaton_type != "dfa" and transition_target_access_string not in states:
                        if self.automaton_type == 'mealy':
                            new_state = MealyState(state_id=f's{state_counter}')
                        else:
                            output = self._query_and_update_cache(transition_target_access_string)
                            new_state = MooreState(state_id=f's{state_counter}', output=output)

                        new_state.prefix = transition_target_access_string
                        states_for_transitions.append(new_state)
                        states[new_state.prefix] = new_state
                        state_counter += 1

                    state.transitions[letter] = states[transition_target_access_string]

                    if self.automaton_type == "mealy":
                        state.output_fun[letter] = self._query_and_update_cache(state.prefix + (letter,))
            else:
                # internal transitions
                for internal_letter in self.alphabet.internal_alphabet:
                    transition_target_node = self._sift(state.prefix + (internal_letter,))
                    transition_target_access_string = transition_target_node.access_string

                    assert transition_target_access_string in states
                    trans = SevpaTransition(target=states[transition_target_access_string],
                                            letter=internal_letter, action=None)
                    state.transitions[internal_letter].append(trans)

                # Add call transitions
                for call_letter in self.alphabet.call_alphabet:
                    # Add return transitions
                    for return_letter in self.alphabet.return_alphabet:
                        # check if exclusive pairs of call and return letters are defined in an alphabets
                        if self.alphabet.exclusive_call_return_pairs and \
                                self.alphabet.exclusive_call_return_pairs[call_letter] != return_letter:
                            continue

                        for other_state in states_for_transitions:
                            # ignore other state if other state is error state
                            if other_state.prefix == self.error_state_prefix:
                                continue
                            transition_target_node = self._sift(
                                other_state.prefix + (call_letter,) + state.prefix + (return_letter,))
                            transition_target_access_string = transition_target_node.access_string

                            trans = SevpaTransition(target=states[transition_target_access_string],
                                                    letter=return_letter,
                                                    action='pop', stack_guard=(other_state.state_id, call_letter))
                            state.transitions[return_letter].append(trans)

        if self.automaton_type == 'vpa':
            hypothesis = Sevpa(initial_state=initial_state, states=list(states.values()))
            if not self.error_state_prefix:
                error_state = hypothesis.get_error_state()
                if error_state:
                    self.error_state_prefix = error_state.prefix
            return hypothesis

        return automaton_class[self.automaton_type](initial_state=initial_state, states=list(states.values()))

    def _least_common_ancestor(self, node_1_id, node_2_id):
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

        def ancestor(parent, node):
            for child in parent.children.values():
                if child.is_leaf():
                    if child.access_string == node:
                        return True
                else:
                    next_ancestor = ancestor(child, node)
                    if next_ancestor:
                        return True
            return False

        def findLCA(n1_id, n2_id):
            node = self.leaf_nodes[n1_id]
            parent = node.parent
            while parent:
                if ancestor(parent, n2_id):
                    return parent
                if parent.parent:
                    parent = parent.parent
                else:
                    return parent
            return None

        return findLCA(node_1_id, node_2_id).distinguishing_string

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
            s_i = self._sift(cex[:i]).access_string
            hypothesis.execute_sequence(hypothesis.initial_state, cex[:i])
            s_star_i = hypothesis.current_state.prefix

            if s_i != s_star_i:
                j = i
                d = self._least_common_ancestor(s_i, s_star_i)
                break
        if j is None and d is None:
            j = len(cex)
            d = []
        assert j is not None and d is not None

        hypothesis.execute_sequence(hypothesis.initial_state, cex[:j - 1] or tuple())

        self._insert_new_leaf(discriminator=(cex[j - 1], *d),
                              old_leaf_access_string=hypothesis.current_state.prefix,
                              new_leaf_access_string=tuple(cex[:j - 1]) or tuple(),
                              new_leaf_position=self.sul.query((*cex[:j - 1], *(cex[j - 1], *d)))[-1])

    def process_counterexample(self, cex: tuple, hypothesis, cex_processing_fun):
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
            cex_processing_fun: string choosing which cex_processing to use

        """
        v = None
        if 'linear' in cex_processing_fun:
            direction = cex_processing_fun[-3:]
            v = linear_cex_processing(self.sul, cex, hypothesis, is_vpa=self.automaton_type == 'vpa',
                                      direction=direction, suffix_closedness=False)[0]
        elif 'exponential' in cex_processing_fun:
            direction = cex_processing_fun[-3:]
            v = exponential_cex_processing(self.sul, cex, hypothesis, is_vpa=self.automaton_type == 'vpa',
                                           direction=direction, suffix_closedness=False)[0]
        elif cex_processing_fun == 'rs':
            v = rs_cex_processing(self.sul, cex, hypothesis, is_vpa=self.automaton_type == 'vpa',
                                  suffix_closedness=False)[0]

        assert v
        a = cex[len(cex) - len(v) - 1]
        u = cex[:len(cex) - len(v) - 1]
        assert (*u, a, *v) == cex

        hypothesis.execute_sequence(hypothesis.initial_state, u)
        u_state = hypothesis.current_state

        top_of_stack = hypothesis.stack[-1] if self.automaton_type == 'vpa' else None

        # get state reached after executing last action => old leaf
        hypothesis.step(a)
        ua_state = hypothesis.current_state

        # get discriminator and new_leaf_access_string
        if self.automaton_type == 'vpa':
            discriminator = (tuple(hypothesis.transform_access_string()), tuple(v))

            if a in self.alphabet.internal_alphabet:
                new_leaf_access_string = (*u_state.prefix, a)
            else:
                assert a in self.alphabet.return_alphabet
                l_prime, call = hypothesis.get_state_by_id(top_of_stack[0]), top_of_stack[1]
                new_leaf_access_string = l_prime.prefix + (call,) + u_state.prefix + (a,)
        else:
            discriminator = v
            new_leaf_access_string = (*u_state.prefix, a)

        if self.automaton_type == 'dfa' or self.automaton_type == 'vpa':
            new_leaf_position = not hypothesis.execute_sequence(hypothesis.initial_state, cex)[-1]
        else:
            new_leaf_position = self.sul.query(cex)[-1]

        self._insert_new_leaf(discriminator=discriminator,
                              old_leaf_access_string=ua_state.prefix,
                              new_leaf_access_string=new_leaf_access_string,
                              new_leaf_position=new_leaf_position)

    def _insert_new_leaf(self, discriminator, old_leaf_access_string, new_leaf_access_string, new_leaf_position):
        """
        Inserts a new leaf in the classification tree by:
        - moving the leaf node specified by <old_leaf_access_string> down one level
        - inserting an internal node  at the former position of the old node (i.e. as the parent of the old node)
        - adding a new leaf node with <new_leaf_access_string> as child of the new internal node/sibling of the old node
        Could also be thought of as 'splitting' the old node into two (one of which keeps the old access string and one
        of which gets the new one) with <discriminator> as the distinguishing string between the two.

        where one of the resulting nodes keeps the old
        node's access string and the other gets new_leaf_access_string.
        Args:
            discriminator: The distinguishing string of the new internal node
            old_leaf_access_string: The access string specifying the leaf node to be 'split' (or rather moved down)
            new_leaf_access_string: The access string of the leaf node that will be created
            new_leaf_position: The path from the new internal node to the new leaf node

        Returns:

        """
        if self.automaton_type == "dfa" or self.automaton_type == 'vpa':
            other_leaf_position = not new_leaf_position
        else:
            # check if this query is in the node cache
            other_leaf_position = self.sul.query((*old_leaf_access_string, *discriminator))[-1]

        old_leaf = self.leaf_nodes[old_leaf_access_string]

        # create an internal node at the same position as the old leaf node
        discriminator_node = CTInternalNode(distinguishing_string=discriminator,
                                            parent=old_leaf.parent, path_to_node=old_leaf.path_to_node)

        # create the new leaf node and add it as child of the internal node
        new_leaf = CTLeafNode(access_string=new_leaf_access_string,
                              parent=discriminator_node,
                              path_to_node=new_leaf_position)
        self.leaf_nodes[new_leaf_access_string] = new_leaf

        # redirect the old nodes former parent to the internal node
        old_leaf.parent.children[old_leaf.path_to_node] = discriminator_node

        # add the internal node as parent of the old leaf
        old_leaf.parent = discriminator_node
        old_leaf.path_to_node = other_leaf_position

        # set the two nodes as children of the internal node
        discriminator_node.children[new_leaf_position] = new_leaf
        discriminator_node.children[other_leaf_position] = old_leaf

        # sifting cache update
        sifting_cache_outdated = []
        if old_leaf in self.sifting_cache.values():
            for prefix, node in self.sifting_cache.items():
                if old_leaf == node:
                    sifting_cache_outdated.append(prefix)

        for to_delete in sifting_cache_outdated:
            del self.sifting_cache[to_delete]

    def _query_and_update_cache(self, word):
        if word in self.query_cache.keys():
            output = self.query_cache[word]
        else:
            output = self.sul.query(word)[-1]
            self.query_cache[word] = output
        return output
