from .ADS import Ads
from .Apartness import Apartness
from ... import Dfa, DfaState, MealyState, MealyMachine, MooreMachine, MooreState

aut_type = ['dfa', 'mealy', 'moore']

class MooreNode:
    _id_counter = 0
    __slots__ = ['id', 'output', 'successors', 'parent', 'input_to_parent']

    def __init__(self, parent=None):
        MooreNode._id_counter += 1
        self.id = MooreNode._id_counter
        self.output = None
        self.successors = {}
        self.parent = parent
        self.input_to_parent = None

    def __hash__(self):
        return hash(self.id)

    def add_successor(self, input_val, output_val, successor_node):
        """ Adds a successor node to the current node based on input """
        self.successors[input_val] = successor_node
        self.successors[input_val].output = output_val

    def get_successor(self, input_val):
        """ Returns the successor node for the given input """
        if input_val in self.successors:
            return self.successors[input_val]
        return None

    def extend_and_get(self, inp, output):
        """ Extend the node with a new successor and return the successor node """
        if inp in self.successors:
            return self.successors[inp]
        successor_node = MooreNode(parent=self)
        self.add_successor(inp, output, successor_node)
        successor_node.input_to_parent = inp
        return successor_node

    @property
    def id_counter(self):
        return self._id_counter


class MealyNode:
    _id_counter = 0
    __slots__ = ['id', 'successors', 'parent', 'input_to_parent']

    def __init__(self, parent=None):
        MealyNode._id_counter += 1
        self.id = MealyNode._id_counter
        self.successors = {}
        self.parent = parent
        self.input_to_parent = None

    def __hash__(self):
        return hash(self.id)

    def add_successor(self, input_val, output_val, successor_node):
        """ Adds a successor node to the current node based on input """
        self.successors[input_val] = (output_val, successor_node)

    def get_successor(self, input_val):
        """ Returns the successor node for the given input """
        if input_val in self.successors:
            return self.successors[input_val][1]
        return None

    def get_output(self, input_val):
        """ Returns the output for the given input """
        if input_val in self.successors:
            return self.successors[input_val][0]
        return None

    def extend_and_get(self, inp, output):
        """ Extend the node with a new successor and return the successor node """
        if inp in self.successors:
            out = self.successors[inp][0]
            if out != output:
                raise Exception(
                    f"observation not consistent with tree with output from tree: {out} and output from call: {output}")
            return self.successors[inp][1]
        successor_node = MealyNode(parent=self)
        self.add_successor(inp, output, successor_node)
        successor_node.input_to_parent = inp
        return successor_node

    @property
    def id_counter(self):
        return self._id_counter


class ObservationTree:
    def __init__(self, alphabet, sul, automaton_type, extension_rule, separation_rule):
        """
        Initialize the tree with a root node and the alphabet
        """
        assert automaton_type in aut_type
        assert alphabet is not None and sul is not None

        self.automaton_type = automaton_type
        self.alphabet = alphabet
        self.sul = sul
        self.extension_rule = extension_rule
        self.separation_rule = separation_rule

        if self.automaton_type == 'mealy':
            self.root = MealyNode()
        else:
            self.root = MooreNode()
            # initialize MooreNode with empty word output
            self.root.output = self.sul.query([])[0]

        self.basis = []
        self.basis.append(self.root)
        self.frontier_to_basis_dict = {}

        # Caches the separating sequences between basis states
        self.witness_cache = {}
        # Maps the basis states to hypothesis states
        self.states_dict = dict()

    def insert_observation(self, inputs, outputs):
        # Insert an observation into the tree using sequences of inputs and outputs
        if len(inputs) != len(outputs):
            raise ValueError("Inputs and outputs must have the same length.")

        current_node = self.root
        for input_val, output_val in zip(inputs, outputs):
            current_node = current_node.extend_and_get(input_val, output_val)

    def get_observation(self, inputs):
        # Retrieve the list of outputs based on a given input sequence
        current_node = self.root
        observation = []
        for input_val in inputs:
            if self.automaton_type == 'mealy':
                output = current_node.get_output(input_val)
                current_node = current_node.get_successor(input_val)
            else:
                current_node = current_node.get_successor(input_val)
                output = current_node.output
            if output is None:
                return None
            observation.append(output)
        return observation

    def get_outputs(self, basis_state, inputs):
        # Retrieve the list of outputs based on a basis state and a given input sequence
        prefix = self.get_transfer_sequence(self.root, basis_state)
        current_node = self.get_successor(prefix)
        observation = []
        for input_val in inputs:
            if self.automaton_type == 'mealy':
                output = current_node.get_output(input_val)
            else:
                output = current_node.output
            if output is None:
                return None
            observation.append(output)
            current_node = current_node.get_successor(input_val)

        return observation

    def get_successor(self, inputs):
        # Retrieve the node (subtree) corresponding to the given input sequence
        current_node = self.root
        for input_val in inputs:
            successor_node = current_node.get_successor(input_val)
            if successor_node is None:
                return None
            current_node = successor_node

        return current_node

    def get_transfer_sequence(self, from_node, to_node):
        # Get the transfer sequence (inputs) that moves from one node to another
        transfer_sequence = []
        current_node = to_node

        while current_node != from_node:
            if current_node.parent is None:
                return None
            transfer_sequence.append(current_node.input_to_parent)
            current_node = current_node.parent

        transfer_sequence.reverse()
        return transfer_sequence

    def get_access_sequence(self, to_node):
        # Get the transfer sequence (inputs) that moves from one node to another
        transfer_sequence = []
        current_node = to_node

        while current_node != self.root:
            if current_node.parent is None:
                return None
            transfer_sequence.append(current_node.input_to_parent)
            current_node = current_node.parent

        transfer_sequence.reverse()
        return tuple(transfer_sequence)

    def get_size(self):
        return self.root.id_counter

    # Functions related to finding new basis and frontier states
    def update_frontier_and_basis(self):
        # Updates the frontier to basis map, promotes a frontier state and checks for consistency
        self.update_frontier_to_basis_dict()
        self.promote_frontier_state()
        self.check_frontier_consistency()
        self.update_frontier_to_basis_dict()

    def update_basis_candidates(self, frontier_state):
        """
        Updates the basis candidates for the specified frontier state.
        Removes basis states that are deemed apart from the frontier state.
        """
        if frontier_state not in self.frontier_to_basis_dict:
            print(
                f"Warning: {frontier_state} not found in frontier_to_basis_dict.")
            return

        basis_list = self.frontier_to_basis_dict[frontier_state]
        self.frontier_to_basis_dict[frontier_state] = [basis_state for basis_state in basis_list
                                                       if not Apartness.states_are_apart(frontier_state, basis_state, self)]

    def update_frontier_to_basis_dict(self):
        """
        Checks for basis candidates (basis states with the same behavior) for each frontier state.
        If a frontier state and a basis state are "apart", the basis state is removed from the basis list.
        """

        for frontier_state, basis_list in self.frontier_to_basis_dict.items():
            self.frontier_to_basis_dict[frontier_state] = [
                basis_state for basis_state in basis_list
                if not Apartness.states_are_apart(frontier_state, basis_state, self)]

    def promote_frontier_state(self):
        """
        Searches for an isolated frontier state and adds it to the basis states if it is not associated with another basis state
        """
        for iso_frontier_state, basis_list in self.frontier_to_basis_dict.items():
            if not basis_list:
                new_basis = iso_frontier_state
                self.basis.append(new_basis)
                self.frontier_to_basis_dict.pop(new_basis)

                for frontier_state, new_basis_list in self.frontier_to_basis_dict.items():
                    if not Apartness.states_are_apart(new_basis, frontier_state, self):
                        new_basis_list.append(new_basis)
                break

    def check_frontier_consistency(self):
        """
        Checks if all the states are correctly defined and creates new frontier states when possible 
        """
        for basis_state in self.basis:
            for i in self.alphabet:
                maybe_frontier = basis_state.get_successor(i)
                if maybe_frontier is None or maybe_frontier in self.basis or maybe_frontier in self.frontier_to_basis_dict:
                    continue

                self.frontier_to_basis_dict[maybe_frontier] = [
                    new_basis_state for new_basis_state in self.basis
                    if not Apartness.states_are_apart(new_basis_state, maybe_frontier, self)
                ]

    def is_observation_tree_adequate(self):
        # Check if the frontier state have only 1 basis candidate, and if all basis
        # states have some output for every input.
        self.check_frontier_consistency()
        for _, basis_list in self.frontier_to_basis_dict.items():
            if len(basis_list) != 1:
                return False

        for basis_state in self.basis:
            for inp in self.alphabet:
                if self.automaton_type == 'mealy':
                    if basis_state.get_output(inp) is None:
                        return False
                else:
                    if basis_state.get_successor(inp) is None:
                        return False

        return True

    def make_basis_complete(self):
        # Explore new frontier states and adding them to the frontier to basis map
        for basis_state in self.basis:
            for inp in self.alphabet:
                if basis_state.get_successor(inp) is None:
                    self.explore_frontier(basis_state, inp)
                    new_frontier = basis_state.get_successor(inp)
                    basis_candidates = self.find_basis_candidates(new_frontier)
                    self.frontier_to_basis_dict[new_frontier] = basis_candidates

    def find_basis_candidates(self, new_frontier):
        return {
            new_basis_state for new_basis_state in self.basis
            if not Apartness.states_are_apart(new_basis_state, new_frontier, self)
        }

    def explore_frontier(self, basis_state, inp):
        # Explores a specific frontier state (basis state + input) by passing a query to the SUL
        if self.extension_rule is None or (self.extension_rule == "SepSeq" and len(self.basis) == 1):
            inputs = self.get_transfer_sequence(self.root, basis_state)
            inputs.append(inp)
            outputs = self.sul.query(inputs)
            self.insert_observation(inputs, outputs)
        elif self.extension_rule == "ADS":
            ads = Ads(self, self.basis)
            ads_in, ads_out = self.adaptive_output_query(
                self.get_transfer_sequence(self.root, basis_state), inp, ads)
            self.insert_observation(ads_in, ads_out)
        elif self.extension_rule == "SepSeq":
            iterator = iter(self.basis)
            basis_two = next(iterator)
            if basis_two == basis_state:
                basis_two = next(iterator)
            witness = self.get_or_compute_witness(basis_state, basis_two)
            inputs = self.get_transfer_sequence(self.root, basis_state)
            inputs.append(inp)
            inputs.extend(witness)
            outputs = self.sul.query(inputs)
            self.insert_observation(inputs, outputs)

    def adaptive_output_query(self, prefix, infix, ads):
        # Adds input to the prefix and calls the base function
        prefix.append(infix)
        return self.adaptive_output_query_base(prefix, ads)

    def adaptive_output_query_base(self, prefix, ads):
        # Query the tree for a result, if unsuccessful query the SUL and update the tree
        from_node = self.get_successor(prefix)
        if from_node:
            tree_in, tree_out = self._answer_ads_from_tree(ads, from_node)
            ads.reset_to_root()

            if tree_out:
                inputs = prefix
                outputs = self.get_observation(prefix)
                inputs.extend(tree_in)
                outputs.extend(tree_out)
                return inputs, outputs


        sul_in, sul_out = self.sul.adaptive_query(prefix, ads)
        if sul_out:
            outputs = sul_out

        self.insert_observation(sul_in, outputs)

        return sul_in, outputs

    def _answer_ads_from_tree(self, ads, from_node):
        # searches the tree based on the inputs returning the inputs/outputs when all ads inputs are used
        prev_output = None
        inputs_sent = []
        outputs_received = []
        current_node = from_node

        while True:
            next_input = ads.next_input(prev_output)
            if next_input is None:
                break
            inputs_sent.append(next_input)

            if self.automaton_type == 'mealy':
                output_from_node = current_node.get_output(next_input)
                successor_from_node = current_node.get_successor(next_input)
                if successor_from_node is None:
                    return None, None
            else:
                successor_from_node = current_node.get_successor(next_input)
                if successor_from_node is None:
                    return None, None
                output_from_node = successor_from_node.output

            prev_output = output_from_node
            outputs_received.append(output_from_node)
            current_node = successor_from_node

        ads.reset_to_root()
        return inputs_sent, outputs_received

    def get_or_compute_witness(self, state_one, state_two):
        """
        Get witness by checking cache and computing it otherwise.
        Only add pairs (a,b) with a < b.
        """
        if state_one.id < state_two.id:
            pair = (state_one.id, state_two.id)
        else:
            pair = (state_two.id, state_one.id)

        if pair in self.witness_cache:
            return self.witness_cache.get(pair)

        witness = Apartness.compute_witness(state_one, state_two, self)
        self.witness_cache[pair] = witness
        return witness

    def make_frontiers_identified(self):
        # Loop over all frontier states to identify them
        for frontier_state in self.frontier_to_basis_dict:
            self.identify_frontier(frontier_state)

    def identify_frontier(self, frontier_state):
        # Identify a specific frontier state
        if frontier_state not in self.frontier_to_basis_dict:
            raise Exception(
                f"Warning: {frontier_state} not found in frontier_to_basis_dict.")

        self.update_basis_candidates(frontier_state)
        old_candidate_size = len(
            self.frontier_to_basis_dict.get(frontier_state))
        if old_candidate_size < 2:
            return

        if self.separation_rule == "SepSeq" or old_candidate_size == 2:
            inputs, outputs = self._identify_frontier_sepseq(frontier_state)
        else:
            inputs, outputs = self._identify_frontier_ads(frontier_state)

        self.insert_observation(inputs, outputs)
        self.update_basis_candidates(frontier_state)
        if len(self.frontier_to_basis_dict.get(frontier_state)) == old_candidate_size:
            raise RuntimeError("Identification did not increase the norm")

    def _identify_frontier_sepseq(self, frontier_state):
        # Specifically identify frontier states using separating sequences
        basis_candidates = self.frontier_to_basis_dict.get(frontier_state)
        basis_one = basis_candidates[0]
        basis_two = basis_candidates[1]

        witness = self.get_or_compute_witness(basis_one, basis_two)
        inputs = self.get_transfer_sequence(self.root, frontier_state)
        inputs.extend(witness)

        outputs = self.sul.query(inputs)

        return inputs, outputs

    def _identify_frontier_ads(self, frontier_state):
        # Specifically identify frontier states using ADS
        basis_candidates = self.frontier_to_basis_dict.get(frontier_state)
        ads = Ads(self, basis_candidates)
        ads.reset_to_root()
        return self.adaptive_output_query_base(self.get_transfer_sequence(self.root, frontier_state), ads)

    def construct_hypothesis_states(self):
        # Construct the hypothesis states from the basis
        self.states_dict = dict()
        state_counter = 0

        for basis_state in self.basis:
            state_id = f's{state_counter}'
            if self.automaton_type == 'dfa':
                self.states_dict[basis_state] = DfaState(state_id)
                self.states_dict[basis_state].is_accepting = basis_state.output
            elif self.automaton_type == 'moore':
                self.states_dict[basis_state] = MooreState(
                    state_id, output=basis_state.output)
            else:
                self.states_dict[basis_state] = MealyState(state_id)
            state_counter += 1

    def construct_hypothesis_transitions(self):
        # Construct the hypothesis transitions from the basis, frontier and basis to frontier mapping
        for basis_state in self.basis:
            for input_val in self.alphabet:
                # set transition
                successor = basis_state.get_successor(input_val)
                if successor in self.frontier_to_basis_dict:
                    # set successor for frontier state
                    candidates = self.frontier_to_basis_dict[successor]
                    if len(candidates) > 1:
                        raise RuntimeError(
                            "Multiple basis candidates for a single frontier state.")
                    successor = next(iter(candidates))
                if successor not in self.states_dict:
                    raise RuntimeError(
                        "Successor is not in the basisToStateMap.")

                destination = self.states_dict[successor]
                self.states_dict[basis_state].transitions[input_val] = destination
                if self.automaton_type == 'mealy':
                    self.states_dict[basis_state].output_fun[input_val] = basis_state.get_output(
                        input_val)

    def construct_hypothesis(self):
        # Construct a hypothesis (Mealy Machine) based on the observation tree
        self.construct_hypothesis_states()
        self.construct_hypothesis_transitions()

        automaton_class = {'dfa': Dfa, 'mealy': MealyMachine, 'moore': MooreMachine}
        hypothesis = automaton_class[self.automaton_type](
            self.states_dict[self.root], list(self.states_dict.values()))
        hypothesis.compute_prefixes()
        hypothesis.characterization_set = hypothesis.compute_characterization_set(raise_warning=False)

        return hypothesis

    def build_hypothesis(self):
        # Builds the hypothesis which will be sent to the SUL and checks consistency
        while True:
            self.make_observation_tree_adequate()
            hypothesis = self.construct_hypothesis()
            counter_example = Apartness.compute_witness_in_tree_and_hypothesis_states(self, self.root, hypothesis.initial_state)

            if not counter_example:
                return hypothesis

            cex_outputs = self.get_observation(counter_example)
            self.process_counter_example(hypothesis, counter_example, cex_outputs)

    def make_observation_tree_adequate(self):
        # Updates the frontier and basis based on extension and separation rule
        self.update_frontier_and_basis()
        while not self.is_observation_tree_adequate():
            self.make_basis_complete()
            self.make_frontiers_identified()
            self.promote_frontier_state()

    # Counterexample Processing

    def process_counter_example(self, hypothesis, cex_inputs, cex_outputs):
        """
        Inserts the counter example into the observation tree and searches for the
        input-output sequence which is different
        """
        self.insert_observation(cex_inputs, cex_outputs)
        hyp_outputs = hypothesis.compute_output_seq(
            hypothesis.initial_state, cex_inputs)
        prefix_index = self._get_counter_example_prefix_index(
            cex_outputs, hyp_outputs)
        self._process_binary_search(
            hypothesis, cex_inputs[:prefix_index], cex_outputs[:prefix_index])

    def _get_counter_example_prefix_index(self, cex_outputs, hyp_outputs):
        """ Checks at which index the output functions differ """
        for index in range(len(cex_outputs)):
            if cex_outputs[index] != hyp_outputs[index]:
                return index
        raise RuntimeError("counterexample and hypothesis outputs are equal")

    def _process_binary_search(self, hypothesis, cex_inputs, cex_outputs):
        """
        use binary search on the counter example to compute a witness between the real system and the hypothesis
        """
        tree_node = self.get_successor(cex_inputs)
        self.update_frontier_and_basis()

        if tree_node in self.frontier_to_basis_dict or tree_node in self.basis:
            return

        hyp_state = self._get_automaton_successor(
            hypothesis, hypothesis.initial_state, cex_inputs)
        hyp_node = list(self.states_dict.keys())[list(
            self.states_dict.values()).index(hyp_state)]

        prefix = []
        current_state = self.root
        for input in cex_inputs:
            if current_state in self.frontier_to_basis_dict:
                break
            current_state = current_state.get_successor(input)
            prefix.append(input)

        h = (len(prefix) + len(cex_inputs)) // 2
        sigma1 = list(cex_inputs[:h])
        sigma2 = list(cex_inputs[h:])

        hyp_state_p = self._get_automaton_successor(
            hypothesis, hypothesis.initial_state, sigma1)
        hyp_node_p = list(self.states_dict.keys())[list(
            self.states_dict.values()).index(hyp_state_p)]
        hyp_p_access = self.get_transfer_sequence(self.root, hyp_node_p)

        witness = Apartness.compute_witness(tree_node, hyp_node, self)
        if witness is None:
            raise RuntimeError("Binary search: There should be a witness")

        query_inputs = hyp_p_access + sigma2 + witness
        query_outputs = self.sul.query(query_inputs)

        self.insert_observation(query_inputs, query_outputs)

        tree_node_p = self.get_successor(sigma1)

        witness_p = Apartness.compute_witness(tree_node_p, hyp_node_p, self)

        if witness_p is not None:
            self._process_binary_search(hypothesis, sigma1, cex_outputs[:h])
        else:
            new_inputs = list(hyp_p_access) + sigma2
            self._process_binary_search(
                hypothesis, new_inputs, query_outputs[:len(new_inputs)])

    def _get_automaton_successor(self, automaton, from_state, inputs):
        automaton.current_state = from_state
        for inp in inputs:
            automaton.current_state = automaton.current_state.transitions[inp]

        return automaton.current_state
