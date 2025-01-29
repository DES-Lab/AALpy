from aalpy.learning_algs.deterministic.Apartness import Apartness
from aalpy.learning_algs.deterministic.ADS import Ads
from aalpy.automata import MealyMachine, MealyState
from aalpy.base import Automaton
from aalpy.learning_algs.deterministic.ObservationTree import ObservationTree
from aalpy.learning_algs.adaptive.StateMatching import TotalStateMatching, ApproximateStateMatching
from aalpy.oracles.WpMethodEqOracle import state_characterization_set


class AdaptiveObservationTree(ObservationTree):
    def __init__(self, alphabet, sul, references, extension_rule, separation_rule, rebuilding=True, state_matching="Approximate"):
        """
        Initialize the tree with a root node and the alphabet
        A temporary new basis is needed for the prioritized promotion rule
        The rebuild states counter counts the number of states found with rebuilding excluding the root
        The matching states counter counts the number of states found with match refinement and match separation (NOT prioritized separation)
        """
        super().__init__(alphabet, sul, extension_rule, separation_rule)
        self.references = references
        self.rebuilding = rebuilding
        self.state_matching = state_matching
        self.prefixes_map = {}
        self.characterization_map = {}
        self.combined_model = self.get_combined_model()

        # We keep track of a new basis to ensure maximal overlap between prefixes in the references and the new model
        self.new_basis = [self.root]
        self.initial_OQs = []

        self.rebuild_states = 0
        self.matching_states = 0

        if self.rebuilding:
            self.rebuildObsTree()

        self.initialize_state_matcher()

    def initialize_state_matcher(self):
        """ 
        Initializes the state matching as "Total" or "Approximate"
        """
        if self.state_matching == "Total":
            self.state_matcher = TotalStateMatching(
                self.alphabet, self.combined_model)
            self.state_matcher.initialize_matching(self)
        if self.state_matching == "Approximate":
            self.state_matcher = ApproximateStateMatching(
                self.alphabet, self.combined_model)
            self.state_matcher.initialize_matching(self)

    def build_hypothesis(self):
        """
        Builds the hypothesis which will be sent to the SUL
        This is either done with or without matching rules
        """
        if self.state_matching == "None":
            super().make_observation_tree_adequate()
        else:
            self.make_observation_tree_adequate_matching()
        return self.construct_hypothesis()

    def make_observation_tree_adequate_matching(self):
        """
        Updates the frontier and basis based on several rules 
        Terminates when the observation tree is adequate and no progress has been made in one round
        The separation rule is only used when prioritized separation did not make progress
        The matching rules are only used when the observation tree is already adequate
        """
        self.update_frontier_and_basis()
        ob_tree_size = self.get_size()
        while (not self.is_observation_tree_adequate() or ob_tree_size != self.get_size()):
            self.make_basis_complete()

            ob_tree_size = self.get_size()
            self.make_frontiers_identified_with_matching()
            if ob_tree_size == self.get_size():
                self.make_frontiers_identified()

            self.promote_frontier_state()
            self.update_frontier_and_basis()
            if self.is_observation_tree_adequate():
                old_basis = len(self.basis)
                self.match_refinement()
                self.match_separation()
                if old_basis < len(self.basis):
                    self.matching_states += len(self.basis) - old_basis

    def make_frontiers_identified_with_matching(self):
        """
        Loop over all frontier states to identify them using prioritized identification,
        Only enabled when L# is running with the SepSeq separation rule
        """
        if self.separation_rule == "SepSeq":
            for frontier_state in self.frontier_to_basis_dict:
                self.identify_frontier_with_matching(frontier_state)

    def identify_frontier_with_matching(self, frontier_state):
        """
        Determines the reference state which matches the frontier state (by looking at the basis parent)
        Then finds the state identifiers for the matched reference state 
        Tries to identify the frontier state using the state identifiers of the matched state
        """
        if frontier_state not in self.frontier_to_basis_dict:
            raise Exception(
                f"Warning: {frontier_state} not found in frontier_to_basis_dict.")

        self.update_basis_candidates(frontier_state)
        old_candidate_size = len(
            self.frontier_to_basis_dict.get(frontier_state))

        parent_basis = frontier_state.parent
        inp = frontier_state.input_to_parent
        match = self.state_matcher.best_match[parent_basis]
        if match[0].output_fun[inp] != 'epsilon': 
            frontier_match = match[0].transitions[inp]
            identifiers = self.characterization_map[frontier_match]
            self.identify_frontier_with_identifiers(
                frontier_state, identifiers)

    def identify_frontier_with_identifiers(self, frontier_state, identifiers):
        """ 
        Loops through all candidates states and checks whether they can be separated using one
        of the state identifiers of the state matched with the frontier state
        """
        basis_candidates = self.frontier_to_basis_dict.get(frontier_state)
        for i in range(0, len(basis_candidates)):
            for j in range(i+1, len(basis_candidates)):
                newest_basis_candidates = self.frontier_to_basis_dict.get(
                    frontier_state)
                basis_one = basis_candidates[i]
                basis_two = basis_candidates[j]
                if (basis_one not in newest_basis_candidates):
                    break
                if (basis_two not in newest_basis_candidates):
                    continue

                witness = self.get_or_compute_witness(basis_one, basis_two)
                if tuple(witness) in identifiers:
                    inputs = self.get_transfer_sequence(
                        self.root, frontier_state)
                    inputs.extend(witness)
                    outputs = self.sul.query(inputs)
                    self.insert_observation(inputs, outputs)
                    self.update_basis_candidates(frontier_state)

                if (len(self.frontier_to_basis_dict.get(frontier_state)) < 2):
                    return

    def match_refinement(self):
        """ Loops over the basis states to refine the match for each basis state """
        old_basis = list(self.basis)
        for basis_state in old_basis:
            matches = self.state_matcher.best_match[basis_state]
            if len(matches) > 1:
                self.refine_matches_basis(basis_state, matches)
                self.update_frontier_and_basis()

    def refine_matches_basis(self, basis_state, matches):
        """ 
        Loops over the matched reference states and separates them using a separating sequence
        Returns when only one matching reference state remains, or some states are not distinguishable
        """
        current_matches = matches
        for i in range(0, len(matches)):
            for j in range(i+1, len(matches)):
                ref_state_one = matches[i]
                ref_state_two = matches[j]

                current_matches = self.state_matcher.best_match[basis_state]
                if ref_state_one not in current_matches:
                    break
                if ref_state_two not in current_matches:
                    continue

                witness = self.combined_model.find_distinguishing_seq(
                    ref_state_one, ref_state_two, self.alphabet)
                if witness == None:
                    continue
                inputs = self.get_transfer_sequence(self.root, basis_state)
                inputs.extend(witness)
                outputs = self.sul.query(inputs)
                self.insert_observation(inputs, outputs)

                current_matches = self.state_matcher.best_match[basis_state]
                if (len(current_matches) < 2):
                    return

    def match_separation(self):
        """ 
        Loops over frontier states and calls the match separation with as goal isolation of the frontier state
        """
        matched_states = []
        all_best_matches = [self.state_matcher.best_match[basis_state]
                            for basis_state in self.basis if len(self.state_matcher.best_match[basis_state]) == 1]
        [matched_states.append(
            ref_state) for matches in all_best_matches for ref_state in matches if ref_state not in matched_states]

        frontier_states = list(self.frontier_to_basis_dict.keys())
        for frontier_state in frontier_states:
            basis_candidates = self.frontier_to_basis_dict[frontier_state]
            self.match_separation_frontier(
                matched_states, frontier_state, basis_candidates)
            self.update_frontier_and_basis()

    def match_separation_frontier(self, matched_states, frontier_state, basis_candidates):
        """ 
        Tries to isolate the frontier state if it matches a reference state that currently is not matched to any basis state
        """
        parent_basis = frontier_state.parent
        inp = frontier_state.input_to_parent
        for match in self.state_matcher.best_match[parent_basis]:
            if match.transitions[inp] in matched_states:
                continue

            frontier_match = match.transitions[inp]
            for basis_state in basis_candidates:
                if basis_state not in self.frontier_to_basis_dict[frontier_state]:
                    continue
                if Apartness.compute_witness_in_tree_and_hypothesis_states(
                    self, frontier_state, self.combined_model, frontier_match):
                    continue

                witness = Apartness.compute_witness_in_tree_and_hypothesis_states(
                    self, basis_state, self.combined_model, frontier_match)
                if witness is None:
                    continue
                inputs = self.get_transfer_sequence(self.root, frontier_state)
                inputs.extend(witness)
                outputs = self.sul.query(inputs)
                self.insert_observation(inputs, outputs)
                self.update_basis_candidates(frontier_state)

    def promote_frontier_state(self):
        """
        Searches for a isolated frontier state and adds it to the basis states if it is not associated with another basis state
        """
        for iso_frontier_state, basis_list in self.frontier_to_basis_dict.items():
            if not basis_list:
                new_basis = iso_frontier_state
                self.basis.append(new_basis)
                self.frontier_to_basis_dict.pop(new_basis)
                if self.state_matching != "None":
                    self.state_matcher.update_matching_basis(
                        new_basis, self.basis, self)

                for frontier_state, new_basis_list in self.frontier_to_basis_dict.items():
                    if not Apartness.states_are_apart(new_basis, frontier_state, self):
                        new_basis_list.append(new_basis)
                break

    def insert_observation(self, inputs, outputs):
        """
        Insert an observation into the tree using sequences of inputs and outputs
        If state matching is enabled, ensure that the matching is updated
        """
        if len(inputs) != len(outputs):
            raise ValueError("Inputs and outputs must have the same length.")

        self.validate_input(inputs)

        if self.state_matching != "None":
            self.extend_node_and_update_matching(inputs, outputs)
        else:
            current_node = self.root
            for input_val, output_val in zip(inputs, outputs):
                current_node = current_node.extend_and_get(
                    input_val, output_val)

    def extend_node_and_update_matching(self, inputs, outputs):
        """ 
        Extends the observation tree with new inputs 
        Splits the input sequence in "already defined" part and the "new inputs" part
        If the inputs are not already present in the tree, we update the matching
        """
        to_recalc = []
        split = None
        current_node = self.root
        for i in range(0, len(inputs)):
            input_val = inputs[i]
            output_val = outputs[i]
            if current_node in self.basis:
                to_recalc.append(current_node)
            if input_val not in current_node.successors and split == None:
                split = (inputs[:i], inputs[i:])
            current_node = current_node.extend_and_get(input_val, output_val)

        if split:
            self.state_matcher.update_matching(to_recalc, split, self)

    ## Functions related to rebuilding the observation tree ##

    def rebuildObsTree(self):
        """ 
        Rebuilds the observation tree by finding pairs of frontier and basis states that occur in the same reference model
        Then posing output queries to try to distinguish them in the SUL
        Try to apply the prioritized promotion rule
        """
        tup = self.find_frontier_new_basis()
        while tup:
            (basis_state_access, frontier_state_access, sep_seq) = tup
            query1 = frontier_state_access + sep_seq
            self.insert_observation_rebuilding(query1, self.sul.query(query1))
            query2 = basis_state_access + sep_seq
            self.insert_observation_rebuilding(query2, self.sul.query(query2))

            self.prioritized_promotion()
            tup = self.find_frontier_new_basis()
        self.basis = self.new_basis
        self.update_frontier_and_basis()

    def prioritized_promotion(self):
        """
        Promotes an isolated frontier state with an access sequence in the prefix set of one of the references
        """
        for reference_id in range(0, len(self.references)):
            for reference_prefix in self.prefixes_map[reference_id]:
                ob_tree_state = self.get_successor(reference_prefix)
                if not ob_tree_state:
                    continue
                basis_parent = ob_tree_state.parent
                if (basis_parent in self.new_basis) and (ob_tree_state not in self.new_basis) and self.apart_from_all(ob_tree_state):
                    self.new_basis.append(ob_tree_state)
                    self.rebuild_states += 1

    def find_frontier_new_basis(self):
        """ 
        This function find a frontier and basis state pair which both occur in one of the reference models 
        Because they occur in the same reference model, we have a separating sequence to distinguish them 
        """
        for basis_state_one in self.new_basis:
            for inp in self.alphabet:
                frontier_state_access = self.get_access_sequence(
                    basis_state_one) + (inp,)
                frontier_state = basis_state_one.get_successor(inp)
                if self.get_successor(frontier_state_access) in self.new_basis:
                    continue
                if self.find_basis_frontier_pair(frontier_state, frontier_state_access):
                    return self.find_basis_frontier_pair(frontier_state, frontier_state_access)
        return None

    def find_basis_frontier_pair(self, frontier_state, frontier_state_access):
        """ 
        Find a basis state and reference model such that the prefixes of the
            basis state and frontier state are in the reference model prefix set
        Find a separating sequence that separates the frontier and basis state
        """
        for basis_state in self.new_basis:
            basis_state_access = self.get_access_sequence(basis_state)
            for reference_id in range(0, len(self.references)):
                reference = self.references[reference_id]
                if (basis_state_access not in self.prefixes_map[reference_id]) or (frontier_state_access not in self.prefixes_map[reference_id]):
                    continue
                if frontier_state and Apartness.compute_witness(basis_state, frontier_state, self) != None:
                    continue

                reference.execute_sequence(
                    reference.initial_state, frontier_state_access)
                state_one = reference.current_state
                reference.execute_sequence(
                    reference.initial_state, basis_state_access)
                state_two = reference.current_state

                sep_seq = tuple(reference.find_distinguishing_seq(
                    state_one, state_two, reference.get_input_alphabet()))
                if sep_seq and (self.get_successor(frontier_state_access+sep_seq) == None or self.get_successor(basis_state_access+sep_seq) == None):
                    return (basis_state_access, frontier_state_access, sep_seq)
        return None

    def insert_observation_rebuilding(self, inputs, outputs):
        """
        Insert an observation into the tree using sequences of inputs and outputs
        """
        if len(inputs) != len(outputs):
            raise ValueError("Inputs and outputs must have the same length.")

        self.validate_input(inputs)

        split = None
        current_node = self.root
        for i in range(0, len(inputs)):
            input_val = inputs[i]
            output_val = outputs[i]
            if input_val not in current_node.successors and split == None:
                split = (inputs[:i], inputs[i:])
            current_node = current_node.extend_and_get(input_val, output_val)
        if split:
            self.initial_OQs.append(split)

    def apart_from_all(self, frontier_state):
        """ 
        Checks if a frontier state is apart from all new basis states
        """
        for basis_state in self.new_basis:
            if not Apartness.states_are_apart(basis_state, frontier_state, self):
                return False
        return True

    # Functions related to finding the combined model
    def add_ref_transitions_to_states(self, reference, reference_id):
        """ 
        Makes a copy of the states of a reference with a unique state id and only transitions with the new input alphabet
        """
        states = [MealyState(f"s({reference_id},{ref_state})")
                  for ref_state in range(0, len(reference.states))]
        for state_id in range(0, len(reference.states)):
            states[state_id].output_fun = reference.states[state_id].output_fun
            for inp in self.alphabet:
                if inp in reference.get_input_alphabet():
                    old_index = reference.states.index(
                        reference.states[state_id].transitions[inp])
                    states[state_id].transitions[inp] = states[old_index]
                else:
                    states[state_id].transitions[inp] = states[state_id]
                    states[state_id].output_fun[inp] = 'epsilon'
        return states

    def compute_prefix_map(self, reference, reference_id):
        """ 
        Computes the prefixes of a reference model and stores them in a prefix map
        """
        for state in reference.states:
            state.prefix = reference.get_shortest_path(
                reference.initial_state, state)
        self.prefixes_map[reference_id] = [state.prefix for state in reference.states if state.prefix is not None]

    def compute_characterization_map(self, reference, states):
        """ 
        Computes the separating sequences of a reference model and stores them in a characterization map
        """
        for state_i in range(0, len(reference.states)):
            all_sepseqs = state_characterization_set(
                reference, reference.get_input_alphabet(), reference.states[state_i])
            unique_sepseqs = []
            [unique_sepseqs.append(
                sepseq) for sepseq in all_sepseqs if sepseq not in unique_sepseqs]
            self.characterization_map[states[state_i]] = unique_sepseqs


    def get_combined_model(self):
        """ 
        Builds a combined model from the reference models
        Compute the prefix and characterization maps used during construction of the combined model
        The resulting mealy machine is made input complete by adding self-loops with output 'epsilon' for all undefined inputs
        """
        all_states = []
        for reference_id in range(0, len(self.references)):
            reference = self.references[reference_id]
            overlap = [
                inp for inp in self.alphabet if inp in reference.get_input_alphabet()]
            if not overlap:
                print(
                    f"Warning: reference model {reference_id} has no common inputs and will not be used as a reference.")
                self.references.remove(reference)
                reference_id -= 1
                continue

            states = self.add_ref_transitions_to_states(
                reference, reference_id)
            all_states += states

            self.compute_prefix_map(MealyMachine(states[0], states), reference_id)
            self.compute_characterization_map(reference, states)

        mm = MealyMachine(all_states[0], all_states)
        mm.make_input_complete()
        return mm
