from abc import abstractmethod


class StateMatching:
    def __init__(self, alphabet, combined_model):
        """
        Initializes the super class for state matching
         """
        self.alphabet = alphabet
        self.combined_model = combined_model

        self.matchings = {}  # map from basis states to reference states
        self.best_match = {}
        self.best_score = {}

    @abstractmethod
    def add_entry_basis(self, basis_state):
        pass

    @abstractmethod
    def update_best_score(self, basis_state):
        pass

    @abstractmethod
    def update_score(self, ob_tree, basis_state, reference_state, basis_state_access, defined_after_access, new_part):
        pass

    @abstractmethod
    def update_best_match(self, basis_state, score):
        pass

    def initialize_matching(self, ob_tree):
        """ 
        Initializes the matching by adding an entry for every basis state found during rebuilding 
        Updates the matching for the output queries posed during rebuilding
        """
        for basis_state in ob_tree.basis:
            self.add_entry_basis(basis_state, ob_tree.automaton_type)

        for defined_part, new_part in ob_tree.initial_OQs:
            to_recalc = []
            for basis_state in ob_tree.basis:
                if self.is_prefix_of(tuple(ob_tree.get_access_sequence(basis_state)), (defined_part + new_part)):
                    to_recalc.append(basis_state)

            self.update_matching(to_recalc, (defined_part, new_part), ob_tree)


    def update_matching(self, to_recalc, split, ob_tree):
        """ 
        Updates the matching for the to be recalculated basis states
        The split contains the already defined part and the new part of an output query

        For every basis state, we determine the defined part and new part AFTER accessing the basis state
        Then for every reference model check if the defined part uses only inputs valid in the reference model
        If these are all valid, we calculate the score of the new part
        """
        defined_part, orig_new_part = split

        for basis_state in to_recalc:
            basis_state_access = ob_tree.get_access_sequence(basis_state)
            defined_after_access = ()

            new_part = orig_new_part
            if len(defined_part) > len(basis_state_access):
                defined_after_access = defined_part[len(basis_state_access):]
            else:
                new_part = (defined_part + orig_new_part)[len(basis_state_access):]

            for reference_state in self.combined_model.states:
                # defined_after_access contains invalid input for reference model, score remains 0
                if self.validate_reference_input(defined_after_access, reference_state):
                    self.update_score(ob_tree, basis_state, reference_state,
                                      basis_state_access, defined_after_access, new_part)

            score = self.update_best_score(basis_state)
            self.update_best_match(basis_state, score)



    def update_matching_basis(self, basis_state, ob_tree):
        """ 
        Initializes and updates the matching for a newly added basis state
        """
        self.add_entry_basis(basis_state, ob_tree.automaton_type)
        longest_words = list(self.find_longest_words(basis_state, ob_tree, []))
        basis_state_access = ob_tree.get_access_sequence(basis_state)

        for i in range(0, len(longest_words)):
            split = basis_state_access, longest_words[i][len(basis_state_access):]

            if i > 0:
                split = self.find_longest_common_part(longest_words[i-1], longest_words[i])

            self.update_matching([basis_state], split, ob_tree)

    def find_longest_words(self, current_state, ob_tree, all_seqs):
        """ 
        Finds prefix closed words in the observation tree starting from the current state
        DFS-like procedure
        """
        leaf = True
        for inp in self.alphabet:
            if inp in current_state.successors:
                new_seqs = self.find_longest_words(
                    current_state.get_successor(inp), ob_tree, all_seqs)
                for new_seq in new_seqs:
                    if new_seq not in all_seqs:
                        all_seqs.add(new_seq)
                leaf = False
        if leaf:
            all_seqs.append(ob_tree.get_access_sequence(current_state))
        return all_seqs

    def validate_reference_input(self, inputs, reference_state):
        """
        Check if all inputs are valid (part of the alphabet of the reference model)
        """
        for input_val in inputs:
            if input_val not in reference_state.transitions:
                return False
        return True

    def is_prefix_of(self, str1, str2):
        """ Checks if input sequence str1 is a prefix of str2 """
        if len(str1) > len(str2):
            return False
        for i in range(0, len(str1)):
            if str2[i] != str1[i]:
                return False
        return True

    def find_longest_common_part(self, str1, str2):
        """ Finds the longest common prefix of input sequences str1 and str2 """
        so_far = []
        for i in range(0, len(str2)):
            if str2[i] == str1[i]:
                so_far.append(str2[i])
            else:
                return tuple(so_far), tuple(str2[i:])
        return tuple(so_far), tuple()

    def print_match_table(self, ob_tree):
        """
        Prints the match table
        Code based on https://stackoverflow.com/questions/13214809/pretty-print-2d-list
        """
        print(f"Mapping of basis state ids to access sequences")
        for basis_state in ob_tree.basis:
            print(f"{basis_state.id}: {ob_tree.get_access_sequence(basis_state)}")
        first_row = ["state", "|", "match", "|", "score", "|"] + \
            [ref_state.state_id for ref_state in self.combined_model.states]
        data = [first_row]
        for basis_state in ob_tree.basis:
            row = [basis_state.id, "|", [
                st.state_id for st in self.best_match[basis_state]], "|", self.best_score[basis_state], "|"]
            row += [sc for sc in self.matchings[basis_state].values()]
            data.append(row)

        s = [[str(e) for e in row] for row in data]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print('\n'.join(table))


class TotalStateMatching(StateMatching):
    """ 
    Total State Matching is an instance of State Matching

    A basis and reference state match if all inputs sequences (over the shared alphabet) defined from the basis state 
    give EXACTLY the same outputs
    """

    def __init__(self, alphabet, combined_model):
        super().__init__(alphabet, combined_model)

    def add_entry_basis(self, basis_state, aut_type):
        """ Initializes a new matching row with 1 or empty word evaluation"""
        if aut_type == 'mealy':
            self.matchings[basis_state] = {ref_state: 1 for ref_state in self.combined_model.states}
        else:
            self.matchings[basis_state] = dict()
            basis_out = basis_state.output 
            for ref_state in self.combined_model.states:
                if aut_type == 'dfa':
                    ref_out = ref_state.is_accepting
                elif aut_type == 'moore':
                    ref_out = ref_state.output
                if ref_out == basis_out:
                    self.matchings[basis_state][ref_state] = 1
                else:
                    self.matchings[basis_state][ref_state] = 0

    def update_best_score(self, basis_state):
        """ Updates the best score for a basis state """
        score = max([self.matchings[basis_state][ref_state] for ref_state in self.combined_model.states])
        self.best_score[basis_state] = score
        return score

    def update_best_match(self, basis_state, score):
        """ Updates the best match for a basis state """
        if score == 0:
            self.best_match[basis_state] = []
        else:
            matches = []
            for ref_state in self.combined_model.states:
                if self.matchings[basis_state][ref_state] == score:
                    matches.append(ref_state)

            self.best_match[basis_state] = matches

    def update_score(self, ob_tree, basis_state, reference_state, basis_state_access, defined_after_access, new_part):
        # calls update score for either mealy or moore/dfa 
        if ob_tree.automaton_type == 'mealy':
            return self.update_score_mealy(ob_tree, basis_state, reference_state, basis_state_access, defined_after_access, new_part)
        else:
            return self.update_score_moore(ob_tree, basis_state, reference_state, basis_state_access, defined_after_access, new_part)


    def update_score_mealy(self, ob_tree, basis_state, reference_state, basis_state_access, defined_after_access, new_part):
        """ 
        Updates the matching score for a basis and reference state based on the: 
            basis access, already defined part after access and the new part
        For every input in the new part, we take a step in the ob_tree and in the combined model
        If the combined model has no transition for some input (because it is not in the reference alphabet), we return
        If the outputs differ, we set the matching to 0
        """
        if self.matchings[basis_state][reference_state] == 0:
            return

        current_ob_state = ob_tree.get_successor(
            tuple(basis_state_access) + tuple(defined_after_access))
        self.combined_model.execute_sequence(
            reference_state, defined_after_access)

        for inp in new_part:
            ob_out = current_ob_state.get_output(inp)
            if inp not in reference_state.output_fun:
                return

            ref_out = self.combined_model.step(inp)
            if ob_out != ref_out:
                self.matchings[basis_state][reference_state] = 0
                return
            current_ob_state = current_ob_state.get_successor(inp)

    def update_score_moore(self, ob_tree, basis_state, reference_state, basis_state_access, defined_after_access, new_part):
        """ 
        Updates the matching score for a basis and reference state based on the: 
            basis access, already defined part after access and the new part
        For every input in the new part, we take a step in the ob_tree and in the combined model
        If the outputs differ, we set the matching to 0
        """
        orig_ref = reference_state
        if self.matchings[basis_state][reference_state] == 0:
            return

        current_ob_state = ob_tree.get_successor(
            tuple(basis_state_access) + tuple(defined_after_access))
        self.combined_model.execute_sequence(reference_state, defined_after_access)
        reference_state = self.combined_model.current_state

        # need to test the empty word
        if defined_after_access == (): 
            if reference_state.is_accepting != current_ob_state.output:
                self.matchings[basis_state][orig_ref] = 0
                return

        for inp in new_part:
            if inp not in reference_state.transitions:
                return
            ref_out = self.combined_model.step(inp)
            reference_state = self.combined_model.current_state

            current_ob_state = current_ob_state.get_successor(inp)
            ob_out = current_ob_state.output

            if ob_out != ref_out:
                self.matchings[basis_state][orig_ref] = 0
                return



class ApproximateStateMatching(StateMatching):
    """ 
    Approximate State Matching is an instance of State Matching

    A basis matches a reference state if the reference state has the highest ratio of matching outputs over 
    all inputs sequences (over the shared alphabet) defined from the basis state compared to the other reference states
    """

    def __init__(self, alphabet, combined_model):
        super().__init__(alphabet, combined_model)
        self.unmatched = set()

    def add_entry_basis(self, basis_state, aut_type):
        """ Initializes a new matching row with [0,0] or empty word evaluation """
        if aut_type == 'mealy':
            self.matchings[basis_state] = {ref_state: [0, 0]
                                        for ref_state in self.combined_model.states}
        else:
            self.matchings[basis_state] = dict()
            basis_out = basis_state.output 
            for ref_state in self.combined_model.states:
                if aut_type == 'dfa':
                    ref_out = ref_state.is_accepting
                elif aut_type == 'moore':
                    ref_out = ref_state.output
                if ref_out == basis_out:
                    self.matchings[basis_state][ref_state] = [1,1]
                else:
                    self.matchings[basis_state][ref_state] = [0,1]

    def get_score(self, basis_state, ref_state):
        """ Gets the score for a basis and reference states """
        if self.matchings[basis_state][ref_state][1] == 0:
            return 0
        return self.matchings[basis_state][ref_state][0]/self.matchings[basis_state][ref_state][1]

    def update_best_score(self, basis_state):
        """ Updates the best score for a basis state """
        score = max([self.get_score(basis_state, ref_state)
                    for ref_state in self.combined_model.states])
        self.best_score[basis_state] = round(score, 2)
        return score

    def update_best_match(self, basis_state, score):
        """ Updates the best match for a basis state """
        if score == 0:
            self.best_match[basis_state] = []
        else:
            matches = []
            for ref_state in self.combined_model.states:
                if self.get_score(basis_state, ref_state) == score:
                    matches.append(ref_state)

            self.best_match[basis_state] = matches

    def update_score(self, ob_tree, basis_state, reference_state, basis_state_access, defined_after_access, new_part):
        # calls update score for either mealy or moore/dfa 
        if ob_tree.automaton_type == 'mealy':
            return self.update_score_mealy(ob_tree, basis_state, reference_state, basis_state_access, defined_after_access, new_part)
        else:
            return self.update_score_moore(ob_tree, basis_state, reference_state, basis_state_access, defined_after_access, new_part)

    def update_score_mealy(self, ob_tree, basis_state, reference_state, basis_state_access, defined_after_access, new_part):
        """ 
        Updates the matching score for a basis and reference state based on the: 
            basis access, already defined part after access and the new part
        For every input in the new part, we take a step in the ob_tree and in the combined model
        If the combined model has no transition for some input (because it is not in the reference alphabet), we return
        If the outputs are equivalent, we add [1,1]
        If the outputs differ, we add [0,1]
        """
        current_ob_state = ob_tree.get_successor(tuple(basis_state_access) + tuple(defined_after_access))

        self.combined_model.execute_sequence(reference_state, defined_after_access)

        for inp in new_part:
            ob_out = current_ob_state.get_output(inp)
            if inp not in self.combined_model.current_state.output_fun:
                break

            ref_out = self.combined_model.step(inp)
            self.matchings[basis_state][reference_state][1] += 1
            if ob_out == ref_out:
                self.matchings[basis_state][reference_state][0] += 1
            current_ob_state = current_ob_state.get_successor(inp)

    def update_score_moore(self, ob_tree, basis_state, reference_state, basis_state_access, defined_after_access, new_part):
        """ 
        Updates the matching score for a basis and reference state based on the: 
            basis access, already defined part after access and the new part
        For every input in the new part, we take a step in the ob_tree and in the combined model
        If the combined model has no transition for some input (because it is not in the reference alphabet), we return
        If the outputs are equivalent, we add [1,1]
        If the outputs differ, we add [0,1]
        """
        orig_ref = reference_state
        current_ob_state = ob_tree.get_successor(tuple(basis_state_access) + tuple(defined_after_access))
        self.combined_model.execute_sequence(reference_state, defined_after_access)
        reference_state = self.combined_model.current_state

        for inp in new_part:
            if inp not in reference_state.transitions:
                return
            ref_out = self.combined_model.step(inp)
            reference_state = self.combined_model.current_state

            current_ob_state = current_ob_state.get_successor(inp)
            ob_out = current_ob_state.output

            self.matchings[basis_state][orig_ref][1] += 1
            if ob_out == ref_out:
                self.matchings[basis_state][orig_ref][0] += 1
