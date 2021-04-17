from collections import defaultdict

from aalpy.automata import Onfsm, OnfsmState
from aalpy.base import Automaton, SUL
from aalpy.utils.HelperFunctions import all_suffixes


class NonDetObservationTable:

    def __init__(self, alphabet: list, sul: SUL, n_sampling=100):
        """
        Construction of the non-deterministic observation table.

        Args:

            alphabet: input alphabet
            sul: system under learning
            n_sampling: number of samples to be performed for each cell
        """
        assert alphabet is not None and sul is not None

        self.A = [tuple([a]) for a in alphabet]
        self.S = list()  # prefixes of S
        self.S_dot_A = []
        self.E = [tuple([a]) for a in alphabet]
        self.T = defaultdict(dict)
        self.n_samples = n_sampling

        self.sul = sul
        empty_word = tuple()

        # Elements of S are in form that is presented in 'Learning Finite State Models of Observable Nondeterministic
        # Systems in a Testing Context'. Each element of S is a (inputs, outputs) tuple, where first element of the
        # tuple are inputs and second element of the tuple are outputs associated with inputs.
        self.S.append((empty_word, empty_word))

    def get_row_to_close(self):
        """
        Get row for that need to be closed.

        Returns:

            row that will be moved to S set and closed
        """
        s_rows = set()
        for s in self.S:
            s_rows.add(self.row_to_hashable(s))

        for t in self.S_dot_A:
            row_t = self.row_to_hashable(t)

            if row_t not in s_rows:
                self.S.append(t)
                self.S_dot_A.remove(t)
                return t

        return None

    def update_extended_S(self, row):
        """
        Helper generator function that returns extended S, or S.A set.
        For all values in the cell, create a new row where inputs is parent input plus element of alphabet, and
        output is parent output plus value in cell.

        Returns:

            New rows of extended S set.
        """
        s_set = set(self.S)
        extension = []
        for a in self.A:
            for t in self.T[row][a]:
                new_row = (row[0] + a, row[1] + tuple([t]))
                if new_row not in s_set:
                    extension.append(new_row)

        self.S_dot_A.extend(extension)
        return extension

    def update_obs_table(self, s_set=None, e_set: list = None):
        """
        Perform the membership queries.
        With  the  all-weather  assumption,  each  output  query  is  tried  a  number  of  times  on  the  system,
        and  the  driver  reports  the  set  of  all  possible  outputs.

        Args:

            s_set: Prefixes of S set on which to preform membership queries (Default value = None)
            e_set: Suffixes of E set on which to perform membership queries


        """

        update_S = s_set if s_set else self.S + self.S_dot_A
        update_E = e_set if e_set else self.E

        for s in update_S:
            for e in update_E:
                if e not in self.T[s].keys():
                    num_s_e_sampled = 0
                    while num_s_e_sampled < self.n_samples:
                        output = tuple(self.sul.query(s[0] + e))
                        # Here I basically say... add just the last element of the output if it e is element of alphabet
                        # else add last len(e) outputs
                        o = output[-1] if len(e) == 1 else tuple(output[-len(e):])
                        self.add_to_T((s[0], output[:len(s[1])]), e, o)
                        if output[:len(s[1])] == s[1]:
                            num_s_e_sampled += 1

    def gen_hypothesis(self) -> Automaton:
        """
        Generate automaton based on the values found in the observation table.

        Returns:

            Current hypothesis

        """
        state_distinguish = dict()
        states_dict = dict()
        initial = None

        stateCounter = 0
        for prefix in self.S:
            state_id = f's{stateCounter}'
            states_dict[prefix] = OnfsmState(state_id)

            states_dict[prefix].prefix = prefix
            state_distinguish[self.row_to_hashable(prefix)] = states_dict[prefix]

            if prefix == self.S[0]:
                initial = states_dict[prefix]
            stateCounter += 1

        for prefix in self.S:
            for a in self.A:
                for t in self.T[prefix][a]:
                    state_in_S = state_distinguish[self.row_to_hashable((prefix[0] + a, prefix[1] + tuple([t])))]
                    assert state_in_S
                    states_dict[prefix].transitions[a[0]].append((t, state_in_S))

        assert initial
        automaton = Onfsm(initial, [s for s in states_dict.values()])
        automaton.characterization_set = self.E

        return automaton

    def row_to_hashable(self, row_prefix):
        """
        Creates the hashable representation of the row. Frozenset is used as the order of element in each cell does not
        matter

        Args:

            row_prefix: prefix of the row in the observation table

        Returns:

            hashable representation of the row

        """
        row_repr = tuple()
        for e in self.E:
            #if e in self.T[row_prefix].keys():
            row_repr += (frozenset(self.T[row_prefix][e]),)
        return row_repr

    def add_to_T(self, s, e, value):
        """
        Add values to the cell at T[s][e].

        Args:

            s: prefix
            e: element of S
            value: value to be added to the cell


        """
        if e not in self.T[s]:
            self.T[s][e] = set()
        self.T[s][e].add(value)

    def cex_processing(self, cex: tuple):
        """
        Suffix processing strategy found in Shahbaz-Groz paper 'Inferring Mealy Machines'.
        It splits the counterexample into prefix and suffix. Prefix is the longest element of the S union S.A that
        matches the beginning of the counterexample. By removing such prefix from counterexample, no consistency check
        is needed.

        Args:

            cex: counterexample (inputs/outputs)

        Returns:
            suffixes to add to the E set

        """
        prefixes = list(self.S + self.S_dot_A)
        prefixes.reverse()
        trimmed_suffix = None

        cex = tuple(cex[0])  # cex[0] are inputs, cex[1] are outputs
        for p in prefixes:
            prefix_inputs = p[0]
            if prefix_inputs == tuple(cex[:len(prefix_inputs)]):
                trimmed_suffix = cex[len(prefix_inputs):]
                break

        if trimmed_suffix:
            suffixes = all_suffixes(trimmed_suffix)
        else:
            suffixes = all_suffixes(cex)
        suffixes.reverse()
        return suffixes
