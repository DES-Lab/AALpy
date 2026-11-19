# Observation table used by the abstracted ONFSM L* learning algorithm.
from collections import defaultdict
from typing import Any

from aalpy.automata import Onfsm, OnfsmState
from aalpy.learning_algs.non_deterministic.OnfsmObservationTable import NonDetObservationTable
from aalpy.learning_algs.non_deterministic.NonDeterministicSULWrapper import NonDeterministicSULWrapper
from aalpy.utils.HelperFunctions import all_suffixes, extend_set


class AbstractedNonDetObservationTable:
    """
    Observation table for learning abstracted observable non-deterministic finite state machines, where outputs
    are grouped into equivalence classes via an abstraction mapping.
    """

    def __init__(self, alphabet: list, sul: NonDeterministicSULWrapper, abstraction_mapping: dict,
                 n_sampling: int = 100) -> None:
        """
        Construction of the abstracted non-deterministic observation table.

        :param list alphabet: Input alphabet.
        :param NonDeterministicSULWrapper sul: System under learning.
        :param dict abstraction_mapping: Map that translates outputs to abstracted outputs.
        :param int n_sampling: Number of samples to be performed for each cell.
        """

        assert alphabet is not None and sul is not None

        self.observation_table = NonDetObservationTable(alphabet, sul, n_sampling)

        self.S = list()
        self.S_dot_A = []
        self.E = []
        self.T = defaultdict(dict)
        self.A = [tuple([a]) for a in alphabet]

        self.abstraction_mapping = abstraction_mapping
        self.sul = sul

        empty_word = tuple()
        self.S.append((empty_word, empty_word))

    def update_obs_table(self, s_set: list[tuple[tuple, tuple]] | None = None,
                          e_set: list[tuple] | None = None) -> None:
        """
        Perform the membership queries and abstraction on observation table.
        With the all-weather assumption, each output query is tried a number of times on the system,
        and the driver reports the set of all possible outputs.

        :param list[tuple[tuple, tuple]] | None s_set: Prefixes of S set on which to perform membership queries
            (Default value = None).
        :param list[tuple] | None e_set: Suffixes of E set on which to perform membership queries.
        """

        if s_set is None:
            s_set = self.S + self.S_dot_A

        self.observation_table.query_missing_observations(s_set, e_set)
        self.abstract_obs_table()
        self.clean_obs_table()

    def abstract_obs_table(self) -> None:
        """
        Creation of abstracted observation table. The provided abstraction mapping is used to
        replace outputs by abstracted outputs.
        """

        self.S = self.observation_table.S
        self.S_dot_A = list(set(self.observation_table.get_extended_S()).union(set(self.S_dot_A) - set(self.S)))
        self.E = self.observation_table.E

        update_S = self.S + self.S_dot_A
        update_E = self.E

        for s in update_S:
            for e in update_E:
                for o_tup in self.get_all_outputs(s, e):
                    abstracted_outputs = []
                    o_tup = tuple([o_tup])
                    for outputs in o_tup:
                        for o in outputs:
                            abstract_output = self.get_abstraction(o)
                            abstracted_outputs.append(abstract_output)
                    self.add_to_T(s, e, tuple(abstracted_outputs))

    def add_to_T(self, s: tuple[tuple, tuple], e: tuple, value: tuple) -> None:
        """
        Add values to the cell at T[s][e].

        :param tuple[tuple, tuple] s: Prefix.
        :param tuple e: Element of E.
        :param tuple value: Value to be added to the cell.
        """
        if e not in self.T[s]:
            self.T[s][e] = set()
        self.T[s][e].add(value)

    # CHANGED
    # helper function
    def get_all_outputs(self, s: tuple[tuple, tuple], e: tuple) -> set:
        """
        Collects all observed output traces for the given row and suffix.

        :param tuple[tuple, tuple] s: Prefix.
        :param tuple e: Element of E.
        :return set: Set of observed output traces.
        """
        cell_outputs = set()
        cell_outputs.update(self.sul.cache.get_all_traces(s, e))
        return cell_outputs

    def update_extended_S(self, row_prefix: tuple[tuple, tuple] | None = None) -> list[tuple[tuple, tuple]]:
        """
        Helper function that returns extended S, or S.A set.
        For all values in the cell, create a new row where inputs is parent input plus element of alphabet, and
        output is parent output plus value in cell.

        :param tuple[tuple, tuple] | None row_prefix: If given, only extend this single row instead of all of S.
        :return list[tuple[tuple, tuple]]: New rows of extended S set.
        """
        return self.observation_table.get_extended_S(row_prefix=row_prefix)

    def get_row_to_close(self) -> tuple[tuple, tuple] | None:
        """
        Get row that needs to be closed.

        :return tuple[tuple, tuple] | None: Row that will be moved to S set and closed, or None if all rows are
            already closed.
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

    def get_row_to_complete(self) -> tuple[tuple, tuple] | None:
        """
        Get row that needs to be completed.

        :return tuple[tuple, tuple] | None: Row that will be added to S.A, or None if the table is complete.
        """

        s_rows = set()
        for s in self.S:
            s_rows.add(tuple((s, self.row_to_hashable(s))))

        for s_row in s_rows:
            similar_s_dot_a_rows = []
            for t in self.S_dot_A:
                row_t = self.row_to_hashable(t)
                if row_t == s_row[1]:
                    similar_s_dot_a_rows.append(t)
            similar_s_dot_a_rows.sort(key=lambda row: len(row[0]))
            for a in self.A:
                complete_outputs = self.get_all_outputs(s_row[0], a)
                for similar_s_dot_a_row in similar_s_dot_a_rows:
                    t_row_outputs = self.get_all_outputs(similar_s_dot_a_row, a)
                    output_difference = t_row_outputs.difference(complete_outputs)
                    if len(output_difference) > 0:
                        for o in output_difference:
                            extension = (similar_s_dot_a_row[0] + a, similar_s_dot_a_row[1] + tuple([o[0]]))
                            if extension not in self.S and extension not in self.S_dot_A:
                                return extension
                            else:
                                complete_outputs = complete_outputs.union(output_difference)

        return None

    def get_row_to_make_consistent(self) -> tuple | None:
        """
        Get row that violates consistency.

        :return tuple | None: Distinguishing input sequence that violates consistency, or None if the table is
            consistent.
        """
        unified_S = self.S + self.S_dot_A
        s_rows = set()
        for s in self.S:
            s_rows.add(tuple((s, self.row_to_hashable(s))))

        for s_row in s_rows:
            similar_s_dot_a_rows = []
            for t in self.S_dot_A:
                row_t = self.row_to_hashable(t)
                if row_t == s_row[1]:
                    similar_s_dot_a_rows.append(t)

            similar_s_dot_a_rows.sort(key=lambda row: len(row[0]))

            for a in self.A:
                # CHANGED
                #                 outputs = self.observation_table.T[s_row[0]][a]
                outputs = self.get_all_outputs(s_row[0], a)
                for o in outputs:
                    extended_s_sequence = (s_row[0][0] + a, s_row[0][1] + tuple([o]))
                    if extended_s_sequence in unified_S:
                        extended_s_sequence_row = self.row_to_hashable(extended_s_sequence)
                        for similar_s_dot_a_row in similar_s_dot_a_rows:
                            extended_s_dot_a_sequence = (
                                similar_s_dot_a_row[0] + a, similar_s_dot_a_row[1] + tuple([o]))
                            if extended_s_dot_a_sequence in unified_S:
                                extended_s_dot_a_sequence_row = self.row_to_hashable(extended_s_dot_a_sequence)
                                if extended_s_sequence_row is not extended_s_dot_a_sequence_row:
                                    return self.get_distinctive_input_sequence(extended_s_sequence,
                                                                               extended_s_dot_a_sequence, a)

        return None

    def get_distinctive_input_sequence(self, first_row: tuple[tuple, tuple], second_row: tuple[tuple, tuple],
                                        inp: tuple) -> tuple | None:
        """
        Get input sequence that leads to a different output sequence for two given input/output sequences.

        :param tuple[tuple, tuple] first_row: Row to be compared.
        :param tuple[tuple, tuple] second_row: Row to be compared.
        :param tuple inp: Appended input to first_row and second_row that leads to different state.
        :return tuple | None: Input sequence that leads to different outputs, or None if none is found.
        """
        for e in self.E:
            if len(self.T[first_row][e].difference(self.T[second_row][e])) > 0:
                return tuple([inp]) + e

        return None

    def update_E(self, seq: tuple) -> list[tuple]:
        """
        Adds a suffix to the E set if not already present.

        :param tuple seq: Suffix to add.
        :return list[tuple]: The newly added suffix as a single-element list, or an empty list if it was
            already present. The caller (run_abstracted_ONFSM_Lstar) passes this straight on as the
            e_set of update_obs_table(), so it must not be None/omitted - without a return statement
            here it was always None, silently making update_obs_table() query every column of E instead
            of just the new one.
        """
        if seq not in self.E:
            self.E.append(seq)
            return [seq]
        return []

    def clean_obs_table(self) -> None:
        """
        Moves duplicates from S to S_dot_A. The entries in S_dot_A which are based on the moved row get deleted.
        The table will be smaller and more efficient.
        """
        # just for testing without cleaning
        # return False

        tmp_S = self.S.copy()
        tmp_both_S = self.S + self.S_dot_A
        hashed_rows_from_s = set()

        tmp_S.sort(key=lambda t: len(t[0]))

        for s in tmp_S:
            hashed_s_row = self.row_to_hashable(s)
            if hashed_s_row in hashed_rows_from_s:
                # self.S is the very same list object as self.observation_table.S (aliased in
                # abstract_obs_table), so removing from one already removes from the other -
                # calling .remove() on both raised ValueError: x not in list on the second call.
                if s in self.S:
                    self.S.remove(s)
                size = len(s[0])
                for row_prefix in tmp_both_S:
                    s_both_row = (row_prefix[0][:size], row_prefix[1][:size])
                    if s != row_prefix and s == s_both_row:
                        if row_prefix in self.S:
                            self.S.remove(row_prefix)
            else:
                hashed_rows_from_s.add(hashed_s_row)

    def row_to_hashable(self, row_prefix: tuple[tuple, tuple]) -> tuple:
        """
        Creates the hashable representation of the row. Frozenset is used as the order of element in each cell does
        not matter.

        :param tuple[tuple, tuple] row_prefix: Prefix of the row in the observation table.
        :return tuple: Hashable representation of the row.
        """
        row_repr = tuple()
        for e in self.E:
            # if e in self.T[row_prefix].keys():
            row_repr += (frozenset(self.T[row_prefix][e]),)
        return row_repr

    def gen_hypothesis(self) -> Onfsm:
        """
        Generate automaton based on the values found in the abstracted observation table.

        :return Onfsm: Current abstracted hypothesis.
        """
        state_distinguish = dict()
        states_dict = dict()
        initial = None

        unified_S = self.S + self.S_dot_A

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
            similar_rows = []
            for row in unified_S:
                if self.row_to_hashable(row) == self.row_to_hashable(prefix):
                    similar_rows.append(row)
            for row in similar_rows:
                for a in self.A:
                    for t in self.get_all_outputs(row, a):
                        s_entry = (row[0] + a, row[1] + t)
                        if s_entry in unified_S:
                            state_in_S = state_distinguish[self.row_to_hashable(s_entry)]

                            if (t[0], state_in_S) not in states_dict[prefix].transitions[a[0]]:
                                states_dict[prefix].transitions[a[0]].append((t[0], state_in_S))

        assert initial
        automaton = Onfsm(initial, [s for s in states_dict.values()])
        automaton.characterization_set = self.E

        return automaton

    def extend_S_dot_A(self, cex_prefixes: list[tuple[tuple, tuple]]) -> list[tuple[tuple, tuple]]:
        """
        Extends S.A based on counterexample prefixes.

        :param list[tuple[tuple, tuple]] cex_prefixes: Input/output sequences that are added to S.A.
        :return list[tuple[tuple, tuple]]: Input/output sequences that have been added to the S.A.
        """
        prefixes = self.S + self.S_dot_A
        prefixes_to_extend = []
        for cex_prefix in cex_prefixes:
            if cex_prefix not in prefixes:
                prefixes_to_extend.append(cex_prefix)
                self.S_dot_A.append(cex_prefix)
        return prefixes_to_extend

    def get_abstraction(self, out: Any) -> Any:
        """
        Get an abstraction for a concrete output. If such abstraction is not defined, return output.

        :param Any out: Output to be abstracted if possible.
        :return Any: Abstracted output or output itself.
        """
        return self.abstraction_mapping[out] if out in self.abstraction_mapping.keys() else out

    def cex_processing(self, cex: tuple[list, list], hypothesis: Onfsm) -> None:
        """
        Add counterexample to the observation table. If the counterexample leads to a state where an output of the
        same equivalence class already exists, the prefixes of the counterexample are added to S.A.
        Otherwise, the postfixes of counterexample are added to E.

        :param tuple[list, list] cex: (inputs, outputs) counterexample that should be added to the observation
            table.
        :param Onfsm hypothesis: ONFSM that implements the counterexample.
        """

        cex_len = len(cex[0])
        hypothesis.reset_to_initial()

        for step in range(0, cex_len - 1):
            hypothesis.step_to(cex[0][step], cex[1][step])

        possible_outputs = hypothesis.outputs_on_input(cex[0][cex_len - 1])

        equivalent_output = False

        for out in possible_outputs:
            abstracted_out = self.get_abstraction(out)
            abstracted_out_cex = self.get_abstraction(cex[1][cex_len - 1])
            if abstracted_out_cex == abstracted_out:
                equivalent_output = True
                break

        if equivalent_output:
            # add prefixes of cex to S_dot_A
            cex_prefixes = [(tuple(cex[0][0:i + 1]), tuple(cex[1][0:i + 1])) for i in range(0, len(cex[0]))]
            prefixes_to_extend = self.extend_S_dot_A(cex_prefixes)

            # CHANGED: REMOVED
            # self.observation_table.S_dot_A.extend(prefixes_to_extend)
            self.update_obs_table(s_set=prefixes_to_extend)
        else:
            # add distinguishing suffixes of cex to E
            # CHANGED CEX PROX
            # TODO: this will now not work as cex processing was changed
            # cex_suffixes = non_det_longest_prefix_cex_processing(self.observation_table, cex)
            cex_suffixes = all_suffixes(cex[0])

            added_suffixes = extend_set(self.observation_table.E, cex_suffixes)
            self.update_obs_table(e_set=added_suffixes)

    def clean_tables(self) -> None:
        """
        Cleans both the underlying observation table and the abstracted table, moving duplicate rows from S to
        S_dot_A while keeping both tables consistent with each other.
        """

        self.observation_table.clean_obs_table()
        self.abstract_obs_table()

        update_S = self.S.copy()
        whole_S = self.S + self.S_dot_A

        update_S.sort()
        update_S.sort(key=lambda t: len(t[0]))

        s_rows = set()
        for s in update_S:
            hashed_s_row = self.row_to_hashable(s)
            if hashed_s_row not in s_rows:
                s_rows.add(hashed_s_row)
            else:
                size = len(s[0])
                for row in whole_S:
                    cmp_row = (row[0][:size], row[1][:size])
                    if s == cmp_row:
                        if row in self.S_dot_A:
                            self.S_dot_A.remove(row)
                        elif row in self.S:
                            self.S.remove(row)

                self.S_dot_A.append(s)
                self.S.remove(s)
