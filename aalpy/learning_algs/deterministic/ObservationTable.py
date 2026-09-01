# Angluin-style observation table (S, E, T) used by the L* learning algorithm.
from collections import defaultdict

from aalpy.base import Automaton, SUL
from aalpy.automata import Dfa, DfaState, MealyState, MealyMachine, MooreMachine, MooreState

aut_type = ['dfa', 'mealy', 'moore']
closing_options = ['shortest_first', 'longest_first', 'single', 'single_longest']


class ObservationTable:
    """
    Angluin-style observation table over an S set of prefixes, an E set of suffixes, and a T function mapping
    prefixes to rows of observed outputs.
    """

    def __init__(self, alphabet: list, sul: SUL, automaton_type: str, prefixes_in_cell: bool = False) -> None:
        """
        Constructor of the observation table. Initial queries are asked in the constructor.

        :param list alphabet: Input alphabet.
        :param SUL sul: System under learning.
        :param str automaton_type: Automaton type, one of ['dfa', 'mealy', 'moore'].
        :param bool prefixes_in_cell: If True add prefixes of each element of E set to a cell, else only add the
            output (Default value = False).
        """
        assert automaton_type in aut_type
        assert alphabet is not None and sul is not None
        self.automaton_type = automaton_type

        # If True add prefixes of each element of E set to a cell, else only add the output
        self.prefixes_in_cell = prefixes_in_cell

        self.A = [tuple([a]) for a in alphabet]
        self.S = list()  # prefixes of S
        # DFA's can also take whole alphabet in E, this convention follows Angluin's paper
        self.E = [] if self.automaton_type != 'mealy' else [tuple([a]) for a in alphabet]
        # For performance reasons, the T function maps S to a tuple where element at index i is the element of the E
        # set of index i. Therefore it is important to keep E set ordered and ask membership queries only when needed
        # and in correct order. It would make more sense to implement it as a defaultdict(dict) where you can access
        # elements via self.T[s][e], but it causes significant performance hit.
        self.T = defaultdict(tuple)

        self.sul = sul
        empty_word = tuple()
        self.S.append(empty_word)

        # DFAs and Moore machines use empty word for identification of accepting states/state outputs
        if self.automaton_type == 'dfa' or self.automaton_type == 'moore':
            self.E.insert(0, empty_word)

    def get_rows_to_close(self, closing_strategy: str = 'longest_first') -> list | None:
        """
        Get rows for that need to be closed. Row selection is done according to closing_strategy.
        The length of the row is defined by the length of the prefix corresponding to the row in the S set.
        longest_first -> get all rows that need to be closed and ask membership queries for the longest row first
        shortest_first -> get all rows that need to be closed and ask membership queries for the shortest row first
        single -> find and ask membership query for the single row
        single_longest -> returns single longest row to close

        :param str closing_strategy: One of ['shortest_first', 'longest_first', 'single'] (Default value =
            'longest_first').
        :return list | None: Rows that will be moved to S set and closed, or None if all rows are already closed.
        """
        assert closing_strategy in closing_options
        rows_to_close = []
        row_values = set()

        s_rows = {self.T[s] for s in self.S}

        for t in self.s_dot_a():
            row_t = self.T[t]
            if row_t not in s_rows and row_t not in row_values:
                rows_to_close.append(t)
                row_values.add(row_t)

                if closing_strategy == 'single':
                    return rows_to_close

        if not rows_to_close:
            return None

        if 'longest' in closing_strategy:
            rows_to_close.sort(key=len, reverse=True)
            if closing_strategy == 'longest_first':
                return rows_to_close
            if closing_strategy == 'single_longest':
                return [rows_to_close[0]]

        return rows_to_close

    def get_causes_of_inconsistency(self) -> list | None:
        """
        If the two rows in the S set are the same, but their one letter extensions are not, this method finds
        the cause of inconsistency and returns it.

        :return list | None: A single-element list containing the a+e value that is the cause of inconsistency,
            or None if the table is consistent.
        """
        for i, s1 in enumerate(self.S):
            for s2 in self.S[i + 1:]:
                if self.T[s1] == self.T[s2]:
                    for a in self.A:
                        if self.T[s1 + a] != self.T[s2 + a]:
                            for index, e in enumerate(self.E):
                                if self.T[s1 + a][index] != self.T[s2 + a][index]:
                                    return [(a + e)]

        return None

    def s_dot_a(self):
        """
        Helper generator function that returns extended S, or S.A set.

        :return: Generator over elements of S.A that are not already in S.
        """
        s_set = set(self.S)
        for s in self.S:
            for a in self.A:
                if s + a not in s_set:
                    yield s + a

    def update_obs_table(self, s_set: list = None, e_set: list = None) -> None:
        """
        Perform the membership queries.

        :param list s_set: Prefixes of S set on which to perform membership queries. If None, then whole S set
            (plus S.A) will be used.
        :param list e_set: Suffixes of E set on which to perform membership queries. If None, then whole E set
            will be used.
        """

        update_S = s_set if s_set else list(self.S) + list(self.s_dot_a())
        update_E = e_set if e_set else self.E

        # This could save few queries
        update_S.reverse()

        for s in update_S:
            for e in update_E:
                if len(self.T[s]) != len(self.E):
                    output = tuple(self.sul.query(s + e))
                    if self.prefixes_in_cell and len(e) > 1:
                        obs_table_entry = tuple([output[-len(e):]],)
                    else:
                        obs_table_entry = (output[-1],)
                    self.T[s] += obs_table_entry

    def gen_hypothesis(self, no_cex_processing_used: bool = False) -> Automaton:
        """
        Generate automaton based on the values found in the observation table.

        :param bool no_cex_processing_used: If True, row representatives (deduplicated by row value) are used
            instead of the full S set, since no counterexample processing has narrowed it (Default value = False).
        :return Automaton: Automaton of type `automaton_type`.
        """
        state_distinguish = dict()
        states_dict = dict()
        initial_state = None
        automaton_class = {'dfa': Dfa, 'mealy': MealyMachine, 'moore': MooreMachine}

        s_set = self.S
        # Added check for the algorithm without counterexample processing
        if no_cex_processing_used:
            s_set = self._get_row_representatives()

        # create states based on S set
        stateCounter = 0
        for prefix in s_set:
            state_id = f's{stateCounter}'

            if self.automaton_type == 'dfa':
                states_dict[prefix] = DfaState(state_id)
                states_dict[prefix].is_accepting = self.T[prefix][0]
            elif self.automaton_type == 'moore':
                states_dict[prefix] = MooreState(state_id, output=self.T[prefix][0])
            else:
                states_dict[prefix] = MealyState(state_id)

            states_dict[prefix].prefix = prefix
            state_distinguish[tuple(self.T[prefix])] = states_dict[prefix]

            if not prefix:
                initial_state = states_dict[prefix]
            stateCounter += 1

        # add transitions based on extended S set
        for prefix in s_set:
            for a in self.A:
                state_in_S = state_distinguish[self.T[prefix + a]]
                states_dict[prefix].transitions[a[0]] = state_in_S
                if self.automaton_type == 'mealy':
                    states_dict[prefix].output_fun[a[0]] = self.T[prefix][self.E.index(a)]

        automaton = automaton_class[self.automaton_type](initial_state, list(states_dict.values()))
        automaton.characterization_set = self.E

        return automaton

    def _get_row_representatives(self) -> list:
        """
        Selects a single representative prefix per distinct row value, preferring the shortest prefix.

        :return list: List of representative prefixes, one per distinct row value.
        """
        self.S.sort(key=len)
        representatives = defaultdict(list)
        for prefix in self.S:
            representatives[self.T[prefix]].append(prefix)

        return [r[0] for r in representatives.values()]
