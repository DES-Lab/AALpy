from collections import defaultdict

from aalpy.base import Automaton, SUL
from aalpy.automata import Dfa, DfaState, MealyState, MealyMachine, MooreMachine, MooreState

aut_type = ['dfa', 'mealy', 'moore']
closing_options = ['shortest_first', 'longest_first', 'single']


class ObservationTable:
    def __init__(self, alphabet: list, sul: SUL, automaton_type="mealy"):
        """
        Constructor of the observation table. Initial queries are asked in the constructor.

        Args:

            alphabet: input alphabet
            sul: system under learning
            automaton_type: automaton type, one of ['dfa', 'mealy', 'moore']

        Returns:

        """
        assert automaton_type in aut_type
        assert alphabet is not None and sul is not None
        self.automaton_type = automaton_type

        self.A = [tuple([a]) for a in alphabet]
        self.S = list()  # prefixes of S
        # DFA's can also take whole alphabet in E, this convention follows Angluin's paper
        self.E = [] if self.automaton_type == 'dfa' else [tuple([a]) for a in alphabet]
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

    def get_rows_to_close(self, closing_strategy='longest_first'):
        """
        Get rows for that need to be closed. Row selection is done according to closing_strategy.
        The length of the row is defined by the length of the prefix corresponding to the row in the S set.
        longest_first -> get all rows that need to be closed and ask membership queries for the longest row first
        shortest_first -> get all rows that need to be closed and ask membership queries for the shortest row first
        single -> find and ask membership query for the single row

        Args:

            closing_strategy: one of ['shortest_first', 'longest_first', 'single'] (Default value = 'longest_first')

        Returns:

            list if non-closed exist, None otherwise: rows that will be moved to S set and closed

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

        if closing_strategy == 'longest_first':
            rows_to_close.reverse()

        return rows_to_close

    def get_causes_of_inconsistency(self):
        """
        If the two rows in the S set are the same, but their one letter extensions are not, this method founds
        the cause of inconsistency and returns it.
        :return:

        Returns:

            a+e values that are the causes of inconsistency

        """
        causes_of_inconsistency = set()
        for i, s1 in enumerate(self.S):
            for s2 in self.S[i + 1:]:
                if self.T[s1] == self.T[s2]:
                    for a in self.A:
                        if self.T[s1 + a] != self.T[s2 + a]:
                            for index, e in enumerate(self.E):
                                if self.T[s1 + a][index] != self.T[s2 + a][index]:
                                    causes_of_inconsistency.add(a + e)

        if not causes_of_inconsistency:
            return None
        return causes_of_inconsistency

    def s_dot_a(self):
        """
        Helper generator function that returns extended S, or S.A set.
        """
        s_set = set(self.S)
        for s in self.S:
            for a in self.A:
                if s + a not in s_set:
                    yield s + a

    def update_obs_table(self, s_set: list = None, e_set: list = None):
        """
        Perform the membership queries.

        Args:

            s_set: Prefixes of S set on which to preform membership queries. If None, then whole S set will be used.

            e_set: Suffixes of E set on which to perform membership queries. If None, then whole E set will be used.

        Returns:

        """

        update_S = s_set if s_set else list(self.S) + list(self.s_dot_a())
        update_E = e_set if e_set else self.E

        # This could save few queries
        update_S.reverse()

        for s in update_S:
            for e in update_E:
                if len(self.T[s]) != len(self.E):
                    output = self.sul.query(s + e)
                    self.T[s] += (output[-1],)

    def gen_hypothesis(self, check_for_duplicate_rows=False) -> Automaton:
        """
        Generate automaton based on the values found in the observation table.
        :return:

        Args:

            check_for_duplicate_rows:  (Default value = False)

        Returns:

            Automaton of type `automaton_type`

        """
        state_distinguish = dict()
        states_dict = dict()
        initial_state = None
        automaton_class = {'dfa': Dfa, 'mealy': MealyMachine, 'moore': MooreMachine}

        # delete duplicate rows, only possible if no counterexample processing is present
        # counterexample processing removes the need for consistency check, as it ensures
        # that no two rows in the S set are the same
        if check_for_duplicate_rows:
            rows_to_delete = set()
            for i, s1 in enumerate(self.S):
                for s2 in self.S[i + 1:]:
                    if self.T[s1] == self.T[s2]:
                        rows_to_delete.add(s2)

            for row in rows_to_delete:
                self.S.remove(row)

        # create states based on S set
        stateCounter = 0
        for prefix in self.S:
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
        for prefix in self.S:
            for a in self.A:
                state_in_S = state_distinguish[self.T[prefix + a]]
                states_dict[prefix].transitions[a[0]] = state_in_S
                if self.automaton_type == 'mealy':
                    states_dict[prefix].output_fun[a[0]] = self.T[prefix][self.E.index(a)]

        automaton = automaton_class[self.automaton_type](initial_state, list(states_dict.values()))
        automaton.characterization_set = self.E

        return automaton
