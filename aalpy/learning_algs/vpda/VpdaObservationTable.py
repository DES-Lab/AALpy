from collections import defaultdict

from aalpy.automata.Vpa import VpaTransition
from aalpy.base import Automaton, SUL
from aalpy.automata import Vpa, VpaState

aut_type = ['pda', 'vpa']
closing_options = ['shortest_first', 'longest_first', 'single', 'single_longest']


class VpdaObservationTable:
    def __init__(self, alphabet: list, sul: SUL, automaton_type, prefixes_in_cell=False):
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

        if self.automaton_type == 'vpa':
            self.call_set = alphabet[0]
            self.return_set = alphabet[1]
            self.internal_set = alphabet[2]
            self.merged_alphabet = list()
            self.merged_alphabet.extend(alphabet[0])
            self.merged_alphabet.extend(alphabet[1])
            self.merged_alphabet.extend(alphabet[2])

        # If True add prefixes of each element of E set to a cell, else only add the output
        self.prefixes_in_cell = prefixes_in_cell

        if automaton_type == 'vpa':
            self.A = [tuple(a) for a in self.merged_alphabet]
        else:
            self.A = [tuple([a]) for a in alphabet]

        self.S = list()  # prefixes of S
        # DFA's can also take whole alphabet in E, this convention follows Angluin's paper
        self.E = []
        # For performance reasons, the T function maps S to a tuple where element at index i is the element of the E
        # set of index i. Therefore it is important to keep E set ordered and ask membership queries only when needed
        # and in correct order. It would make more sense to implement it as a defaultdict(dict) where you can access
        # elements via self.T[s][e], but it causes significant performance hit.
        self.T = defaultdict(tuple)

        self.sul = sul
        empty_word = tuple()
        self.S.append(empty_word)

        # DFAs and Moore machines use empty word for identification of accepting states/state outputs
        self.E.insert(0, empty_word)

    def get_rows_to_close(self, closing_strategy='longest_first'):
        """
        Get rows for that need to be closed. Row selection is done according to closing_strategy.
        The length of the row is defined by the length of the prefix corresponding to the row in the S set.
        longest_first -> get all rows that need to be closed and ask membership queries for the longest row first
        shortest_first -> get all rows that need to be closed and ask membership queries for the shortest row first
        single -> find and ask membership query for the single row
        single_longest -> returns single longest row to close

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

        if 'longest' in closing_strategy:
            rows_to_close.sort(key=len, reverse=True)
            if closing_strategy == 'longest_first':
                return rows_to_close
            if closing_strategy == 'single_longest':
                return [rows_to_close[0]]

        return rows_to_close

    def get_causes_of_inconsistency(self):
        """
        If the two rows in the S set are the same, but their one letter extensions are not, this method founds
        the cause of inconsistency and returns it.
        :return:

        Returns:

            a+e values that are the causes of inconsistency

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
                    output = tuple(self.sul.query(s + e))
                    if self.prefixes_in_cell and len(e) > 1:
                        obs_table_entry = tuple([output[-len(e):]],)
                    else:
                        obs_table_entry = (output[-1],)
                    self.T[s] += obs_table_entry

    def get_action_type(self, letter) -> str:
        if letter in self.call_set:
            return 'push'
        elif letter in self.return_set:
            return 'pop'
        elif letter in self.internal_set:
            return ''
        else:
            assert False

    def get_stack_guard(self, prefix, letter, action):
        """

        Gets the stack guard based on the action and word (prefix + letter)

        """
        out = self.sul.query(prefix + letter)
        out_pre = self.sul.query(prefix)
        if action == 'push':
            if out_pre[-1][1] == out[-1][1] and out_pre[-1][1] == '_':  # stack doesn't change on push action
                stack_guard = '?'
            else:                           # stack changed so we know the push action worked
                stack_guard = out[-1][1]
        elif action == 'pop':
            if out_pre[-1][1] == out[-1][1]:  # stack doesn't change on pop action
                stack_guard = '?'
            else:                           # stack changed so we know the pop operation worked
                stack_guard = out_pre[-1][1]
        else:
            stack_guard = ''

        return stack_guard


    def gen_hypothesis(self, no_cex_processing_used=False) -> Automaton:
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
        automaton_class = {'vpa': Vpa}

        s_set = self.S
        # Added check for the algorithm without counterexample processing
        if no_cex_processing_used:
            s_set = self._get_row_representatives()

        # create states based on S set
        stateCounter = 0
        for prefix in s_set:
            state_id = f's{stateCounter}'

            states_dict[prefix] = VpaState(state_id)
            states_dict[prefix].is_accepting = self.T[prefix][0][0]

            states_dict[prefix].prefix = prefix
            state_distinguish[tuple(self.T[prefix])] = states_dict[prefix]

            if not prefix:
                initial_state = states_dict[prefix]
            stateCounter += 1

        for prefix in s_set:
            for a in self.A:
                prev_state = state_distinguish[self.T[prefix]]
                target_state = state_distinguish[self.T[prefix + a]]
                action = self.get_action_type(a[0])
                stack_guard = self.get_stack_guard(prefix, a, action)
                if stack_guard == '?':
                    target_state = Vpa.error_state
                trans = VpaTransition(start=prev_state, target=target_state, symbol=a[0], action=action, stack_guard=stack_guard)
                states_dict[prefix].transitions[a[0]].append(trans)

        if self.automaton_type == 'vpa':
            automaton = automaton_class[self.automaton_type](initial_state, list(states_dict.values()), self.call_set, self.return_set, self.internal_set)
        else:
            automaton = automaton_class[self.automaton_type](initial_state, list(states_dict.values()))

        automaton.characterization_set = self.E

        return automaton

    def _get_row_representatives(self):
        self.S.sort(key=len)
        representatives = defaultdict(list)
        for prefix in self.S:
            representatives[self.T[prefix]].append(prefix)

        return [r[0] for r in representatives.values()]
