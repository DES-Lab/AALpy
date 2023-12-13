import random
from collections import defaultdict, deque
from typing import Union

from aalpy.base import Automaton, AutomatonState

from typing import List, Dict


class SevpaAlphabet:
    """
    The Alphabet of a 1-SEVPA.

    Attributes:
        internal_alphabet (List[str]): Letters for internal transitions.
        call_alphabet (List[str]): Letters for push transitions.
        return_alphabet (List[str]): Letters for pop transitions.
        exclusive_call_return_pairs (Dict[str, str]): A dictionary representing exclusive pairs
            of call and return symbols.
    """

    def __init__(self, internal_alphabet: List[str], call_alphabet: List[str], return_alphabet: List[str],
                 exclusive_call_return_pairs: Dict[str, str] = None):
        self.internal_alphabet = internal_alphabet
        self.call_alphabet = call_alphabet
        self.return_alphabet = return_alphabet
        self.exclusive_call_return_pairs = exclusive_call_return_pairs

    def get_merged_alphabet(self) -> List[str]:
        """
        Get the merged alphabet, including internal, call, and return symbols.

        Returns:
            List[str]: A list of all symbols in the alphabet.
        """
        alphabet = list()
        alphabet.extend(self.internal_alphabet)
        alphabet.extend(self.call_alphabet)
        alphabet.extend(self.return_alphabet)
        return alphabet

    def __str__(self) -> str:
        """
        Returns:
            str: A string representation of the alphabet.
        """
        return f'Internal: {self.internal_alphabet} Call: {self.call_alphabet} Return: {self.return_alphabet}'


class SevpaState(AutomatonState):
    """
    Single state of a 1-SEVPA.
    """

    def __init__(self, state_id, is_accepting=False):
        super().__init__(state_id)
        self.transitions = defaultdict(list[SevpaTransition])
        self.is_accepting = is_accepting


class SevpaTransition:
    """
    Represents a transition in a 1-SEVPA.

    Attributes:
        target (SevpaState): The target state of the transition.
        letter: The symbol associated with the transition.
        action: The action performed during the transition (pop | None).
        stack_guard: Pair of (automaton_state_id, call_letter)
    """

    def __init__(self, target: SevpaState, letter, action, stack_guard=None):
        self.target_state = target
        self.letter = letter
        self.action = action
        self.stack_guard = stack_guard

    def __str__(self):
        """
        Returns:
            str: A string representation of the transition.
        """
        return f'{self.letter} --> {self.target_state.state_id}' + \
               f' | {self.action}: {self.stack_guard}' if self.stack_guard else ''


class Sevpa(Automaton):
    """
    1-Module Single Entry Visibly Pushdown Automaton.
    """
    empty = "_"

    def __init__(self, initial_state: SevpaState, states: list[SevpaState]):
        super().__init__(initial_state, states)
        self.initial_state = initial_state
        self.states = states
        self.input_alphabet = self.get_input_alphabet()
        self.current_state = None
        self.stack = []
        self.error_state_reached = False

        # alphabet sets for faster inclusion checks (as in SevpaAlphabet we have lists, for reproducibility)
        self.internal_set = set(self.input_alphabet.internal_alphabet)
        self.call_set = set(self.input_alphabet.call_alphabet)
        self.return_set = set(self.input_alphabet.return_alphabet)

    def reset_to_initial(self):
        super().reset_to_initial()
        self.current_state = self.initial_state
        self.stack = [self.empty]
        self.error_state_reached = False
        return self.current_state.is_accepting and self.stack[-1] == self.empty

    def step(self, letter):
        """
        Perform a single step on the 1-SEVPA by transitioning with the given input letter.

        Args:
            letter: A single input that is looked up in the transition table of the SevpaState.

        Returns:
            bool: True if the reached state is an accepting state and the stack is empty, False otherwise.
        """
        if self.error_state_reached:
            return False

        if letter is None:
            return self.current_state.is_accepting and self.stack[-1] == self.empty

        if letter in self.call_set:
            self.stack.append((self.current_state.state_id, letter))
            self.current_state = self.initial_state
            return self.current_state.is_accepting and self.stack[-1] == self.empty

        # get possible transitions
        transitions = self.current_state.transitions[letter]
        taken_transition = None
        for t in transitions:
            if t.letter in self.return_set:
                if t.stack_guard == self.stack[-1]:
                    taken_transition = t
                    break
            elif t.letter in self.internal_set:
                taken_transition = t
                break
            else:
                assert False

        # No transition is possible
        if not taken_transition:
            self.error_state_reached = True
            return False

        self.current_state = taken_transition.target_state

        if taken_transition.action == 'pop':
            # empty stack elem should always be on the stack
            if len(self.stack) <= 1:
                self.error_state_reached = True
                return False
            self.stack.pop()

        return self.current_state.is_accepting and self.stack[-1] == self.empty

    def get_state_by_id(self, state_id) -> Union[SevpaState, None]:
        for state in self.states:
            if state.state_id == state_id:
                return state
        return None

    def is_input_complete(self) -> bool:
        pass

    def execute_sequence(self, origin_state, seq):
        if origin_state.prefix != self.initial_state.prefix:
            assert False, 'execute_sequence for Sevpa only is only supported from the initial state.'
        self.reset_to_initial()
        self.current_state = origin_state
        return [self.step(s) for s in seq]

    def to_state_setup(self):
        state_setup_dict = {}

        sorted_states = sorted(self.states, key=lambda x: x.state_id)
        for state in sorted_states:
            transitions_for_symbol = {}
            for symbol, transition_list in state.transitions.items():
                trans_list_for_setup = []
                for transition in transition_list:
                    trans_list_for_setup.append(
                        (transition.target_state.state_id, transition.action, transition.stack_guard))
                if trans_list_for_setup:
                    transitions_for_symbol[symbol] = trans_list_for_setup
            state_setup_dict[state.state_id] = (state.is_accepting, transitions_for_symbol)

        return state_setup_dict

    @staticmethod
    def from_state_setup(state_setup: dict, **kwargs):

        init_state_id = kwargs['init_state_id']

        # build states with state_id and output
        states = {key: SevpaState(key, val[0]) for key, val in state_setup.items()}
        # states[Sevpa.error_state.state_id] = Sevpa.error_state  # PdaState(Pda.error_state,False)

        # add transitions to states
        for state_id, state in states.items():
            for _input, trans_spec in state_setup[state_id][1].items():
                for (target_state_id, action, stack_guard) in trans_spec:
                    if action == 'pop':
                        stack_guard = (stack_guard[0], stack_guard[1])
                        trans = SevpaTransition(target=states[target_state_id], letter=_input,
                                                action=action, stack_guard=stack_guard)
                    elif action is None:
                        trans = SevpaTransition(target=states[target_state_id], letter=_input,
                                                action=None, stack_guard=None)
                    else:
                        assert False, 'Action must either be "pop" or None, note that there are no push actions ' \
                                      'definitions in SEVPA'

                    state.transitions[_input].append(trans)

        init_state = states[init_state_id]
        return Sevpa(init_state, [state for state in states.values()])

    def transform_access_string(self, state=None, stack_content=None) -> List[str]:
        """
        Transform the access string by omitting redundant call and return letters, as well as internal letters.

        This function creates the following word:
            For every element in the stack (except the first element '_'):
                - Append the state prefix from where the stack element was pushed
                - Append the call letter
            Append the state prefix from the state where you are calling this function from.

        Args:
            state: The state from which the transformation is initiated (default: initial state).
            stack_content: The content of the stack for transformation (default: Current Stack content).

        Returns:
            List[str]: The transformed access string.
        """
        word = []
        calling_state = self.initial_state if not state else state
        stack = self.stack if not stack_content else stack_content

        for index, stack_elem in enumerate(stack):
            # Skip the first element because it's the start of the stack '_'
            if index == 0:
                continue
            from_state_id = stack_elem[0]  # The corresponding state where the stack element was pushed from
            call_letter = stack_elem[1]  # The call letter that was pushed on the stack
            from_state = self.get_state_by_id(from_state_id)
            if from_state.prefix != ():
                word.extend(from_state.prefix)
            word.append(call_letter)
        word.extend(calling_state.prefix)
        return word

    @staticmethod
    def create_daisy_hypothesis(initial_state, alphabet):
        """
        Create a Daisy Hypothesis 1-SEVPA using the given initial state and alphabet.

        This function creates self-loop transitions for the internal state on every internal letter.
        Additionally, it creates self-loop transitions with a pop action for every call letter.

        Args:
            initial_state (SevpaState): The initial state of the 1-SEVPA.
            alphabet (SevpaAlphabet): The alphabet for the 1-SEVPA.

        Returns:
            Sevpa: The created 1-SEVPA with the specified initial state and alphabet.
        """
        for i in alphabet.internal_alphabet:
            trans = SevpaTransition(target=initial_state, letter=i, action=None)
            initial_state.transitions[i].append(trans)

        for c in alphabet.call_alphabet:
            for r in alphabet.return_alphabet:
                trans = SevpaTransition(target=initial_state, letter=r, action='pop',
                                        stack_guard=(initial_state.state_id, c))
                initial_state.transitions[r].append(trans)

        return Sevpa(initial_state, [initial_state])

    def get_input_alphabet(self):

        int_alphabet, ret_alphabet, call_alphabet = [], [], []
        for state in self.states:
            for transition_list in state.transitions.values():
                for transition in transition_list:
                    if transition.action == 'pop':
                        if transition.letter not in ret_alphabet:
                            ret_alphabet.append(transition.letter)
                        if transition.stack_guard[1] not in call_alphabet:
                            call_alphabet.append(transition.stack_guard[1])
                    else:
                        if transition.letter not in int_alphabet:
                            int_alphabet.append(transition.letter)

        return SevpaAlphabet(int_alphabet, call_alphabet, ret_alphabet)

    def get_error_state(self):
        """
        A state is an error state iff:
            - if all transitions self loop to itself
            - if the pop transitions from the corresponding stack symbol lead to the same state
            - for example:
                - all q2 transitions lead to q2
                - the pop transitions from the initial state which pop the q2+call-symbol from the stack lead to q2 as well

            - Not an error state if it is the initial state or an accepting state
        """

        for state in self.states:

            is_error_state = True
            if state.is_accepting or state == self.initial_state:
                continue

            state_target = None
            # check internal and return transitions
            ret_int_al = []
            ret_int_al.extend(self.input_alphabet.internal_alphabet)
            ret_int_al.extend(self.input_alphabet.return_alphabet)
            for letter in ret_int_al:
                for transition in state.transitions[letter]:
                    if state_target is None:
                        state_target = transition.target_state
                    else:
                        if state_target != transition.target_state:
                            is_error_state = False
                            break
                if not is_error_state:
                    break

            # check return transitions from the initial state
            if is_error_state:
                for return_letter in self.input_alphabet.return_alphabet:
                    for transition in self.initial_state.transitions[return_letter]:
                        if transition.stack_guard[0] == state_target.state_id:
                            if transition.target_state != state_target:
                                is_error_state = False
                                break
                    if not is_error_state:
                        break
            else:
                continue

            if is_error_state:
                return state

        return None

    def delete_state(self, state_to_remove):

        if state_to_remove is not None:
            self.states.remove(state_to_remove)
        else:
            return

        for state in self.states:
            ret_int_al = []
            ret_int_al.extend(self.input_alphabet.internal_alphabet)
            ret_int_al.extend(self.input_alphabet.return_alphabet)
            for letter in ret_int_al:
                cleaned_transitions = []
                for transition in state.transitions[letter]:
                    if transition.stack_guard is not None:
                        if transition.stack_guard[0] == state_to_remove.state_id:
                            continue
                    if transition.target_state.state_id == state_to_remove.state_id:
                        continue

                    cleaned_transitions.append(transition)
                del state.transitions[letter]
                state.transitions[letter] = cleaned_transitions

    def get_allowed_call_transitions(self):
        """
        Returns a dict of states that are allowed to push a call letters on the stack.

        For all states that are connected via internal transitions from the initial state on, the state_id and
        call_letter of the stack_guard from every return transition is used.

        States are not allowed to push something somthing on the stack if there is no possibility to pop the
        stack guard, where their state_id is used, from the stack, which would lead into a dead-end otherwise.

        Returns:
        - dict: A dictionary where keys are the call_letters and values are sets of the allowed states.
        """

        # get all states that are connected via internal transitions by using BFS
        connected_states = set()
        queue = deque([self.initial_state])
        while queue:
            current_state = queue.popleft()
            connected_states.add(current_state)

            for internal_letter in self.input_alphabet.internal_alphabet:
                for internal_trans in current_state.transitions[internal_letter]:
                    target_state = internal_trans.target_state
                    if target_state not in connected_states:
                        queue.append(target_state)

        allowed_call_transitions = defaultdict(set)
        for state in connected_states:
            for return_letter in self.input_alphabet.return_alphabet:
                for trans in state.transitions[return_letter]:
                    allowed_call_transitions[trans.stack_guard[1]].add(trans.stack_guard[0])

        return allowed_call_transitions

    def get_accepting_words_bfs(self, min_word_length: int = 0, num_words: int = 1) -> list:
        """
        Generate a list of random words that are accepted by the automaton using the breadth-first search approach.

        Args:
        - min_word_length (int): Minimum length of the generated words.
        - amount_words (int): Number of words to generate.

        Returns:
        - set: A set of randomly generated words that are accepted by the automaton.
        """
        allowed_call_trans = self.get_allowed_call_transitions()
        self.reset_to_initial()
        queue = deque()
        shuffled_alphabet = self.input_alphabet.get_merged_alphabet()
        random.shuffle(shuffled_alphabet)
        for letter in shuffled_alphabet:
            queue.append([letter])

        found_words = set()
        while queue:
            word = queue.popleft()
            self.reset_to_initial()
            self.execute_sequence(self.initial_state, word)
            # skipping words that lead into the error state will also shorten growth of the queue
            if self.error_state_reached:
                continue
            if self.current_state.is_accepting and self.stack[-1] == self.empty and len(word) >= min_word_length:
                found_words.add(tuple(word))
            if len(found_words) >= num_words:
                found_words = list(found_words)
                found_words.sort(key=len)
                return found_words
            shuffled_alphabet = self.input_alphabet.get_merged_alphabet()
            for letter in shuffled_alphabet:
                if letter in allowed_call_trans:
                    # skip words where it's not possible to pop the stack_guard
                    if self.current_state.state_id not in allowed_call_trans[letter]:
                        continue
                new_word = word + [letter]
                queue.append(new_word)

    def get_random_accepting_word(self, return_letter_prob: float = 0.5, min_len: int = 2) -> list:
        """
        Generate a random word that is accepted by the automaton.

        Only internal letters and return letters will be chosen. If a return letter is randomly chosen a random
        stack guard will be selected. Then the stack needed stack configuration will be searched by using BFS

        Args:
        - return_letter_prob (float): Probability for selecting a letter from the return alphabet.
        - min_len (int): Minimum length of the generated word.

        Returns:
        - list: A randomly generated word that gets accepted by the automaton.
        """
        assert return_letter_prob <= 1.0
        word = []

        internal_letter_prob = 0.0
        if len(self.input_alphabet.internal_alphabet) != 0:
            internal_letter_prob = 1.0 - return_letter_prob
        else:
            return_letter_prob = 1.0

        assert (return_letter_prob + internal_letter_prob) == 1.0

        return_letter_boarder = return_letter_prob
        internal_letter_boarder = return_letter_boarder + internal_letter_prob

        allowed_call_trans = self.get_allowed_call_transitions()

        self.reset_to_initial()

        while True:
            letter_type = random.uniform(0.0, 1.0)
            is_return_letter = False
            if letter_type <= return_letter_boarder:
                possible_letters = self.input_alphabet.return_alphabet
                is_return_letter = True
            elif return_letter_boarder < letter_type <= internal_letter_boarder:
                possible_letters = self.input_alphabet.internal_alphabet
            else:
                assert False

            assert len(possible_letters) > 0

            random_trans_letter_index = random.randint(0, len(possible_letters) - 1)
            letter_for_word = possible_letters[random_trans_letter_index]

            # find the sub-word for the needed stack guard beginning from the initial state
            # the new word will be: letter_prefix + word + letter
            if is_return_letter:
                # randomly select one of the return transitions with the respective return symbol
                if len(self.current_state.transitions[letter_for_word]) == 0:
                    continue
                elif len(self.current_state.transitions[letter_for_word]) == 1:
                    random_stack_guard = self.current_state.transitions[letter_for_word][0].stack_guard
                else:
                    random_stack_guard_index = random.randint(0,
                                                              len(self.current_state.transitions[letter_for_word]) - 1)
                    random_stack_guard = self.current_state.transitions[letter_for_word][
                        random_stack_guard_index].stack_guard

                # start from the initial state
                self.reset_to_initial()

                letter_prefix = []
                needed_stack = self.stack.copy()
                needed_stack.append(random_stack_guard)
                queue = deque()
                for letter in self.input_alphabet.get_merged_alphabet():
                    queue.append([letter])

                while queue:
                    letter_prefix = queue.popleft()
                    self.reset_to_initial()
                    self.execute_sequence(self.initial_state, letter_prefix)
                    if self.error_state_reached:
                        continue
                    if self.stack == needed_stack:
                        break

                    for letter in self.input_alphabet.get_merged_alphabet():
                        if letter in allowed_call_trans:
                            # skip words where it's not possible to pop the stack_guard
                            if self.current_state.state_id not in allowed_call_trans[letter]:
                                continue
                        new_word = letter_prefix + [letter]
                        queue.append(new_word)

                for letter in word:
                    self.step(letter)
                self.step(letter_for_word)
                if not self.error_state_reached:
                    word = letter_prefix + word
                    word.append(letter_for_word)
                else:
                    self.execute_sequence(self.initial_state, word)

            else:
                self.step(letter_for_word)
                if not self.error_state_reached:
                    word.append(letter_for_word)
                else:
                    self.execute_sequence(self.initial_state, word)

            if self.current_state.is_accepting and self.stack[-1] == self.empty and len(word) >= min_len \
                    and random.random() < 0.2:
                break

        self.reset_to_initial()
        return word
