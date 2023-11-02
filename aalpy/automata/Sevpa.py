import random
from collections import defaultdict
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
        start (SevpaState): The starting state of the transition.
        target (SevpaState): The target state of the transition.
        symbol: The symbol associated with the transition.
        action: The action performed during the transition (push | pop | None).
        stack_guard: The stack symbol to be pushed/popped.
    """
    def __init__(self, start: SevpaState, target: SevpaState, symbol, action, stack_guard=None):
        self.start = start
        self.target = target
        self.symbol = symbol
        self.action = action
        self.stack_guard = stack_guard

    def __str__(self):
        """
        Returns:
            str: A string representation of the transition.
        """
        return f"{self.symbol}: {self.start.state_id} --> {self.target.state_id} | {self.action}: {self.stack_guard}"


class Sevpa(Automaton):
    """
    1-Module Single Entry Visibly Pushdown Automaton.
    """
    empty = "_"

    def __init__(self, initial_state: SevpaState, states: list[SevpaState], input_alphabet: SevpaAlphabet):
        super().__init__(initial_state, states)
        self.initial_state = initial_state
        self.states = states
        self.input_alphabet = input_alphabet
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
            if t.symbol in self.return_set:
                if t.stack_guard == self.stack[-1]:
                    taken_transition = t
                    break
            elif t.symbol in self.internal_set:
                taken_transition = t
                break
            else:
                assert False

        # No transition is possible
        if not taken_transition:
            self.error_state_reached = True
            return False

        self.current_state = taken_transition.target

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

    def execute_sequence(self, origin_state, seq):
        if origin_state.prefix != self.initial_state.prefix:
            assert False, 'execute_sequence for Sevpa only is only supported from the initial state.'
        self.reset_to_initial()
        self.current_state = origin_state
        return [self.step(s) for s in seq]

    def to_state_setup(self):
        state_setup_dict = {}

        # ensure prefixes are computed
        # self.compute_prefixes()
        # TODO
        sorted_states = sorted(self.states, key=lambda x: len(x.state_id))
        for s in sorted_states:
            state_setup_dict[s.state_id] = (
                s.is_accepting, {k: (v.target.state_id, v.action) for k, v in s.transitions.items()})

        return state_setup_dict

    @staticmethod
    def from_state_setup(state_setup: dict, init_state_id, input_alphabet: SevpaAlphabet):

        # build states with state_id and output
        states = {key: SevpaState(key, val[0]) for key, val in state_setup.items()}
        states[Sevpa.error_state.state_id] = Sevpa.error_state  # PdaState(Pda.error_state,False)

        # add transitions to states
        for state_id, state in states.items():
            if state_id == Sevpa.error_state.state_id:
                continue
            for _input, trans_spec in state_setup[state_id][1].items():
                for (target_state_id, action, stack_guard) in trans_spec:
                    if action == 'pop':
                        assert stack_guard[0] in states
                        assert stack_guard[1] in input_alphabet.call_alphabet
                        stack_guard = (stack_guard[0], stack_guard[1])
                        trans = SevpaTransition(start=state, target=states[target_state_id], symbol=_input,
                                                action=action, stack_guard=stack_guard)
                    elif action is None:
                        trans = SevpaTransition(start=state, target=states[target_state_id], symbol=_input,
                                                action=None, stack_guard=None)
                    else:
                        assert False, 'Action must either be "pop" or None, note that there are no push actions ' \
                                      'definitions in SEVPA'

                    state.transitions[_input].append(trans)

        init_state = states[init_state_id]
        return Sevpa(init_state, [state for state in states.values()], input_alphabet)

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
            trans = SevpaTransition(start=initial_state, target=initial_state, symbol=i, action=None)
            initial_state.transitions[i].append(trans)

        for c in alphabet.call_alphabet:
            for r in alphabet.return_alphabet:
                trans = SevpaTransition(start=initial_state, target=initial_state, symbol=r, action='pop',
                                        stack_guard=(initial_state.state_id, c))
                initial_state.transitions[r].append(trans)

        return Sevpa(initial_state, [initial_state], alphabet)

    def gen_random_accepting_word(self, return_letter_prob: float = 0.0, call_letter_prob: float = 0.0,
                                  early_finish: bool = True):
        """
        Create a random word that gets accepted by the automaton.

        Args:

        Returns:
        """
        assert return_letter_prob + call_letter_prob <= 1.0
        word = []
        if return_letter_prob == 0.0 and call_letter_prob == 0.0:
            return_letter_prob = 0.34
            call_letter_prob = 0.33
        elif return_letter_prob == 0.0 and call_letter_prob != 0.0:
            return_letter_prob = (1.0 - call_letter_prob) / 2
        elif return_letter_prob != 0.0 and call_letter_prob == 0.0:
            call_letter_prob = (1.0 - return_letter_prob) / 2

        if len(self.input_alphabet.internal_alphabet) != 0:
            internal_letter_prob = 1.0 - return_letter_prob - call_letter_prob
        else:
            internal_letter_prob = 0.0
            if return_letter_prob == 0.0 and call_letter_prob == 0.0:
                return_letter_prob = 0.5
                call_letter_prob = 0.5
            elif return_letter_prob == 0.0 and call_letter_prob != 0.0:
                return_letter_prob = (1.0 - call_letter_prob)
            elif return_letter_prob != 0.0 and call_letter_prob == 0.0:
                call_letter_prob = (1.0 - return_letter_prob)

        assert (call_letter_prob + return_letter_prob + internal_letter_prob) == 1.0

        call_letter_boarder = call_letter_prob
        return_letter_boarder = call_letter_boarder + return_letter_prob
        internal_letter_boarder = return_letter_boarder + internal_letter_prob

        self.reset_to_initial()
        while True:
            letter_type = random.uniform(0.0, 1.0)
            if 0.0 <= letter_type <= call_letter_boarder:
                possible_letters = self.input_alphabet.call_alphabet
            elif call_letter_boarder < letter_type <= return_letter_boarder:
                # skip return letters if stack is empty or if the word is empty
                if self.stack[-1] == self.empty or word == []:
                    continue
                possible_letters = self.input_alphabet.return_alphabet
            elif return_letter_boarder < letter_type <= internal_letter_boarder:
                possible_letters = self.input_alphabet.internal_alphabet
            else:
                assert False

            assert len(possible_letters) > 0

            letter = ''
            if early_finish:
                for l in possible_letters:
                    for transition in self.current_state.transitions[l]:
                        if transition.target.is_accepting:
                            letter = l
                            break
                    break
            if letter == '':
                random_trans_letter_index = random.randint(0, len(possible_letters) - 1)
                letter = possible_letters[random_trans_letter_index]
            self.step(letter)
            if not self.error_state_reached:
                word.append(letter)
            else:
                self.execute_sequence(self.initial_state, word)

            if self.current_state.is_accepting and self.stack[-1] == self.empty:
                break

        return word
