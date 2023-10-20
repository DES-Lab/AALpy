import random
from collections import defaultdict
from aalpy.base import Automaton, AutomatonState


class SevpaAlphabet:
    def __init__(self, internal_alphabet: list, call_alphabet: list, return_alphabet: list):
        self.internal_alphabet = internal_alphabet
        self.call_alphabet = call_alphabet
        self.return_alphabet = return_alphabet

    def __str__(self):
        return f'Internal: {self.internal_alphabet} Call: {self.call_alphabet} Return: {self.return_alphabet}'

    def get_merged_alphabet(self) -> list:
        alphabet = list()
        alphabet.extend(self.internal_alphabet)
        alphabet.extend(self.call_alphabet)
        alphabet.extend(self.return_alphabet)
        return alphabet


class SevpaState(AutomatonState):
    """
    Single state of a deterministic finite automaton.
    """

    def __init__(self, state_id, is_accepting=False):
        super().__init__(state_id)
        self.transitions = defaultdict(list[SevpaTransition])
        self.is_accepting = is_accepting


class SevpaTransition:
    def __init__(self, start: SevpaState, target: SevpaState, symbol, action, stack_guard=None):
        self.start = start
        self.target = target
        self.symbol = symbol
        self.action = action
        self.stack_guard = stack_guard

    def __str__(self):
        return f"{self.symbol}: {self.start.state_id} --> {self.target.state_id} | {self.action}: {self.stack_guard}"


class Sevpa(Automaton):
    empty = "_"
    error_state = SevpaState("ErrorSinkState", False)

    def __init__(self, initial_state: SevpaState, states: list[SevpaState], input_alphabet: SevpaAlphabet):
        super().__init__(initial_state, states)
        self.initial_state = initial_state
        self.states = states
        self.input_alphabet = input_alphabet
        self.current_state = None
        self.call_balance = 0
        self.stack = []

    def reset_to_initial(self):
        super().reset_to_initial()
        self.reset()

    def reset(self):
        self.current_state = self.initial_state
        self.stack = [self.empty]
        self.call_balance = 0
        return self.current_state.is_accepting and self.top() == self.empty

    def top(self):
        return self.stack[-1]

    def pop(self):
        return self.stack.pop()

    def possible(self, letter):
        """
        Checks if a certain step on the automaton is possible

        TODO: Adaptation for Stack content ?
        """
        if self.current_state == Sevpa.error_state:
            return True
        # push is always possible
        if letter in self.input_alphabet.call_alphabet:
            return True
        if letter is not None:
            transitions = self.current_state.transitions[letter]
            possible_trans = []
            for t in transitions:
                if t.symbol in self.input_alphabet.call_alphabet:
                    possible_trans.append(t)
                elif t.symbol in self.input_alphabet.return_alphabet:
                    if t.stack_guard == self.top():
                        possible_trans.append(t)
                elif t.symbol in self.input_alphabet.internal_alphabet:
                    possible_trans.append(t)
                else:
                    assert False
            # trans = [t for t in transitions if t.stack_guard is None or self.top() == t.stack_guard]
            assert len(possible_trans) < 2
            if len(possible_trans) == 0:
                return False
            else:
                return True
        return False

    def step(self, letter):
        if self.current_state == Sevpa.error_state:
            return False
        if not self.possible(letter):
            self.current_state = Sevpa.error_state
            return False
        if letter is not None:
            transitions = self.current_state.transitions[letter]
            possible_trans = []
            for t in transitions:
                if t.symbol in self.input_alphabet.call_alphabet:
                    possible_trans.append(t)
                elif t.symbol in self.input_alphabet.return_alphabet:
                    if t.stack_guard == self.top():
                        possible_trans.append(t)
                elif t.symbol in self.input_alphabet.internal_alphabet:
                    possible_trans.append(t)
                else:
                    assert False

            if letter in self.input_alphabet.call_alphabet:
                assert(letter in self.input_alphabet.call_alphabet)     # push letters must be in call set
                self.stack.append((self.current_state.state_id, letter))
                self.current_state = self.initial_state
                return self.current_state.is_accepting and self.top() == self.empty

            assert len(possible_trans) < 2
            trans = possible_trans[0]
            self.current_state = trans.target

            if trans.action == 'pop':
                assert(letter in self.input_alphabet.return_alphabet)     # pop letters must be in return set
                if len(self.stack) <= 1:  # empty stack elem should always be there
                    self.current_state = Sevpa.error_state
                    return False
                self.stack.pop()

        return self.current_state.is_accepting and self.top() == self.empty

    def get_state_by_id(self, state_id) -> SevpaState:
        for state in self.states:
            if state.state_id == state_id:
                return state
        return None

    def execute_sequence(self, origin_state, seq):
        if origin_state.prefix != self.initial_state.prefix:
            assert False, 'execute_sequance for Sevpa only is only supported from the initial state.'
        self.reset_to_initial()
        self.current_state = origin_state
        return [self.step(s) for s in seq]

    def to_state_setup(self):
        state_setup_dict = {}

        # ensure prefixes are computed
        # self.compute_prefixes()

        sorted_states = sorted(self.states, key=lambda x: len(x.prefix))
        for s in sorted_states:
            state_setup_dict[s.state_id] = (
                s.is_accepting, {k: (v.target.state_id, v.action) for k, v in s.transitions.items()})

        return state_setup_dict

    @staticmethod
    def from_state_setup(state_setup: dict, init_state_id, input_alphabet: SevpaAlphabet):
        """
            First state in the state setup is the initial state.
            Example state setup:
            state_setup = {
                    "a": (True, {"x": ("b1",PUSH), "y": ("a", NONE)}),
                    "b1": (False, {"x": ("b2", PUSH), "y": "a"}),
                    "b2": (True, {"x": "b3", "y": "a"}),
                    "b3": (False, {"x": "b4", "y": "a"}),
                    "b4": (False, {"x": "c", "y": "a"}),
                    "c": (True, {"x": "a", "y": "a"}),
                }

            Args:

                state_setup: map from state_id to tuple(output and transitions_dict)

            Returns:

                PDA
            """
        # state_setup should map from state_id to tuple(is_accepting and transitions_dict)

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
                    elif action == 'push':  # In SEVPA you can only define return transitions and internal transitions
                        assert False
                    else:
                        trans = SevpaTransition(start=state, target=states[target_state_id], symbol=_input,
                                                action=None, stack_guard=None)

                    state.transitions[_input].append(trans)

            # add call transitions
            for call_letter in input_alphabet.call_alphabet:
                trans = SevpaTransition(start=state, target=states[init_state_id], symbol=call_letter, action='push',
                                        stack_guard=f'{state_id}{call_letter}')
                state.transitions[call_letter].append(trans)

        init_state = states[init_state_id]
        # states to list
        states = [state for state in states.values()]

        sevpa = Sevpa(init_state, states, input_alphabet)
        return sevpa

    def transform_access_sequance(self, state=None, stack_content=None) -> list[str]:

        word = []
        calling_state = self.initial_state if not state else state
        stack = self.stack if not stack_content else stack_content

        for index, stack_elem in enumerate(stack):
            # skip the first element because it's the start of the stack '_
            if index == 0:
                continue
            from_state_id = stack_elem[0]  # the corresponding state where the stack element got pushed from
            call_letter = stack_elem[1]  # the call letter that was pushed on the stack
            from_state = self.get_state_by_id(from_state_id)
            if from_state.prefix != ():
                word.extend(from_state.prefix)
            word.append(call_letter)
        word.extend(calling_state.prefix)
        return word

    def is_balanced(self, x):
        counter = 0
        for i in x:
            if i in self.input_alphabet.call_alphabet:
                counter += 1
            if i in self.input_alphabet.return_alphabet:
                counter -= 1
            if counter < 0:
                return False
        return counter == 0

def has_transition(state: SevpaState, transition_letter, stack_guard) -> bool:
    transitions = state.transitions[transition_letter]
    if transitions is not None:
        if stack_guard is None:     # internal transition
            for transition in transitions:
                if transition.symbol == transition_letter:
                    return True
        else:       # return transition
            for transition in transitions:
                if transition.stack_guard == stack_guard and transition.symbol == transition_letter:
                    return True

    return False


def generate_random_sevpa(alphabet: SevpaAlphabet, amount_states, acceptance_prob, return_transition_prob):
    # TODO: for some reason the alphabet attributes get
    #  treated as sets which are don't have accessible elements via index
    internal_alphabet = list(alphabet.internal_alphabet)
    return_alphabet = list(alphabet.return_alphabet)
    call_alphabet = list(alphabet.call_alphabet)

    state_list = [SevpaState('q0', random.uniform(0.0, 1.0) < acceptance_prob)]
    for i in range(1, amount_states):  # add a return transition
        if internal_alphabet == 0 or random.uniform(0.0, 1.0) < return_transition_prob:
            while True:
                from_state = state_list[random.randint(0, len(state_list)-1)]
                return_letter = return_alphabet[random.randint(0, len(return_alphabet)-1)]
                stack_state = state_list[random.randint(0, len(state_list)-1)]
                call_letter = call_alphabet[random.randint(0, len(call_alphabet)-1)]
                stack_guard = f'{stack_state}{call_letter}'
                if not has_transition(from_state, return_letter, stack_guard):
                    break
            target_state = SevpaState(f'q{i}', random.uniform(0.0, 1.0) < acceptance_prob)
            state_list.append(target_state)
            from_state.transitions[return_letter].append(SevpaTransition(from_state, target_state, return_letter, 'pop', stack_guard))
        else:  # add an internal transition
            while True:
                from_state = state_list[random.randint(0, len(state_list)-1)]
                internal_letter = internal_alphabet[random.randint(0, len(internal_alphabet)-1)]
                if not has_transition(from_state, internal_letter, None):
                    break
            target_state = SevpaState(f'q{i}', random.uniform(0.0, 1.0) < acceptance_prob)
            state_list.append(target_state)
            from_state.transitions[internal_letter].append(SevpaTransition(from_state, target_state, internal_letter, None, None))

    assert len(state_list) == amount_states
    initial_state_id = random.randint(0, amount_states)
    initial_state = state_list[initial_state_id]

    for state in state_list:
        for internal_letter in internal_alphabet:
            if state.transitions[internal_letter] is None:
                target_state = state_list[random.randint(0, len(state_list)-1)]
                state.transitions[internal_letter].append(SevpaTransition(state, target_state, internal_letter, None, None))

        for call_letter in call_alphabet:
            for stack_state in state_list:
                stack_guard = f'{stack_state.state_id}{call_letter}'
                for return_letter in return_alphabet:
                    if not has_transition(state, return_letter, stack_guard):
                        target_state = state_list[random.randint(0, len(state_list)-1)]
                        state.transitions[return_letter].append(SevpaTransition(state, target_state, return_letter, 'pop', stack_guard))

        # add call transitions
        for call_letter in call_alphabet:
            trans = SevpaTransition(start=state, target=initial_state, symbol=call_letter, action='push',
                                    stack_guard=f'{state.state_id}{call_letter}')
            state.transitions[call_letter].append(trans)

    return Sevpa(initial_state, state_list, alphabet)
