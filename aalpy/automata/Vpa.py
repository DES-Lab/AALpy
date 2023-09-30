from collections import defaultdict

from aalpy.base import Automaton, AutomatonState


class VpaState(AutomatonState):
    """
    Single state of a deterministic finite automaton.
    """

    def __init__(self, state_id, is_accepting=False):
        super().__init__(state_id)
        self.transitions = defaultdict(list)
        self.is_accepting = is_accepting


class VpaTransition:
    def __init__(self, start: VpaState, target: VpaState, symbol, action, stack_guard=None):
        self.start = start
        self.target = target
        self.symbol = symbol
        self.action = action
        self.stack_guard = stack_guard

    def __str__(self):
        return f"{self.symbol}: {self.start.state_id} --> {self.target.state_id} | {self.action}: {self.stack_guard}"


class Vpa(Automaton):
    empty = "_"
    error_state = VpaState("ErrorSinkState", False)

    def __init__(self, initial_state: VpaState, states, call_set, return_set, internal_set):
        super().__init__(initial_state, states)
        self.initial_state = initial_state
        self.states = states
        self.call_set = call_set
        self.return_set = return_set
        self.internal_set = internal_set
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
        if self.current_state == Vpa.error_state:
            return True
        if letter is not None:
            transitions = self.current_state.transitions[letter]
            trans = [t for t in transitions if t.stack_guard is None or self.top() == t.stack_guard]
            assert len(trans) < 2
            if len(trans) == 0:
                return False
            else:
                return True
        return False

    def step(self, letter):
        if self.current_state == Vpa.error_state:
            return False
        if not self.possible(letter):
            self.current_state = Vpa.error_state
            return False
        if letter is not None:
            transitions = self.current_state.transitions[letter]
            trans = [t for t in transitions if t.stack_guard is None or self.top() == t.stack_guard][0]
            self.current_state = trans.target
            if trans.action == 'push':
                assert(letter in self.call_set)     # push letters must be in call set
                self.stack.append(letter)
            elif trans.action == 'pop':
                assert(letter in self.return_set)     # pop letters must be in return set
                if len(self.stack) <= 1:  # empty stack elem should always be there
                    self.current_state = Vpa.error_state
                    return False
                self.stack.pop()

        return self.current_state.is_accepting and self.top() == self.empty

    # def compute_output_seq(self, state, sequence):
    #     if not sequence:
    #         return [state.is_accepting]
    #     return super(Dfa, self).compute_output_seq(state, sequence)

    def get_input_alphabet(self) -> list:
        alphabet_list = list()
        alphabet_list.append(self.call_set)
        alphabet_list.append(self.return_set)
        alphabet_list.append(self.internal_set)
        return alphabet_list

    def get_input_alphabet_merged(self) -> list:
        alphabet = list()
        alphabet.extend(self.call_set)
        alphabet.extend(self.return_set)
        alphabet.extend(self.internal_set)
        return alphabet

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
    def from_state_setup(state_setup: dict, init_state_id, call_set, return_set, internal_set):
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
        states = {key: VpaState(key, val[0]) for key, val in state_setup.items()}
        states[Vpa.error_state.state_id] = Vpa.error_state  # PdaState(Pda.error_state,False)
        # add transitions to states
        for state_id, state in states.items():
            if state_id == Vpa.error_state.state_id:
                continue
            for _input, trans_spec in state_setup[state_id][1].items():
                for (target_state_id, action, stack_guard) in trans_spec:
                    # action = Action[action_string]
                    trans = VpaTransition(start=state, target=states[target_state_id], symbol=_input, action=action,
                                          stack_guard=stack_guard)
                    state.transitions[_input].append(trans)

        init_state = states[init_state_id]
        # states to list
        states = [state for state in states.values()]

        pda = Vpa(init_state, states, call_set, return_set, internal_set)
        return pda


def generate_data_from_pda(automaton, num_examples, lens=None, classify_states=False, stack_limit=None,
                           break_on_impossible=False, possible_prob=0.75):
    import random
    from itertools import product

    input_al = automaton.get_input_alphabet()

    if lens is None:
        lens = list(range(1, 15))

    sum_lens = sum(lens)
    # key is length, value is number of examples for said length
    ex_per_len = dict()

    additional_seq = 0
    for l in lens:
        ex_per_len[l] = int(num_examples * (l / sum_lens)) + 1
        if ex_per_len[l] > pow(len(input_al), l):
            additional_seq += ex_per_len[l] - pow(len(input_al), l)
            ex_per_len[l] = 'comb'

    additional_seq = additional_seq // len([i for i in ex_per_len.values() if i != 'comb'])

    training_data = []
    for l in ex_per_len.keys():
        seqs = []
        if ex_per_len[l] == 'comb':

            seqs = list(product(input_al, repeat=l))
            for seq in seqs:

                out = automaton.reset()
                nr_steps = 0
                for inp in seq:
                    if automaton.possible(inp) or not break_on_impossible:
                        nr_steps += 1
                    if stack_limit and len(automaton.stack) > stack_limit:
                        break
                    if break_on_impossible and not automaton.possible(inp):
                        break
                    out = automaton.step(inp)
                seq = seq[:nr_steps]
                training_data.append((tuple(seq), out if not classify_states else automaton.current_state.state_id))

        else:
            for _ in range(ex_per_len[l] + additional_seq):
                # seq = [random.choice(input_al) for _ in range(l)]
                out = automaton.reset()
                nr_steps = 0
                seq = []
                for i in range(l):
                    possible_inp = [inp for inp in input_al if automaton.possible(inp)]
                    if len(possible_inp) == 0:
                        inp = random.choice(input_al)
                    else:
                        if random.random() <= possible_prob:
                            inp = random.choice(possible_inp)
                        else:
                            inp = random.choice(input_al)
                    seq.append(inp)
                    if automaton.possible(inp) or not break_on_impossible:
                        nr_steps += 1
                    if stack_limit and len(automaton.stack) > stack_limit:
                        break
                    if break_on_impossible and not automaton.possible(inp):
                        break
                    out = automaton.step(inp)
                seq = seq[:nr_steps]
                training_data.append((tuple(seq), out))

    return training_data
