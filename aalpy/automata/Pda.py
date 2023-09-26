from collections import defaultdict

from aalpy.base import Automaton, AutomatonState


class PdaState(AutomatonState):
    """
    Single state of a deterministic finite automaton.
    """

    def __init__(self, state_id, is_accepting=False):
        super().__init__(state_id)
        self.transitions = defaultdict(list)
        self.is_accepting = is_accepting


class PdaTransition:
    def __init__(self, start: PdaState, target: PdaState, symbol, action, stack_guard=None):
        self.start = start
        self.target = target
        self.symbol = symbol
        self.action = action
        self.stack_guard = stack_guard


class Pda(Automaton):
    empty = "$"
    error_state = PdaState("ErrorSinkState", False)

    def __init__(self, initial_state: PdaState, states):
        super().__init__(initial_state, states)
        self.initial_state = initial_state
        self.states = states
        self.current_state = None
        self.stack = []

    def reset_to_initial(self):
        super().reset_to_initial()
        self.reset()

    def reset(self):
        self.current_state = self.initial_state
        self.stack = [self.empty]
        return self.current_state.is_accepting and self.top() == self.empty

    def top(self):
        return self.stack[-1]

    def pop(self):
        return self.stack.pop()

    def possible(self, letter):
        if self.current_state == Pda.error_state:
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
        if self.current_state == Pda.error_state or not self.possible(letter):
            return False
        if letter is not None:
            transitions = self.current_state.transitions[letter]
            trans = [t for t in transitions if t.stack_guard is None or self.top() == t.stack_guard][0]
            self.current_state = trans.target
            if trans.action == 'push':
                self.stack.append(letter)
            elif trans.action == 'pop':
                if len(self.stack) <= 1:  # empty stack elem should always be there
                    self.current_state = Pda.error_state
                    return False
                self.stack.pop()

        return self.current_state.is_accepting and self.top() == self.empty

    # def compute_output_seq(self, state, sequence):
    #     if not sequence:
    #         return [state.is_accepting]
    #     return super(Dfa, self).compute_output_seq(state, sequence)

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
    def from_state_setup(state_setup: dict, init_state_id):
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
        states = {key: PdaState(key, val[0]) for key, val in state_setup.items()}
        states[Pda.error_state.state_id] = Pda.error_state  # PdaState(Pda.error_state,False)
        # add transitions to states
        for state_id, state in states.items():
            if state_id == Pda.error_state.state_id:
                continue
            for _input, trans_spec in state_setup[state_id][1].items():
                for (target_state_id, action, stack_guard) in trans_spec:
                    # action = Action[action_string]
                    trans = PdaTransition(start=state, target=states[target_state_id], symbol=_input, action=action,
                                          stack_guard=stack_guard)
                    state.transitions[_input].append(trans)

        init_state = states[init_state_id]
        # states to list
        states = [state for state in states.values()]

        pda = Pda(init_state, states)
        return pda


# def generate_data_from_pda(automaton, num_examples, lens=None, classify_states=False, stack_limit=None,
#                            break_on_impossible=False, possible_prob=0.75):
#     input_al = automaton.get_input_alphabet()
#     output_al = [False, True]
#     if classify_states:
#         output_al = [s.state_id for s in automaton.states]
#
#     if lens is None:
#         lens = list(range(1, 15))
#
#     sum_lens = sum(lens)
#     # key is length, value is number of examples for said length
#     ex_per_len = dict()
#
#     additional_seq = 0
#     for l in lens:
#         ex_per_len[l] = int(num_examples * (l / sum_lens)) + 1
#         if ex_per_len[l] > pow(len(input_al), l):
#             additional_seq += ex_per_len[l] - pow(len(input_al), l)
#             ex_per_len[l] = 'comb'
#
#     additional_seq = additional_seq // len([i for i in ex_per_len.values() if i != 'comb'])
#
#     training_data = []
#     for l in ex_per_len.keys():
#         seqs = []
#         if ex_per_len[l] == 'comb':
#             seqs = list(product(input_al, repeat=l))
#             for seq in seqs:
#
#                 out = automaton.reset()
#                 nr_steps = 0
#                 for inp in seq:
#                     if automaton.possible(inp) or not break_on_impossible:
#                         nr_steps += 1
#                     if stack_limit and len(automaton.stack) > stack_limit:
#                         break
#                     if break_on_impossible and not automaton.possible(inp):
#                         break
#                     out = automaton.step(inp)
#                 seq = seq[:nr_steps]
#                 training_data.append((tuple(seq), out if not classify_states else automaton.current_state.state_id))
#
#         else:
#             for _ in range(ex_per_len[l] + additional_seq):
#                 # seq = [random.choice(input_al) for _ in range(l)]
#                 out = automaton.reset()
#                 nr_steps = 0
#                 seq = []
#                 for i in range(l):
#                     possible_inp = [inp for inp in input_al if automaton.possible(inp)]
#                     if len(possible_inp) == 0:
#                         inp = random.choice(input_al)
#                     else:
#                         if random.random() <= possible_prob:
#                             inp = random.choice(possible_inp)
#                         else:
#                             inp = random.choice(input_al)
#                     seq.append(inp)
#                     if automaton.possible(inp) or not break_on_impossible:
#                         nr_steps += 1
#                     if stack_limit and len(automaton.stack) > stack_limit:
#                         break
#                     if break_on_impossible and not automaton.possible(inp):
#                         break
#                     out = automaton.step(inp)
#                 seq = seq[:nr_steps]
#                 training_data.append((tuple(seq), out if not classify_states else automaton.current_state.state_id))
#
#     return training_data, input_al, output_al
