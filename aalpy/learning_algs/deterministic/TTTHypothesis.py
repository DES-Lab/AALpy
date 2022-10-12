from aalpy.automata import Dfa, DfaState


class TTTHypothesisState(DfaState):
    def __init__(self, state_id):
        super().__init__(state_id)
        self.dtree_node = None

    def __repr__(self):
        return f'{self.__class__.__name__} \'{self.state_id}\''

    @property
    def prefix(self):
        if self.dtree_node:
            return self.dtree_node.prefix
        else:
            return (None,)

    @prefix.setter
    def prefix(self, value):
        pass


class TTTHypothesis:
    def __init__(self, alphabet):
        self.states = dict()
        self.initial_state = self.new_state()
        self.alphabet = alphabet

    def new_state(self):
        state_id = f's{len(self.states)}'
        new_state = TTTHypothesisState(state_id)
        self.states[state_id] = new_state
        return new_state

    @property
    def dfa(self):
        return Dfa(self.initial_state, list(self.states.values()))

    def get_open_transitions(self):
        '''
        Get all open transitions in the hypothesis.

        Returns:

            list of dict(state: TTTHypothesisState, letter),
            where state is the starting state and letter is the missing
            transition

        '''
        open_transitions = []
        for state in self.states.values():
            for letter in self.alphabet:
                if letter not in state.transitions:
                    open_transitions.append((state, letter))

        return open_transitions

    def __str__(self):
        return str(self.dfa)