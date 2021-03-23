from abc import ABC, abstractmethod


class AutomatonState(ABC):

    def __init__(self, state_id):
        """
        Single state of an automaton. Each state consists of a state id, a dictionary of transitions, where the keys are
        inputs and the values are the corresponding target states, and a prefix that leads to the state from the initial
        state.

        Args:

            state_id(Any): used for graphical representation of the state. A good practice is to keep it unique.

        """
        self.state_id = state_id
        self.transitions = dict()
        self.prefix = None

    def get_diff_state_transitions(self) -> list:
        """
        Returns a list of transitions that lead to new states, not same-state transitions.
        """
        transitions = []
        for trans, state in self.transitions.items():
            if state != self:
                transitions.append(trans)
        return transitions

    def get_same_state_transitions(self) -> list:
        """
        Get all transitions that lead to the same state (self loops).
        """
        dst = self.get_diff_state_transitions()
        all_trans = set(self.transitions.keys())
        return [t for t in all_trans if t not in dst]


class Automaton(ABC):
    """
    Abstract class representing an automaton.
    """

    def __init__(self, initial_state, states: list):
        """
        Args:

            initial_state (AutomatonState): initial state of the automaton
            states (list) : list containing all states of the automaton

        """
        self.initial_state = initial_state
        self.states = states
        self.characterization_set: list
        self.current_state = initial_state

    def reset_to_initial(self):
        """
        Resets the current state of the automaton to the initial state
        """
        self.current_state = self.initial_state

    @abstractmethod
    def step(self, letter):
        """
        Performs a single step on the automaton changing its current state.

        Args:

            letter: element of the input alphabet to be executed on the system under learning

        Returns:

            Output produced when executing the input letter from the current state

        """
        pass

    def get_shortest_path(self, origin_state: AutomatonState, target_state: AutomatonState) -> tuple:
        """
        Breath First Search over the automaton

        Args:

            origin_state (AutomatonState): state from which the BFS will start
            target_state (AutomatonState): state that will be reached with the return value

        Returns:

            sequence of inputs that lead from origin_state to target state

        """
        if origin_state not in self.states or target_state not in self.states:
            raise SystemExit("State not in the automaton.")

        explored = []
        queue = [[origin_state]]

        if origin_state == target_state:
            return ()

        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node not in explored:
                neighbours = node.transitions.values()
                for neighbour in neighbours:
                    new_path = list(path)
                    new_path.append(neighbour)
                    queue.append(new_path)
                    # return path if neighbour is goal
                    if neighbour == target_state:
                        acc_seq = new_path[:-1]
                        inputs = []
                        for ind, state in enumerate(acc_seq):
                            inputs.append(next(key for key, value in state.transitions.items()
                                               if value == new_path[ind + 1]))
                        return tuple(inputs)

                # mark node as explored
                explored.append(node)
        return ()

    def is_strongly_connected(self) -> bool:
        """
        Check whether the automaton is strongly connected,
        meaning that every state can be reached from every other state.

        Returns:

            True if strongly connected, False otherwise

        """
        import itertools

        state_comb_list = itertools.combinations(self.states, 2)
        for state_comb in state_comb_list:
            if not self.get_shortest_path(state_comb[0], state_comb[1]):
                return False
        return True

    def is_input_complete(self) -> bool:
        """
        Check whether all states have defined transition for all inputs
        :return: true if automaton is input complete

        Returns:

            True if input complete, False otherwise

        """
        alphabet = set(self.initial_state.transitions.keys())
        for state in self.states:
            if state.transitions.keys() != alphabet:
                return False
        return True

    def get_input_alphabet(self) -> list:
        """
        Returns the input alphabet
        """
        assert self.is_input_complete()
        return list(self.initial_state.transitions.keys())

    def __str__(self):
        """
        :return: A string representation of the automaton
        """
        from aalpy.utils import save_automaton_to_file
        return save_automaton_to_file(self, path='learnedModel', file_type='string')
