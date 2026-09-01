# Deterministic finite automaton (DFA) state and automaton implementation.
from collections.abc import Hashable
from typing import Generic

from aalpy.base import AutomatonState, DeterministicAutomaton
from aalpy.base.Automaton import InputType


class DfaState(AutomatonState, Generic[InputType]):
    """
    Single state of a deterministic finite automaton.
    """

    def __init__(self, state_id: Hashable, is_accepting: bool = False) -> None:
        """
        Creates a DFA state.

        :param Hashable state_id: Unique identifier of the state.
        :param bool is_accepting: Whether the state is an accepting state.
        """
        super().__init__(state_id)
        self.transitions: dict[InputType, DfaState[InputType]] = dict()
        self.is_accepting = is_accepting

    @property
    def output(self) -> bool:
        """
        :return bool: True if the state is accepting, False otherwise.
        """
        return self.is_accepting


class Dfa(DeterministicAutomaton[DfaState[InputType]]):
    """
    Deterministic finite automaton.
    """

    def __init__(self, initial_state: DfaState, states: list[DfaState]) -> None:
        """
        Creates a DFA.

        :param DfaState initial_state: Initial state of the DFA.
        :param list[DfaState] states: All states of the DFA.
        """
        super().__init__(initial_state, states)

    def step(self, letter: InputType | None) -> bool:
        """
        Performs a single step on the DFA.

        :param InputType | None letter: Single input that is looked up in the transition table of the DfaState.
        :return bool: True if the reached state is an accepting state, False otherwise.
        """
        if letter is not None:
            self.current_state = self.current_state.transitions[letter]
        return self.current_state.is_accepting

    def compute_characterization_set(self, char_set_init: list[tuple] | None = None, online_suffix_closure: bool = True,
                                     split_all_blocks: bool = True, return_same_states: bool = False,
                                     raise_warning: bool = True) -> list[tuple] | None:
        """
        Computes a characterization set for the DFA. See DeterministicAutomaton.compute_characterization_set for details.

        :param list[tuple] | None char_set_init: Sequences to include in the characterization set.
        :param bool online_suffix_closure: If true, ensures suffix closedness at every computation step.
        :param bool split_all_blocks: If true, sequences are used to distinguish all states.
        :param bool return_same_states: If true, a single non-distinguishable pair of states is returned.
        :param bool raise_warning: Whether to print a warning if the characterization set cannot be computed.
        :return list[tuple] | None: The characterization set, or None if it cannot be computed.
        """
        return super(Dfa, self).compute_characterization_set(char_set_init if char_set_init else [()],
                                                             online_suffix_closure, split_all_blocks,
                                                             return_same_states, raise_warning)

    def compute_output_seq(self, state: DfaState, sequence: list[InputType]) -> list[bool]:
        """
        Computes the output response of the DFA for a given input sequence from a given state.

        :param DfaState state: State from which the output response shall be computed.
        :param list[InputType] sequence: Input sequence over the alphabet.
        :return list[bool]: The output response.
        """
        if not sequence:
            return [state.is_accepting]
        return super(Dfa, self).compute_output_seq(state, sequence)

    def execute_sequence(self, origin_state: DfaState, seq: list[InputType]) -> list[bool] | bool:
        """
        Executes an input sequence on the DFA starting from a given state.

        :param DfaState origin_state: State from which the sequence execution starts.
        :param list[InputType] seq: Input sequence to execute.
        :return list[bool] | bool: The output response for the executed sequence.
        """
        if not seq:
            self.current_state = origin_state
            return self.current_state.output
        return super(Dfa, self).execute_sequence(origin_state, seq)

    def to_state_setup(self) -> dict:
        """
        Converts the DFA to a state setup dictionary.

        :return dict: Map from state_id to tuple(is_accepting, transitions_dict).
        """
        state_setup_dict = {}

        # ensure prefixes are computed
        self.compute_prefixes()

        sorted_states = sorted(self.states, key=lambda x: len(x.prefix) if x.prefix is not None else len(self.states))
        for s in sorted_states:
            state_setup_dict[s.state_id] = (s.is_accepting, {k: v.state_id for k, v in s.transitions.items()})

        return state_setup_dict

    @staticmethod
    def from_state_setup(state_setup: dict, **kwargs) -> 'Dfa':
        """
        Creates a DFA from a state setup dictionary. The first state in the state setup is the initial state.

        Example state setup::

            state_setup = {
                    "a": (True, {"x": "b1", "y": "a"}),
                    "b1": (False, {"x": "b2", "y": "a"}),
                    "b2": (True, {"x": "b3", "y": "a"}),
                    "b3": (False, {"x": "b4", "y": "a"}),
                    "b4": (False, {"x": "c", "y": "a"}),
                    "c": (True, {"x": "a", "y": "a"}),
                }

        :param dict state_setup: Map from state_id to tuple(is_accepting, transitions_dict).
        :return Dfa: The constructed DFA.
        """
        # state_setup should map from state_id to tuple(is_accepting and transitions_dict)

        # build states with state_id and output
        states = {key: DfaState(key, val[0]) for key, val in state_setup.items()}

        # add transitions to states
        for state_id, state in states.items():
            for _input, target_state_id in state_setup[state_id][1].items():
                state.transitions[_input] = states[target_state_id]

        # states to list
        states = [state for state in states.values()]

        # build moore machine with first state as starting state
        dfa = Dfa(states[0], states)

        for state in states:
            state.prefix = dfa.get_shortest_path(dfa.initial_state, state)

        return dfa
