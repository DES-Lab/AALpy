# Moore machine state and automaton implementation, where outputs are associated with states.
from collections.abc import Hashable
from typing import Generic

from aalpy.automata.Dfa import Dfa, DfaState
from aalpy.base import AutomatonState, DeterministicAutomaton
from aalpy.base.Automaton import InputType, OutputType


class MooreState(AutomatonState, Generic[InputType, OutputType]):
    """
    Single state of a Moore machine. Each state has an output value.
    """

    def __init__(self, state_id: Hashable, output: OutputType | None = None) -> None:
        """
        Creates a Moore machine state.

        :param Hashable state_id: Unique identifier of the state.
        :param OutputType | None output: Output value associated with the state.
        """
        super().__init__(state_id)
        self.output: OutputType | None = output
        self.transitions: dict[InputType, MooreState[InputType, OutputType]] = dict()


class MooreMachine(DeterministicAutomaton[MooreState[InputType, OutputType]]):
    """
    Deterministic Moore machine, where outputs depend only on the current state.
    """

    def __init__(self, initial_state: AutomatonState, states: list) -> None:
        """
        Creates a Moore machine.

        :param AutomatonState initial_state: Initial state of the Moore machine.
        :param list states: All states of the Moore machine.
        """
        super().__init__(initial_state, states)

    def step(self, letter: InputType | None) -> OutputType:
        """
        Performs a single step on the Moore machine. In Moore machines outputs depend on the current state.

        :param InputType | None letter: Single input that is looked up in the transition function leading to a
            new state.
        :return OutputType: The output of the reached state.
        """
        if letter is not None:
            self.current_state = self.current_state.transitions[letter]
        return self.current_state.output

    def compute_characterization_set(self, char_set_init: list[tuple] | None = None, online_suffix_closure: bool = True,
                                     split_all_blocks: bool = True, return_same_states: bool = False,
                                     raise_warning: bool = True) -> list[tuple] | None:
        """
        Computes a characterization set for the Moore machine. See
        DeterministicAutomaton.compute_characterization_set for details.

        :param list[tuple] | None char_set_init: Sequences to include in the characterization set.
        :param bool online_suffix_closure: If true, ensures suffix closedness at every computation step.
        :param bool split_all_blocks: If true, sequences are used to distinguish all states.
        :param bool return_same_states: If true, a single non-distinguishable pair of states is returned.
        :param bool raise_warning: Whether to print a warning if the characterization set cannot be computed.
        :return list[tuple] | None: The characterization set, or None if it cannot be computed.
        """
        return super(MooreMachine, self).compute_characterization_set(char_set_init if char_set_init else [()],
                                                                      online_suffix_closure, split_all_blocks,
                                                                      return_same_states, raise_warning)

    def compute_output_seq(self, state: MooreState, sequence: list[InputType]) -> list[OutputType]:
        """
        Computes the output response of the Moore machine for a given input sequence from a given state.

        :param MooreState state: State from which the output response shall be computed.
        :param list[InputType] sequence: Input sequence over the alphabet.
        :return list[OutputType]: The output response.
        """
        if not sequence:
            return [state.output]
        return super(MooreMachine, self).compute_output_seq(state, sequence)

    def execute_sequence(self, origin_state: MooreState, seq: list[InputType]) -> list[OutputType] | OutputType:
        """
        Executes an input sequence on the Moore machine starting from a given state.

        :param MooreState origin_state: State from which the sequence execution starts.
        :param list[InputType] seq: Input sequence to execute.
        :return list[OutputType] | OutputType: The output response for the executed sequence.
        """
        if not seq:
            self.current_state = origin_state
            return self.current_state.output
        return super(MooreMachine, self).execute_sequence(origin_state, seq)

    def to_state_setup(self) -> dict:
        """
        Converts the Moore machine to a state setup dictionary.

        :return dict: Map from state_id to tuple(output, transitions_dict).
        """
        state_setup_dict = {}

        # ensure prefixes are computed
        self.compute_prefixes()

        sorted_states = sorted(self.states, key=lambda x: len(x.prefix) if x.prefix is not None else len(self.states))
        for s in sorted_states:
            state_setup_dict[s.state_id] = (s.output, {k: v.state_id for k, v in s.transitions.items()})

        return state_setup_dict

    def find_distinguishing_seq(self, state1: MooreState[InputType, OutputType], state2: MooreState[InputType, OutputType], alphabet: list) -> list | None:
        if state1.output != state2.output:
            return []
        return super().find_distinguishing_seq(state1, state2, alphabet)

    @staticmethod
    def from_state_setup(state_setup: dict, **kwargs) -> 'MooreMachine':
        """
        Creates a Moore machine from a state setup dictionary. The first state in the state setup is the initial
        state.

        Example state setup::

            state_setup = {
                    "a": ("a", {"x": "b1", "y": "a"}),
                    "b1": ("b", {"x": "b2", "y": "a"}),
                    "b2": ("b", {"x": "b3", "y": "a"}),
                    "b3": ("b", {"x": "b4", "y": "a"}),
                    "b4": ("b", {"x": "c", "y": "a"}),
                    "c": ("c", {"x": "a", "y": "a"}),
                }

        :param dict state_setup: Map from state_id to tuple(output, transitions_dict).
        :return MooreMachine: The constructed Moore machine.
        """

        # build states with state_id and output
        states = {key: MooreState(key, val[0]) for key, val in state_setup.items()}

        # add transitions to states
        for state_id, state in states.items():
            for _input, target_state_id in state_setup[state_id][1].items():
                state.transitions[_input] = states[target_state_id]

        # states to list
        states = [state for state in states.values()]

        # build moore machine with first state as starting state
        mm = MooreMachine(states[0], states)

        for state in states:
            state.prefix = mm.get_shortest_path(mm.initial_state, state)

        return mm

    @staticmethod
    def to_dfa(moore_machine: 'MooreMachine') -> Dfa:
        """
        Converts a Moore machine with boolean state outputs to a DFA.

        :param MooreMachine moore_machine: Moore machine to convert. All states must have boolean outputs.
        :return Dfa: The equivalent DFA.
        """
        if not all(isinstance(state.output, bool) for state in moore_machine.states):
            raise ValueError('Only Moore machines with boolean state outputs can be cast to a Dfa.')

        dfa_state_map = {}
        for moore_state in moore_machine.states:
            dfa_state = DfaState(moore_state.state_id, is_accepting=moore_state.output)
            dfa_state.prefix = moore_state.prefix
            dfa_state_map[moore_state] = dfa_state

        for moore_state in moore_machine.states:
            for letter, target in moore_state.transitions.items():
                dfa_state_map[moore_state].transitions[letter] = dfa_state_map[target]

        dfa = Dfa(dfa_state_map[moore_machine.initial_state], list(dfa_state_map.values()))
        dfa.current_state = dfa.initial_state
        return dfa
