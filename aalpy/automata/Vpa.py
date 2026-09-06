# Visibly Pushdown Automaton (VPA) state and automaton implementation.
import random
from collections import defaultdict, deque
from collections.abc import Hashable

from aalpy.automata.Sevpa import Sevpa, SevpaAlphabet
from aalpy.base import Automaton, AutomatonState


class VpaAlphabet:
    """
    The Alphabet of a VPA.

    Attributes:
        internal_alphabet (list[str]): Letters for internal transitions.
        call_alphabet (list[str]): Letters for push transitions.
        return_alphabet (list[str]): Letters for pop transitions.
        exclusive_call_return_pairs (dict[str, str]): A dictionary representing exclusive pairs
            of call and return symbols.
    """

    def __init__(self, internal_alphabet: list[str], call_alphabet: list[str], return_alphabet: list[str],
                 exclusive_call_return_pairs: dict[str, str] | None = None) -> None:
        """
        Creates a VPA alphabet.

        :param list[str] internal_alphabet: Letters for internal transitions.
        :param list[str] call_alphabet: Letters for push transitions.
        :param list[str] return_alphabet: Letters for pop transitions.
        :param dict[str, str] | None exclusive_call_return_pairs: Exclusive pairs of call and return symbols.
        """
        self.internal_alphabet = internal_alphabet
        self.call_alphabet = call_alphabet
        self.return_alphabet = return_alphabet
        self.exclusive_call_return_pairs = exclusive_call_return_pairs

    def get_merged_alphabet(self) -> list[str]:
        """
        Get the merged alphabet, including internal, call, and return symbols.

        :return list[str]: A list of all symbols in the alphabet.
        """
        alphabet = list()
        alphabet.extend(self.internal_alphabet)
        alphabet.extend(self.call_alphabet)
        alphabet.extend(self.return_alphabet)
        return alphabet

    def __str__(self) -> str:
        """
        :return str: A string representation of the alphabet.
        """
        return f'Internal: {self.internal_alphabet} Call: {self.call_alphabet} Return: {self.return_alphabet}'


class VpaState(AutomatonState):
    """
    Single state of a VPA.
    """

    def __init__(self, state_id: Hashable, is_accepting: bool = False) -> None:
        """
        Creates a VPA state.

        :param Hashable state_id: Unique identifier of the state.
        :param bool is_accepting: Whether the state is an accepting state.
        """
        super().__init__(state_id)
        self.transitions: dict[str, list[VpaTransition]] = defaultdict(list)
        self.is_accepting = is_accepting


class VpaTransition:
    """
    Represents a transition in a VPA.

    Attributes:
        start (VpaState): The starting state of the transition.
        target (VpaState): The target state of the transition.
        symbol: The symbol associated with the transition.
        action: The action performed during the transition (push | pop | None).
        stack_guard: The stack symbol to be pushed/popped.
    """

    def __init__(self, start: VpaState, target: VpaState, symbol: str, action: str | None,
                 stack_guard: str | None = None) -> None:
        """
        Creates a VPA transition.

        :param VpaState start: The starting state of the transition.
        :param VpaState target: The target state of the transition.
        :param str symbol: The symbol associated with the transition.
        :param str | None action: The action performed during the transition (push | pop | None).
        :param str | None stack_guard: The stack symbol to be pushed/popped.
        """
        self.start = start
        self.target_state = target
        self.letter = symbol
        self.action = action
        self.stack_guard = stack_guard

    def __str__(self) -> str:
        """
        :return str: A string representation of the transition.
        """
        return f"{self.letter}: {self.start.state_id} --> {self.target_state.state_id} | {self.action}: {self.stack_guard}"


class Vpa(Automaton):
    """
    Visibly Pushdown Automaton.
    """
    error_state = VpaState("ErrorSinkState", False)

    def __init__(self, initial_state: VpaState, states: list[VpaState],
                 input_alphabet: VpaAlphabet | None = None) -> None:
        """
        Creates a VPA.

        :param VpaState initial_state: Initial state of the VPA.
        :param list[VpaState] states: All states of the VPA.
        :param VpaAlphabet | None input_alphabet: Input alphabet of the VPA (Default value = None, meaning that it
            is recovered from the transitions, which loses the symbols that no transition happens to use).
        """
        super().__init__(initial_state, states)
        self.initial_state = initial_state
        self.states = states
        self.input_alphabet = input_alphabet if input_alphabet else self.get_input_alphabet()
        self.current_state = None
        self.stack = []

        # alphabet sets for faster inclusion checks (as in VpaAlphabet we have lists, for reproducibility)
        self.internal_set = set(self.input_alphabet.internal_alphabet)
        self.call_set = set(self.input_alphabet.call_alphabet)
        self.return_set = set(self.input_alphabet.return_alphabet)

    def reset_to_initial(self) -> None:
        """
        Resets the current state and stack of the VPA to the initial configuration.
        """
        self.current_state = self.initial_state
        self.stack = []

    def top(self) -> str | list:
        """
        :return str | list: The top of the stack, or an empty list if the stack is empty.
        """
        return self.stack[-1] if self.stack else []

    def step(self, letter: str | None) -> bool:
        """
        Perform a single step on the VPA by transitioning with the given input letter.

        :param str | None letter: A single input that is looked up in the transition table of the VpaState.
        :return bool: True if the reached state is an accepting state and the stack is empty, False otherwise.
        """
        if self.current_state == Vpa.error_state:
            return False

        if letter is None:
            return self.current_state.is_accepting and self.stack == []

        # read without inserting: transitions is a defaultdict, and probing helpers such as
        # vpa_step_configuration would otherwise permanently add empty entries to the model
        transitions = self.current_state.transitions.get(letter, ())

        taken_transition = None

        for t in transitions:
            if t.action == 'push' or t.action is None:
                taken_transition = t
                break
            else:
                if t.stack_guard == self.top():
                    taken_transition = t
                    break

        if taken_transition is None:
            self.current_state = Vpa.error_state
            return False

        self.current_state = taken_transition.target_state
        if taken_transition.action == 'push':
            self.stack.append(taken_transition.stack_guard)
        elif taken_transition.action == 'pop':
            # empty stack elem should always be there
            if not self.stack:
                self.current_state = Vpa.error_state
                return False
            self.stack.pop()

        return self.current_state.is_accepting and self.stack == []

    def execute_sequence(self, origin_state: VpaState, seq: list[str], stack: list) -> list[bool]:
        """
        Executes an input sequence on the VPA starting from a given state and stack configuration.

        A VPA's actual configuration is the pair (state, stack), not just the state, since the stack is
        what makes call/return symbols visibly balanced. The generic Automaton.execute_sequence only
        resets current_state, so it cannot start from a well-defined configuration on its own; stack is
        therefore a required parameter here rather than defaulted, so callers always state explicitly
        which configuration they mean to start from (pass [] to start fresh from origin_state).

        :param VpaState origin_state: State from which the sequence execution starts.
        :param list[str] seq: Input sequence to execute.
        :param list stack: Stack content to start from.
        :return list[bool]: The output response for the executed sequence.
        """
        self.current_state = origin_state
        self.stack = list(stack)
        return [self.step(s) for s in seq]

    def to_state_setup(self) -> dict:
        """
        Converts the VPA to a state setup dictionary.

        :return dict: Map from state_id to tuple(is_accepting, transitions_dict).
        """
        state_setup_dict = {}

        # ensure prefixes are computed
        # self.compute_prefixes()

        sorted_states = sorted(self.states, key=lambda x: len(x.prefix) if x.prefix is not None else len(self.states))
        for s in sorted_states:
            state_setup_dict[s.state_id] = (
                s.is_accepting,
                {k: [(t.target_state.state_id, t.action, t.stack_guard) for t in v] for k, v in s.transitions.items()})

        return state_setup_dict

    def get_input_alphabet(self) -> VpaAlphabet:
        """
        Computes the input alphabet of the VPA from its transitions.

        :return VpaAlphabet: The input alphabet.
        """
        int_alphabet, ret_alphabet, call_alphabet = [], [], []
        for state in self.states:
            for transition_list in state.transitions.values():
                for transition in transition_list:
                    if transition.action == 'pop':
                        if transition.letter not in ret_alphabet:
                            ret_alphabet.append(transition.letter)
                    elif transition.action == 'push':
                        if transition.letter not in call_alphabet:
                            call_alphabet.append(transition.letter)
                    elif transition.letter not in int_alphabet:
                        int_alphabet.append(transition.letter)

        return VpaAlphabet(int_alphabet, call_alphabet, ret_alphabet)

    def is_input_complete(self) -> bool:
        """
        Check whether all states have defined transition for all inputs.

        :return bool: True if automaton is input complete, False otherwise.
        """
        alphabet = set(self.get_input_alphabet().get_merged_alphabet())
        for state in self.states:
            if set(state.transitions.keys()) != alphabet:
                return False
        return True

    @staticmethod
    def from_state_setup(state_setup: dict, **kwargs) -> 'Vpa':
        """
        Create a VPA from a state setup.

        Example state setup:
            state_setup = {
                "q0": (False, {"(": [("q1", 'push', "(")],
                               "[": [("q1", 'push', "[")],  # exclude empty seq
                               }),
                "q1": (False, {"(": [("q1", 'push', "(")],
                               "[": [("q1", 'push', "[")],
                               ")": [("q2", 'pop', "(")],
                               "]": [("q2", 'pop', "[")]}),
                "q2": (True, {
                    ")": [("q2", 'pop', "(")],
                    "]": [("q2", 'pop', "[")]
                }),

        :param dict state_setup: A dictionary mapping from state IDs to tuples containing
            (is_accepting: bool, transitions_dict: dict), where transitions_dict maps input symbols to
            lists of tuples (target_state_id, action, stack_guard).
        :param init_state_id: The state ID for the initial state of the VPA, passed via kwargs.
        :param input_alphabet: Input alphabet of the VPA, passed via kwargs (Default value = None, meaning that it
            is recovered from the transitions, which loses the symbols that no transition happens to use).
        :return Vpa: The constructed Visibly Pushdown Automaton.
        """
        # state_setup should map from state_id to tuple(is_accepting and transitions_dict)

        init_state_id = kwargs['init_state_id']

        # build states with state_id and output
        states = {key: VpaState(key, val[0]) for key, val in state_setup.items()}
        states[Vpa.error_state.state_id] = Vpa.error_state  # PdaState(Pda.error_state,False)
        # add transitions to states
        for state_id, state in states.items():
            if state_id == Vpa.error_state.state_id:
                continue
            for _input, trans_spec in state_setup[state_id][1].items():
                for (target_state_id, action, stack_guard) in trans_spec:
                    trans = VpaTransition(start=state, target=states[target_state_id], symbol=_input, action=action,
                                          stack_guard=stack_guard)
                    state.transitions[_input].append(trans)

        init_state = states[init_state_id]
        # states to list
        states = [state for state in states.values()]

        # the alphabet is forwarded rather than recovered from the transitions, which would lose the symbols
        # that no transition of the model happens to use
        return Vpa(init_state, states, kwargs.get('input_alphabet'))

    def is_balanced(self, seq: list[str]) -> bool:
        """
        Checks whether an input sequence has balanced call and return symbols with respect to the VPA's alphabet.

        :param list[str] seq: The input sequence to check.
        :return bool: True if the sequence is balanced, False otherwise.
        """
        from aalpy.utils import is_balanced
        return is_balanced(seq, self.input_alphabet)

    def compute_characterizing_set(self, congruence: 'VpaCongruence | None' = None) -> list[tuple]:
        """
        Computes a characterizing set of the VPA, that is, a set of contexts that tells apart all congruence classes
        of the visibly pushdown language it accepts.

        The congruence is the one of Alur, Kumar, Madhusudan and Viswanathan, "Congruences for Visibly Pushdown
        Languages" (ICALP 2005), which characterizes the unique minimal single-entry VPA of a well-matched visibly
        pushdown language. It is defined on *well-matched* words and uses *two-sided* contexts:

            w1 ~ w2  iff  for all u, v: u w1 v is in the language exactly when u w2 v is

        Both halves matter. A one-sided suffix is not enough because only well-matched words are members of the
        language, so a suffix can only be appended to a word whose stack it empties, and words reaching different
        stack heights then have no common suffix at all. Taking the left half u into the context removes that
        problem: u fixes one configuration, and since a well-matched word leaves the stack it is executed on
        untouched, both w1 and w2 are compared from the very same stack, so a separating right half v always exists
        whenever the two classes differ.

        The classes told apart are those of the transition cover, not only those of the state cover, so that the
        characterizing set also witnesses where every transition of the canonical model leads. The contexts are
        found by VpaCongruence, which decides the congruence exactly rather than by searching words up to a bound,
        so the returned set separates *every* pair of classes that any context whatsoever separates.

        :param VpaCongruence | None congruence: Congruence of this VPA (Default value = None, meaning that it is
            computed here). Deciding the congruence dominates the runtime, so an already computed one is worth
            passing.
        :return list[tuple]: The characterizing set, a list of (left, right) context pairs.
        """
        congruence = congruence or VpaCongruence(self)

        # a class is represented by the transduction of its words, and by a shortest word realizing it
        representatives = {}
        for word in vpa_transition_cover(self, congruence):
            representatives.setdefault(vpa_word_transduction(self, word), word)

        characterizing_set = [((), ())]
        transductions = list(representatives)

        for index, transduction_1 in enumerate(transductions):
            for transduction_2 in transductions[index + 1:]:
                word_1, word_2 = representatives[transduction_1], representatives[transduction_2]
                if any(vpa_context_separates_words(self, word_1, word_2, c) for c in characterizing_set):
                    continue
                context = congruence.separating_context(transduction_1, transduction_2)
                if context is not None:
                    characterizing_set.append(context)

        return characterizing_set

    def generate_random_accepting_word(self, min_steps: int = 4, max_steps: int = 20) -> list[str] | None:
        """
        Generate a random valid sequence for a given VPDA.

        :param int min_steps: Minimum number of steps.
        :param int max_steps: Maximum number of steps before the process terminates.
        :return list[str] | None: A list of input symbols (the generated sequence) leading to an accepting state,
            or None if a sequence could not be generated.
        """

        sequence = []
        self.reset_to_initial()

        for step_count in range(max_steps):
            current_state = self.current_state

            # If we have met the min_steps requirement and are in an accepting state with an empty stack, stop
            if step_count >= min_steps and current_state.is_accepting and not self.stack:
                return sequence

            # Get all possible transitions from the current state
            possible_transitions = []
            for letter, transitions in current_state.transitions.items():
                for t in transitions:
                    if t.action == 'pop' and self.stack and t.stack_guard == self.top():
                        possible_transitions.append(t)
                    elif t.action == 'push' or t.action is None:
                        possible_transitions.append(t)

            # If no valid transitions exist, return an incomplete sequence or error
            if not possible_transitions:
                break

            # Randomly choose a valid transition
            chosen_transition = random.choice(possible_transitions)

            # Perform the transition
            self.step(chosen_transition.letter)

            # Add the chosen letter to the sequence
            sequence.append(chosen_transition.letter)

        # None indicates that a sequance was not successfully generated
        return None



def vpa_step_configuration(vpa: Vpa, configuration: tuple, letter: str) -> tuple | None:
    """
    Performs a single step of a VPA on a configuration, that is, a (state, stack) pair.

    A configuration whose state is None is the error configuration reached once an undefined transition is taken. It
    is absorbing, but its stack height keeps being tracked, so that a word that ran into the error configuration can
    still be completed to a well-matched word (which is a perfectly usable negative example).

    :param Vpa vpa: VPA on which the step is performed.
    :param tuple configuration: Configuration (state, stack) from which the step is performed.
    :param str letter: Input letter.
    :return tuple | None: Reached configuration, or None if the letter is a return symbol that would pop from an
        empty stack, meaning that no well-matched word starts with this sequence.
    """
    state, stack = configuration

    if letter in vpa.return_set and not stack:
        return None

    if state is not None:
        vpa.current_state = state
        vpa.stack = list(stack)
        vpa.step(letter)
        if vpa.current_state is not Vpa.error_state:
            return vpa.current_state, tuple(vpa.stack)

    # error configuration: only the stack height still matters, so a placeholder is pushed
    if letter in vpa.call_set:
        return None, stack + (None,)
    if letter in vpa.return_set:
        return None, stack[:-1]
    return None, stack


def vpa_configuration_output(configuration: tuple) -> bool:
    """
    Computes the output of a VPA configuration, which is True only for accepting states with an empty stack.

    :param tuple configuration: Configuration (state, stack).
    :return bool: True if the configuration is accepting, False otherwise.
    """
    state, stack = configuration
    return state is not None and state.is_accepting and not stack


def vpa_configuration_key(configuration: tuple) -> tuple:
    """
    Computes a hashable key identifying a VPA configuration.

    :param tuple configuration: Configuration (state, stack).
    :return tuple: Hashable representation of the configuration.
    """
    state, stack = configuration
    return state.state_id if state is not None else None, stack


def apply_vpa_context(vpa: Vpa, configuration: tuple, context: tuple) -> tuple | None:
    """
    Applies a sequence of inputs to a VPA configuration.

    :param Vpa vpa: VPA on which the context is executed.
    :param tuple configuration: Configuration (state, stack) from which the execution starts.
    :param tuple context: Input sequence to execute.
    :return tuple | None: Reached configuration, or None if the sequence would pop from an empty stack.
    """
    for letter in context:
        configuration = vpa_step_configuration(vpa, configuration, letter)
        if configuration is None:
            return None
    return configuration


def vpa_word_transduction(vpa: Vpa, word: tuple) -> tuple:
    """
    Computes the transduction of a well-matched word, that is, the map sending every state of the VPA to the state
    the word leads to when it is executed from that state.

    A well-matched word never pops below the stack it starts on, so its transduction does not depend on that stack,
    and two well-matched words with the same transduction are interchangeable in every context.

    :param Vpa vpa: VPA on which the word is executed.
    :param tuple word: Well-matched word.
    :return tuple: The state_id reached from every state, None where the word runs into the error configuration.
    """
    reached = [apply_vpa_context(vpa, (state, ()), word) for state in vpa_states(vpa)]
    return tuple(configuration[0].state_id if configuration is not None and configuration[0] is not None else None
                 for configuration in reached)


def vpa_states(vpa: Vpa) -> list[VpaState]:
    """
    Lists the states of a VPA in a fixed order, without the shared error state.

    :param Vpa vpa: VPA whose states are listed.
    :return list[VpaState]: The states of the VPA.
    """
    return [state for state in vpa.states if state is not Vpa.error_state]


class VpaCongruence:
    """
    Decides the congruence characterizing the visibly pushdown language of a VPA, and produces witnesses for it.

    The congruence is the one of Alur, Kumar, Madhusudan and Viswanathan, "Congruences for Visibly Pushdown
    Languages" (ICALP 2005): for well-matched words, w1 ~ w2 iff u w1 v is in the language exactly when u w2 v is,
    for all u and v. Deciding it by enumerating words up to some bound can only ever be a heuristic, since neither
    the nesting depth of u nor the length of v is bounded a priori. This class decides it exactly instead, by three
    fixpoints over finite domains, so a pair of words reported as inseparable really is congruent.

    The three fixpoints are:

    1. The transductions of well-matched words. Well-matched words are generated by the grammar W -> eps | i W |
       c W r W, so their transductions are the closure of the identity and of the internal symbols under
       composition and under wrapping a transduction into a call/return pair. They live in a finite domain, so
       the closure terminates and is exact.
    2. The pairs of states that a right half of a context tells apart, per stack. With an empty stack the right
       half must itself be well-matched, so those pairs are read off the transductions. A right half emptying a
       stack with the symbol g on top splits as a well-matched part, the return popping g, and a right half for
       the remaining stack, which turns the pairs for a stack into the pairs for that stack extended by g.
    3. The left halves. A left half has no unmatched returns, so it is a sequence of well-matched parts separated
       by calls. Tracking, alongside the state it reaches, the separable pairs of the stack it built, gives a
       fixpoint over pairs of a state and a set of pairs of states, which is again finite.

    Attributes:
        transductions (dict): Map from transduction to a witness well-matched word realizing it.
    """

    def __init__(self, vpa: Vpa) -> None:
        """
        Decides the congruence of a VPA by computing the three fixpoints.

        :param Vpa vpa: VPA whose congruence is decided.
        """
        self.vpa = vpa
        self.states = vpa_states(vpa)
        self.state_index = {state.state_id: index for index, state in enumerate(self.states)}
        self.state_by_id = {state.state_id: state for state in self.states}
        self.state_ids = [state.state_id for state in self.states] + [None]

        self.stack_symbols = []
        for state in vpa.states:
            for transitions in state.transitions.values():
                for transition in transitions:
                    if transition.action == 'push' and transition.stack_guard not in self.stack_symbols:
                        self.stack_symbols.append(transition.stack_guard)

        self.transductions = self._compute_transductions()
        self._separable_on_empty_stack = self._compute_separable_on_empty_stack()
        self._pre_cache = {}
        self._left_halves = self._compute_left_halves()

    def apply(self, transduction: tuple, state_id: Hashable) -> Hashable:
        """
        Applies a transduction to a state.

        :param tuple transduction: Transduction to apply.
        :param Hashable state_id: State to apply it to, None for the error state.
        :return Hashable: The reached state_id, None for the error state.
        """
        return transduction[self.state_index[state_id]] if state_id is not None else None

    def separating_context(self, transduction_1: tuple, transduction_2: tuple) -> tuple | None:
        """
        Searches for a context telling two well-matched words apart, given their transductions.

        The search is exact: None means that no context whatsoever tells the two words apart, that is, that they
        belong to the same congruence class.

        :param tuple transduction_1: Transduction of the first well-matched word.
        :param tuple transduction_2: Transduction of the second well-matched word.
        :return tuple | None: A (left, right) context pair separating the two words, or None if they are congruent.
        """
        for state_id, separable, left in self._left_halves.values():
            pair = (self.apply(transduction_1, state_id), self.apply(transduction_2, state_id))
            if pair in separable:
                return left, separable[pair]

        return None

    def _step(self, state_id: Hashable, stack: tuple, letter: str) -> tuple:
        """
        Takes a single step from a state with a given stack, mapping the error state to None.

        :param Hashable state_id: State to step from, None for the error state.
        :param tuple stack: Stack to step with.
        :param str letter: Input letter.
        :return tuple: Pair of the reached state_id (None for the error state) and the reached stack.
        """
        if state_id is None:
            return None, stack
        reached = vpa_step_configuration(self.vpa, (self.state_by_id[state_id], stack), letter)
        if reached is None:
            return None, stack
        return (reached[0].state_id if reached[0] is not None else None), reached[1]

    def _compute_transductions(self) -> dict:
        """
        Computes the transductions of all well-matched words, by closing the identity and the internal symbols
        under composition and under wrapping into a call/return pair.

        :return dict: Map from transduction to a witness well-matched word realizing it.
        """
        identity = tuple(state.state_id for state in self.states)
        transductions = {identity: ()}
        queue = deque([identity])

        for internal in self.vpa.input_alphabet.internal_alphabet:
            transduction = tuple(self._step(state.state_id, (), internal)[0] for state in self.states)
            if transduction not in transductions:
                transductions[transduction] = (internal,)
                queue.append(transduction)

        while queue:
            transduction = queue.popleft()
            new_transductions = [(self._wrap(transduction, call, ret), (call,) + transductions[transduction] + (ret,))
                                 for call in self.vpa.input_alphabet.call_alphabet
                                 for ret in self.vpa.input_alphabet.return_alphabet]
            # the transductions of well-matched words are closed under concatenation of the words
            for other in list(transductions):
                new_transductions.append((self._compose(transduction, other),
                                          transductions[transduction] + transductions[other]))
                new_transductions.append((self._compose(other, transduction),
                                          transductions[other] + transductions[transduction]))

            for new_transduction, witness in new_transductions:
                if new_transduction not in transductions:
                    transductions[new_transduction] = witness
                    queue.append(new_transduction)

        return transductions

    def _compose(self, transduction: tuple, other: tuple) -> tuple:
        """
        Composes two transductions, applying the first one and then the second one.

        :param tuple transduction: Transduction applied first.
        :param tuple other: Transduction applied second.
        :return tuple: The composed transduction.
        """
        return tuple(self.apply(other, state_id) for state_id in transduction)

    def _wrap(self, transduction: tuple, call: str, ret: str) -> tuple:
        """
        Computes the transduction of the word c w r, given the transduction of the well-matched word w.

        :param tuple transduction: Transduction of the wrapped well-matched word.
        :param str call: Call symbol pushing onto the stack.
        :param str ret: Return symbol popping it again.
        :return tuple: The transduction of the wrapped word.
        """
        wrapped = []
        for state in self.states:
            state_id, stack = self._step(state.state_id, (), call)
            state_id, stack = self._step(self.apply(transduction, state_id), stack, ret)
            wrapped.append(state_id)

        return tuple(wrapped)

    def _compute_separable_on_empty_stack(self) -> dict:
        """
        Computes the pairs of states that a right half of a context tells apart when the stack is already empty.

        With an empty stack the right half has to be well-matched itself, so it is one of the words behind the
        transductions and the pairs can be read off them directly.

        :return dict: Map from a pair of state_ids to a witness right half telling the two states apart.
        """
        separable = {}
        for transduction, witness in self.transductions.items():
            for state_id_1 in self.state_ids:
                for state_id_2 in self.state_ids:
                    reached_1, reached_2 = self.apply(transduction, state_id_1), self.apply(transduction, state_id_2)
                    if self._is_accepting(reached_1) != self._is_accepting(reached_2):
                        separable.setdefault((state_id_1, state_id_2), witness)

        return separable

    def _is_accepting(self, state_id: Hashable) -> bool:
        """
        :param Hashable state_id: State, None for the error state.
        :return bool: True if the state is accepting.
        """
        return state_id is not None and self.state_by_id[state_id].is_accepting

    def _pre(self, stack_symbol: Hashable, separable: dict) -> dict:
        """
        Turns the pairs separable with some stack into the pairs separable with that stack extended by one symbol.

        A right half emptying the extended stack splits into a well-matched part executed above the new symbol, the
        return popping it, and a right half emptying the remaining stack.

        :param Hashable stack_symbol: Symbol the stack is extended by.
        :param dict separable: Pairs separable with the remaining stack, mapped to their witnesses.
        :return dict: Pairs separable with the extended stack, mapped to their witnesses.
        """
        # the witnesses are part of the key: two stacks can separate the very same pairs by different right
        # halves, and a right half built for the other stack has the wrong number of returns for this one
        key = (stack_symbol, frozenset(separable.items()))
        if key in self._pre_cache:
            return self._pre_cache[key]

        extended = {}
        for transduction, witness in self.transductions.items():
            for ret in self.vpa.input_alphabet.return_alphabet:
                for state_id_1 in self.state_ids:
                    for state_id_2 in self.state_ids:
                        popped = tuple(self._step(self.apply(transduction, state_id), (stack_symbol,), ret)[0]
                                       for state_id in (state_id_1, state_id_2))
                        if popped in separable:
                            extended.setdefault((state_id_1, state_id_2), witness + (ret,) + separable[popped])

        self._pre_cache[key] = extended
        return extended

    def _compute_left_halves(self) -> dict:
        """
        Computes the left halves of the contexts, as a fixpoint over pairs of a reachable state and the pairs of
        states that are separable with the stack that left half has built.

        A left half cannot contain an unmatched return, as that sends the VPA into the error state, from which no
        context tells anything apart. It is therefore a sequence of well-matched parts separated by calls, which is
        exactly what the two steps of this fixpoint apply.

        :return dict: Map from a key identifying the reached state and the separable pairs to a triple of the
            reached state_id, those separable pairs with their witnesses, and a witness left half.
        """
        def key_of(state_id, separable):
            return state_id, frozenset(separable)

        initial = (self.vpa.initial_state.state_id, self._separable_on_empty_stack, ())
        left_halves = {key_of(initial[0], initial[1]): initial}
        queue = deque([initial])

        while queue:
            state_id, separable, left = queue.popleft()

            successors = [(self.apply(transduction, state_id), separable, left + witness)
                          for transduction, witness in self.transductions.items()]
            for call in self.vpa.input_alphabet.call_alphabet:
                reached, stack = self._step(state_id, (), call)
                if reached is not None:
                    successors.append((reached, self._pre(stack[-1], separable), left + (call,)))

            for successor in successors:
                if successor[0] is None or key_of(successor[0], successor[1]) in left_halves:
                    continue
                left_halves[key_of(successor[0], successor[1])] = successor
                queue.append(successor)

        return left_halves


def vpa_well_matched_cover(vpa: Vpa, congruence: 'VpaCongruence | None' = None) -> dict:
    """
    Enumerates the transductions of the well-matched words of a VPA, with a witness word for each.

    The words are the access sequences of the congruence classes of the accepted language: the states of the minimal
    single-entry VPA are the classes of well-matched words, so covering all transductions covers all of them.

    :param Vpa vpa: VPA whose well-matched words are enumerated.
    :param VpaCongruence | None congruence: Congruence of the VPA (Default value = None, meaning that it is
        computed here). Deciding the congruence dominates the runtime, so an already computed one is worth passing.
    :return dict: Map from transduction to a witness well-matched word realizing it.
    """
    return (congruence or VpaCongruence(vpa)).transductions


def vpa_transition_cover(vpa: Vpa, congruence: 'VpaCongruence | None' = None) -> list[tuple]:
    """
    Computes a transition cover of the minimal single-entry VPA of the language of a VPA, as a list of well-matched
    words.

    The single-entry VPA has an internal transition [w] -i-> [w i] and a return transition taking [w] to
    [w' c w r] when the popped stack symbol was pushed on the call c taken in [w']. Extending the well-matched cover
    by w i and by w' c w r for all covered w, w' therefore exercises every transition of it.

    :param Vpa vpa: VPA whose transition cover is computed.
    :param VpaCongruence | None congruence: Congruence of the VPA (Default value = None, meaning that it is
        computed here). Deciding the congruence dominates the runtime, so an already computed one is worth passing.
    :return list[tuple]: Well-matched words covering all states and all transitions.
    """
    alphabet = vpa.input_alphabet
    cover_words = list(vpa_well_matched_cover(vpa, congruence).values())

    transition_cover, seen = list(cover_words), set(cover_words)
    for word in cover_words:
        extensions = [word + (internal,) for internal in alphabet.internal_alphabet]
        extensions += [other + (call,) + word + (ret,) for other in cover_words
                       for call in alphabet.call_alphabet for ret in alphabet.return_alphabet]
        for extension in extensions:
            if extension not in seen:
                seen.add(extension)
                transition_cover.append(extension)

    return transition_cover


def vpa_context_separates_words(vpa: Vpa, word_1: tuple, word_2: tuple, context: tuple) -> bool:
    """
    Checks whether a context tells two well-matched words apart, that is, whether exactly one of the two words
    plugged into the context is accepted.

    :param Vpa vpa: VPA on which the words are executed.
    :param tuple word_1: First well-matched word.
    :param tuple word_2: Second well-matched word.
    :param tuple context: Context, a (left, right) pair of sequences.
    :return bool: True if exactly one of the two resulting words is accepted.
    """
    left, right = context
    outputs = []
    for word in (word_1, word_2):
        reached = apply_vpa_context(vpa, (vpa.initial_state, ()), left + tuple(word) + right)
        outputs.append(reached is not None and vpa_configuration_output(reached))

    return outputs[0] != outputs[1]


def vpa_call_symbol_conflicts(vpa: Vpa) -> dict:
    """
    Looks up the call symbols that push more than one stack symbol.

    A VPA that pushes only one stack symbol per call symbol carries no information about the state a call was read
    in, so it needs no stack alphabet beyond its call alphabet. A learner that identifies the two cannot return a
    model equivalent to a VPA with such a conflict, no matter which data it is given. PAPNI is not such a learner,
    as the stack symbols of the model it learns are (state at the call, call symbol) pairs.

    :param Vpa vpa: VPA to inspect.
    :return dict: Map from call symbol to the set of stack symbols it pushes, for call symbols pushing more than one.
    """
    pushed_symbols = defaultdict(set)
    for state in vpa.states:
        for transitions in state.transitions.values():
            for transition in transitions:
                if transition.action == 'push':
                    pushed_symbols[transition.letter].add(transition.stack_guard)

    return {letter: symbols for letter, symbols in pushed_symbols.items() if len(symbols) > 1}


def find_vpa_counterexample(vpa: Vpa, other: Vpa, max_stack_height: int = 5,
                            max_search_nodes: int = 200000) -> tuple | None:
    """
    Searches for a shortest well-matched word on which two VPAs disagree, via a breadth-first search over the product
    of their configurations. Both stacks are bounded, so a negative result only means that the two VPAs agree on all
    well-matched words within the searched bounds.

    :param Vpa vpa: First VPA, whose alphabet and stack bound drive the search.
    :param Vpa other: Second VPA.
    :param int max_stack_height: Maximal stack height explored by the search.
    :param int max_search_nodes: Maximal number of configuration pairs explored before the search gives up.
    :return tuple | None: A well-matched word on which the two VPAs disagree, or None if none was found.
    """
    alphabet = vpa.input_alphabet.get_merged_alphabet()
    configuration, other_configuration = (vpa.initial_state, ()), (other.initial_state, ())

    visited = {(vpa_configuration_key(configuration), vpa_configuration_key(other_configuration))}
    queue = deque([((), configuration, other_configuration)])

    while queue and len(visited) < max_search_nodes:
        word, configuration, other_configuration = queue.popleft()
        # only well-matched words are members/non-members of the accepted language. other_configuration is None
        # once the word popped from an empty stack in the other VPA, which the two need not agree on, since the
        # call/return alphabets of the two VPAs need not be the same. No continuation is then accepted by it.
        other_output = other_configuration is not None and vpa_configuration_output(other_configuration)
        if not configuration[1] and vpa_configuration_output(configuration) != other_output:
            return word

        for letter in alphabet:
            reached = vpa_step_configuration(vpa, configuration, letter)
            if reached is None or len(reached[1]) > max_stack_height:
                continue
            other_reached = vpa_step_configuration(other, other_configuration, letter) \
                if other_configuration is not None else None
            key = (vpa_configuration_key(reached),
                   vpa_configuration_key(other_reached) if other_reached is not None else None)
            if key in visited:
                continue
            visited.add(key)
            queue.append((word + (letter,), reached, other_reached))

    return None


def vpa_from_sevpa(sevpa: Sevpa, vpa_alphabet: SevpaAlphabet | VpaAlphabet | None = None) -> Vpa:
    """
    Converts a 1-SEVPA into an equivalent Vpa.

    A 1-SEVPA is a VPA in which the call transitions are implicit: every call symbol pushes the pair (state the
    call is read in, call symbol) and continues from the initial state. Those transitions are made explicit here,
    while the internal and pop transitions are carried over unchanged.

    :param Sevpa sevpa: The 1-SEVPA to convert.
    :param SevpaAlphabet | VpaAlphabet | None vpa_alphabet: Alphabet of the resulting VPA (Default value = None,
        meaning that the alphabet of the 1-SEVPA is used). Passing it keeps the symbols that no transition of the
        1-SEVPA happens to use.
    :return Vpa: The constructed VPA.
    """
    if vpa_alphabet is None:
        vpa_alphabet = sevpa.get_input_alphabet()

    vpa_states = {state.state_id: VpaState(state_id=state.state_id, is_accepting=state.is_accepting)
                  for state in sevpa.states}
    initial_state = vpa_states[sevpa.initial_state.state_id]

    for state in sevpa.states:
        origin_state = vpa_states[state.state_id]

        # the call transitions of a 1-SEVPA are not stored, as they are the same for every state
        for call_symbol in vpa_alphabet.call_alphabet:
            origin_state.transitions[call_symbol].append(
                VpaTransition(origin_state, initial_state, call_symbol, 'push', (state.state_id, call_symbol)))

        for transitions in state.transitions.values():
            for transition in transitions:
                reached_state = vpa_states[transition.target_state.state_id]
                origin_state.transitions[transition.letter].append(
                    VpaTransition(origin_state, reached_state, transition.letter,
                                  transition.action, transition.stack_guard))

    return Vpa(initial_state, list(vpa_states.values()),
               VpaAlphabet(list(vpa_alphabet.internal_alphabet), list(vpa_alphabet.call_alphabet),
                           list(vpa_alphabet.return_alphabet)))
