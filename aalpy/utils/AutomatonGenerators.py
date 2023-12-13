import random
import warnings

from aalpy.automata import Dfa, DfaState, MdpState, Mdp, MealyMachine, MealyState, \
    MooreMachine, MooreState, OnfsmState, Onfsm, MarkovChain, McState, StochasticMealyState, StochasticMealyMachine, \
    Sevpa, SevpaState, SevpaAlphabet, SevpaTransition


def generate_random_deterministic_automata(automaton_type,
                                           num_states,
                                           input_alphabet_size,
                                           output_alphabet_size=None,
                                           ensure_minimality=True,
                                           **kwargs
                                           ):
    """
    Generates a random deterministic automata of 'automaton_type'.

    Args:
        automaton_type: type of automaton, either 'dfa', 'mealy', or 'moore'
        num_states: number of states
        input_alphabet_size: size of input alphabet
        output_alphabet_size: size of output alphabet. (ignored for DFAs)
        ensure_minimality: ensure that the automaton is minimal
        **kwargs:
            : 'num_accepting_states' number of accepting states for DFA generation. If not defined, half of states will
            be accepting

    Returns:

        Random deterministic automaton of user defined type, size. If ensure_minimality is set to False returned
        automaton is not necessarily minimal. If minimality is reacquired and random automaton cannot be produced in
        multiple interactions, non-minimal automaton will be returned and a warning message printed.
    """

    assert automaton_type in {'dfa', 'mealy', 'moore'}
    if output_alphabet_size and output_alphabet_size < 2 or output_alphabet_size is None:
        output_alphabet_size = 2

    state_class_map = {'dfa': DfaState, 'mealy': MealyState, 'moore': MooreState}
    automaton_class_map = {'dfa': Dfa, 'mealy': MealyMachine, 'moore': MooreMachine}

    input_alphabet = [f'i{i + 1}' for i in range(input_alphabet_size)]
    output_alphabet = [f'o{i + 1}' for i in range(output_alphabet_size)] if automaton_type != 'dfa' else [True, False]

    # For backwards comparability or if uses passes custom input output functions
    if 'custom_input_alphabet' in kwargs:
        input_alphabet = kwargs.get('custom_input_alphabet')
        if len(input_alphabet) != input_alphabet_size:
            assert False, 'Lenght of input_alphabet_size and custom input alphabet should be equal.'
    if 'custom_output_alphabet' in kwargs:
        output_alphabet = kwargs.get('custom_output_alphabet')
        if len(output_alphabet) != output_alphabet_size:
            assert False, 'Lenght of output_alphabet_size and custom output alphabet should be equal.'

    num_accepting_states = None
    if 'num_accepting_states' in kwargs:
        num_accepting_states = kwargs.get('num_accepting_states')
    if num_accepting_states is None:
        num_accepting_states = num_states // 2

    num_random_outputs = num_states if automaton_type != 'mealy' else num_states * input_alphabet_size

    output_list = []
    output_al_copy = output_alphabet.copy()

    if automaton_type != 'dfa':
        for _ in range(num_random_outputs):
            if output_al_copy:
                o = random.choice(output_al_copy)
                output_al_copy.remove(o)
            else:
                o = random.choice(output_alphabet)
            output_list.append(o)
    else:
        output_list = [True] * num_accepting_states + [False] * (num_states - num_accepting_states)
        random.shuffle(output_list)

    states = [state_class_map[automaton_type](state_id=f's{i + 1}') for i in range(num_states)]

    # define an output function
    for state_index, state in enumerate(states):
        if automaton_type == 'dfa':
            state.is_accepting = output_list[state_index]
        if automaton_type == 'moore':
            state.output = output_list[state_index]
        if automaton_type == 'mealy':
            for i in input_alphabet:
                state.output_fun[i] = output_list.pop(0)

    state_buffer = []

    state_buffer.extend(states)
    while len(state_buffer) < num_states * input_alphabet_size:
        state_buffer.append(random.choice(states))

    random_automaton = None

    # keep changing transitions until all states are reachable (update in future )
    all_states_reachable = False
    while not all_states_reachable:
        random.shuffle(state_buffer)

        transition_index = 0
        for state in states:
            for i in input_alphabet:
                state.transitions[i] = state_buffer[transition_index]
                transition_index += 1

        random_automaton = automaton_class_map[automaton_type](states[0], states)

        unreachable_state_exits = False
        for state in random_automaton.states:
            state.prefix = random_automaton.get_shortest_path(random_automaton.initial_state, state)
            if state != random_automaton.initial_state and state.prefix is None:
                unreachable_state_exits = True
                break
        all_states_reachable = not unreachable_state_exits

    if ensure_minimality:
        minimality_iterations = 1
        while not random_automaton.is_minimal() or random_automaton.size != num_states:
            # to avoid infinite loops
            if minimality_iterations == 100:
                warnings.warn(f'Non-minimal automaton ({automaton_type}, num_states : {num_states}) returned.')
                break

            custom_args = {}
            if 'custom_input_alphabet' in kwargs:
                custom_args['custom_input_alphabet'] = kwargs.get('custom_input_alphabet')
            if 'custom_output_alphabet' in kwargs:
                custom_args['custom_output_alphabet'] = kwargs.get('custom_output_alphabet')
            if 'num_accepting_states' in kwargs:
                custom_args['num_accepting_states'] = kwargs.get('num_accepting_states')

            random_automaton = generate_random_deterministic_automata(automaton_type,
                                                                      num_states,
                                                                      input_alphabet_size,
                                                                      output_alphabet_size,
                                                                      False,  # ensure minimality
                                                                      **custom_args)

    return random_automaton


def generate_random_mealy_machine(num_states, input_alphabet, output_alphabet,
                                  compute_prefixes=False, ensure_minimality=True) -> MealyMachine:
    """
    Generates a random Mealy machine. Kept for backwards compatibility.

    Args:

        num_states: number of states
        input_alphabet: input alphabet
        output_alphabet: output alphabet
        compute_prefixes: if true, shortest path to reach each state will be computed (Default value = False)
        ensure_minimality: returned automaton will be minimal

    Returns:

        Mealy machine with num_states states
    """

    random_mealy_machine = generate_random_deterministic_automata('mealy', num_states,
                                                                  input_alphabet_size=len(input_alphabet),
                                                                  output_alphabet_size=len(output_alphabet),
                                                                  ensure_minimality=ensure_minimality,
                                                                  compute_prefixes=compute_prefixes,
                                                                  custom_input_alphabet=input_alphabet,
                                                                  custom_output_alphabet=output_alphabet)

    return random_mealy_machine


def generate_random_moore_machine(num_states, input_alphabet, output_alphabet,
                                  compute_prefixes=False, ensure_minimality=True) -> MooreMachine:
    """
    Generates a random Moore machine.

    Args:

        num_states: number of states
        input_alphabet: input alphabet
        output_alphabet: output alphabet
        compute_prefixes: if true, shortest path to reach each state will be computed (Default value = False)
        ensure_minimality: returned automaton will be minimal

    Returns:

        Random Moore machine with num_states states

    """
    random_moore_machine = generate_random_deterministic_automata('moore', num_states,
                                                                  input_alphabet_size=len(input_alphabet),
                                                                  output_alphabet_size=len(output_alphabet),
                                                                  ensure_minimality=ensure_minimality,
                                                                  compute_prefixes=compute_prefixes,
                                                                  custom_input_alphabet=input_alphabet,
                                                                  custom_output_alphabet=output_alphabet)

    return random_moore_machine


def generate_random_dfa(num_states, alphabet, num_accepting_states=1,
                        compute_prefixes=False, ensure_minimality=True) -> Dfa:
    """
    Generates a random DFA.

    Args:

        num_states: number of states
        alphabet: input alphabet
        num_accepting_states: number of accepting states (Default value = 1)
        compute_prefixes: if true, shortest path to reach each state will be computed (Default value = False)
        ensure_minimality: returned automaton will be minimal

    Returns:

        Randomly generated DFA

    """
    if num_states <= num_accepting_states:
        num_accepting_states = num_states // 2

    random_dfa = generate_random_deterministic_automata('dfa', num_states,
                                                        input_alphabet_size=len(alphabet),
                                                        output_alphabet_size=2,
                                                        ensure_minimality=ensure_minimality,
                                                        compute_prefixes=compute_prefixes,
                                                        custom_input_alphabet=alphabet,
                                                        num_accepting_states=num_accepting_states)

    return random_dfa


def generate_random_mdp(num_states, input_size, output_size, possible_probabilities=None):
    """
    Generates random MDP.

    Args:

        num_states: number of states
        input_size: number of inputs
        output_size: user predefined outputs
        possible_probabilities: list of possible probability pairs to choose from

    Returns:

        random MDP

    """

    deterministic_model = generate_random_deterministic_automata('moore', num_states, input_size, output_size)

    if input_size > output_size:
        assert False, 'Cannot create deterministic MDP (in all states, all input-output pairs leads to a single state)' \
                      ', if number of inputs is smaller than number of outputs)'

    if not possible_probabilities:
        possible_probabilities = [(1.,), (1.,), (1.,), (0.9, 0.1),
                                  (0.8, 0.2), (0.7, 0.3), (0.8, 0.1, 0.1), (0.7, 0.2, 0.1), (0.6, 0.2, 0.1, 0.1)]
        # ensure that there are no infinite loops
        max_prob_num = min(num_states, input_size)
        possible_probabilities = [p for p in possible_probabilities if len(p) <= max_prob_num]

    mdp_states = []
    state_id_state_map = {}
    for state in deterministic_model.states:
        mdp_state = MdpState(state.state_id, state.output)
        state_id_state_map[state.state_id] = mdp_state
        mdp_states.append(mdp_state)

    input_al = deterministic_model.get_input_alphabet()
    for deterministic_state in deterministic_model.states:
        for i in input_al:
            state_from_det_model = state_id_state_map[deterministic_state.transitions[i].state_id]
            prob = random.choice(possible_probabilities)

            reached_states = [state_from_det_model]
            for _ in range(len(prob) - 1):
                while True:
                    new_state = random.choice(mdp_states)

                    # ensure determinism
                    if new_state.output not in {s.output for s in reached_states}:
                        break

                reached_states.append(new_state)

            for prob, reached_state in zip(prob, reached_states):
                mdp_origin_state = state_id_state_map[deterministic_state.state_id]
                mdp_origin_state.transitions[i].append((reached_state, prob))

    # deterministically labeled check
    for state in mdp_states:
        for _, transition_values in state.transitions.items():
            reached_outputs = [s.output for s, _ in transition_values]
            assert len(reached_outputs) == len(set(reached_outputs))

    return Mdp(mdp_states[0], mdp_states)


def generate_random_smm(num_states, input_size, output_size, possible_probabilities=None):
    """
    Generates random SMM.

    Args:

        num_states: number of states
        input_size: number of inputs
        output_size: number of outputs
        possible_probabilities: list of possible probability pairs to choose from

    Returns:

        random SMM

    """

    deterministic_model = generate_random_deterministic_automata('mealy', num_states, input_size, output_size)
    input_al = deterministic_model.get_input_alphabet()
    output_al = list(set([o for state in deterministic_model.states for o in state.output_fun.values()]))
    output_al.sort()

    if input_size > output_size:
        assert False, 'Cannot create deterministic SMM (in all states, all input-output pairs leads to a single state)' \
                      ', if number of inputs is smaller than number of outputs)'

    if not possible_probabilities:
        possible_probabilities = [(1.,), (1.,), (1.,), (0.9, 0.1),
                                  (0.8, 0.2), (0.7, 0.3), (0.8, 0.1, 0.1), (0.7, 0.2, 0.1), (0.6, 0.2, 0.1, 0.1)]
        # ensure that there are no infinite loops
        max_prob_num = min(num_states, input_size)
        possible_probabilities = [p for p in possible_probabilities if len(p) <= max_prob_num]

    smm_states = []
    state_id_state_map = {}
    for state in deterministic_model.states:
        smm_state = StochasticMealyState(state.state_id)
        state_id_state_map[state.state_id] = smm_state
        smm_states.append(smm_state)

    for deterministic_state in deterministic_model.states:
        for i in input_al:
            state_from_det_model = state_id_state_map[deterministic_state.transitions[i].state_id]
            output_from_det_model = deterministic_state.output_fun[i]

            prob = random.choice(possible_probabilities)

            state_id_state_map[deterministic_state.state_id].transitions[i].append(
                (state_from_det_model, output_from_det_model, prob[0]))

            observed_outputs = [output_from_det_model]
            for prob_index in range(1, len(prob)):
                while True:
                    new_state = random.choice(smm_states)
                    new_output = random.choice(output_al)

                    # ensure determinism
                    if new_output not in observed_outputs:
                        state_id_state_map[deterministic_state.state_id].transitions[i].append(
                            (new_state, new_output, prob[prob_index]))
                        break

    return StochasticMealyMachine(smm_states[0], smm_states)


def generate_random_ONFSM(num_states, num_inputs, num_outputs, multiple_out_prob=0.33):
    """
    Randomly generate an observable non-deterministic finite-state machine.

    Args:

      num_states: number of states
      num_inputs: number of inputs
      num_outputs: number of outputs
      multiple_out_prob: probability that state will have multiple outputs (Default value = 0.5)

    Returns:

        randomly generated ONFSM

    """
    inputs = [f'i{i + 1}' for i in range(num_inputs)]
    outputs = [f'o{i + 1}' for i in range(num_outputs)]

    states = []
    for i in range(num_states):
        state = OnfsmState(f's{i}')
        states.append(state)

    state_buffer = states.copy()

    for state in states:
        for i in inputs:
            state_outputs = 1
            if random.random() <= multiple_out_prob and num_outputs >= 2:
                state_outputs = random.randint(2, num_outputs)

            random_out = random.sample(outputs, state_outputs)
            for index in range(state_outputs):
                if state_buffer:
                    new_state = random.choice(state_buffer)
                    state_buffer.remove(new_state)
                else:
                    new_state = random.choice(states)
                state.transitions[i].append((random_out[index], new_state))

    return Onfsm(states[0], states)


def generate_random_markov_chain(num_states):
    assert num_states >= 3
    possible_probabilities = [1.0, 1.0, 0.8, 0.5, 0.9]
    states = []

    for i in range(num_states):
        states.append(McState(f'q{i}', i))

    for index, state in enumerate(states[:-1]):
        prob = random.choice(possible_probabilities)
        if prob == 1.:
            new_state = states[index + 1]
            state.transitions.append((new_state, prob))
        else:
            next_state = states[index + 1]
            up_states = list(states)
            up_states.remove(next_state)
            rand_state = random.choice(up_states)

            state.transitions.append((next_state, prob))
            state.transitions.append((rand_state, round(1 - prob, 2)))

    return MarkovChain(states[0], states)


def _has_transition(state: SevpaState, transition_letter, stack_guard) -> bool:
    transitions = state.transitions[transition_letter]
    if transitions is not None:
        if stack_guard is None:  # internal transition
            for transition in transitions:
                if transition.letter == transition_letter:
                    return True
        else:  # return transition
            for transition in transitions:
                if transition.stack_guard == stack_guard and transition.letter == transition_letter:
                    return True

    return False


def generate_random_sevpa(num_states, internal_alphabet_size, call_alphabet_size, return_alphabet_size
                          , acceptance_prob, return_transition_prob):
    """
    Generate a random Single Entry Visibly Pushdown Automaton (SEVPA).

    Args:
        num_states (int): The number of states in the SEVPA.
        internal_alphabet_size (int): The size of the internal alphabet.
        call_alphabet_size (int): The size of the call alphabet.
        return_alphabet_size (int): The size of the return alphabet.
        acceptance_prob (float): The probability of a state being an accepting state.
        return_transition_prob (float): The probability of generating a return transition.

    Returns:
        Sevpa: A randomly generated SEVPA.
    """

    internal_alphabet = [f'i{i}' for i in range(internal_alphabet_size)]
    call_alphabet = [f'c{i}' for i in range(call_alphabet_size)]
    return_alphabet = [f'r{i}' for i in range(return_alphabet_size)]

    sevpa_alphabet = SevpaAlphabet(internal_alphabet, call_alphabet, return_alphabet)

    states = [SevpaState(f'q{i}', random.random() < acceptance_prob) for i in range(num_states)]
    state_buffer = states.copy()

    for state in states:
        if not internal_alphabet or random.uniform(0.0, 1.0) < return_transition_prob:
            # add return transition
            while True:
                return_letter = random.choice(return_alphabet)
                stack_state = random.choice(states) if not state_buffer else random.choice(state_buffer)
                if stack_state in state_buffer:
                    state_buffer.remove(stack_state)

                call_letter = random.choice(call_alphabet)
                stack_guard = (stack_state.state_id, call_letter)

                if not _has_transition(state, return_letter, stack_guard):
                    break

            target_state = random.choice(states)
            state.transitions[return_letter].append(
                SevpaTransition(target_state, return_letter, 'pop', stack_guard))
        else:
            # add an internal transition
            while True:
                internal_letter = random.choice(internal_alphabet)
                if not _has_transition(state, internal_letter, None):
                    break

            target_state = random.choice(states) if not state_buffer else random.choice(state_buffer)
            if target_state in state_buffer:
                state_buffer.remove(target_state)
            state.transitions[internal_letter].append(
                SevpaTransition(target_state, internal_letter, None, None))

    assert len(states) == num_states
    initial_state = random.choice(states)

    for state in states:
        for internal_letter in internal_alphabet:
            if state.transitions[internal_letter] is None:
                target_state = random.choice(states)
                state.transitions[internal_letter].append(
                    SevpaTransition(target_state, internal_letter, None, None))

        for call_letter in call_alphabet:
            for stack_state in states:
                stack_guard = (stack_state.state_id, call_letter)
                for return_letter in return_alphabet:
                    if not _has_transition(state, return_letter, stack_guard):
                        target_state = states[random.randint(0, len(states) - 1)]
                        state.transitions[return_letter].append(
                            SevpaTransition(target_state, return_letter, 'pop', stack_guard))

    return Sevpa(initial_state, states)
