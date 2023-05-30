import random
import warnings

from aalpy.automata import Dfa, DfaState, MdpState, Mdp, MealyMachine, MealyState, \
    MooreMachine, MooreState, OnfsmState, Onfsm, MarkovChain, McState, StochasticMealyState, StochasticMealyMachine


def generate_random_deterministic_automata(automaton_type,
                                           num_states,
                                           input_alphabet_size,
                                           output_alphabet_size,
                                           compute_prefixes=False,
                                           ensure_minimality=True,
                                           **kwargs
                                           ):
    """
    Generates a random deterministic automata of 'automaton_type'.

    Args:
        automaton_type: type of automaton, either 'dfa', 'mealy', or 'moore'
        num_states: number of states
        input_alphabet_size: size of input alphabet
        output_alphabet_size: size of output alphabet (ignored for DFAs)
        compute_prefixes: compute prefixes leading to each state
        ensure_minimality: ensure that the automaton is minimal
        **kwargs:
            : 'custom_input_alphabet'  a list of custom input alphabet values
            : 'custom_output_alphabet' a list of custom output alphabet values
            : 'num_accepting_states' number of accepting states for DFA generation

    Returns:

        Random deterministic automaton of user defined type, size. If ensure_minimality is set to False returned
        automaton is not necessarily minimal. If minimality is reacquired and random automaton cannot be produced in
        multiple interactions, non-minimal automaton will be returned and a warning message printed.
    """

    assert automaton_type in {'dfa', 'mealy', 'moore'}
    if output_alphabet_size < 2 or output_alphabet_size is None:
        output_alphabet_size = 2

    state_class_map = {'dfa': DfaState, 'mealy': MealyState, 'moore': MooreState}
    automaton_class_map = {'dfa': Dfa, 'mealy': MealyMachine, 'moore': MooreMachine}

    input_alphabet = [f'i{i + 1}' for i in range(input_alphabet_size)]
    output_alphabet = [f'o{i + 1}' for i in range(output_alphabet_size)] if automaton_type != 'dfa' else [True, False]

    # For backwards comparability or if uses passes custom input output functions
    if 'custom_input_alphabet' in kwargs:
        input_alphabet = kwargs.get('custom_input_alphabet')
    if 'custom_output_alphabet' in kwargs:
        output_alphabet = kwargs.get('custom_output_alphabet')
    accepting_state_ids = None
    if 'num_accepting_states' in kwargs:
        num_accepting_states = kwargs.get('num_accepting_states')
        accepting_state_ids = [f's{i + 1}' for i in random.sample(list(range(num_states)), k=num_accepting_states)]

    states = [state_class_map[automaton_type](state_id=f's{i + 1}') for i in range(num_states)]
    state_id_state_map = {state.state_id: state for state in states}

    if automaton_type != 'mealy':
        for state in states:
            output = random.choice(output_alphabet)
            if automaton_type == 'dfa':
                if accepting_state_ids is not None:
                    output = state.state_id in accepting_state_ids
                state.is_accepting = output
            else:
                state.output = output

    state_buffer = [state.state_id for state in states]
    queue = [states[0]]
    state_buffer.remove(states[0].state_id)
    visited_states = set()

    while queue:
        state = queue.pop(0)
        visited_states.add(state.state_id)
        for i in input_alphabet:
            # states from which to choose next state (while all states have not be reached)
            if state_buffer:
                new_state_candidates = [state_id_state_map[state_id] for state_id in state_buffer]
            else:
                new_state_candidates = states

            new_state = random.choice(new_state_candidates)

            if new_state.state_id in state_buffer:
                state_buffer.remove(new_state.state_id)

            state.transitions[i] = new_state

            if automaton_type == 'mealy':
                state.output_fun[i] = random.choice(output_alphabet)

        for child in state.transitions.values():
            if child.state_id not in visited_states and child not in queue:
                queue.append(child)

    random_automaton = automaton_class_map[automaton_type](states[0], states)

    if compute_prefixes:
        for state in random_automaton.states:
            state.prefix = random_automaton.get_shortest_path(random_automaton.initial_state, state)
            if state != random_automaton.initial_state and not state.prefix:
                print('Non-reachable state:', state.state_id)

    if ensure_minimality:
        minimality_iterations = 1
        while not random_automaton.is_minimal():
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
                                                                      compute_prefixes,  # compute prefixes
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
        num_accepting_states = num_states - 1

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

    inputs = [f'i{i + 1}' for i in range(input_size)]
    outputs = [f'o{i + 1}' for i in range(output_size)]

    if not possible_probabilities:
        possible_probabilities = [(1.,), (1.,), (1.,), (0.9, 0.1),
                                  (0.8, 0.2), (0.7, 0.3), (0.8, 0.1, 0.1), (0.7, 0.2, 0.1), (0.6, 0.2, 0.1, 0.1)]
        # ensure that there are no infinite loops
        max_prob_num = min(num_states, input_size)
        possible_probabilities = [p for p in possible_probabilities if len(p) <= max_prob_num]

    state_outputs = outputs.copy()
    states = []
    for i in range(num_states):
        curr_output = state_outputs.pop(0) if state_outputs else random.choice(outputs)
        states.append(MdpState(f'q{i}', curr_output))

    state_buffer = list(states)
    for state in states:
        for i in inputs:
            prob = random.choice(possible_probabilities)
            reached_states = []
            for _ in prob:
                while True:
                    new_state = random.choice(state_buffer) if state_buffer else random.choice(states)

                    # ensure determinism
                    if new_state.output not in {s.output for s in reached_states}:
                        if state_buffer:
                            state_buffer.remove(new_state)
                        break

                reached_states.append(new_state)

            for prob, reached_state in zip(prob, reached_states):
                state.transitions[i].append((reached_state, prob))

    return Mdp(states[0], states)


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

    inputs = [f'i{i + 1}' for i in range(input_size)]
    outputs = [f'o{i + 1}' for i in range(output_size)]

    if not possible_probabilities:
        possible_probabilities = [(1.,), (1.,), (1.,), (0.9, 0.1),
                                  (0.8, 0.2), (0.7, 0.3), (0.8, 0.1, 0.1), (0.7, 0.2, 0.1), (0.6, 0.2, 0.1, 0.1)]
        # ensure that there are no infinite loops
        max_prob_num = min(num_states, input_size)
        possible_probabilities = [p for p in possible_probabilities if len(p) <= max_prob_num]

    states = []
    for i in range(num_states):
        states.append(StochasticMealyState(f'q{i}'))

    state_buffer = list(states)
    output_buffer = outputs.copy()
    for state in states:
        for i in inputs:
            prob = random.choice(possible_probabilities)
            reached_states = []
            transition_outputs = []
            for _ in prob:
                while True:
                    o = random.choice(output_buffer) if output_buffer else random.choice(outputs)
                    new_state = random.choice(state_buffer) if state_buffer else random.choice(states)

                    # ensure determinism
                    if o not in transition_outputs:
                        if output_buffer:
                            output_buffer.remove(o)
                        if state_buffer:
                            state_buffer.remove(new_state)
                        break

                reached_states.append(new_state)
                transition_outputs.append(o)

            for index in range(len(prob)):
                state.transitions[i].append((reached_states[index], transition_outputs[index], prob[index]))

    return StochasticMealyMachine(states[0], states)


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


def dfa_from_state_setup(state_setup) -> Dfa:
    """
        First state in the state setup is the initial state.
        Example state setup:
        state_setup = {
                "a": (True, {"x": "b1", "y": "a"}),
                "b1": (False, {"x": "b2", "y": "a"}),
                "b2": (True, {"x": "b3", "y": "a"}),
                "b3": (False, {"x": "b4", "y": "a"}),
                "b4": (False, {"x": "c", "y": "a"}),
                "c": (True, {"x": "a", "y": "a"}),
            }

        Args:

            state_setup: map from state_id to tuple(output and transitions_dict)

        Returns:

            DFA
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


def moore_from_state_setup(state_setup) -> MooreMachine:
    """
    First state in the state setup is the initial state.
    Example state setup:
    state_setup = {
            "a": ("a", {"x": "b1", "y": "a"}),
            "b1": ("b", {"x": "b2", "y": "a"}),
            "b2": ("b", {"x": "b3", "y": "a"}),
            "b3": ("b", {"x": "b4", "y": "a"}),
            "b4": ("b", {"x": "c", "y": "a"}),
            "c": ("c", {"x": "a", "y": "a"}),
        }

    Args:

        state_setup: map from state_id to tuple(output and transitions_dict)

    Returns:

        Moore machine
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


def mealy_from_state_setup(state_setup) -> MealyMachine:
    """
        First state in the state setup is the initial state.
        state_setup = {
            "a": {"x": ("o1", "b1"), "y": ("o2", "a")},
            "b1": {"x": ("o3", "b2"), "y": ("o1", "a")},
            "b2": {"x": ("o1", "b3"), "y": ("o2", "a")},
            "b3": {"x": ("o3", "b4"), "y": ("o1", "a")},
            "b4": {"x": ("o1", "c"), "y": ("o4", "a")},
            "c": {"x": ("o3", "a"), "y": ("o5", "a")},
        }


    Args:

        state_setup:
            state_setup should map from state_id to tuple(transitions_dict).

    Returns:

        Mealy Machine
    """
    # state_setup should map from state_id to tuple(transitions_dict).
    # Each entry in transition dict is <input> : <output, new_state_id>

    # build states with state_id and output
    states = {key: MealyState(key) for key, _ in state_setup.items()}

    # add transitions to states
    for state_id, state in states.items():
        for _input, (output, new_state) in state_setup[state_id].items():
            state.transitions[_input] = states[new_state]
            state.output_fun[_input] = output

    # states to list
    states = [state for state in states.values()]

    # build moore machine with first state as starting state
    mm = MealyMachine(states[0], states)

    for state in states:
        state.prefix = mm.get_shortest_path(mm.initial_state, state)

    return mm


def mdp_from_state_setup(state_setup):
    from aalpy.automata import MdpState, Mdp
    states_map = {key: MdpState(key, output=value[0]) for key, value in state_setup.items()}

    for key, values in state_setup.items():
        source = states_map[key]
        for i, transitions in values[1].items():
            for node, prob in transitions:
                source.transitions[i].append((states_map[node], prob))

    initial_state = states_map[list(state_setup.keys())[0]]
    return Mdp(initial_state, list(states_map.values()))


def smm_from_state_setup(state_setup):
    from aalpy.automata import StochasticMealyState, StochasticMealyMachine
    states_map = {key: StochasticMealyState(key) for key in state_setup.keys()}

    for key, values in state_setup.items():
        source = states_map[key]
        for i, transitions in values.items():
            for node, output, prob in transitions:
                source.transitions[i].append((states_map[node], output, prob))

    initial_state = states_map[list(state_setup.keys())[0]]
    return StochasticMealyMachine(initial_state, list(states_map.values()))
