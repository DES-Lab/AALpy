import random

from aalpy.automata import Dfa, DfaState, MdpState, Mdp, MealyMachine, MealyState, \
    MooreMachine, MooreState, OnfsmState, Onfsm, MarkovChain, McState, StochasticMealyState, StochasticMealyMachine
from aalpy.utils.HelperFunctions import random_string_generator


def generate_random_mealy_machine(num_states, input_alphabet, output_alphabet, compute_prefixes=False) -> MealyMachine:
    """
    Generates a random Mealy machine.

    Args:

        num_states: number of states
        input_alphabet: input alphabet
        output_alphabet: output alphabet
        compute_prefixes: if true, shortest path to reach each state will be computed (Default value = False)

    Returns:
        Mealy machine with num_states states

    """
    states = list()

    for i in range(num_states):
        states.append(MealyState(i))

    state_buffer = list(states)
    for state in states:
        for a in input_alphabet:
            if state_buffer:
                new_state = random.choice(state_buffer)
                state_buffer.remove(new_state)
            else:
                new_state = random.choice(states)
            state.transitions[a] = new_state
            state.output_fun[a] = random.choice(output_alphabet)

    mm = MealyMachine(states[0], states)
    if compute_prefixes:
        for state in states:
            state.prefix = mm.get_shortest_path(mm.initial_state, state)

    return mm


def generate_random_moore_machine(num_states, input_alphabet, output_alphabet, compute_prefixes=False) -> MooreMachine:
    """
    Generates a random Moore machine.

    Args:

        num_states: number of states
        input_alphabet: input alphabet
        output_alphabet: output alphabet
        compute_prefixes: if true, shortest path to reach each state will be computed (Default value = False)

    Returns:

        Moore machine with num_states states

    """
    states = list()

    for i in range(num_states):
        states.append(MooreState(i, random.choice(output_alphabet)))

    state_buffer = list(states)
    for state in states:
        for a in input_alphabet:
            if state_buffer:
                new_state = random.choice(state_buffer)
                state_buffer.remove(new_state)
            else:
                new_state = random.choice(states)
            state.transitions[a] = new_state

    mm = MooreMachine(states[0], states)
    if compute_prefixes:
        for state in states:
            state.prefix = mm.get_shortest_path(mm.initial_state, state)

    return mm


def generate_random_dfa(num_states, alphabet, num_accepting_states=1, compute_prefixes=False) -> Dfa:
    """
    Generates a random DFA.

    Args:

        num_states: number of states
        alphabet: input alphabet
        num_accepting_states: number of accepting states (Default value = 1)
        compute_prefixes: if true, shortest path to reach each state will be computed (Default value = False)

    Returns:

        DFA

    """
    assert num_states >= num_accepting_states
    states = list()

    for i in range(num_states):
        states.append(DfaState(i))

    state_buffer = list(states)
    for state in states:
        for a in alphabet:
            if state_buffer:
                new_state = random.choice(state_buffer)
                state_buffer.remove(new_state)
            else:
                new_state = random.choice(states)
            state.transitions[a] = new_state

    for _ in range(num_accepting_states):
        random.choice(states).is_accepting = True

    dfa = Dfa(states[0], states)
    if compute_prefixes:
        for state in states:
            state.prefix = dfa.get_shortest_path(dfa.initial_state, state)

    return dfa


def generate_random_mdp(num_states, len_input, custom_outputs=None, num_unique_outputs=None):
    """
    Generates random MDP.

    Args:

        num_states: number of states
        len_input: number of inputs
        custom_outputs: user predefined outputs
        num_unique_outputs: number of outputs

    Returns:

        random MDP

    """
    num_unique_outputs = num_states if not num_unique_outputs else num_unique_outputs
    outputs = [random_string_generator(random.randint(3, 7)) for _ in range(num_unique_outputs)]
    outputs = custom_outputs if custom_outputs else outputs

    while len(outputs) < num_states:
        outputs.append(random.choice(outputs))

    possible_probabilities = [1.0, 1.0, 1.0, 1.0, 0.8, 0.5, 0.9]
    states = []
    for i in range(num_states):
        states.append(MdpState(f'q{i}', outputs.pop()))

    state_buffer = list(states)
    for state in states:
        for i in range(len_input):
            prob = random.choice(possible_probabilities)
            if state_buffer:
                new_state = random.choice(state_buffer)
                state_buffer.remove(new_state)
            else:
                new_state = random.choice(states)

            if prob == 1.:
                state.transitions[i].append((new_state, prob))
            else:
                new_states = list(states)
                s1 = new_state
                new_states.remove(s1)

                state.transitions[i].append((s1, prob))
                state.transitions[i].append((random.choice(new_states), round(1 - prob, 2)))

    return Mdp(states[0], states), list(range(len_input))


def generate_random_smm(num_states, num_inputs, num_output):
    """
    Generates random MDP.

    Args:

        num_states: number of states
        num_inputs: number of inputs
        num_output: number of outputs

    Returns:

        random SMM

    """
    import string
    inputs = list(range(num_inputs))
    outputs = list(string.ascii_uppercase)[:num_output]

    possible_probabilities = [1.0, 1.0, 1.0, 1.0, 0.75, 0.5, 0.9]

    states = []
    for i in range(num_states):
        states.append(StochasticMealyState(f'q{i}'))

    state_buffer = list(states)
    for state in states:
        for i in range(num_inputs):
            if state_buffer:
                new_state = random.choice(state_buffer)
                state_buffer.remove(new_state)
            else:
                new_state = random.choice(states)

            prob = random.choice(possible_probabilities)
            if prob == 1.:
                state.transitions[i].append((new_state, random.choice(outputs), prob))
            else:
                new_states, new_outputs = list(states), list(outputs)
                s1 = new_state
                o1 = random.choice(new_outputs)
                new_states.remove(s1)
                new_outputs.remove(o1)

                state.transitions[i].append((s1, o1, prob))
                state.transitions[i].append((random.choice(new_states), random.choice(outputs), round(1 - prob, 2)))

    return StochasticMealyMachine(states[0], states)


def generate_random_ONFSM(num_states, num_inputs, num_outputs, multiple_out_prob=0.1):
    """
    Randomly generate an observable non-deterministic finite-state machine.

    Args:

      num_states: number of states
      num_inputs: number of inputs
      num_outputs: number of outputs
      multiple_out_prob: probability that state will have multiple outputs (Default value = 0.1)

    Returns:

        randomly generated ONFSM

    """
    inputs = [random_string_generator(random.randint(1, 3)) for _ in range(num_inputs)]
    outputs = [random_string_generator(random.randint(3, 7)) for _ in range(num_outputs)]

    states = []
    for i in range(num_states):
        state = OnfsmState(f's{i}')
        states.append(state)

    for state in states:
        for i in inputs:
            state_outputs = 1
            if random.random() <= multiple_out_prob and num_outputs > 1:
                state_outputs = random.randint(2, num_outputs)

            random_out = random.sample(outputs, state_outputs)
            for index in range(state_outputs):
                state.transitions[i].append((random_out[index], random.choice(states)))

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
