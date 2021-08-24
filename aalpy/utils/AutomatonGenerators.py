import string
from aalpy.automata.IotsMachine import IotsMachine, IotsState
import random

from aalpy.automata import Dfa, DfaState, MdpState, Mdp, MealyMachine, MealyState, \
    MooreMachine, MooreState, OnfsmState, Onfsm
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

    for state in states:
        for a in input_alphabet:
            state.transitions[a] = random.choice(states)
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

    for state in states:
        for a in input_alphabet:
            state.transitions[a] = random.choice(states)

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


def generate_random_mdp(num_states, len_input, num_unique_outputs=None):
    """
    Generates random MDP.

    Args:

        num_states: number of states
        len_input: number of inputs
        num_unique_outputs: number of outputs

    Returns:

        random MDP

    """
    num_unique_outputs = num_states if not num_unique_outputs else num_unique_outputs
    outputs = [random_string_generator(random.randint(3, 7)) for _ in range(num_unique_outputs)]

    while len(outputs) < num_states:
        outputs.append(random.choice(outputs))

    possible_probabilities = [1.0, 1.0, 1.0, 1.0, 0.8, 0.5, 0.9]
    states = []
    for i in range(num_states):
        states.append(MdpState(f'q{i}', outputs.pop()))

    for state in states:
        for i in range(len_input):
            prob = random.choice(possible_probabilities)
            if prob == 1.:
                state.transitions[i].append((random.choice(states), prob))
            else:
                new_states = list(states)
                s1 = random.choice(new_states)
                new_states.remove(s1)

                state.transitions[i].append((s1, prob))
                state.transitions[i].append((random.choice(new_states), round(1 - prob, 2)))

    return Mdp(states[0], states), list(range(len_input))


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


def generate_random_iots(num_states, num_inputs, num_outputs, max_num_inputs_per_state=1, max_num_outputs_per_state=1,
                         deterministic=True) -> IotsMachine:
    """
    Randomly generate an input-output-tansition-system machine.

    The automaton may not be connected.

    Args:

      num_states: number of states
      num_inputs: number of inputs
      num_outputs: number of outputs
      max_num_inputs_per_state: maximal number of inputs per state (Default value = 1)
      max_num_outputs_per_state: maximal number of outputs per state (Default value = 1)
      deterministic: if true the automata is deterministic if false the automata may be non-deterministic
     
    Returns:

        randomly generated Iots

    """
    inputs = ['?' + string.ascii_uppercase[i] for i in range(num_inputs)]
    outputs = ['!' + string.ascii_lowercase[::-1][i] for i in range(num_outputs)]
    states = [IotsState(f's{i}') for i in range(num_states)]

    for state in states:
        while True:
            num_state_inputs = random.randint(0, max_num_inputs_per_state)
            num_state_outputs = random.randint(0, max_num_outputs_per_state)

            if num_state_inputs <= 1:
                break

            if num_state_outputs != 0:
                break

        if deterministic:
            for input in random.sample(inputs, num_state_inputs):
                state.add_input(input, random.choice(states))

            for output in random.sample(outputs, num_state_outputs):
                state.add_output(output, random.choice(states))
        else:
            raise Exception("Not implemented yet")

    return IotsMachine(states[0], states)
