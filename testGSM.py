import random

from aalpy.automata import MooreMachine
from aalpy.learning_algs.deterministic_passive.GeneralizedStateMerging import GeneralizedStateMerging
from aalpy.utils import convert_i_o_traces_for_RPNI, generate_random_deterministic_automata
from aalpy.utils.ModelChecking import bisimilar, compare_automata


def get_characterizing_set(automaton, prefix_closed=False):
    # sanity check
    automaton.minimize()

    alphabet = automaton.get_input_alphabet()
    data_set = []

    if isinstance(automaton, MooreMachine):
        data_set.append(((),automaton.initial_state.output))
    char_set = automaton.compute_characterization_set()
    for i in alphabet:
        if (i,) not in char_set:
            char_set.append((i,))

    assert char_set

    input_sequances = []
    for state in automaton.states:
        for suffix in char_set:
            input_sequances.append(state.prefix + suffix)
            for i in alphabet:
                input_sequances.append(state.prefix + (i,) + suffix)

    for input_seq in input_sequances:
        if input_seq:
            outputs = automaton.execute_sequence(automaton.initial_state, input_seq)
        else:
            outputs = [automaton.initial_state.output]

        if prefix_closed:
            i_o = list(zip(input_seq, outputs))
            data_set.extend(convert_i_o_traces_for_RPNI([i_o]))
        else:
            data_set.append((input_seq, outputs[-1]))

    return data_set


def get_random_samples(automaton, num_random_seq, length, prefix_closed=False):
    alphabet = automaton.get_input_alphabet()
    data_set = []

    if isinstance(automaton, MooreMachine):
        data_set.append(((), automaton.initial_state.output))

    for _ in range(num_random_seq):
        k = random.randint(1, length)
        random_seq = random.choices(alphabet, k=k)
        outputs = automaton.compute_output_seq(automaton.initial_state, random_seq)

        if prefix_closed:
            i_o = list(zip(random_seq, outputs))
            data_set.extend(convert_i_o_traces_for_RPNI([i_o]))
        else:
            data_set.append((random_seq, outputs[-1]))

    return data_set


num_states = 10
num_inputs = 4
num_outputs = 5
num_random_samples = num_states * 1000
sample_length = num_states * 2

num_tests = 1000

automata_types = ['mealy', 'moore']

for automaton_type in automata_types:
    for i in range(num_tests):
        random.seed(i)
        print(f'{automaton_type} seed: {i}')
        ground_truth = generate_random_deterministic_automata(automaton_type, num_states, num_inputs, num_outputs)
        # samples = get_random_samples(ground_truth, num_random_samples, sample_length,)
        samples = get_characterizing_set(ground_truth, prefix_closed=True)

        gsm_state = GeneralizedStateMerging(samples, automaton_type, print_info=False)
        learned_model = gsm_state.run()

        cex = bisimilar(learned_model, ground_truth)
        cex_sanity_check = compare_automata(ground_truth, learned_model, num_cex=1)
        if cex is not None or cex_sanity_check:
            if cex is None:
                print("Warning: Bisimilarity found none")
                cex = cex_sanity_check[0]
            print(f"Counterexample found: {cex}")
            print(ground_truth.execute_sequence(ground_truth.initial_state, cex))
            print(learned_model.execute_sequence(learned_model.initial_state, cex))

            learned_model.visualize("learned")
            ground_truth.visualize("truth")
            exit()
