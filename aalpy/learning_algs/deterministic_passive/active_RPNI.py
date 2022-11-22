import random
import time

from aalpy.SULs import DfaSUL
from aalpy.base.SUL import CacheSUL
from aalpy.learning_algs import run_RPNI, run_Lstar
from aalpy.oracles import RandomWalkEqOracle
from aalpy.utils import convert_i_o_traces_for_RPNI, get_Angluin_dfa, generate_random_deterministic_automata
from aalpy.utils.HelperFunctions import print_learning_info
from aalpy.learning_algs.deterministic.CounterExampleProcessing import rs_cex_processing


def ensure_input_completeness(alphabet, hypothesis, input_completeness_method):
    hypothesis_alphabet = hypothesis.get_input_alphabet()
    if set(alphabet) == set(hypothesis_alphabet):
        return hypothesis

    for i in alphabet:
        if i not in hypothesis_alphabet:
            hypothesis.initial_state.transitions[i] = hypothesis.initial_state

    hypothesis.make_input_complete(input_completeness_method)
    return hypothesis


def extend_data_set(inputs, outputs, data):
    io_trace = [[(i, o) for i, o in zip(inputs, outputs)]]
    data.extend(convert_i_o_traces_for_RPNI(io_trace))


def run_active_RPNI(alphabet, sul, eq_oracle, automaton_type, num_random_samples, print_level=2):
    data = []

    start_time = time.time()
    eq_query_time = 0

    char_set = [(a,) for a in alphabet]

    sul = CacheSUL(sul)
    eq_oracle.sul = sul

    for a in char_set:
        outputs = sul.query(a)
        extend_data_set(a, outputs, data)

    learning_rounds = 0
    while True:
        learning_rounds += 1
        hypothesis = run_RPNI(data, automaton_type, input_completeness='self_loop', print_info=False)

        for state in hypothesis.states:
            state.prefix = hypothesis.get_shortest_path(hypothesis.initial_state, state)
        # hypothesis = ensure_input_completeness(alphabet, hypothesis, 'self_loop')

        if print_level > 1:
            print(f'Learning round {learning_rounds}: {hypothesis.size} states.')

        eq_query_start = time.time()
        cex = eq_oracle.find_cex(hypothesis)
        eq_query_time += time.time() - eq_query_start

        if cex is None:
            break

        dist_suffix = rs_cex_processing(sul, cex, hypothesis, suffix_closedness=False)[0]
        char_set.append(dist_suffix)

        cex_output = sul.query(cex)
        extend_data_set(cex, cex_output, data)

        for state in hypothesis.states:
            for suffix in char_set:
                inputs = state.prefix + suffix
                outputs = sul.query(inputs)
                extend_data_set(inputs, outputs, data)

        if print_level == 3:
            print('Counterexample', cex)

    total_time = round(time.time() - start_time, 2)
    eq_query_time = round(eq_query_time, 2)
    learning_time = round(total_time - eq_query_time, 2)

    info = {
        'learning_rounds': learning_rounds,
        'automaton_size': hypothesis.size,
        'queries_learning': sul.num_queries,
        'steps_learning': sul.num_steps,
        'queries_eq_oracle': eq_oracle.num_queries,
        'steps_eq_oracle': eq_oracle.num_steps,
        'learning_time': learning_time,
        'eq_oracle_time': eq_query_time,
        'total_time': total_time,
    }

    if print_level > 0:
        print_learning_info(info)

    return hypothesis


model = generate_random_deterministic_automata('dfa', 5, input_alphabet_size=4, output_alphabet_size=2)
sul = DfaSUL(model)
alphabet = model.get_input_alphabet()
eq_oracle = RandomWalkEqOracle(alphabet, sul, )

print('Active RPNI')
learned_model = run_active_RPNI(alphabet, sul, eq_oracle, 'dfa', num_random_samples=1)

print('L*')
eq_oracle = RandomWalkEqOracle(alphabet, sul, )
learned_model = run_Lstar(alphabet, sul, eq_oracle, 'dfa',)

