import os
from statistics import mean

from aalpy.SULs import AutomatonSUL
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import StatePrefixEqOracle
from aalpy.utils import load_automaton_from_file

dfa_1000_states_20_inputs = '../DotModels/DFA_1000_states_20_inp'
dfa_2000_states_10_inputs = '../DotModels/DFA_2000_states_10_inp'
moore_1000_states_20_inputs = '../DotModels/Moore_1000_states_20_inp_out'
moore_2000_states_10_inputs = '../DotModels/Moore_2000_states_10_inp_out'
run_times = []

# change on which folder to perform experiments
exp = dfa_2000_states_10_inputs

benchmarks = os.listdir(exp)
benchmarks = benchmarks[:10]

caching_opt = [True, False]
closing_options = ['shortest_first', 'longest_first', 'single']
suffix_processing = ['all', 'single']
counter_example_processing = ['rs', 'longest_prefix', None]
e_closedness = ['prefix', 'suffix']

for b in benchmarks:
    automaton = load_automaton_from_file(f'{exp}/{b}', automaton_type='dfa')
    input_al = automaton.get_input_alphabet()

    sul_dfa = AutomatonSUL(automaton)

    state_origin_eq_oracle = StatePrefixEqOracle(input_al, sul_dfa, walks_per_state=5, walk_len=25)

    learned_dfa, data = run_Lstar(input_al, sul_dfa, state_origin_eq_oracle, automaton_type='dfa',
                            cache_and_non_det_check=False, cex_processing='rs', return_data=True, print_level=0)
    run_times.append(data['total_time'])

print(run_times)
print(mean(run_times))
