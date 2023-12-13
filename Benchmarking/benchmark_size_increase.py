from statistics import mean
import csv

from aalpy.SULs import AutomatonSUL
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import RandomWalkEqOracle
from aalpy.utils import generate_random_dfa, generate_random_mealy_machine, generate_random_moore_machine

num_states = 100
alphabet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

repeat = 15
num_increases = 50

states = ['num_states']
times_dfa = ['dfa_pypy_rs']
times_mealy = ['mealy_pypy_rs']
times_moore = ['moore_pypyrs']
total_dfa = ['dfa_total']
total_mealy = ['mealy_total']
total__moore = ['moore_total']

cex_processing = 'rs'
for i in range(num_increases):
    print(i)
    learning_time_dfa = []
    learning_time_mealy = []
    learning_time_moore = []

    total_time_dfa = []
    total_time_mealy = []
    total_time_moore = []

    states.append(num_states)

    for _ in range(repeat):
        dfa = generate_random_dfa(num_states, alphabet=alphabet, num_accepting_states=num_states // 2)
        sul = AutomatonSUL(dfa)

        # eq_oracle = StatePrefixEqOracle(alphabet, sul, walks_per_state=5, walk_len=40)
        eq_oracle = RandomWalkEqOracle(alphabet, sul, num_steps=9000, reset_prob=0.09)

        _, data = run_Lstar(alphabet, sul, eq_oracle, cex_processing=cex_processing, cache_and_non_det_check=False,
                            return_data=True, automaton_type='dfa')

        learning_time_dfa.append(data['learning_time'])
        total_time_dfa.append(data['total_time'])

        del sul
        del eq_oracle
        del dfa

        mealy = generate_random_mealy_machine(num_states, input_alphabet=alphabet, output_alphabet=alphabet)
        sul_mealy = AutomatonSUL(mealy)

        # eq_oracle = StatePrefixEqOracle(alphabet, sul_mealy, walks_per_state=5, walk_len=40)
        eq_oracle = RandomWalkEqOracle(alphabet, sul_mealy, num_steps=9000, reset_prob=0.09)

        _, data = run_Lstar(alphabet, sul_mealy, eq_oracle, cex_processing=cex_processing,
                            cache_and_non_det_check=False,
                            return_data=True, automaton_type='mealy')

        learning_time_mealy.append(data['learning_time'])
        total_time_mealy.append(data['total_time'])

        del mealy
        del sul_mealy
        del eq_oracle

        moore = generate_random_moore_machine(num_states, input_alphabet=alphabet, output_alphabet=alphabet)
        moore_sul = AutomatonSUL(moore)

        # eq_oracle = StatePrefixEqOracle(alphabet, moore_sul, walks_per_state=5, walk_len=40)
        eq_oracle = RandomWalkEqOracle(alphabet, moore_sul, num_steps=9000, reset_prob=0.09)

        _, data = run_Lstar(alphabet, moore_sul, eq_oracle, cex_processing=cex_processing,
                            cache_and_non_det_check=False,
                            return_data=True, automaton_type='moore')

        learning_time_moore.append(data['learning_time'])
        total_time_moore.append(data['total_time'])

    # save data and keep averages
    times_dfa.append(round(mean(learning_time_dfa), 4))
    times_mealy.append(round(mean(learning_time_mealy), 4))
    times_moore.append(round(mean(learning_time_moore), 4))

    total_dfa.append(round(mean(total_time_dfa), 4))
    total_mealy.append(round(mean(total_time_mealy), 4))
    total__moore.append(round(mean(total_time_moore), 4))

    num_states += 100

with open('increasing_size_experiments.csv', 'w') as f:
    wr = csv.writer(f, dialect='excel')
    wr.writerow(states)
    wr.writerow(times_dfa)
    wr.writerow(times_mealy)
    wr.writerow(times_moore)
    wr.writerow(total_dfa)
    wr.writerow(total_mealy)
    wr.writerow(times_moore)
