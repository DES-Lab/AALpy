import random
import time

import aalpy.paths

from aalpy.SULs import AutomatonSUL
from aalpy.learning_algs import run_stochastic_Lstar
from aalpy.oracles.RandomWordEqOracle import RandomWordEqOracle
from aalpy.utils import load_automaton_from_file, get_properties_file, get_correct_prop_values
from aalpy.utils import model_check_experiment
from aalpy.automata.StochasticMealyMachine import smm_to_mdp_conversion

seeds = [29334,1554,9430459,92344168,55451679,569315,7776892,3875261,811,51,766603,778438967,9819877,6755560,52903,5257,4635,358,1441,838]

path_to_dir = '../DotModels/MDPs/'
# files = ['first_grid.dot', 'second_grid.dot', 'slot_machine.dot', 'mqtt.dot', 'tcp.dot'] # 'slot_machine.dot' ,
# 'shared_coin.dot'
files = ['second_grid.dot', 'mqtt.dot'] # 'slot_machine.dot' ,'shared_coin.dot'

prop_folder = '../prism_eval_props/'

# TODO Change the path to your PRIMS executable

aalpy.paths.path_to_prism =      "C:/Program Files/prism-4.6/bin/prism.bat"
aalpy.paths.path_to_properties = "../prism_eval_props/"

n_c = 10
n_resample = 1000
min_rounds = 25
max_rounds = 500
experiment_repetition = 10

uniform_parameters = False
strategy = ["normal", "chi2"] # chi_square
cex_sampling = ['bfs',] # random:100:0.15
cex_processing = [None, 'longest_prefix'] # add a single prefix
start = time.time()

model_dict = {m.split('.')[0] : load_automaton_from_file(path_to_dir + m, automaton_type='mdp') for m in files}

for strat in strategy:
    for cex_stat in cex_sampling:
        for cex_proc in cex_processing:
            print(strat, cex_stat, cex_proc)
            benchmark_dir = f'FM_mdp_smm/benchmark_22_04_{strat}_{cex_proc}/'
            for seed in range(experiment_repetition):
                print(seed)
                random.seed(seeds[seed])
                import os

                if not os.path.exists(benchmark_dir):
                    os.makedirs(benchmark_dir)
                text_file = open(f"{benchmark_dir}/exp_{seed}.csv", "w")

                for file in files:
                    print(file)

                    exp_name = file.split('.')[0]

                    original_mdp = model_dict[exp_name]
                    input_alphabet = original_mdp.get_input_alphabet()

                    mdp_sul = AutomatonSUL(original_mdp)

                    eq_oracle = RandomWordEqOracle(input_alphabet, mdp_sul, num_walks=150, min_walk_len=5,
                                                               max_walk_len=16, reset_after_cex=True)

                    learned_mdp, data_mdp = run_stochastic_Lstar(input_alphabet, mdp_sul, eq_oracle, automaton_type='mdp',
                                                                 min_rounds=min_rounds, strategy=strat,
                                                                 max_rounds=max_rounds, return_data=True, samples_cex_strategy=cex_stat,
                                                                 print_level=1, cex_processing=cex_proc, target_unambiguity=0.99)

                    del mdp_sul
                    del eq_oracle
                    random.seed(seeds[seed])
                    mdp_sul = AutomatonSUL(original_mdp)

                    eq_oracle = RandomWordEqOracle(input_alphabet, mdp_sul, num_walks=150, min_walk_len=5,
                                                               max_walk_len=15, reset_after_cex=True)

                    learned_smm, data_smm = run_stochastic_Lstar(input_alphabet, mdp_sul, eq_oracle, automaton_type='smm',
                                                                 min_rounds=min_rounds, strategy=strat,
                                                                 max_rounds=max_rounds, return_data=True, samples_cex_strategy=cex_stat,
                                                                 print_level=1, cex_processing=cex_proc, target_unambiguity=0.99)

                    smm_2_mdp = smm_to_mdp_conversion(learned_smm)

                    mdp_results, mdp_err = model_check_experiment(get_properties_file(exp_name),
                                                                  get_correct_prop_values(exp_name), learned_mdp)
                    smm_results, smm_err = model_check_experiment(get_properties_file(exp_name),
                                                                  get_correct_prop_values(exp_name), smm_2_mdp)

                    properties_string_header = ",".join([f'{key}_val,{key}_err' for key in mdp_results.keys()])

                    property_string_mdp = ",".join([f'{str(mdp_results[p])},{str(mdp_err[p])}' for p in mdp_results.keys()])
                    property_string_smm = ",".join([f'{str(smm_results[p])},{str(smm_err[p])}' for p in smm_results.keys()])

                    text_file.write('Exp_Name,n_c,n_resample,Final Hypothesis Size,Learning time,'
                                    'Eq. Query Time,Learning Rounds,#MQ Learning,# Steps Learning,'
                                    f'# MQ Eq.Queries,# Steps Eq.Queries ,{properties_string_header}\n')

                    text_file.write(f'learned_mdp_{exp_name},{n_c},{n_resample},{data_mdp["automaton_size"]},'
                                    f'{data_mdp["learning_time"]},{data_mdp["eq_oracle_time"]},'
                                    f'{data_mdp["learning_rounds"]},{data_mdp["queries_learning"]},{data_mdp["steps_learning"]},'
                                    f'{data_mdp["queries_eq_oracle"]},{data_mdp["steps_eq_oracle"]},'
                                    f'{property_string_mdp}\n')

                    text_file.write(f'learned_smm_{exp_name},{n_c},{n_resample},{data_smm["automaton_size"]},'
                                    f'{data_smm["learning_time"]},{data_smm["eq_oracle_time"]}, '
                                    f'{data_smm["learning_rounds"]},{data_smm["queries_learning"]},{data_smm["steps_learning"]},'
                                    f'{data_smm["queries_eq_oracle"]},{data_smm["steps_eq_oracle"]},'
                                    f'{property_string_smm}\n')

                    text_file.flush()

                text_file.close()
print('Exp duration', time.time() - start)
