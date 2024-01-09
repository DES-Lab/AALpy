import random
import os

import aalpy.paths

from aalpy.SULs import AutomatonSUL
from aalpy.learning_algs import run_stochastic_Lstar
from aalpy.oracles.RandomWordEqOracle import UnseenOutputRandomWordEqOracle
from aalpy.utils import load_automaton_from_file, get_properties_file, get_correct_prop_values
from aalpy.utils import model_check_experiment
from aalpy.automata.StochasticMealyMachine import smm_to_mdp_conversion

seeds = [1212,4557,19059,468,43,654,235345,6546,76768,4563,543526,777676,5555,776767,87878787,98989,60967553,3866677,1555841,8638]

path_to_dir = '../DotModels/MDPs/'
files = ['first_grid.dot', 'second_grid.dot'] # 'slot_machine.dot' ,'shared_coin.dot'  'mqtt.dot', 'tcp.dot'

prop_folder = 'prism_eval_props/'

# TODO Change the path to your PRIMS executable and change the path_to_prism in the stop_based_on_confidence method in ModelChecking.py.
prism_executable = "/home/mtappler/Programs/prism-4.4-linux64/bin/prism"

aalpy.paths.path_to_prism =      "C:/Program Files/prism-4.6/bin/prism.bat"
aalpy.paths.path_to_properties = "prism_eval_props/"

n_c = 20
n_resample = 1000
min_rounds = 10
max_rounds = 300
experiment_repetition = 5

uniform_parameters = False
strategy = ["normal"] # chi_square
cex_sampling = [None] # random:100:0.15
cex_processing = [None] # add a single prefix

for strat in strategy:
    for cex_stat in cex_sampling:
        for cex_proc in cex_processing:
            print(strat, cex_stat, cex_proc)
            benchmark_dir = f'FM_mdp_smm_error_based_stop/benchmark_{strat}_{cex_stat}_{cex_proc}/'
            if not os.path.exists(benchmark_dir):
                os.makedirs(benchmark_dir)
            for seed in range(experiment_repetition):
                print(seed)
                random.seed(seeds[seed])

                text_file = open(f"{benchmark_dir}/exp_{seed}.csv", "w")

                for file in files:
                    print(file)

                    exp_name = file.split('.')[0]
                    if uniform_parameters:
                        if exp_name == 'first_grid':
                            n_c, n_resample = n_c, n_resample
                        elif exp_name == 'second_grid':
                            n_c, n_resample = n_c, n_resample
                        elif exp_name == 'shared_coin':
                            n_c, n_resample = n_c, n_resample
                        elif exp_name == 'slot_machine':
                            n_c, n_resample = n_c, n_resample
                        elif exp_name == 'mqtt':
                            n_c, n_resample = n_c, n_resample
                        elif exp_name == 'tcp':
                            n_c, n_resample = n_c, n_resample
                    else:
                        if exp_name == 'first_grid':
                            n_c, n_resample = 20, 1000
                        elif exp_name == 'second_grid':
                            n_c, n_resample = 20, 2000
                        elif exp_name == 'shared_coin':
                            n_c, n_resample = 25, 2500
                        elif exp_name == 'slot_machine':
                            n_c, n_resample = 30, 5000
                        elif exp_name == 'mqtt':
                            n_c, n_resample = 20, 1000
                        elif exp_name == 'tcp':
                            n_c, n_resample = 20, 1000

                    stopping_data = (get_properties_file(exp_name), get_correct_prop_values(exp_name), 0.02)

                    original_mdp = load_automaton_from_file(path_to_dir + file, automaton_type='mdp')
                    input_alphabet = original_mdp.get_input_alphabet()

                    mdp_sul = AutomatonSUL(original_mdp)

                    eq_oracle = UnseenOutputRandomWordEqOracle(input_alphabet, mdp_sul, num_walks=150, min_walk_len=5,
                                                               max_walk_len=15, reset_after_cex=True)

                    learned_mdp, data_mdp = run_stochastic_Lstar(input_alphabet, mdp_sul, eq_oracle, automaton_type='mdp',
                                                                 n_c=n_c, n_resample=n_resample, min_rounds=min_rounds, strategy=strat,
                                                                 max_rounds=max_rounds, return_data=True, samples_cex_strategy=cex_stat,
                                                                 print_level=1, cex_processing=cex_proc, property_based_stopping=stopping_data)

                    del mdp_sul
                    del eq_oracle
                    random.seed(seeds[seed])
                    mdp_sul = AutomatonSUL(original_mdp)

                    eq_oracle = UnseenOutputRandomWordEqOracle(input_alphabet, mdp_sul, num_walks=150, min_walk_len=5,
                                                               max_walk_len=15, reset_after_cex=True)

                    learned_smm, data_smm = run_stochastic_Lstar(input_alphabet, mdp_sul, eq_oracle, automaton_type='smm',
                                                                 n_c=n_c, n_resample=n_resample, min_rounds=min_rounds, strategy=strat,
                                                                 max_rounds=max_rounds, return_data=True, samples_cex_strategy=cex_stat,
                                                                 print_level=1, cex_processing=cex_proc, property_based_stopping=stopping_data)

                    smm_2_mdp = smm_to_mdp_conversion(learned_smm)

                    mdp_results, mdp_err = model_check_experiment(get_properties_file(exp_name),
                                                                  get_correct_prop_values(exp_name), learned_mdp)
                    smm_results, smm_err = model_check_experiment(get_properties_file(exp_name),
                                                                  get_correct_prop_values(exp_name), smm_2_mdp)

                    properties_string_header = ",".join([f'{key}_val,{key}_err' for key in mdp_results.keys()])

                    property_string_mdp = ",".join([f'{str(mdp_results[p])},{str(mdp_err[p])}' for p in mdp_results.keys()])
                    property_string_smm = ",".join([f'{str(smm_results[p])},{str(smm_err[p])}' for p in smm_results.keys()])

                    text_file.write('Exp_Name, n_c, n_resample, Final Hypothesis Size, Learning time,'
                                    'Eq. Query Time, Learning Rounds, #MQ Learning, # Steps Learning,'
                                    f'# MQ Eq.Queries, # Steps Eq.Queries , {properties_string_header}\n')

                    text_file.write(f'learned_mdp_{exp_name},{n_c},{n_resample}, {data_mdp["automaton_size"]}, '
                                    f'{data_mdp["learning_time"]}, {data_mdp["eq_oracle_time"]}, '
                                    f'{data_mdp["learning_rounds"]}, {data_mdp["queries_learning"]}, {data_mdp["steps_learning"]},'
                                    f'{data_mdp["queries_eq_oracle"]}, {data_mdp["steps_eq_oracle"]},'
                                    f'{property_string_mdp}\n')

                    text_file.write(f'learned_smm_{exp_name},{n_c},{n_resample}, {data_smm["automaton_size"]}, '
                                    f'{data_smm["learning_time"]}, {data_smm["eq_oracle_time"]}, '
                                    f'{data_smm["learning_rounds"]}, {data_smm["queries_learning"]}, {data_smm["steps_learning"]},'
                                    f'{data_smm["queries_eq_oracle"]}, {data_smm["steps_eq_oracle"]},'
                                    f'{property_string_smm}\n')

                    text_file.flush()

                text_file.close()
