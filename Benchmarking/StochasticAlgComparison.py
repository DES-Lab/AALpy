import random
import time
from statistics import mean

import aalpy.paths

from aalpy.SULs import AutomatonSUL
from aalpy.learning_algs import run_stochastic_Lstar, run_Alergia
from aalpy.oracles.RandomWordEqOracle import RandomWordEqOracle
from aalpy.utils import load_automaton_from_file, get_properties_file, get_correct_prop_values
from aalpy.utils import model_check_experiment
from aalpy.automata.StochasticMealyMachine import smm_to_mdp_conversion

path_to_dir = '../DotModels/MDPs/'
files = ['first_grid.dot', 'second_grid.dot', 'slot_machine.dot', 'mqtt.dot', 'tcp.dot', 'bluetooth.dot']  #

prop_folder = 'prism_eval_props/'

aalpy.paths.path_to_prism = "C:/Program Files/prism-4.6/bin/prism.bat"
aalpy.paths.path_to_properties = "prism_eval_props/"

model_dict = {m.split('.')[0]: load_automaton_from_file(path_to_dir + m, automaton_type='mdp') for m in files}

for file in files:
    print(file)

    exp_name = file.split('.')[0]

    print('--------------------------------------------------')
    print('Experiment:', exp_name)

    original_mdp = model_dict[exp_name]
    input_alphabet = original_mdp.get_input_alphabet()

    mdp_sul = AutomatonSUL(original_mdp)

    eq_oracle = RandomWordEqOracle(input_alphabet, mdp_sul, num_walks=500, min_walk_len=5,
                                   max_walk_len=16, reset_after_cex=True)

    learned_classic_mdp, data_mdp = run_stochastic_Lstar(input_alphabet, mdp_sul, eq_oracle, automaton_type='mdp',
                                                         min_rounds=10, strategy='classic', n_c=20, n_resample=2000,
                                                         stopping_range_dict={},
                                                         max_rounds=200, return_data=True, target_unambiguity=0.98,
                                                         print_level=1)

    del mdp_sul
    del eq_oracle

    mdp_sul = AutomatonSUL(original_mdp)

    eq_oracle = RandomWordEqOracle(input_alphabet, mdp_sul, num_walks=150, min_walk_len=5,
                                   max_walk_len=15, reset_after_cex=True)

    learned_smm, data_smm = run_stochastic_Lstar(input_alphabet, mdp_sul, eq_oracle, automaton_type='smm',
                                                 min_rounds=10, strategy='normal',
                                                 max_rounds=200, return_data=True, target_unambiguity=0.98,
                                                 print_level=1)

    smm_2_mdp = smm_to_mdp_conversion(learned_smm)

    mdp_results, mdp_err = model_check_experiment(get_properties_file(exp_name),
                                                  get_correct_prop_values(exp_name), learned_classic_mdp)
    smm_results, smm_err = model_check_experiment(get_properties_file(exp_name),
                                                  get_correct_prop_values(exp_name), smm_2_mdp)

    num_alergia_samples = max([data_mdp["queries_learning"] + data_mdp["queries_eq_oracle"],
                               data_smm["queries_learning"] + data_smm["queries_eq_oracle"]])

    alergia_samples = []
    for _ in range(num_alergia_samples):
        sample = [mdp_sul.pre()]
        for _ in range(random.randint(10, 30)):
            action = random.choice(input_alphabet)
            output = mdp_sul.step(action)
            sample.append((action, output))
        alergia_samples.append(sample)

    alergia_model = run_Alergia(alergia_samples, automaton_type='mdp')

    alergia_results, alergia_error = model_check_experiment(get_properties_file(exp_name),
                                                            get_correct_prop_values(exp_name), alergia_model)

    print('Classic MDP learning', mean(mdp_err.values()), mdp_err)
    print('SMM learning', mean(smm_err.values()), smm_err)
    print('Alergia learning', mean(alergia_error.values()), alergia_error)

    print('Classic MDP traces', data_mdp["queries_learning"] + data_mdp["queries_eq_oracle"])
    print('SMM learning traces', data_smm["queries_learning"] + data_smm["queries_eq_oracle"])