import pickle
import random
import time
from collections import defaultdict
from statistics import mean

import aalpy.paths

from aalpy.SULs import AutomatonSUL
from aalpy.learning_algs import run_stochastic_Lstar, run_Alergia
from aalpy.oracles.RandomWordEqOracle import RandomWordEqOracle
from aalpy.utils import load_automaton_from_file, get_properties_file, get_correct_prop_values, model_check_properties
from aalpy.utils import model_check_experiment
from aalpy.automata.StochasticMealyMachine import smm_to_mdp_conversion

path_to_dir = '../DotModels/MDPs/'
files = ['slot_machine.dot', 'bluetooth.dot']  #
files = ['first_grid.dot', 'second_grid.dot', 'tcp.dot', 'mqtt.dot', 'bluetooth.dot', 'slot_machine.dot']

prop_folder = 'prism_eval_props/'

aalpy.paths.path_to_prism = "C:/Program Files/prism-4.6/bin/prism.bat"
aalpy.paths.path_to_properties = "prism_eval_props/"

model_dict = {m.split('.')[0]: load_automaton_from_file(path_to_dir + m, automaton_type='mdp') for m in files}

model_type = ['smm']
cex_processing = [None, 'longest_prefix', 'rs']
# model_type.reverse()

res = defaultdict(list)

# for file in files:
#     for mt in model_type:
#         for cp in cex_processing:
#             for _ in range(4):
#
#                 exp_name = file.split('.')[0]
#
#                 print('--------------------------------------------------')
#                 print('Experiment:', exp_name, cp)
#
#                 original_mdp = model_dict[exp_name]
#                 input_alphabet = original_mdp.get_input_alphabet()
#
#                 mdp_sul = AutomatonSUL(original_mdp)
#
#                 eq_oracle = RandomWordEqOracle(input_alphabet, mdp_sul, num_walks=500, min_walk_len=5,
#                                                max_walk_len=15, reset_after_cex=True)
#
#                 pbs = ((get_properties_file(exp_name),
#                         get_correct_prop_values(exp_name), 0.02 if exp_name != 'bluetooth' else 0.03))
#                 learned_classic_mdp, data_mdp = run_stochastic_Lstar(input_alphabet, mdp_sul, eq_oracle, automaton_type=mt,
#                                                                      min_rounds=10,
#                                                                      #property_based_stopping=pbs,
#                                                                      cex_processing=cp,
#                                                                      samples_cex_strategy=None,
#                                                                      return_data=True, target_unambiguity=0.98,
#                                                                      print_level=1)
#
#                 res[exp_name].append((cp, data_mdp['queries_learning'] + data_mdp['queries_eq_oracle']))

# with open('cex_processing_res.pickle', 'wb') as handle:
#     pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('cex_processing_res.pickle', 'rb') as handle:
    res = pickle.load(handle)

for key, val in res.items():
    print(key)
    sorted_by_cp = defaultdict(list)
    for cp, data in val:
        sorted_by_cp[cp].append(data)

    for cp_method, data in sorted_by_cp.items():
        print(cp_method)
        print(mean(data), min(data), max(data))