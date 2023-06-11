import random
import time
from statistics import mean

import aalpy.paths

from aalpy.SULs import MdpSUL
from aalpy.learning_algs import run_stochastic_Lstar, run_Alergia
from aalpy.oracles.RandomWordEqOracle import RandomWordEqOracle
from aalpy.utils import load_automaton_from_file, get_properties_file, get_correct_prop_values, model_check_properties
from aalpy.utils import model_check_experiment
from aalpy.automata.StochasticMealyMachine import smm_to_mdp_conversion

path_to_dir = '../DotModels/MDPs/'
files = ['slot_machine.dot', 'bluetooth.dot']  #
files.reverse()

prop_folder = 'prism_eval_props/'

aalpy.paths.path_to_prism = "C:/Program Files/prism-4.6/bin/prism.bat"
aalpy.paths.path_to_properties = "prism_eval_props/"

model_dict = {m.split('.')[0]: load_automaton_from_file(path_to_dir + m, automaton_type='mdp') for m in files}

model_type = ['mdp', 'smm']
model_type.reverse()

for file in files:
    for mt in model_type:
        exp_name = file.split('.')[0]

        print('--------------------------------------------------')
        print('Experiment:', exp_name, )

        original_mdp = model_dict[exp_name]
        input_alphabet = original_mdp.get_input_alphabet()

        mdp_sul = MdpSUL(original_mdp)

        eq_oracle = RandomWordEqOracle(input_alphabet, mdp_sul, num_walks=500, min_walk_len=5,
                                       max_walk_len=20, reset_after_cex=True)

        pbs = ((get_properties_file(exp_name),
                get_correct_prop_values(exp_name), 0.02))
        learned_classic_mdp, data_mdp = run_stochastic_Lstar(input_alphabet, mdp_sul, eq_oracle, automaton_type=mt,
                                                             min_rounds=10,
                                                             property_based_stopping=pbs,
                                                             return_data=True, target_unambiguity=1.1,
                                                             print_level=2)
