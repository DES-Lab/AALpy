import random

import aalpy.paths
from aalpy.SULs import AutomatonSUL
from aalpy.automata.StochasticMealyMachine import smm_to_mdp_conversion
from aalpy.learning_algs import run_Alergia
from aalpy.utils import load_automaton_from_file, get_correct_prop_values, get_properties_file
from aalpy.utils import model_check_experiment

path_to_dir = '../DotModels/MDPs/'
files = ['first_grid.dot', 'second_grid.dot',
         'slot_machine.dot', 'mqtt.dot', 'tcp.dot']  # 'shared_coin.dot'

aalpy.paths.path_to_prism = "C:/Program Files/prism-4.7/bin/prism.bat"
aalpy.paths.path_to_properties = "prism_eval_props/"


def writeSamplesToFile(samples, path="alergiaSamples.txt"):
    isSMM = False
    if isinstance(samples[0][0], tuple):
        isSMM = True
    with open(path, 'a') as f:
        for sample in samples:
            s = "" if isSMM else f'{str(sample.pop(0))}'
            for i, o in sample:
                s += f',{i},{o}'
            f.write(s + '\n')

    f.close()
    # samples.clear()


def deleteSampleFile(path="alergiaSamples.txt"):
    import os
    if os.path.exists(path):
        os.remove(path)


num_traces = 100000

for file in ['first_grid.dot']:

    exp_name = file.split('.')[0]

    original_mdp = load_automaton_from_file(path_to_dir + file, automaton_type='mdp')
    input_alphabet = original_mdp.get_input_alphabet()

    mdp_sul = AutomatonSUL(original_mdp)

    for _ in range(1):

        data = []
        for _ in range(num_traces):
            sample = [mdp_sul.pre()]
            for _ in range(random.randint(10, 50)):
                i = random.choice(input_alphabet)
                o = mdp_sul.step(i)
                sample.append((i, o))
            data.append(sample)
            mdp_sul.post()

        learned_mdp = run_Alergia(data, automaton_type='mdp')

        for s in data:
            s.pop(0)

        learned_smm = run_Alergia(data, automaton_type='smm')

        smm_2_mdp = smm_to_mdp_conversion(learned_smm)

        mdp_results, mdp_err = model_check_experiment(get_properties_file(exp_name),
                                                      get_correct_prop_values(exp_name), learned_mdp)
        smm_results, smm_err = model_check_experiment(get_properties_file(exp_name),
                                                      get_correct_prop_values(exp_name), smm_2_mdp)

        print(learned_mdp.size, learned_smm.size, smm_2_mdp.size)
        print(f'-------{exp_name}---------')
        print(f'MDP Error:       {mdp_err}')
        print(f'SMM Error:       {smm_err}')
        smm_diff = {}
        for key, val in mdp_err.items():
            if key not in smm_err.keys() or smm_err[key] == 0:
                continue
            smm_diff[key] = round(smm_err[key] - val, 2)
        print(f'SMM differance: {smm_diff}')
