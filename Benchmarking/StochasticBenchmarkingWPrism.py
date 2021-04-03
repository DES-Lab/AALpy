import csv
import io
import os
import random
import re
import subprocess

from aalpy.SULs import MdpSUL
from aalpy.utils import load_automaton_from_file
from aalpy.utils.HelperFunctions import smm_to_mdp_conversion
from aalpy.utils.FileHandler import mdp_2_prism_format
from aalpy.oracles.RandomWalkEqOracle import UnseenOutputRandomWalkEqOracle
from aalpy.learning_algs import run_stochastic_Lstar

prism_prob_output_regex = re.compile("Result: (.+?) \\(value in the initial state\\)")


def count_properties(properties_content):
    return len(list(filter(lambda line: len(line.strip()) > 0, properties_content)))


def properties_string(properties_sorted, property_data, property_data_original):
    property_string = ""
    for p in properties_sorted:
        distance = abs(property_data[p] - property_data_original[p])
        property_string += "," + "{:.4f}".format(property_data[p]) + "|{:.4f}".format(distance)
    return property_string


def eval_property(shell_name, prism_executable, prism_file_name, properties_file_name, property_index):
    proc = subprocess.Popen(
        [shell_name, prism_executable, prism_file_name, properties_file_name, "-prop", str(property_index)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
        if not line:
            break
        else:
            match = prism_prob_output_regex.match(line)
            if match:
                return float(match.group(1))
    return 0.0


def eval_properties(shell_name, prism_executable, prism_file_name, properties_file_name, data):
    with open(properties_file_name, 'r') as properties_file:
        properties_content = properties_file.readlines()
        nr_properties = count_properties(properties_content)
        for property_index in range(1, nr_properties + 1):
            probability = eval_property(shell_name, prism_executable, prism_file_name, properties_file_name,
                                        property_index)
            data[f"prob{property_index}"] = probability


path_to_dir = '../DotModels/MDPs/'
files = ['shared_coin.dot','first_grid.dot', 'second_grid.dot',
         'slot_machine.dot']

# TODO Change the path to your PRIMS executable
prism_executable = "/home/mtappler/Programs/prism-4.4-linux64/bin/prism"
shell_name = "bash"
property_dir = "prism_eval_props"

n_c = 20
n_resample = 1000
min_rounds = 10
max_rounds = 8000
do_check_properties = True
strategy = "chi-square"

seed = 12313412
for seed in range(1, 4):
    random.seed(seed)
    benchmark_dir = f"benchmark_complete_impl_chi2/benchmark_data_{seed}"
    import os

    if not os.path.exists(benchmark_dir):
        os.makedirs(benchmark_dir)
    text_file = open(f"{benchmark_dir}/StochasticExperiments.csv", "w")
    uniform_parameters = True

    for file in files:
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
        else:
            if exp_name == 'first_grid':
                n_c, n_resample = 20, 1000
            elif exp_name == 'second_grid':
                n_c, n_resample = 20, 1000
            elif exp_name == 'shared_coin':
                n_c, n_resample = 50, 5000
            elif exp_name == 'slot_machine':
                n_c, n_resample = 100, 10000

        original_mdp = load_automaton_from_file(path_to_dir + file, automaton_type='mdp')
        input_alphabet = original_mdp.get_input_alphabet()

        original_prism_file_name = f'{benchmark_dir}/original_{exp_name}.prism'
        mdp_2_prism_format(original_mdp, name=exp_name, output_path=f'{original_prism_file_name}')

        mdp_sul = MdpSUL(original_mdp)

        eq_oracle = UnseenOutputRandomWalkEqOracle(input_alphabet, mdp_sul, num_steps=n_resample * (1 / 0.25),
                                                   reset_after_cex=True, reset_prob=0.25)

        learned_mdp, data_mdp = run_stochastic_Lstar(input_alphabet, mdp_sul, eq_oracle, automaton_type='mdp',
                                                     n_c=n_c, n_resample=n_resample, min_rounds=min_rounds, strategy=strategy,
                                                     max_rounds=max_rounds, return_data=True, samples_cex_strategy="bfs")

        mdp_2_prism_format(learned_mdp, f'learned_mdp_{exp_name}',
                           output_path=f'{benchmark_dir}/learned_mdp_{exp_name}.prism')

        mdp_sul.num_steps = 0
        mdp_sul.num_queries = 0

        learned_smm, data_smm = run_stochastic_Lstar(input_alphabet, mdp_sul, eq_oracle, automaton_type='smm',
                                                     n_c=n_c, n_resample=n_resample, min_rounds=min_rounds, strategy=strategy,
                                                     max_rounds=max_rounds, return_data=True, samples_cex_strategy="bfs")

        mdp_from_smm = smm_to_mdp_conversion(learned_smm)
        mdp_2_prism_format(mdp_from_smm, f'learned_smm_{exp_name}',
                           output_path=f'{benchmark_dir}/learned_smm_{exp_name}.prism')

        # paths to prism files
        original_mdp_prism = f'{benchmark_dir}/original_{exp_name}.prism'
        learned_mdp_prism = f'{benchmark_dir}/learned_mdp_{exp_name}.prism'
        learned_smm_prism = f'{benchmark_dir}/learned_smm_{exp_name}.prism'

        if exp_name == 'first_grid':
            properties_file_name = f"{property_dir}/first_eval.props"
        elif exp_name == 'second_grid':
            properties_file_name = f"{property_dir}/second_eval.props"
        elif exp_name == 'shared_coin':
            properties_file_name = f"{property_dir}/shared_coin_eval.props"
        elif exp_name == 'slot_machine':
            properties_file_name = f"{property_dir}/slot_machine_eval.props"
        else:
            properties_file_name = None

        property_string_smm = ""
        property_string_mdp = ""
        properties_string_header = ""
        if do_check_properties and properties_file_name:
            property_data_mdp = dict()
            property_data_smm = dict()
            property_data_original = dict()
            eval_properties(shell_name, prism_executable, learned_mdp_prism, properties_file_name, property_data_mdp)
            eval_properties(shell_name, prism_executable, learned_smm_prism, properties_file_name, property_data_smm)
            eval_properties(shell_name, prism_executable, original_prism_file_name, properties_file_name,
                            property_data_original)
            properties_sorted = list(property_data_smm.keys())
            properties_sorted.sort()
            properties_string_header = "," + ",".join(properties_sorted)
            property_string_mdp = properties_string(properties_sorted, property_data_mdp, property_data_original)
            property_string_smm = properties_string(properties_sorted, property_data_smm, property_data_original)

        text_file.write('Exp_Name, n_c, n_resample, Final Hypothesis Size, Learning time,'
                        'Eq. Query Time, Learning Rounds, Learning # MQ, Learning # Steps'
                        f'Eq.Oracle # MQ, Eq.Oracle # Steps,Num Steps {properties_string_header}\n')

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
