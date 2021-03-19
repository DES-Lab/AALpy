import random

from aalpy.learning_algs import run_stochastic_Lstar
from aalpy.oracles.RandomWalkEqOracle import UnseenOutputRandomWalkEqOracle
from aalpy.SULs import MdpSUL
from aalpy.utils import load_automaton_from_file
from aalpy.utils.FileHandler import mdp_2_prism_format
from aalpy.utils.HelperFunctions import smm_to_mdp_conversion

path_to_dir = '../DotModels/MDPs/'
files = ['first_grid.dot', 'second_grid.dot',
         'shared_coin.dot',
         'slot_machine.dot']

n_c = 20
n_resample = 1000
min_rounds = 10
max_rounds = 500

random.seed(123)
text_file = open("StochasticExperiments_no_cq.csv", "w")

text_file.write('Exp_Name, n_c, n_resample, Final Hypothesis Size, Learning time, '
                'Eq. Query Time, Learning Rounds, Learning # MQ, Learning # Steps'
                'Eq.Oracle # MQ, Eq.Oracle # Steps\n')
for file in files:
    exp_name = file.split('.')[0]

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

    mdp_2_prism_format(original_mdp, name=exp_name, output_path=f'original_{exp_name}.prism')

    mdp_sul = MdpSUL(original_mdp)

    eq_oracle = UnseenOutputRandomWalkEqOracle(input_alphabet, mdp_sul, num_steps=9000,
                                               reset_after_cex=True, reset_prob=0.8)

    learned_mdp, data_mdp = run_stochastic_Lstar(input_alphabet, mdp_sul, eq_oracle, automaton_type='mdp',
                                                 n_c=n_c, n_resample=n_resample, min_rounds=min_rounds,
                                                 max_rounds=max_rounds, return_data=True)

    #mdp_2_prism_format(learned_mdp, f'learned_mdp_{exp_name}', output_path=f'learned_mdp_{exp_name}.prism')

    # to ensure that everything is reseted
    mdp_sul = MdpSUL(original_mdp)

    learned_smm, data_smm = run_stochastic_Lstar(input_alphabet, mdp_sul, eq_oracle, automaton_type='smm',
                                                 n_c=n_c, n_resample=n_resample, min_rounds=min_rounds,
                                                 max_rounds=max_rounds, return_data=True)

    mdp_from_smm = smm_to_mdp_conversion(learned_smm)
    #mdp_2_prism_format(mdp_from_smm, f'learned_smm_{exp_name}', output_path=f'learned_smm_{exp_name}.prism')

    text_file.write(f'learned_mdp_{exp_name},{n_c},{n_resample}, {data_mdp["automaton_size"]}, '
                    f'{data_mdp["learning_time"]}, {data_mdp["eq_oracle_time"]}, '
                    f'{data_mdp["learning_rounds"]}, {data_mdp["queries_learning"]}, {data_mdp["steps_learning"]},'
                    f'{data_mdp["queries_eq_oracle"]}, {data_mdp["steps_eq_oracle"]}\n')

    text_file.write(f'learned_smm_{exp_name},{n_c},{n_resample}, {data_smm["automaton_size"]}, '
                    f'{data_smm["learning_time"]}, {data_smm["eq_oracle_time"]}, '
                    f'{data_smm["learning_rounds"]}, {data_smm["queries_learning"]}, {data_smm["steps_learning"]},'
                    f'{data_smm["queries_eq_oracle"]}, {data_smm["steps_eq_oracle"]}\n')

    # paths to prism files
    original_mdp_prism = f'original_{exp_name}.prism'
    learned_mdp_prism = f'learned_mdp_{exp_name}.prism'
    learned_smm_prims = f'learned_smm_{exp_name}.prism'

    if exp_name == 'first_grid':
        pass
    elif exp_name == 'second_grid':
        pass
    elif exp_name == 'shared_coin':
        pass
    elif exp_name == 'slot_machine':
        pass

text_file.close()

