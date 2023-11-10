import unittest

import aalpy.paths
from aalpy.SULs import AutomatonSUL
from aalpy.learning_algs import run_stochastic_Lstar
from aalpy.oracles import RandomWalkEqOracle
from aalpy.utils import load_automaton_from_file


class StochasticTest(unittest.TestCase):

    def test_learning_based_on_accuracy_based_stopping(self):

        example = 'first_grid'
        mdp = load_automaton_from_file(f'../DotModels/MDPs/{example}.dot', automaton_type='mdp')

        min_rounds = 10
        max_rounds = 500

        from aalpy.automata import StochasticMealyMachine
        from aalpy.utils import model_check_experiment, get_properties_file, \
            get_correct_prop_values
        from aalpy.automata.StochasticMealyMachine import smm_to_mdp_conversion

        aalpy.paths.path_to_prism = "C:/Program Files/prism-4.6/bin/prism.bat"
        aalpy.paths.path_to_properties = "../Benchmarking/prism_eval_props/"

        stopping_based_on_prop = (get_properties_file(example), get_correct_prop_values(example), 0.02)

        input_alphabet = mdp.get_input_alphabet()

        automaton_type = ['mdp', 'smm']
        similarity_strategy = ['classic', 'normal', 'chi2']
        cex_processing = [None, 'longest_prefix']
        samples_cex_strategy = [None, 'bfs', 'random:200:0.3']

        for aut_type in automaton_type:
            for strategy in similarity_strategy:
                for cex in cex_processing:
                    for sample_cex in samples_cex_strategy:

                        sul = AutomatonSUL(mdp)

                        eq_oracle = RandomWalkEqOracle(input_alphabet, sul=sul, num_steps=200,
                                                                   reset_prob=0.25,
                                                                   reset_after_cex=True)

                        learned_model = run_stochastic_Lstar(input_alphabet=input_alphabet, eq_oracle=eq_oracle,
                                                             sul=sul, n_c=20,
                                                             n_resample=1000, min_rounds=min_rounds,
                                                             max_rounds=max_rounds,
                                                             automaton_type=aut_type, strategy=strategy,
                                                             cex_processing=cex,
                                                             samples_cex_strategy=sample_cex, target_unambiguity=0.99,
                                                             property_based_stopping=stopping_based_on_prop,
                                                             print_level=0)

                        if isinstance(learned_model, StochasticMealyMachine):
                            mdp = smm_to_mdp_conversion(learned_model)
                        else:
                            mdp = learned_model

                        results, diff = model_check_experiment(get_properties_file(example),
                                                               get_correct_prop_values(example), mdp)

                        for d in diff.values():
                            if d > stopping_based_on_prop[2]:
                                assert False

        assert True
