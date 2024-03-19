import unittest

from aalpy.utils import generate_random_markov_chain, load_automaton_from_file
from aalpy.utils.BenchmarkSULs import *


class TestFileHandler(unittest.TestCase):

    def test_saving_loading(self):
        try:
            type_model_pairs = [
                ("dfa", get_Angluin_dfa()),
                ("mealy", load_automaton_from_file('../DotModels/Angluin_Mealy.dot', automaton_type='mealy')),
                ("moore", load_automaton_from_file('../DotModels/Angluin_Moore.dot', automaton_type='moore')),
                ("onfsm", get_benchmark_ONFSM()),
                ("mdp", get_small_pomdp()),
                ("mdp", load_automaton_from_file('../DotModels/MDPs/first_grid.dot', automaton_type='mdp')),
                ("smm", get_faulty_coffee_machine_SMM()),
                ("mc", generate_random_markov_chain(num_states=10)),
            ]

            for type, model in type_model_pairs:
                model.save()
                print(model)
                loaded_model = load_automaton_from_file('LearnedModel.dot', type)
                loaded_model.save()
                loaded_model2 = load_automaton_from_file('LearnedModel.dot', type)

                if type != 'mc':
                    assert set(model.get_input_alphabet()) == set(loaded_model.get_input_alphabet())
                    assert set(model.get_input_alphabet()) == set(loaded_model2.get_input_alphabet())

                if type in {'dfa', 'moore', 'mealy'}:
                    assert model.compute_characterization_set() == loaded_model2.compute_characterization_set()

            assert True
        except:
            assert False
