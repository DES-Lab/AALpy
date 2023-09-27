import pickle

from aalpy.learning_algs import run_Alergia

with open('alergia_dev_by_zero.pickle', 'rb') as handle:
    data = pickle.load(handle)

learned_model = run_Alergia(data, automaton_type='mdp', print_info=True)