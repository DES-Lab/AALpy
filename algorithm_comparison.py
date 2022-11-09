import json
import os
import pickle
import sys
from functools import partial

from aalpy.SULs import DfaSUL
from aalpy.learning_algs.deterministic.LStar import run_Lstar
from aalpy.learning_algs.deterministic.KV import run_KV
from aalpy.oracles import RandomWalkEqOracle, WMethodEqOracle, StatePrefixEqOracle
from aalpy.utils import load_automaton_from_file
from kv_test import checkConformance


results = dict(random=dict(),
               wmethod=dict())

automata_data = {}

def run_algorithm(algorithm_name, oracle_name, dfa):
    sul, oracle = setup_oracle(dfa, oracle_name)
    alphabet = dfa.get_input_alphabet()

    if algorithm_name == 'kv':
        algorithm = partial(run_KV, alphabet, sul, oracle, automaton_type='dfa', return_data=True, print_level=0, reuse_counterexamples=True, cex_processing=None)
    elif algorithm_name == 'kv_rs':
        algorithm = partial(run_KV, alphabet, sul, oracle, automaton_type='dfa', return_data=True, print_level=0, reuse_counterexamples=True, cex_processing='rs')
    elif algorithm_name == 'lstar':
        algorithm = partial(run_Lstar, alphabet, sul, oracle, automaton_type='dfa', return_data=True, print_level=0, cex_processing=None)
    elif algorithm_name == 'lstar_rs':
        algorithm = partial(run_Lstar, alphabet, sul, oracle, automaton_type='dfa', return_data=True, print_level=0, cex_processing='rs')

    print(f'{algorithm_name} ({oracle_name})... ', end='')
    result, info = algorithm()
    correct_learning = checkConformance(dfa.get_input_alphabet(), result, dfa.size, sul)
    print(f'correctly learned model: {correct_learning} ')
    info.pop('classification_tree', None)
    info.pop('characterization set', None)
    if algorithm_name in results[oracle_name]:
        results[oracle_name][algorithm_name] = {k: results[oracle_name][algorithm_name][k] + info[k] for k in info}
    else:
        results[oracle_name][algorithm_name] = info

def setup_oracle(dfa, type):
    alphabet = dfa.get_input_alphabet()
    sul = DfaSUL(dfa)
    if type == 'random':
        oracle = RandomWalkEqOracle(alphabet, sul, 500, reset_after_cex=True,reset_prob=0.25)
        #oracle = StatePrefixEqOracle
    else:
        oracle = WMethodEqOracle(alphabet, sul, dfa.size)

    return sul, oracle

def main():
    sys.setrecursionlimit(10000)
    folder_path = '/home/maxi/Downloads/DFA/principle/BenchmarkRandomDFAs/DFArandomChamparnaudParanthon_1000States_20Inputs'
    #folder_path = '/Users/andrea/PhD/bachelor_thesis/maximilian_rindler/DFA/principle/BenchmarkRandomDFAs/DFArandomChamparnaudParanthon_1000States_20Inputs'
    dir = os.listdir(folder_path)
    with open('automata.pickle', "rb") as automata_file:
        automata_data = pickle.load(automata_file)
    learned = 0
    to_learn = 2
    for filename in dir:
        print(f"loading {filename}... ", end='')
        if automata_data and filename in automata_data:
            print("load from pickle", end='')
            dfa = automata_data[filename]
        else:
            print("load from file and save as pickle", end='')
            dfa = load_automaton_from_file(os.path.join(folder_path, filename), 'dfa')
            automata_data[filename] = dfa
            with open('automata.pickle', "wb") as automata_file:
                pickle.dump(automata_data, automata_file)

        print("running algorithms... ", end='')
        run_algorithm('kv', 'random', dfa)
        run_algorithm('kv_rs', 'random', dfa)
        run_algorithm('lstar', 'random', dfa)
        run_algorithm('lstar_rs', 'random', dfa)
        #print('done')

        learned += 1
        print(f"Finished algorithm {learned}/{to_learn}")
        if learned >= to_learn:
            break

    with open('results.json', 'w+') as file:
        file.write(json.dumps(results, indent=4))



if __name__ == "__main__":
    main()