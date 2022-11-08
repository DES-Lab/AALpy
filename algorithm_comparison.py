import json
import os
from functools import partial

import aalpy
from aalpy.SULs import DfaSUL
from aalpy.learning_algs.deterministic.LStar import run_Lstar
from aalpy.learning_algs.deterministic.KV import run_KV
from aalpy.oracles import RandomWalkEqOracle, WMethodEqOracle
from aalpy.utils import load_automaton_from_file

results = dict(random=dict(),
               wmethod=dict())

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
        oracle = RandomWalkEqOracle(alphabet, sul, 500, reset_after_cex=True)
    else:
        oracle = WMethodEqOracle(alphabet, sul, dfa.size)

    return sul, oracle

def main():
    folder_path = '/home/maxi/Downloads/DFA/principle/BenchmarkRandomDFAs/DFArandomChamparnaudParanthon_1000States_20Inputs'
    dir = os.listdir(folder_path)
    learned = 0
    to_learn = 2
    for filename in dir:
        print(f"loading {filename}... ", end='')
        dfa = load_automaton_from_file(os.path.join(folder_path, filename), 'dfa')
        print("done")

        print("running algorithms... ", end='')
        run_algorithm('kv', 'random', dfa)
        run_algorithm('kv_rs', 'random', dfa)
        run_algorithm('lstar', 'random', dfa)
        run_algorithm('lstar_rs', 'random', dfa)
        print('done')

        learned += 1
        print(f"Finished algorithm {learned}/{to_learn}")
        if learned >= to_learn:
            break

    with open('results.json', 'w+') as file:
        file.write(json.dumps(results, indent=4))



if __name__ == "__main__":
    main()