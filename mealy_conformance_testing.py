import os
import pandas as pd
import pickle
import random
import json
from pathlib import Path
from aalpy import oracles
from aalpy.SULs.AutomataSUL import AutomatonSUL
from aalpy.learning_algs.deterministic.LStar import run_Lstar
from aalpy.learning_algs.deterministic.KV import run_KV
from aalpy.utils.FileHandler import load_automaton_from_file

resultsdir = "results"

attributes = ['learning_rounds',
              'automaton_size',
              'queries_learning',
              'steps_learning',
              'queries_eq_oracle',
              'steps_eq_oracle',
              'learning_time',
              'eq_oracle_time',
              'total_time',
              'characterization_set',
              'intermediate_hypotheses']

if os.path.exists(resultsdir):
    with open(resultsdir + '/results.pkl', 'rb') as f:
        results = pickle.load(f)
    # do not print the characterisation set
    print(json.dumps(results, indent=4))
    exit()

class StatePrefixEqOracleTrue(oracles.StatePrefixEqOracle):
    def __init__(self, alphabet, sul):
        super().__init__(alphabet, sul, walks_per_state=200, walk_len=50, depth_first=True)

class StatePrefixEqOracleFalse(oracles.StatePrefixEqOracle):
    def __init__(self, alphabet, sul):
        super().__init__(alphabet, sul, walks_per_state=200, walk_len=50, depth_first=False)

os.makedirs(resultsdir)

root = os.getcwd() + "/DotModels"
# for mqtt and tls, walks_per_state=200, walk_len=50
# for tcp, a whole lot more
# protocols = ["MQTT", "TLS", "TCP"]
protocols = ["MQTT", "TLS"]
true = {key : 0 for key in protocols}
total = {key : 0 for key in protocols}
directories = [Path(root + '/' + prot) for prot in protocols]
files = [file for dir in directories for file in dir.iterdir()]
models = [load_automaton_from_file(f, 'mealy') for f in files]

eqos = [
        # "KWayStateCoverageEqOracle",
        # "CacheBasedEqOracle",
        # "TransitionFocusOracle",
        "RandomWMethodEqOracle",
        "StatePrefixEqOracleFalse",
        "StatePrefixEqOracleTrue",
        ]

# dictionary for experiment results
results = {}
for file, model in zip(files, models):
    alphabet = model.get_input_alphabet()
    sul = AutomatonSUL(model)
    correct_size = model.size
    
    # initialize empty dict for specific model
    results[file.stem] = {}
    for index, eqo in enumerate(eqos):
        # equivalence checking algorithm name
        checking = eqo
        # initialize empty dictionary
        results[file.stem][checking] = {}
        for learning in ["lstar", "kv"]:
            total[file.parent.name] += 1
            orcs = [
                    # oracles.KWayStateCoverageEqOracle(alphabet, sul, random_walk_len=100, k=4, method='permutations'),
                    # oracles.CacheBasedEqOracle(alphabet, sul, num_walks=10000),
                    # oracles.TransitionFocusOracle(alphabet, sul, num_random_walks=10000, walk_len=100),
                    oracles.RandomWMethodEqOracle(alphabet, sul, walks_per_state=200, walk_len=50),
                    StatePrefixEqOracleFalse(alphabet, sul),
                    StatePrefixEqOracleTrue(alphabet, sul),
                    ]
            orc = orcs[index]
            assert orc.__class__.__name__ == eqo

            if learning == "lstar":
                model, info = run_Lstar(alphabet, sul, orc, 'mealy', print_level=0, return_data=True)
            else:
                model, info = run_KV(alphabet, sul, orc, 'mealy', print_level=0, return_data=True)

            # directory follows structure of dict
            infodir = resultsdir + '/' + file.stem + '/' + checking + '/' + learning
            if not os.path.exists(infodir):
                os.makedirs(infodir)

            # save intermediate hypotheses for inspection
            intermediate = info['intermediate_hypotheses']
            for number, hyp in enumerate(intermediate):
                hyp.save(file_path=(infodir+ f"/h{number}.dot"))

            # calculate diffs between hypotheses for inspection
            for i in range(1, len(intermediate)):
                os.system(f"diff {infodir}/h{i - 1}.dot {infodir}/h{i}.dot > {infodir}/diff_{i - 1}_{i}.txt")

            # save experiment results expect for intermediate hypotheses
            results[file.stem][checking][learning] = { k : v for k , v in info.items() 
                                                        if not k == "intermediate_hypotheses" }

            if not info['automaton_size'] == correct_size:
                print(f"{file.stem} {checking} {learning} => Wrong size {info['automaton_size']} / {correct_size}")
            else:
                true[file.parent.name] += 1

for key, value in total.items():
    print(f"{key} : {true[key]} / {value}")

with open(resultsdir + '/results.pkl', 'wb') as f:
    pickle.dump(results, f)

