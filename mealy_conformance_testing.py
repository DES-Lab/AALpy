import os
import pandas as pd
import pickle
import random
from pathlib import Path
from aalpy import oracles
from aalpy.SULs.AutomataSUL import AutomatonSUL
from aalpy.learning_algs.deterministic.LStar import run_Lstar
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

if not os.path.exists(resultsdir):

    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)

    root = os.getcwd() + "/DotModels"
    protocols = ["MQTT", "TLS"]
    directories = [Path(root + '/' + prot) for prot in protocols]
    files = [file for dir in directories for file in dir.iterdir()]
    models = [load_automaton_from_file(f, 'mealy') for f in files]

    assert len(files) == len(models)


    class StatePrefixEqOracleTrue(oracles.StatePrefixEqOracle):
        def __init__(self, alphabet, sul):
            super().__init__(alphabet, sul, walks_per_state=50, depth_first=True)

    class StatePrefixEqOracleFalse(oracles.StatePrefixEqOracle):
        def __init__(self, alphabet, sul):
            super().__init__(alphabet, sul, walks_per_state=50, depth_first=False)

    results = {} 
    for file, model in zip(files, models):
        alphabet = model.get_input_alphabet()
        sul = AutomatonSUL(model)

        oracle_status = []

        # deterministic
        orcs = [
                StatePrefixEqOracleFalse(alphabet, sul),
                StatePrefixEqOracleTrue(alphabet, sul),
                oracles.KWayStateCoverageEqOracle(alphabet, sul, random_walk_len=40),
                oracles.TransitionFocusOracle(alphabet, sul, walk_len=40),
                oracles.CacheBasedEqOracle(alphabet, sul),
                oracles.PacOracle(alphabet, sul),
                oracles.RandomWMethodEqOracle(alphabet, sul, walks_per_state=50, walk_len=40)
                ]

        correct = model.size
        states = []
        for eqo in orcs:

            l_star_model, info = run_Lstar(alphabet, sul, eqo, 'mealy', print_level=0, return_data=True)
            name = eqo.__class__.__name__
            infodir = resultsdir + '/' + name + '/' + file.stem
            if not os.path.exists(infodir):
                os.makedirs(infodir)

            infofile = resultsdir + '/' + name + '/' + file.stem + '/info.pkl'
            with open(infofile, 'wb') as f:
                pickle.dump(info, f)
                
            intermediate = info['intermediate_hypotheses']
            for number, hyp in enumerate(intermediate):
                hyp.save(file_path=(infodir+ f"/h{number}.dot"))

            # also save the diffs of consecutive hypotheses
            # this can be done by executing diff on the command line
            # diff hyp1.dot hyp2.dot > diff1_2.txt
            for i in range(1, len(intermediate)):
                os.system(f"diff {infodir}/h{i - 1}.dot {infodir}/h{i}.dot > {infodir}/diff_{i - 1}_{i}.txt")
            
            size = info['automaton_size']
            states.append(size)
            oracle_status.append((name, size == correct))

        results[file.stem] = oracle_status

    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)
else:
    with open('results.pkl', 'rb') as f:
        results = pickle.load(f)

randkey = random.choice(list(results.keys()))
index = [ first for first, _ in results[randkey] ]
new_results= { key: [second for _, second in value] for key, value in results.items() } 
df = pd.DataFrame(new_results, index=index)
print(df)
df.to_pickle("results-pd.pkl")

for index, row in df.iterrows():
    truths = row.value_counts()[True]
    falses = len(row) - truths
    print(f"{index} learned correctly {truths} times and failed {falses} times")
