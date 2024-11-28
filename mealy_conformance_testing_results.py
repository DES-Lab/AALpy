import pickle
import os
from pathlib import Path
import matplotlib.pyplot as plt

# the results folder
resultsdir = "results"

# the file names of learned models
root = os.getcwd() + "/DotModels"
protocols = ["MQTT", "TLS"]
directories = [Path(root + '/' + prot) for prot in protocols]
files = [file for dir in directories for file in dir.iterdir()]

# the equivalence oracles that were tested
orcs = [
        "StatePrefixEqOracleFalse",
        "StatePrefixEqOracleTrue",
        "KWayStateCoverageEqOracle",
        "TransitionFocusOracle",
        "CacheBasedEqOracle",
        "PacOracle",
        "RandomWMethodEqOracle"
        ]

# results dir for each oracle:
# resultsdir + '/' + orc + '/' + file.stem 

# the attributes inside the info.pkl dictionary
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
    print(f"No {resultsdir} directory. Run mealy_conformance_testing.py first.")

with open(resultsdir + '/results-pd.pkl', 'rb') as f:
    # this is a pandas dataframe
    results_df = pickle.load(f)

print(results_df)

for index, row in results_df.iterrows():
    truths = row.value_counts()[True]
    falses = len(row) - truths
    print(f"{index} learned correctly {truths} times and failed {falses} times")

for orc in orcs:
    sizes = []
    queries_learning = []
    queries_checking = []
    for file in files:
        infofile = resultsdir + '/' + orc + '/' + file.stem + '/info.pkl'
        with open(infofile, 'rb') as f:
            info = pickle.load(f)

        sizes.append(info['automaton_size'])
        queries_learning.append(info['queries_learning'])
        queries_checking.append(info['queries_eq_oracle'])

    # create a bar plot for the queries during learning
    # and another for the queries during checking
    fig, ax = plt.subplots()
    ax.bar([f.stem for f in files], queries_learning)
    ax.set_ylabel('Queries')
    ax.set_title(f'Queries during learning for {orc}')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    ax.bar([f.stem for f in files], queries_checking)
    ax.set_ylabel('Queries')
    ax.set_title(f'Queries during checking for {orc}')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


