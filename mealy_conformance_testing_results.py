import pickle
import os
import json
from pathlib import Path
import numpy as np
import math
import matplotlib.pyplot as plt
import sys

# read the results dir from first argument
if len(sys.argv) > 1:
    resultsdir = sys.argv[1]
    if not os.path.exists(resultsdir):
        print(f"No {resultsdir} directory. Run mealy_conformance_testing.py first.")
        exit(1)
else:
    print("Please provide the results directory as an argument.")
    exit(1)

# get the names of the directories in the results folder
files = [file for file in Path(resultsdir).iterdir() if file.is_dir()]

# the equivalence oracles that were tested
# iterate over a file in the results directory
# get the basename
orcs = [equiv_oracle.name for equiv_oracle in files[0].iterdir() if equiv_oracle.is_dir()]

# results dir for each oracle:
# resultsdir + '/' + file.stem + '/' + checking + '/' + learning

if not os.path.exists(resultsdir):
    print(f"No {resultsdir} directory. Run mealy_conformance_testing.py first.")

with open(resultsdir + '/results.pkl', 'rb') as f:
    # this is a pandas dataframe
    results = pickle.load(f)

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

# the results have the following structure:
# results[model][equivalence_oracle][learning_alg] = infos
# where infos is a dict of the attributes above

# for every equivalence oracle and learning algorithm,
# plot the number of equivalence and membership queries
# across all models

width = 0.4  # Width of a bar
xaxis = np.arange(len(files))
for orc in orcs:
    for learning in ["lstar", "kv"]:
        checking_queries = [results[model][orc][learning]['queries_eq_oracle'] for model in results]
        learning_queries = [results[model][orc][learning]['queries_learning'] for model in results]

        # plot the bars side by side
        plt.bar(xaxis,         checking_queries, width, label="Checking queries")
        plt.bar(xaxis + width, learning_queries, width, label="Learning queries")
        plt.xticks(xaxis + width / 2, [f.stem.split('_')[0] for f in files])
        plt.xlabel("Model")
        plt.ylabel("Queries")
        plt.title(f"Number of queries for {orc} and {learning} learning")
        plt.xticks(rotation=20)
        plt.legend()
        plt.savefig(f"{resultsdir}/{orc}_{learning}_queries.png")
        plt.clf()

for learning in ["lstar", "kv"]:
    plt.xticks(xaxis, [f.stem.split('_')[0] for f in files])
    plt.xlabel("Model")
    plt.ylabel("Queries")
    plt.xticks(rotation=20)

    index = 0
    for orc in orcs:
        checking_steps = [results[model][orc][learning]['steps_eq_oracle'] for model in results]
        plt.bar(xaxis + index * width / 3, checking_steps, width, label=orc)
        index += 1
    plt.title(f"Number of queries for {learning} learning")
    plt.legend()
    plt.savefig(f"{resultsdir}/{learning}_steps_eq_oracle.png")
    plt.clf()


