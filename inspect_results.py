import numpy as np
import pandas as pd
import os
import pathlib
import matplotlib.pyplot as plt
from aalpy.utils.FileHandler import load_automaton_from_file

# print up to 3 decimal point
np.set_printoptions(precision=3)
# do not print in scientific notation
np.set_printoptions(suppress=True)
# print up to 3 decimal point
pd.options.display.float_format = '{:.3f}'.format

TEACHER = "RandomWMethodEqOracle"
# TEACHER = "PerfectKnowledgeEqOracle"
ROOT = os.getcwd() + "/DotModels"
# PROTOCOLS = ["ASML", "TLS", "MQTT", "EMV", "TCP"]
# TLS is too small to learn
PROTOCOLS = ["TCP", "MQTT"]
DIRS = [pathlib.Path(ROOT + '/' + prot) for prot in PROTOCOLS]
FILES = [file for dir in DIRS for file in dir.iterdir()]
ORACLES = { 
           'Random'               : 0,
           'InterleavedRandom'    : 1,
           'InterleavedNewFirst'  : 2,
           'InterleavedOldFirst'  : 3,
           'StochasticLinear'     : 4,
           'StochasticSquare'     : 5,
           'StochasticExponential': 6
           }

# for every model, read the measurements npy file
# and print the minimum and maximum values
for file in FILES:
    NAME = file.stem
    PROT = file.parent.name
    RESULT_DIR = f"{TEACHER}/{PROT}/{NAME}"
    measurements = np.load(f"{RESULT_DIR}/measurements.npy")
    # measurements are of shape (num_experiements, num_oracles, num_hypotheses)
    xaxis = np.linspace(0, measurements.shape[0], measurements.shape[0])
    for oracle, index in ORACLES.items():
        if os.path.exists(f"{RESULT_DIR}/{oracle}_queries.png"):
            continue
        # mark data points with red dots
        numbers = measurements[:, index, -1]
        plt.plot(xaxis, numbers)
        plt.plot(xaxis, numbers, 'ro')
        # print arithmetic mean, median and geometric mean
        # as horizontal lines
        plt.axhline(np.mean(numbers), color='r', linestyle='--', label='Mean')
        plt.axhline(np.median(numbers), color='g', linestyle='--', label='Median')
        plt.axhline(np.exp(np.mean(np.log(numbers))), color='b', linestyle='--', label='Geometric Mean')
        plt.xlabel('Experiment')
        plt.ylabel(f'{oracle} Number of Queries')
        plt.title(f"Number of Queries for {NAME}")
        plt.legend()
        plt.savefig(f"{RESULT_DIR}/{oracle}_queries.png")
        plt.clf()

mean_steps_to_learning = np.zeros((len(ORACLES), len(FILES)))
for i, file in enumerate(FILES):
    NAME = file.stem
    PROT = file.parent.name
    RESULT_DIR = f"{TEACHER}/{PROT}/{NAME}"
    measurements = np.load(f"{RESULT_DIR}/measurements.npy")
    for oracle, j in ORACLES.items():
        mean_steps_to_learning[j, i] = np.mean(np.sum(measurements[:, j, :], axis=1))

max_steps_per_model = np.zeros(len(FILES))
for i, file in enumerate(FILES):
    max_steps_per_model[i] = np.max(mean_steps_to_learning[:, i])

s1_scores = {oracle: 
                np.sum(mean_steps_to_learning[i, :])
             for oracle, i in ORACLES.items()}
s2_scores = {oracle: 
                np.sum(mean_steps_to_learning[i, :] / max_steps_per_model) 
             for oracle, i in ORACLES.items()}

# print the scores as panda dataframes and save them to csv files
print("S1 Scores")
df = pd.DataFrame.from_dict(s1_scores, orient='index', columns=['Score'])
print(df)
df.to_csv(f"{TEACHER}/s1_scores.csv")

print("S2 Scores")
df = pd.DataFrame.from_dict(s2_scores, orient='index', columns=['Score'])
print(df)
df.to_csv(f"{TEACHER}/s2_scores.csv")

# the measurements have already been read into the numpy array
# in the order that they appear in the file structure
# therefore, just figure out the span of each protocol
PROT_TO_SPAN = {}
previous = 0
for dir in DIRS:
    prot = dir.name
    files = [file for file in dir.iterdir()]
    PROT_TO_SPAN[prot] = (previous, previous + len(files) - 1)
    previous += len(files)

# now calculate the more detailed scores for each protocol
for prot, span in PROT_TO_SPAN.items():
    s1_scores = {oracle: 
                    np.sum(mean_steps_to_learning[i, span[0]:span[1]])
                 for oracle, i in ORACLES.items()}
    s2_scores = {oracle: 
                    np.sum(mean_steps_to_learning[i, span[0]:span[1]] / max_steps_per_model[span[0]:span[1]]) 
                 for oracle, i in ORACLES.items()}

    print(f"S1 Scores for {prot}")
    df = pd.DataFrame.from_dict(s1_scores, orient='index', columns=['Score'])
    print(df)
    df.to_csv(f"{TEACHER}/{prot}/s1_scores.csv")

    print(f"S2 Scores for {prot}")
    df = pd.DataFrame.from_dict(s2_scores, orient='index', columns=['Score'])
    print(df)
    df.to_csv(f"{TEACHER}/{prot}/s2_scores.csv")



