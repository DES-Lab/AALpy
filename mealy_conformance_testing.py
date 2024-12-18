import os
import pathlib
import pandas as pd
import numpy as np
from aalpy.oracles import RandomWMethodEqOracle
from aalpy.oracles import PerfectKnowledgeEqOracle
from aalpy.oracles import StatePrefixEqOracle
from aalpy.oracles.SortedStateCoverageEqOracle import SortedStateCoverageEqOracle
from aalpy.oracles.InterleavedStateCoverageEqOracle import InterleavedStateCoverageEqOracle
from aalpy.oracles.StochasticStateCoverageEqOracle import StochasticStateCoverageEqOracle
from aalpy.SULs.AutomataSUL import AutomatonSUL
from aalpy.learning_algs.deterministic.LStar import run_Lstar
from aalpy.utils.FileHandler import load_automaton_from_file

# print up to 1 decimal point
np.set_printoptions(precision=1)
# do not print in scientific notation
np.set_printoptions(suppress=True)
# print up to 3 decimal point
pd.options.display.float_format = '{:.3f}'.format

# class NewFirst(SortedStateCoverageEqOracle):
#     def __init__(self, alphabet, sul, walks_per_state=4000, walk_len=150, mode='newest'):
#         super().__init__(alphabet, sul, walks_per_state, walk_len, mode)
#
# class OldFirst(SortedStateCoverageEqOracle):
#     def __init__(self, alphabet, sul, walks_per_state=4000, walk_len=150, mode='oldest'):
#         super().__init__(alphabet, sul, walks_per_state, walk_len, mode)

# class Random(SortedStateCoverageEqOracle):
#     def __init__(self, alphabet, sul, walks_per_round=200000, walks_per_state=4000, walk_len=150, mode='random'):
#         super().__init__(alphabet, sul, walks_per_round, walks_per_state, walk_len, mode)

class Random(StatePrefixEqOracle):
    def __init__(self, alphabet, sul, walks_per_round=200000, walks_per_state=4000, walk_len=150, depth_first=False):
        super().__init__(alphabet, sul, walks_per_round, walks_per_state, walk_len, depth_first)

class InterleavedRandom(InterleavedStateCoverageEqOracle):
    def __init__(self, alphabet, sul, walks_per_round=200000, walks_per_state=4000, walk_len=150, mode='random'):
        super().__init__(alphabet, sul, walks_per_round, walks_per_state, walk_len, mode)

class InterleavedNewFirst(InterleavedStateCoverageEqOracle):
    def __init__(self, alphabet, sul, walks_per_round=200000, walks_per_state=4000, walk_len=150, mode='newest'):
        super().__init__(alphabet, sul, walks_per_round, walks_per_state, walk_len, mode)

class InterleavedOldFirst(InterleavedStateCoverageEqOracle):
    def __init__(self, alphabet, sul, walks_per_round=200000,  walks_per_state=4000, walk_len=150, mode='oldest'):
        super().__init__(alphabet, sul, walks_per_round, walks_per_state, walk_len, mode)

class StochasticLinear(StochasticStateCoverageEqOracle):
    def __init__(self, alphabet, sul, walks_per_round=200000, walk_len=150, prob_function='linear'):
        super().__init__(alphabet, sul, walks_per_round, walk_len, prob_function)

class StochasticSquare(StochasticStateCoverageEqOracle):
    def __init__(self, alphabet, sul, walks_per_round=200000, walk_len=150, prob_function='square'):
        super().__init__(alphabet, sul, walks_per_round, walk_len, prob_function)

class StochasticExponential(StochasticStateCoverageEqOracle):
    def __init__(self, alphabet, sul, walks_per_round=200000, walk_len=150, prob_function='exponential'):
        super().__init__(alphabet, sul, walks_per_round, walk_len, prob_function)

TEACHER = "RandomWMethodEqOracle"
# TEACHER = "PerfectKnowledgeEqOracle"

if not os.path.exists(TEACHER):
    os.makedirs(TEACHER)

def learn_model(alphabet, sul, model, name):
    directory = f"{TEACHER}/{name}"
    if not os.path.exists(directory):
        os.makedirs(directory)

        if TEACHER == "PerfectKnowledgeEqOracle":
            oracle = PerfectKnowledgeEqOracle(alphabet, sul, model)
        elif TEACHER == "RandomWMethodEqOracle":
            oracle = RandomWMethodEqOracle(alphabet, sul, walks_per_state=10000, walk_len=150)
        else:
            raise ValueError("Unknown teacher")
        _, learning_info = run_Lstar(alphabet, sul, oracle, 'mealy', return_data=True, print_level=0)
        intermediate_hypotheses = learning_info['intermediate_hypotheses']
        for num, hyp in enumerate(intermediate_hypotheses):
            hyp.save(file_path=(f"{directory}/h{num}.dot"))
    else: # if hypotheses exist, load them
        intermediate_hypotheses = []
        dots = list(pathlib.Path(directory).glob('*.dot'))
        for num in range(len(dots)):
            intermediate_hypotheses.append(load_automaton_from_file(f'{directory}/h{num}.dot', 'mealy'))

    return intermediate_hypotheses

def test_oracles(oracles, hyps, name):
    queries = np.zeros((len(oracles), len(hyps)))
    failures = np.zeros((len(oracles), len(hyps)))
    # counterexamples are immutable, so we can't store them in a numpy array
    cexs = [[None for _ in range(len(hyps))] for _ in range(len(oracles))]
    for i, hyp in enumerate(hyps):
        for j, oracle in enumerate(oracles):
            cexs[j][i] = oracle.find_cex(hyp)
            queries[j, i] = oracle.num_queries
            failures[j][i] = 1 if cexs[j][i] is None else 0
    return queries, failures


ROOT = os.getcwd() + "/DotModels"
# PROTOCOLS = ["ASML", "TLS", "MQTT", "EMV", "TCP"]
PROTOCOLS = ["TCP"]
DIRS = [pathlib.Path(ROOT + '/' + prot) for prot in PROTOCOLS]
FILES = [file for dir in DIRS for file in dir.iterdir()]
MODELS = [load_automaton_from_file(f, 'mealy') for f in FILES]
MODELS.sort(key=lambda x: x.size)

# We will learn every model using the RandomWpEqOracle and we will retain the
# intermediate hypotheses of the learning experiment.
# Then, for every intermediate hypotheses, we will try to find a counterexample
# using the StatePrefixEqOracleTrue and StatePrefixEqOracleFalse.
# If counterexamples are found successfully, we will store them and keep doing
# this, either until the final hypothesis is reached or until an oracle fails.

TIMES = 20
NUM_ORACLES = 7
huge = []
tiny = []
for index, (model, file) in enumerate(zip(MODELS, FILES)):
    name = file.stem
    alphabet = model.get_input_alphabet()
    if model.size > 150 or len(alphabet) > 100:
        huge.append((name, model.size, len(alphabet)))
        continue
    sul = AutomatonSUL(model)

    # this has the side effect of saving the intermediate hypotheses
    # in the directory "stefanos_test/{name}"
    hypotheses = learn_model(alphabet, sul, model, name)
    # indicator that the model has been learned correctly
    if not hypotheses[-1].size == model.size:
        print(f"Model not learned successfully {hypotheses[-1].size} / {model.size}. Skipping ... ")
        continue

    if len(hypotheses) == 1:
        tiny.append((name, model.size, len(alphabet)))
        continue

    print(f"==================={file.stem}====================")
    print(f"Number of hypotheses: {len(hypotheses)} {[h.size for h in hypotheses]}")
    intermediate = hypotheses[:-1]
    NUM_HYPS = len(intermediate)

    # check if the measurements file exists and load it if it does
    flag = os.path.exists(f"{TEACHER}/{name}/measurements.npy") and os.path.exists(f"{TEACHER}/{name}/failures.csv")
    if flag:
        measurements = np.load(f"{TEACHER}/{name}/measurements.npy")
        failures = pd.read_csv(f"{TEACHER}/{name}/failures.csv", index_col=0)
        # the index of the dataframe holds the oracle names
        index = failures.index.values
        failures = failures.to_numpy()
    else:
        # We will now try to find counterexamples for every intermediate hypothesis
        # using both StatePrefixEqOracles.
        measurements = np.zeros((TIMES, NUM_ORACLES, NUM_HYPS))
        failures = np.zeros((NUM_ORACLES, NUM_HYPS))
        for trial in range(TIMES):
            oracle1 = Random(alphabet, sul)
            oracle2 = InterleavedRandom(alphabet, sul)
            oracle3 = InterleavedNewFirst(alphabet, sul)
            oracle4 = InterleavedOldFirst(alphabet, sul)
            oracle5 = StochasticLinear(alphabet, sul)
            oracle6 = StochasticSquare(alphabet, sul)
            oracle7 = StochasticExponential(alphabet, sul)
            oracles = [oracle1, oracle2, oracle3, oracle4, oracle5, oracle6, oracle7]
            queries, fails = test_oracles(oracles, intermediate, name)
            measurements[trial] = queries
            failures += fails
        
        index = [o.__class__.__name__ for o in oracles]
        # save all measurements so that they can be inspected later
        np.save(f"{TEACHER}/{name}/measurements.npy", measurements)
    averages = np.mean(measurements, axis=0)
    deviations = np.std(measurements, axis=0, mean=averages)
    # also save the geometric mean and the median
    geometric_means = np.exp(np.mean(np.log(measurements), axis=0))
    medians = np.median(measurements, axis=0)
    columns = [f"h{i}" for i in range(NUM_HYPS)]

    failures /= TIMES

    for stat, stat_name in zip([averages, deviations, geometric_means, medians, failures],
                          ['averages', 'deviations', 'geometric_means', 'medians', 'failures']):
        df = pd.DataFrame(stat, columns=columns, index=index)
        df.index.name = 'oracle'
        print(df)
        if not flag:
            df.to_csv(f"{TEACHER}/{name}/{stat_name}.csv")

print("These models are too big to learn:")
print(huge)
print("These models are too small to learn:")
print(tiny)

