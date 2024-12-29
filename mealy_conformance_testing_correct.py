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

WALKS_PER_ROUND = 50000
WALKS_PER_STATE = 10000 # make it big so that it is bounded by WALKS_PER_ROUND
WALK_LEN = 100

class Random(StatePrefixEqOracle):
    def __init__(self, alphabet, sul, walks_per_round=WALKS_PER_ROUND, walks_per_state=WALKS_PER_STATE, walk_len=WALK_LEN, depth_first=False):
        super().__init__(alphabet, sul, walks_per_round, walks_per_state, walk_len, depth_first)

class RandomWMethod(RandomWMethodEqOracle):
    def __init__(self, alphabet, sul, walks_per_round=WALKS_PER_ROUND, walks_per_state=WALKS_PER_STATE, walk_len=WALK_LEN):
        super().__init__(alphabet, sul, walks_per_round, walk_len)

class StochasticLinear(StochasticStateCoverageEqOracle):
    def __init__(self, alphabet, sul, walks_per_round=WALKS_PER_ROUND, walk_len=WALK_LEN, prob_function='linear'):
        super().__init__(alphabet, sul, walks_per_round, walk_len, prob_function)

class StochasticSquare(StochasticStateCoverageEqOracle):
    def __init__(self, alphabet, sul, walks_per_round=WALKS_PER_ROUND, walk_len=WALK_LEN, prob_function='square'):
        super().__init__(alphabet, sul, walks_per_round, walk_len, prob_function)

class StochasticExponential(StochasticStateCoverageEqOracle):
    def __init__(self, alphabet, sul, walks_per_round=WALKS_PER_ROUND, walk_len=WALK_LEN, prob_function='exponential'):
        super().__init__(alphabet, sul, walks_per_round, walk_len, prob_function)

ROOT = os.getcwd() + "/DotModels"
# PROTOCOLS = ["ASML", "TLS", "MQTT", "EMV", "TCP"]
PROTOCOLS = ["TCP",]
DIRS = [pathlib.Path(ROOT + '/' + prot) for prot in PROTOCOLS]
FILES = [file for dir in DIRS for file in dir.iterdir()]
MODELS = [load_automaton_from_file(f, 'mealy') for f in FILES]

TIMES = 10
NUM_ORACLES = 5
EQ_QUERIES = np.zeros((len(MODELS), TIMES, NUM_ORACLES))
MB_QUERIES = np.zeros((len(MODELS), TIMES, NUM_ORACLES))
FAILURES   = np.zeros((len(MODELS), TIMES, NUM_ORACLES))
# iterate over the models
for index, (model, file) in enumerate(zip(MODELS, FILES)):
    # repeat the experiments to gather statistics
    for t in range(TIMES):
        sul = AutomatonSUL(model)
        correct_size = model.size
        alphabet = list(model.get_input_alphabet())
        # reinitialize the oracles
        eq_oracles = [Random(alphabet, sul),
                      RandomWMethod(alphabet, sul),
                      StochasticLinear(alphabet, sul),
                      StochasticSquare(alphabet, sul),
                      StochasticExponential(alphabet, sul)]
        print(f'Learning {file.stem} for each oracle {t + 1}/{TIMES}')
        for i in range(NUM_ORACLES):
            oracle = eq_oracles[i]
            name = oracle.__class__.__name__
            learned_model, info = run_Lstar(alphabet, sul, oracle, 'mealy', return_data=True, print_level=0)
            # store total queries during hypothesis validation
            EQ_QUERIES[index, t, i] = info['queries_eq_oracle']
            # store total queries during learning
            MB_QUERIES[index, t, i] = info['queries_learning']
            # store 1 if the learned model is different from the correct model
            FAILURES[index, t, i]   = 1 if info['automaton_size'] != correct_size else 0

# store them as numpy arrays
np.save('eq_queries_10_tcp.npy', EQ_QUERIES)
np.save('mb_queries_10_tcp.npy', MB_QUERIES)
np.save('failures_10_tcp.npy', FAILURES)

