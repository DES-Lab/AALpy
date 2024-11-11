import os
import itertools
from pathlib import Path

from aalpy.utils.FileHandler import load_automaton_from_file
from aalpy.learning_algs.deterministic import KV
import aalpy.oracles

root = os.getcwd() + "/DotModels"
protocols = ["MQTT", "TCP", "TLS"]
directories = [ Path(root + '/' + prot) for prot in protocols]
files = itertools.chain.from_iterable(dir.iterdir() for dir in directories)
models = iter(load_automaton_from_file(f, 'mealy') for f in files)

equivalence_oracles = [
        aalpy.oracles.RandomWalkEqOracle,
        aalpy.oracles.RandomWordEqOracle
        ]

for model in models:
    print(model.get_input_alphabet())


