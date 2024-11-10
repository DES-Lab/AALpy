import os
from pathlib import Path

from aalpy.learning_algs.deterministic import KV
import aalpy.oracles
from aalpy.utils.FileHandler import (
    load_automaton_from_file,
    save_automaton_to_file,
    visualize_automaton,
)

root = os.getcwd() + "/DotModels"

protocols = ["MQTT", "TCP", "TLS"]

directories = [ Path(root + '/' + prot) for prot in protocols]

# files = itertools.chain.from_iterable(dir.iterdir() for dir in directories)
# 
# models = (load_automaton_from_file(f, 'mealy') for f in files)

equivalence_oracles = [
        aalpy.oracles.RandomWalkEqOracle,
        aalpy.oracles.RandomWordEqOracle
        ]

for dir in directories:
    for f in dir.iterdir():
        print(f)
        try:
            model = load_automaton_from_file(f, 'mealy')
        except ValueError:
            print(f"Value error reading file {f}")
            continue
        alphabet = model.get_input_alphabet()
        print(alphabet)
