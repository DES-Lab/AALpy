import os
from pathlib import Path

from aalpy.SULs.AutomataSUL import AutomatonSUL
from aalpy.learning_algs.deterministic.LStar import run_Lstar
from aalpy.oracles import RandomWordEqOracle
from aalpy.utils.FileHandler import load_automaton_from_file

root = os.getcwd() + "/DotModels"
protocols = ["MQTT", "TCP", "TLS"]
directories = [Path(root + '/' + prot) for prot in protocols]
files = [file for dir in directories for file in dir.iterdir()]
models = [load_automaton_from_file(f, 'mealy') for f in files]

assert len(files) == len(models)

for file, model in zip(files, models):
    alphabet = model.get_input_alphabet()
    sul = AutomatonSUL(model)
    eqo = RandomWordEqOracle(alphabet, sul, num_walks=5000, min_walk_len=10, max_walk_len=100)

    l_star_model, info = run_Lstar(alphabet, sul, eqo, 'mealy', print_level=0, return_data=True)
    size = info['automaton_size']
    steps = info['steps_learning']
    queries = info['queries_learning']
    intermediate = info['intermediate_hypotheses']

    for number, hyp in enumerate(intermediate):
        hyp.save(file_path=f"intermediate_hypotheses/{file.stem}{number}.dot")
    
    print(f"# steps         = {steps}")
    print(f"# queries       = {queries}")
    print(f"# intermediate  = {len(intermediate)}")
    print(f"# states        = {size}")


