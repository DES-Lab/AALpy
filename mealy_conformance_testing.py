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

if not os.path.exists('intermediate_hypotheses'):
    os.makedirs('intermediate_hypotheses')

for file, model in zip(files, models):
    alphabet = model.get_input_alphabet()
    sul = AutomatonSUL(model)
    eqo = RandomWordEqOracle(alphabet, sul, num_walks=5000, min_walk_len=10, max_walk_len=100)

    l_star_model, info = run_Lstar(alphabet, sul, eqo, 'mealy', print_level=0, return_data=True)
    size = info['automaton_size']
    steps = info['steps_learning']
    queries = info['queries_learning']
    intermediate = info['intermediate_hypotheses']

    path = f"intermediate_hypotheses/{file.stem}"
    for number, hyp in enumerate(intermediate):
        # check if directory exists
        if not os.path.exists(path):
            os.makedirs(path)
        hyp.save(file_path=(path + f"/h{number}.dot"))

    # also save the diffs of consecutive hypotheses
    # this can be done by executing diff on the command line
    # diff hyp1.dot hyp2.dot > diff1_2.txt
    for i in range(1, len(intermediate)):
        os.system(f"diff {path}/h{i - 1}.dot {path}/h{i}.dot > {path}/diff_{i - 1}_{i}.txt")
    
    print(f"# steps         = {steps}")
    print(f"# queries       = {queries}")
    print(f"# intermediate  = {len(intermediate)}")
    print(f"# states        = {size}")


