import os
import pandas as pd
import pickle
from pathlib import Path

from aalpy import oracles
from aalpy.SULs.AutomataSUL import AutomatonSUL
from aalpy.learning_algs.deterministic.LStar import run_Lstar
from aalpy.utils.FileHandler import load_automaton_from_file


if not os.path.exists('results.pkl'):

    root = os.getcwd() + "/DotModels"
    protocols = ["MQTT", "TCP", "TLS"]
    directories = [Path(root + '/' + prot) for prot in protocols]
    files = [file for dir in directories for file in dir.iterdir()]
    models = [load_automaton_from_file(f, 'mealy') for f in files]
    
    assert len(files) == len(models)
    
    if not os.path.exists('intermediate_hypotheses'):
        os.makedirs('intermediate_hypotheses')
    
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
                    oracles.BreadthFirstExplorationEqOracle(alphabet, sul),
                    oracles.KWayStateCoverageEqOracle(alphabet, sul),
                    oracles.TransitionFocusOracle(alphabet, sul),
                    oracles.CacheBasedEqOracle(alphabet, sul),
                    oracles.PacOracle(alphabet, sul),
                    oracles.RandomWMethodEqOracle(alphabet, sul, walks_per_state=25, walk_len=20)
                    ]
    
        # eqo = RandomWordEqOracle(alphabet, sul, num_walks=5000, min_walk_len=10, max_walk_len=100)
        correct = model.size
        states = []
        for eqo in orcs:
    
            l_star_model, info = run_Lstar(alphabet, sul, eqo, 'mealy', print_level=0, return_data=True)
            size = info['automaton_size']
            steps = info['steps_learning']
            queries = info['queries_learning']
            intermediate = info['intermediate_hypotheses']
            
            name = eqo.__class__.__name__
            path = f"intermediate_hypotheses/{file.stem}/{name}"
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
           
            with open(f"{path}/info.txt", 'w') as f:
                f.write(f"# steps         = {steps}\n")
                f.write(f"# queries       = {queries}\n")
                f.write(f"# intermediate  = {len(intermediate)}\n")
                f.write(f"# states        = {size}\n")
            states.append(size)
            oracle_status.append((name, size == correct))
    
        results[f'{file.stem}'] = oracle_status
    
        if not all([s == correct for s in states]):
            print(f"correct is {correct} : {states}")
    
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)
else:
    with open('results.pkl', 'rb') as f:
        results = pickle.load(f)

index = [ first for first, _ in results["TCP_Linux_Client"] ]
new_results= { key: [second for _, second in value] for key, value in results.items() } 
df = pd.DataFrame(new_results, index=index)
print(df)
df.to_pickle("results-pd.pkl")
