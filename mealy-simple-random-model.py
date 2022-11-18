import os
from random import seed

from aalpy.SULs import MealySUL
from aalpy.utils import load_automaton_from_file
from aalpy.oracles import RandomWalkEqOracle, StatePrefixEqOracle
from aalpy.learning_algs.deterministic.KV import run_KV
from aalpy.learning_algs.deterministic.LStar import run_Lstar



def main():
    folder_path = '/Users/andrea/PhD/bachelor_thesis/maximilian_rindler/TTT'
    filename = 'mealy-simple-random-model.dot'
    mealy = load_automaton_from_file(os.path.join(folder_path, filename), 'mealy')

    seed(1)

    sul = MealySUL(mealy)
    alphabet = mealy.get_input_alphabet()

    # eq_oracle = RandomWalkEqOracle(alphabet, sul, 1000)
    eq_oracle = StatePrefixEqOracle(alphabet, sul, walks_per_state=15, walk_len=10)

    learned_mealy = run_KV(alphabet, sul, eq_oracle, automaton_type='mealy', print_level=1, cex_processing=None)

    #sul = MealySUL(mealy)
    #eq_oracle = RandomWalkEqOracle(alphabet, sul, 1000)
    #learned_mealy = run_Lstar(alphabet, sul, eq_oracle, automaton_type='mealy', print_level=1, cex_processing='rs')

    learned_mealy.visualize()

if __name__ == "__main__":
    main()