from random import seed

from aalpy.SULs import DfaSUL
from aalpy.learning_algs.deterministic.KV import run_KV
from aalpy.oracles import StatePrefixEqOracle
from aalpy.utils import generate_random_deterministic_automata

automata_sizes = [1000,]
input_al_sizes = [2, 4, 6, 8, 10]
num_repeats = 1

i = 0
for automata_size in automata_sizes:
    for input_size in input_al_sizes:
        for _ in range(num_repeats):
            i += 1
            seed(i)
            model = generate_random_deterministic_automata('dfa', num_states=automata_size,
                                                           input_alphabet_size=input_size,
                                                           output_alphabet_size=2)
            input_al = model.get_input_alphabet()

            sul = DfaSUL(model)
            eq_oracle = StatePrefixEqOracle(input_al, sul, walks_per_state=15, walk_len=10)

            learned_model = run_KV(input_al, sul, eq_oracle, cex_processing='rs', print_level=2)
            if len(learned_model.states) != automata_size:
                print('Failing seed', i)