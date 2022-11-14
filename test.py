from aalpy.SULs import DfaSUL
from aalpy.learning_algs import run_Lstar
from aalpy.learning_algs.deterministic.KV import run_KV
from aalpy.oracles import RandomWordEqOracle, RandomWMethodEqOracle, StatePrefixEqOracle
from aalpy.utils import generate_random_deterministic_automata, get_Angluin_dfa


from random import seed
seed(1)
#dfa = get_Angluin_dfa()
dfa = generate_random_deterministic_automata('dfa', num_states=500, input_alphabet_size=2, output_alphabet_size=2,num_accepting_states=20)
input_al = dfa.get_input_alphabet()
sul = DfaSUL(dfa)

#eq_oracle = RandomWordEqOracle(input_al, sul, min_walk_len=10, max_walk_len=15)
eq_oracle = StatePrefixEqOracle(input_al, sul, walks_per_state=20, walk_len=20)
learned_model = run_KV(input_al, sul, eq_oracle, cex_processing='rs', print_level=2)

exit()
sul1 = DfaSUL(dfa)
eq_oracle = StatePrefixEqOracle(input_al, sul1, walks_per_state=10, walk_len=20)
learned_model = run_Lstar(input_al, sul1, eq_oracle, automaton_type='dfa', cex_processing='rs', print_level=2)
