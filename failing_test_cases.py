import random
from random import seed

from regex import D
from torch import eq

from aalpy.SULs import DfaSUL
from aalpy.learning_algs.deterministic.KV import run_KV
from aalpy.learning_algs.deterministic.LStar import run_Lstar
from aalpy.oracles import RandomWalkEqOracle
from aalpy.oracles.StatePrefixEqOracle import StatePrefixEqOracle
from aalpy.utils import generate_random_dfa
from kv_test import checkConformance

from random import seed
# 31, 56, 57, 62, 85
seed(31)

alphabet_size = 4
maximum_number_states = 100
alphabet = [f'i{inp}' for inp in range(alphabet_size)]
num_states = 100
num_accepting_states = 20

dfa = generate_random_dfa(num_states, alphabet, num_accepting_states, ensure_minimality=True)

# Get its input alphabet
alphabet = dfa.get_input_alphabet()

# Create a SUL instance wrapping the random automaton
sul = DfaSUL(dfa)

# create a random walk equivalence oracle that will perform up to 500 steps every learning round
eq_oracle = StatePrefixEqOracle(alphabet, sul)

learned_dfa_kv = run_KV(alphabet, sul, eq_oracle, automaton_type='dfa',
                        print_level=3, reuse_counterexamples=True, cex_processing='rs')

eq_oracle = RandomWalkEqOracle(alphabet, sul, 2000, reset_after_cex=True)
#learned_dfa_l_star = run_Lstar(alphabet,sul,eq_oracle,"dfa")

learning_result = checkConformance(alphabet, learned_dfa_kv, len(dfa.states), sul)
#assert len(learned_dfa_kv.states) == len(dfa.states)
assert learning_result
