
from aalpy.SULs.AutomataSUL import PdaSUL
from aalpy.automata.Pda import generate_data_from_pda
from aalpy.learning_algs import run_vpda_Lstar
from aalpy.oracles import RandomWMethodEqOracle
from aalpy.utils.BenchmarkPdaModels import *


pda = pda_for_L12()
pda.visualize()

input_alphabet = pda.get_input_alphabet()
sul = PdaSUL(pda, include_top=True, check_balance=True)
print(sul.query(('(',')')))


# pda_sequances = generate_data_from_pda(pda, 10000)
# accepting_seq, rejecting_seq = [x[0] for x in pda_sequances if x[1]], [x[0] for x in pda_sequances if not x[1]]
# accepting_seq.sort(key=len)
# print('Positive')
# for i in range(10):
#     print(accepting_seq[i])
# exit()

# eq_oracle = RandomWMethodEqOracle(alphabet=input_alphabet, sul=sul, walks_per_state=100, walk_len=10)
# model = run_vpda_Lstar(alphabet=input_alphabet, sul=sul, eq_oracle=eq_oracle, automaton_type="dfa", print_level=3,
#                        max_learning_rounds=1)