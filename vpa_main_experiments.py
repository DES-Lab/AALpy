from aalpy.SULs.AutomataSUL import VpaSUL
from aalpy.automata.Pda import generate_data_from_pda
from aalpy.learning_algs import run_vpda_Lstar
from aalpy.oracles import RandomWMethodEqOracle
from aalpy.utils.BenchmarkPdaModels import *
from aalpy.utils.BenchmarkVpaModels import *


vpa = vpa_for_L12()

vpa.visualize()

input_alphabet = vpa.get_input_alphabet()
merged_input_alphabet = vpa.get_input_alphabet_merged()
# print("Call: " + str(input_alphabet[0]) + "\nReturn: " + str(input_alphabet[1]) + "\nInternal: " + str(input_alphabet[2]))

sul = VpaSUL(vpa, include_top=True, check_balance=True)
print(sul.query(('(',')')))

assert sul.query(('(', ')'))[-1][0] == True

# pda_sequences = generate_data_from_pda(vpa, 10000)
# accepting_seq, rejecting_seq = [x[0] for x in pda_sequences if x[1]], [x[0] for x in pda_sequences if not x[1]]
# accepting_seq.sort(key=len)
# print('Positive')
# for i in range(10):
#     print(accepting_seq[i])
#
# eq_oracle = RandomWMethodEqOracle(alphabet=merged_input_alphabet, sul=sul, walks_per_state=100, walk_len=10)
# model = run_vpda_Lstar(alphabet=input_alphabet, sul=sul, eq_oracle=eq_oracle, automaton_type="vpa", print_level=3,
#                        max_learning_rounds=1)
#
# model.visualize()

