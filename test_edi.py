from aalpy.SULs.AutomataSUL import SevpaSUL, VpaSUL
from aalpy.automata.Pda import generate_data_from_pda
from aalpy.learning_algs import run_KV_vpda, run_KV
from aalpy.oracles import RandomWordEqOracle, RandomWalkEqOracle
from aalpy.utils import visualize_automaton, get_Angluin_dfa
from aalpy.utils.BenchmarkPdaModels import *
from aalpy.utils.BenchmarkVpaModels import *
from aalpy.utils.BenchmarkSevpaModels import *

#
# sevpa = sevpa_for_L12_refined()
#
# # visualize_automaton(sevpa, path="InitialModel")
#
# print(sevpa.input_alphabet)

model_under_learning = vpa_for_L12()

merged_input_alphabet = SevpaAlphabet(list(model_under_learning.internal_set),
                                      list(model_under_learning.call_set),
                                      list(model_under_learning.return_set))

sul = VpaSUL(model_under_learning, include_top=False, check_balance=False)

assert sul.query(('(', ')'))[-1] == True

eq_oracle = RandomWordEqOracle(alphabet=merged_input_alphabet.get_merged_alphabet(), sul=sul, num_walks=1000)
model = run_KV_vpda(alphabet=merged_input_alphabet, sul=sul, eq_oracle=eq_oracle, print_level=3, max_learning_rounds=3)

model_sul = SevpaSUL(model, include_top=True, check_balance=True)
print(model_sul.query(('(', ')')))
print(model_sul.query(('[', ')')))
print(model_sul.query(('[', '(', ')', ']')))
