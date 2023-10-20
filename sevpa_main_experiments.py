from aalpy.SULs.AutomataSUL import SevpaSUL, DfaSUL
from aalpy.automata.Pda import generate_data_from_pda
from aalpy.automata.Sevpa import generate_random_sevpa
from aalpy.learning_algs import run_KV_vpda, run_KV
from aalpy.oracles import RandomWordEqOracle, RandomWalkEqOracle
from aalpy.utils import visualize_automaton, get_Angluin_dfa
from aalpy.utils.BenchmarkPdaModels import *
from aalpy.utils.BenchmarkVpaModels import *
from aalpy.utils.BenchmarkSevpaModels import *

# Example for normal KV

# dfa = get_Angluin_dfa()
#
# visualize_automaton(dfa, path="InitialModel")
#
# alphabet = dfa.get_input_alphabet()
#
# sul = DfaSUL(dfa)
# eq_oracle = RandomWalkEqOracle(alphabet, sul, 500)
#
# learned_dfa = run_KV(alphabet, sul, eq_oracle, automaton_type='dfa', cache_and_non_det_check=True, cex_processing=None, print_level=3)
#
# learned_dfa.visualize()

########################################

call_set = {'(', '['}
return_set = {')', ']'}
internal_set = {'x'}
input_alphabet = SevpaAlphabet(internal_alphabet=internal_set, call_alphabet=call_set, return_alphabet=return_set)

random_sevpa = generate_random_sevpa(input_alphabet, 3, 0.5, 0.1)
visualize_automaton(random_sevpa, path="Random Sevpa")

sevpa = sevpa_for_L12_refined()

visualize_automaton(sevpa, path="InitialModel")

print(sevpa.input_alphabet)
merged_input_alphabet = sevpa.input_alphabet.get_merged_alphabet()

sul = SevpaSUL(sevpa, include_top=True, check_balance=True)
print(sul.query(('(', ')')))
print(sul.query(('[', ')')))
print(sul.query(('[', '(', ')', ']')))


eq_oracle = RandomWordEqOracle(alphabet=merged_input_alphabet, sul=sul)
model = run_KV_vpda(alphabet=sevpa.input_alphabet, sul=sul, eq_oracle=eq_oracle, print_level=3, max_learning_rounds=10)

model_sul = SevpaSUL(model, include_top=True, check_balance=True)
print(model_sul.query(('(', ')')))
print(model_sul.query(('[', ')')))
print(model_sul.query(('[', '(', ')', ']')))

model.visualize()

