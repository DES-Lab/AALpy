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

for i, vpa in enumerate([vpa_for_L1(), vpa_for_L2(), vpa_for_L3(), vpa_for_L4(), vpa_for_L5(), vpa_for_L7(), vpa_for_L8(),
            vpa_for_L9(), vpa_for_L10(), vpa_for_L11(), vpa_for_L12(),vpa_for_L13(), vpa_for_L14(), vpa_for_L15()]):

    print(f'VPA {i + 1 if i < 6 else i + 2}')
    model_under_learning = vpa

    merged_input_alphabet = SevpaAlphabet(list(model_under_learning.internal_set),
                                          list(model_under_learning.call_set),
                                          list(model_under_learning.return_set))

    sul = VpaSUL(model_under_learning, include_top=False, check_balance=False)

    eq_oracle = RandomWordEqOracle(alphabet=merged_input_alphabet.get_merged_alphabet(), sul=sul, num_walks=10000)
    model = run_KV_vpda(alphabet=merged_input_alphabet, sul=sul, eq_oracle=eq_oracle, print_level=3, max_learning_rounds=3)
