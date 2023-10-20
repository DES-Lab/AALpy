from Examples import learning_context_free_grammar_example
from aalpy.SULs.AutomataSUL import SevpaSUL, VpaSUL
from aalpy.learning_algs import run_KV
from aalpy.oracles import RandomWordEqOracle, RandomWalkEqOracle
from aalpy.utils import visualize_automaton, get_Angluin_dfa, generate_random_sevpa
from aalpy.utils.BenchmarkVpaModels import *
from aalpy.utils.BenchmarkSevpaModels import *


# learning_context_free_grammar_example()

# TODOs
# 1. Make update function of KV work, update_rs works and most likely should be similar
# 2. Create VpaStateCoverageOracle, that behaves like StatePrefix oracle but for VPAs
# 3. Add all 15 langs as SVEPA
# 4. Implement to state setup
# 5. Create an active interface to learn a grammar of some language, like simple C or Java


# def test_on_random_svepa():
#     random_svepa = generate_random_sevpa(num_states=10, internal_alphabet_size=2,
#                                          call_alphabet_size=2,
#                                          return_alphabet_size=2,
#                                          acceptance_prob=0.4,
#                                          return_transition_prob=0.5)
#
#     alphabet = random_svepa.input_alphabet
#
#     sul = SevpaSUL(random_svepa, include_top=False, check_balance=False)
#
#     eq_oracle = RandomWordEqOracle(alphabet=alphabet.get_merged_alphabet(), sul=sul, num_walks=10000,
#                                    min_walk_len=10, max_walk_len=30)
#     # model = run_KV_vpda(alphabet=alphabet, sul=sul, eq_oracle=eq_oracle, print_level=3,)
#     model = run_KV(alphabet=alphabet, sul=sul, eq_oracle=eq_oracle, automaton_type='vpa',
#                    print_level=2, cex_processing='rs')
#
# test_on_random_svepa()
# exit()

from random import seed

for i, vpa in enumerate(
        [vpa_for_L1(), vpa_for_L2(), vpa_for_L3(), vpa_for_L4(), vpa_for_L5(), vpa_for_L7(), vpa_for_L8(),
         vpa_for_L9(), vpa_for_L10(), vpa_for_L11(), vpa_for_L12(), vpa_for_L13(), vpa_for_L14(), vpa_for_L15()]):

    print(f'VPA {i + 1 if i < 6 else i + 2}')
    # 16 works
    for i in range(10):
        print(i)
        model_under_learning = vpa

        alphabet = SevpaAlphabet(list(model_under_learning.internal_set),
                                 list(model_under_learning.call_set),
                                 list(model_under_learning.return_set))

        sul = VpaSUL(model_under_learning, include_top=False, check_balance=False)

        eq_oracle = RandomWordEqOracle(alphabet=alphabet.get_merged_alphabet(), sul=sul, num_walks=10000)
        # model = run_KV_vpda(alphabet=alphabet, sul=sul, eq_oracle=eq_oracle, print_level=3,)
        model = run_KV(alphabet=alphabet, sul=sul, eq_oracle=eq_oracle, automaton_type='vpa',
                       print_level=2, cex_processing='rs')


        # exit()
