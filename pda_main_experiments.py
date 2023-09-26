from aalpy.SULs.AutomataSUL import PdaSUL
from aalpy.learning_algs import run_vpda_Lstar
from aalpy.oracles import RandomWMethodEqOracle
from aalpy.utils.BenchmarkPdaModels import pda_for_L12

pda = pda_for_L12()

# pda.visualize()

input_alphabet = pda.get_input_alphabet()
sul = PdaSUL(pda, include_top=True)

eq_oracle = RandomWMethodEqOracle(alphabet=input_alphabet, sul=sul, walks_per_state=100, walk_len=10)
model = run_vpda_Lstar(alphabet=input_alphabet, sul=sul, eq_oracle=eq_oracle, automaton_type="dfa", print_level=3,
                       max_learning_rounds=3)
model.visualize()
