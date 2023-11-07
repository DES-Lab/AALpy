from aalpy.utils import generate_random_deterministic_automata, bisimilar
from aalpy.SULs import MealySUL
from aalpy.oracles import RandomWMethodEqOracle
from aalpy.learning_algs import run_KV, run_Lstar

for x in ['exponential_fwd', 'exponential_bwd']:
    for i in range(5):
        print(i)
        model_type = 'mealy'  # or 'moore', 'dfa'

        # for random dfa's you can also define num_accepting_states
        random_model = generate_random_deterministic_automata(automaton_type=model_type, num_states=100,
                                                              input_alphabet_size=3, output_alphabet_size=4)

        sul = MealySUL(random_model)
        input_alphabet = random_model.get_input_alphabet()

        # select any of the oracles
        eq_oracle = RandomWMethodEqOracle(input_alphabet, sul, walks_per_state=10, walk_len=20)

        learned_model = run_KV(input_alphabet, sul, eq_oracle, model_type, cex_processing=x, print_level=0)
        assert bisimilar(random_model, learned_model)
