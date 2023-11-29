from aalpy.utils import generate_random_deterministic_automata, bisimilar
from aalpy.SULs import MealySUL
from aalpy.oracles import RandomWMethodEqOracle
from aalpy.learning_algs import run_KV, run_Lstar

for x in ['rs', 'linear_fwd', 'linear_bwd', 'exponential_fwd', 'exponential_bwd']:
    for at in ['moore', 'dfa', 'mealy']:
        for i in range(50):
            print(x, at, i)
            model_type = at  # or 'moore', 'dfa'

            # for random dfa's you can also define num_accepting_states
            random_model = generate_random_deterministic_automata(automaton_type=model_type, num_states=75,
                                                                  input_alphabet_size=4, output_alphabet_size=5)

            sul = MealySUL(random_model)
            input_alphabet = random_model.get_input_alphabet()

            # select any of the oracles
            eq_oracle = RandomWMethodEqOracle(input_alphabet, sul, walks_per_state=10, walk_len=15)

            learned_model = run_Lstar(input_alphabet, sul, eq_oracle, model_type, cex_processing=x, print_level=0)
            if not bisimilar(random_model, learned_model):
                print(x, at)
                print(bisimilar(random_model, learned_model, return_cex=True))
