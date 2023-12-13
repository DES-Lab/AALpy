from aalpy.SULs import AutomatonSUL
from aalpy.learning_algs import run_Lstar, run_KV
from aalpy.oracles import RandomWordEqOracle
from aalpy.utils import generate_random_deterministic_automata

automata_type = ['dfa', 'mealy', 'moore']
automata_size = [10, 100, 500, 1000,]
input_sizes = [2, 3]
output_sizes = [2, 3, 5, 10]

test_models = []
for model_type in automata_type:
    for size in automata_size:
        for i in input_sizes:
            for o in output_sizes:
                random_model = generate_random_deterministic_automata(model_type, size, i, o, num_accepting_states=size//8)
                input_al = random_model.get_input_alphabet()

                print('------------------------------------------')
                if model_type != 'dfa':
                    print(f'Type: {model_type}, size: {size}, # inputs: {i}, # outputs: {o}')
                else:
                    print(f'Type: {model_type}, size: {size}, # inputs: {i}, # accepting: {size//8}')

                # Lstar
                sul = AutomatonSUL(random_model)
                eq_oracle = RandomWordEqOracle(input_al, sul, num_walks=5000, min_walk_len=10, max_walk_len=40)
                l_star_model, l_star_info = run_Lstar(input_al, sul, eq_oracle, model_type, print_level=0, return_data=True)

                l_star_steps, l_star_queries = l_star_info['steps_learning'], l_star_info['queries_learning']

                # KV
                sul = AutomatonSUL(random_model)
                eq_oracle = RandomWordEqOracle(input_al, sul, num_walks=5000, min_walk_len=10, max_walk_len=40)
                kv_model, kv_info = run_KV(input_al, sul, eq_oracle, model_type, print_level=0, return_data=True)

                kv_steps, kv_queries = kv_info['steps_learning'], kv_info['queries_learning']

                if l_star_model.size != random_model.size:
                    print('L* did not learn correctly.')
                if kv_model.size != random_model.size:
                    print('KV did not learn correctly.')

                print(f'L* steps: {l_star_steps}')
                print(f'KV steps: {kv_steps}')
                if kv_steps < l_star_steps:
                    print(f'KV is {round((l_star_steps / kv_steps) * 100 - 100, 2)}% more step efficient')
                else:
                    print(f'L* is {round((kv_steps / l_star_steps) * 100 - 100, 2)}% more step efficient')




