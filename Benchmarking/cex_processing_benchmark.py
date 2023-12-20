from collections import defaultdict
from statistics import mean, stdev

from aalpy.learning_algs import run_KV, run_Lstar

from aalpy.SULs import AutomatonSUL
from aalpy.oracles import RandomWalkEqOracle
from aalpy.utils import generate_random_deterministic_automata, bisimilar

counterexample_processing_strategy = ['rs', 'linear_fwd', 'linear_bwd', 'exponential_fwd', 'exponential_bwd']
algorithms = ['l_star', 'kv']
model_sizes = [500]
model_type = ['mealy', 'moore']
# alphabet_sizes = [(3,2), (3, 5), (3, 10), (5, 2), (5, 5), (5, 20)]
alphabet_sizes = [(5, 3)]

num_repetitions = 5

for learning_alg in algorithms:
    results = defaultdict(list)
    for model in model_type:
        for model_size in model_sizes:
            for input_size, output_size in alphabet_sizes:
                for cex_processing in counterexample_processing_strategy:

                    for _ in range(num_repetitions):
                        random_model = generate_random_deterministic_automata(model, num_states=model_size,
                                                                              input_alphabet_size=input_size,
                                                                              output_alphabet_size=output_size)
                        sul = AutomatonSUL(random_model)
                        input_al = random_model.get_input_alphabet()
                        eq_oracle = RandomWalkEqOracle(input_al, sul, num_steps=20000, reset_prob=0.09)

                        if learning_alg == 'kv':
                            learned_model, info = run_KV(input_al, sul, eq_oracle,
                                                         automaton_type=model, cex_processing=cex_processing,
                                                         return_data=True, print_level=0)
                        else:
                            learned_model, info = run_Lstar(input_al, sul, eq_oracle,
                                                            automaton_type=model, cex_processing=cex_processing,
                                                            return_data=True, print_level=0)
                        results[cex_processing].append(info['steps_learning'])

                        if not bisimilar(learned_model, random_model):
                            print(learning_alg, cex_processing, 'mismatch')

    print(learning_alg)
    for k, v in results.items():
        print(k, mean(v), stdev(v), min(v), max(v))
