import pickle
from collections import defaultdict
from random import seed
from statistics import mean

from aalpy.SULs import AutomatonSUL
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import StatePrefixEqOracle, RandomWMethodEqOracle, RandomWalkEqOracle, RandomWordEqOracle
from aalpy.utils import generate_random_deterministic_automata

#closing_strategies = ['shortest_first']

closing_strategies = ['shortest_first', 'longest_first', 'single', 'single_longest']
obs_table_cell_prefixes = [True, False]
suffix_closed = [True, False]
cex_processing = [None, 'longest_prefix', 'rs']

automata_size = [400 ]
input_sizes = [2,]
output_sizes = [3, ]
num_repeats = 10

test_models = []
for size in automata_size:
    for i in input_sizes:
        for o in output_sizes:
            random_model = generate_random_deterministic_automata('dfa', size, i, o, num_accepting_states=30)
            test_models.append(random_model)

tc = 0
num_exp = len(test_models) * len(closing_strategies) * len(suffix_closed) * num_repeats * len(cex_processing) * len(obs_table_cell_prefixes)
stats = defaultdict(list)
for test_model in test_models:
    input_al = test_model.get_input_alphabet()
    for closedness_type in suffix_closed:
        for closing_strategy in closing_strategies:
            for cex in cex_processing:
                for prefix_in_cell in obs_table_cell_prefixes:
                    for _ in range(num_repeats):
                        tc += 1
                        print(round(tc / num_exp * 100, 2))
                        # seed(tc)
                        sul = AutomatonSUL(test_model)
                        eq_oracle = RandomWordEqOracle(input_al, sul, num_walks=5000, min_walk_len=10, max_walk_len=40)
                        model, info = run_Lstar(input_al, sul, eq_oracle, 'dfa',
                                                closing_strategy=closing_strategy,
                                                cex_processing=cex,
                                                e_set_suffix_closed=closedness_type,
                                                all_prefixes_in_obs_table=prefix_in_cell,
                                                print_level=0,
                                                return_data=True)

                        config_name = f'suffix_closed:{closedness_type},closing_strategy:{closing_strategy},' \
                                      f'cex:{cex},prefixes_in_cell_{prefix_in_cell}'
                        stats[config_name].append(
                            (info['queries_learning'],
                             info['steps_learning'],
                             model.size == test_model.size))

with open('stats.pickle', 'wb') as handle:
    pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('stats.pickle', 'rb') as handle:
#     stats = pickle.load(handle)

statistics_sorted = []
for k, v in stats.items():
    mean_queries, mean_steps, num_correct = mean([x[0] for x in v]), mean([x[1] for x in v]), sum([x[2] for x in v])
    statistics_sorted.append((k, mean_queries, mean_steps, num_correct))

statistics_sorted.sort(key=lambda x: x[2])

for k, q, s, c in statistics_sorted:
    print(k, int(q), int(s), c)

# suffix_closed:True,closing_strategy:longest_first,cex:rs,prefixes_in_cell_True 6702 122491 10
# suffix_closed:False,closing_strategy:single,cex:rs,prefixes_in_cell_True 8339 150049 10
# suffix_closed:False,closing_strategy:longest_first,cex:rs,prefixes_in_cell_True 8087 154159 10
# suffix_closed:True,closing_strategy:single,cex:rs,prefixes_in_cell_True 8711 159552 10
# suffix_closed:False,closing_strategy:shortest_first,cex:longest_prefix,prefixes_in_cell_True 9032 162634 10
# suffix_closed:True,closing_strategy:shortest_first,cex:rs,prefixes_in_cell_True 9024 163924 10
# suffix_closed:False,closing_strategy:shortest_first,cex:rs,prefixes_in_cell_True 8685 164049 10
# suffix_closed:False,closing_strategy:longest_first,cex:longest_prefix,prefixes_in_cell_True 9011 167036 10
# suffix_closed:False,closing_strategy:single,cex:longest_prefix,prefixes_in_cell_True 9099 176007 10
# suffix_closed:True,closing_strategy:single,cex:longest_prefix,prefixes_in_cell_True 9291 178341 10
# suffix_closed:True,closing_strategy:shortest_first,cex:longest_prefix,prefixes_in_cell_True 9317 179006 10
# suffix_closed:False,closing_strategy:shortest_first,cex:None,prefixes_in_cell_True 10491 193517 10
# suffix_closed:True,closing_strategy:single,cex:None,prefixes_in_cell_True 11139 213554 10
# suffix_closed:True,closing_strategy:longest_first,cex:longest_prefix,prefixes_in_cell_True 10865 214022 10
# suffix_closed:True,closing_strategy:shortest_first,cex:None,prefixes_in_cell_True 12039 221108 10
# suffix_closed:False,closing_strategy:longest_first,cex:None,prefixes_in_cell_True 11107 222398 10
# suffix_closed:False,closing_strategy:single_longest,cex:rs,prefixes_in_cell_True 8998 227650 10
# suffix_closed:True,closing_strategy:longest_first,cex:None,prefixes_in_cell_True 12042 230262 10
# suffix_closed:False,closing_strategy:single,cex:None,prefixes_in_cell_True 12176 239711 10
# suffix_closed:True,closing_strategy:shortest_first,cex:rs,prefixes_in_cell_False 15891 319699 10
# suffix_closed:True,closing_strategy:single_longest,cex:None,prefixes_in_cell_True 10758 321832 10
# suffix_closed:False,closing_strategy:single_longest,cex:None,prefixes_in_cell_True 11331 330290 10
# suffix_closed:True,closing_strategy:single,cex:rs,prefixes_in_cell_False 16014 331618 10
# suffix_closed:False,closing_strategy:shortest_first,cex:rs,prefixes_in_cell_False 15315 335046 10
# suffix_closed:False,closing_strategy:single_longest,cex:rs,prefixes_in_cell_False 15289 339521 10
# suffix_closed:True,closing_strategy:longest_first,cex:rs,prefixes_in_cell_False 16933 343434 10
# suffix_closed:False,closing_strategy:single,cex:rs,prefixes_in_cell_False 16561 356579 10
# suffix_closed:False,closing_strategy:longest_first,cex:rs,prefixes_in_cell_False 17112 367300 10
# suffix_closed:False,closing_strategy:longest_first,cex:None,prefixes_in_cell_False 21305 421926 10
# suffix_closed:True,closing_strategy:single,cex:None,prefixes_in_cell_False 21228 439700 10
# suffix_closed:False,closing_strategy:single,cex:longest_prefix,prefixes_in_cell_False 21119 446616 10
# suffix_closed:True,closing_strategy:longest_first,cex:longest_prefix,prefixes_in_cell_False 20241 449658 10
# suffix_closed:False,closing_strategy:single,cex:None,prefixes_in_cell_False 22933 454032 10
# suffix_closed:False,closing_strategy:shortest_first,cex:longest_prefix,prefixes_in_cell_False 20601 456783 10
# suffix_closed:True,closing_strategy:longest_first,cex:None,prefixes_in_cell_False 21736 457516 10
# suffix_closed:False,closing_strategy:shortest_first,cex:None,prefixes_in_cell_False 22398 469775 10
# suffix_closed:True,closing_strategy:shortest_first,cex:longest_prefix,prefixes_in_cell_False 22365 477132 10
# suffix_closed:True,closing_strategy:single,cex:longest_prefix,prefixes_in_cell_False 21557 505135 10
# suffix_closed:True,closing_strategy:shortest_first,cex:None,prefixes_in_cell_False 24000 518014 10
# suffix_closed:False,closing_strategy:longest_first,cex:longest_prefix,prefixes_in_cell_False 23837 548645 10
# suffix_closed:False,closing_strategy:single_longest,cex:None,prefixes_in_cell_False 24994 583869 10
# suffix_closed:True,closing_strategy:single_longest,cex:None,prefixes_in_cell_False 24839 588329 10
# suffix_closed:False,closing_strategy:single_longest,cex:longest_prefix,prefixes_in_cell_True 10322 620607 10
# suffix_closed:True,closing_strategy:single_longest,cex:rs,prefixes_in_cell_True 9930 651615 10
# suffix_closed:True,closing_strategy:single_longest,cex:rs,prefixes_in_cell_False 16761 729912 10
# suffix_closed:True,closing_strategy:single_longest,cex:longest_prefix,prefixes_in_cell_True 11419 779192 10
# suffix_closed:True,closing_strategy:single_longest,cex:longest_prefix,prefixes_in_cell_False 19734 919089 10
# suffix_closed:False,closing_strategy:single_longest,cex:longest_prefix,prefixes_in_cell_False 19842 939866 10