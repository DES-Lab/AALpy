import csv
import os
from collections import defaultdict
from statistics import mean

directory = 'FM_mdp_smm_error_based_stop/benchmark_no_cq_bfs_longest_prefix/'

benchmarks = os.listdir(directory)[:-1]

benchmarks.remove('exp_14.csv')

values = dict()

for file in benchmarks:
    with open(directory + file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

        for i in range(0, len(data), 3):
            header = data[i]
            mdp, smm = data[i + 1], data[i + 2]

            for formalism in [mdp, smm]:
                for i, val in enumerate(formalism[1:]):
                    if formalism[0] not in values.keys():
                        values[formalism[0]] = defaultdict(list)
                    values[formalism[0]][header[i + 1]].append(round(float(val), 2))

min_values_dict = dict()
max_values_dict = dict()
avr_values_dict = dict()

for exp in values:
    exp_name = exp[12:]
    formalism = 'smm' if 'smm' in exp else 'mdp'

    name = f'{exp_name}_{formalism}'
    min_values_dict[name] = dict()
    max_values_dict[name] = dict()
    avr_values_dict[name] = dict()

    for category, value in values[exp].items():
        min_values_dict[name][category] = min(value)
        max_values_dict[name][category] = max(value)
        avr_values_dict[name][category] = round(mean(value), 2)

interesting_fields = [' Learning time', ' Learning Rounds', ' #MQ Learning', ' # Steps Learning']

print('ALL ERRORS ARE LESS THAN 2%. THAT WAS USED AS STOPPING CRITERION')
experiments = list(min_values_dict.keys())
for e_index in range(0, len(experiments), 2):
    for i in interesting_fields:
        print(f'{experiments[e_index]} vs {experiments[e_index + 1]} = > {i}')
        min_eff = round(min_values_dict[experiments[e_index]][i] / min_values_dict[experiments[e_index + 1]][i]*100 , 2)
        print(f'Min : {min_values_dict[experiments[e_index]][i]} vs {min_values_dict[experiments[e_index + 1]][i]} | SMM efficiency : {min_eff}')
        max_eff = round(max_values_dict[experiments[e_index]][i] / max_values_dict[experiments[e_index + 1]][i]*100 , 2)
        print(f'Max : {max_values_dict[experiments[e_index]][i]} vs {max_values_dict[experiments[e_index + 1]][i]} | SMM efficiency : {max_eff}')
        avr_eff = round(avr_values_dict[experiments[e_index]][i] / avr_values_dict[experiments[e_index + 1]][i]*100 , 2)
        print(f'Avr : {avr_values_dict[experiments[e_index]][i]} vs {avr_values_dict[experiments[e_index + 1]][i]}| SMM efficiency : {avr_eff}')

    print('-------------------------------------------------')

with open('error_benchmark.csv', 'w',newline='') as file:
    writer = csv.writer(file)

    experiments = list(min_values_dict.keys())
    for e_index in range(0, len(experiments), 2):
        writer.writerow([experiments[e_index][:-4], 'mdp', 'smm', 'smm compared to mdp efficiency %'])
        for i in interesting_fields:
            print(f'{experiments[e_index]} vs {experiments[e_index + 1]} = > {i}')
            min_eff = round(min_values_dict[experiments[e_index]][i] / min_values_dict[experiments[e_index + 1]][i]*100 , 2)
            writer.writerow([i + '_min', min_values_dict[experiments[e_index]][i], min_values_dict[experiments[e_index + 1]][i], min_eff])
            max_eff = round(max_values_dict[experiments[e_index]][i] / max_values_dict[experiments[e_index + 1]][i]*100 , 2)
            writer.writerow([i + '_max', max_values_dict[experiments[e_index]][i], max_values_dict[experiments[e_index + 1]][i], max_eff])
            avr_eff = round(avr_values_dict[experiments[e_index]][i] / avr_values_dict[experiments[e_index + 1]][i]*100 , 2)
            writer.writerow([i + '_avr', avr_values_dict[experiments[e_index]][i], avr_values_dict[experiments[e_index + 1]][i], avr_eff])
        writer.writerow([])

        print('-------------------------------------------------')

