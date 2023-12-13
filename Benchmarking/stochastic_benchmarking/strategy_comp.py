from statistics import mean

from aalpy.learning_algs.stochastic.DifferenceChecker import AdvancedHoeffdingChecker, HoeffdingChecker

import aalpy.paths

from aalpy.SULs import AutomatonSUL
from aalpy.learning_algs import run_stochastic_Lstar

from aalpy.oracles import RandomWordEqOracle
from aalpy.utils import load_automaton_from_file, model_check_experiment, get_properties_file, get_correct_prop_values

aalpy.paths.path_to_prism = "C:/Program Files/prism-4.6/bin/prism.bat"
aalpy.paths.path_to_properties = "../../Benchmarking/prism_eval_props/"

example = 'mqtt'

mdp = load_automaton_from_file(f'../../DotModels/MDPs/{example}.dot', automaton_type='mdp')

strategies = [AdvancedHoeffdingChecker(alpha=0.001), 'chi2']


def learn(strategy):
    input_al = mdp.get_input_alphabet()
    sul = AutomatonSUL(mdp)
    eq_oracle = RandomWordEqOracle(input_al, sul, num_walks=1000, min_walk_len=4, max_walk_len=20)
    model, data = run_stochastic_Lstar(input_al, sul, eq_oracle, automaton_type='smm', strategy=strategy,
                                       cex_processing=None, print_level=0, return_data=True)

    num_queries = data['queries_learning'] + data['queries_eq_oracle']

    _, diff = model_check_experiment(get_properties_file(example), get_correct_prop_values(example), model.to_mdp())

    avg_error = mean(diff.values())

    return num_queries, avg_error


# strategies = [HoeffdingChecker(alpha=0.001)]
# for s in strategies:
#     plot_points = []
#     for _ in range(10):
#         plot_points.append(learn(s))
#     print(s)
#     print(plot_points)

# mqtt
normal_data_mqtt = [(38482, 0.01964), (31372, 0.01652), (31170, 0.00624), (31590, 0.01738), (30987, 0.01796),
               (46244, 0.0222), (48533, 0.01496), (31979, 0.02714), (33219, 0.00758), (32295, 0.015560000000000001)]
chi2_data_mqtt = [(155242, 0.01054), (225776, 0.00348), (230385, 0.01838), (70482, 0.01374), (85096, 0.01284),
             (64644, 0.0354), (68898, 0.03556), (61591, 0.02464), (61773, 0.0067), (35876, 0.01562)]


#
# bluetooth
normal_data_bt = [(51490, 0.011823076923076924), (66996, 0.015015384615384616), (96563, 0.003523076923076923),
               (41964, 0.030623076923076923), (40395, 0.026623076923076923), (62838, 0.012146153846153846),
               (123363, 0.03417692307692308), (100228, 0.029423076923076923), (51425, 0.009438461538461538),
               (67885, 0.0241)]
chi2_data_bt = [(19523, 0.009261538461538462), (20467, 0.038446153846153845), (34288, 0.058253846153846156),
             (30030, 0.03462307692307692), (18773, 0.038215384615384616), (16949, 0.04113846153846154),
             (18195, 0.03705384615384615), (19437, 0.030615384615384614), (16834, 0.0487), (16699, 0.09320769230769231)]


def plot():
    from matplotlib import pyplot as plt

    normal_data, chi2_data = normal_data_mqtt, chi2_data_mqtt
    normal_x = [p[0] // 1000 for p in normal_data]
    normal_y = [p[1] for p in normal_data]

    chi2_x = [p[0] // 1000 for p in chi2_data]
    chi2_y = [p[1] for p in chi2_data]

    plt.figure()
    plt.scatter(normal_x, normal_y, label='Hoeffding')
    plt.scatter(chi2_x, chi2_y, label='Chi2')

    plt.title('Bluetooth')
    plt.xlabel('Number of Queries (in thousands)')
    plt.ylabel('Average Error')
    plt.legend()
    plt.grid()

    #plt.show()
    import tikzplotlib
    tikzplotlib.save('bluetooth_strategy_comp.tex')


def plot_side_by_side():
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))

    normal_x = [p[0] // 1000 for p in normal_data_mqtt]
    normal_y = [p[1] for p in normal_data_mqtt]

    chi2_x = [p[0] // 1000 for p in chi2_data_mqtt]
    chi2_y = [p[1] for p in chi2_data_mqtt]

    axes[0].scatter(normal_x, normal_y, label='Hoeffding')
    axes[0].scatter(chi2_x, chi2_y, label='Chi2')

    normal_x = [p[0] // 1000 for p in normal_data_bt]
    normal_y = [p[1] for p in normal_data_bt]

    chi2_x = [p[0] // 1000 for p in chi2_data_bt]
    chi2_y = [p[1] for p in chi2_data_bt]

    axes[1].scatter(normal_x, normal_y, label='Hoeffding')
    axes[1].scatter(chi2_x, chi2_y, label='Chi2')

    axes[0].set_title('MQTT')
    axes[1].set_title('Bluetooth')

    axes[0].set_xlabel('Number of Queries (in thousands)')
    axes[0].set_ylabel('Average Error')
    axes[0].legend()

    axes[1].set_xlabel('Number of Queries (in thousands)')
    # axes[1].set_ylabel('Average Error')
    axes[1].legend()

    axes[0].grid()
    axes[1].grid()

    fig.tight_layout()

    plt.show()
    import tikzplotlib
    # tikzplotlib.save('strategy_comp_side_by_side.tex')


plot_side_by_side()
