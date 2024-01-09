import matplotlib.pyplot as plt

from aalpy.utils import load_automaton_from_file
from aalpy.utils import statistical_model_checking

model = load_automaton_from_file('../../DotModels/MDPs/bluetooth.dot', automaton_type='mdp')

steps = [3, 5, 8, 11, 14, 17, 20]


# statistical_tests= []
# for s in steps:
#     e = statistical_model_checking(model, {'no_response'}, s)
#     statistical_tests.append(e)
#
# print(statistical_tests)

def crash_plot():
    original_values_crash = [0, 0.16800000000000004, 0.3926480000000001, 0.5572338000000001, 0.6772233874640001,
                             0.7646958490393682, 0.8284632739463244]

    random_input_crash = [0, 0.001981749035076958, 0.0281691469985939, 0.03959723310087103, 0.046344616720299714,
                          0.04758085064218106, 0.05136504760916134]

    # smm learning took 34k queries
    smm_crash_property = [0, 0.174, 0.4046, 0.5714, 0.6915, 0.778, 0.8402, ]
    # mdp learning 180k queries
    mdp_crash_property = [0, 0.1899, 0.4624, 0.6471, 0.7684, 0.848, 0.9002, ]

    plt.plot(steps, original_values_crash, label='Correct Values')
    plt.plot(steps, smm_crash_property, label='SMM Values')
    plt.plot(steps, mdp_crash_property, label='MDP Values')
    plt.plot(steps, random_input_crash, label='Random Inputs')

    plt.xticks(steps)

    plt.grid()
    plt.legend()
    plt.show()


def no_response():
    original_values_no_response = [0.36000000000000004, 0.5904, 0.7902848, 0.8926258176000001,
                                   0.9450244186112, 0.9718525023289344, 0.9855884811924145]
    mdp_no_response = [0.3993, 0.6407, 0.8315, 0.921, 0.9629, 0.9826, 0.9918]
    smm_no_response = [0.3652, 0.5966, 0.7956, 0.8965, 0.9476, 0.9734, 0.9865]
    random_input_no_response = [0.29130767125614576, 0.30205630054639654, 0.3069446148329197, 0.31200279332244946,
                                0.3154095142827484, 0.3145318825672143, 0.3125784442326385]

    plt.plot(steps, original_values_no_response, label='Correct Values')
    plt.plot(steps, smm_no_response, label='SMM Values')
    plt.plot(steps, mdp_no_response, label='MDP Values')
    plt.plot(steps, random_input_no_response, label='Random Inputs')

    plt.xticks(steps)

    plt.grid()
    plt.legend()
    plt.show()

def side_by_side():
    original_values_crash = [0, 0.16800000000000004, 0.3926480000000001, 0.5572338000000001, 0.6772233874640001,
                             0.7646958490393682, 0.8284632739463244]

    random_input_crash = [0, 0.001981749035076958, 0.0281691469985939, 0.03959723310087103, 0.046344616720299714,
                          0.04758085064218106, 0.05136504760916134]

    # smm learning took 34k queries
    smm_crash_property = [0, 0.174, 0.4046, 0.5714, 0.6915, 0.778, 0.8402, ]
    # mdp learning 180k queries
    mdp_crash_property = [0, 0.1899, 0.4624, 0.6471, 0.7684, 0.848, 0.9002, ]

    original_values_no_response = [0.36000000000000004, 0.5904, 0.7902848, 0.8926258176000001,
                                   0.9450244186112, 0.9718525023289344, 0.9855884811924145]
    mdp_no_response = [0.3993, 0.6407, 0.8315, 0.921, 0.9629, 0.9826, 0.9918]
    smm_no_response = [0.3652, 0.5966, 0.7956, 0.8965, 0.9476, 0.9734, 0.9865]
    random_input_no_response = [0.29130767125614576, 0.30205630054639654, 0.3069446148329197, 0.31200279332244946,
                                0.3154095142827484, 0.3145318825672143, 0.3125784442326385]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))

    axes[0].plot_side_by_side()
    axes[0].plot_side_by_side()
    axes[0].plot_side_by_side()
    axes[0].plot_side_by_side()
    axes[0].set_xlabel('Steps to \'crash\'')
    axes[0].set_ylabel('Property Value')

    axes[1].plot_side_by_side()
    axes[1].plot_side_by_side()
    axes[1].plot_side_by_side()
    axes[1].plot_side_by_side()
    axes[1].set_xlabel('Steps to \'no_response\'')
    axes[1].set_ylabel('Property Value')

    axes[0].set_xticks(steps)
    axes[1].set_xticks(steps)

    axes[0].grid()
    axes[1].grid()
    axes[0].legend()

    fig.tight_layout()

    plt.show()

    # import tikzplotlib
    # tikzplotlib.save("properties_over_time.tex")

crash_plot()