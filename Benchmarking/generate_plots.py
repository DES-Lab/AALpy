from random import random, randint

import matplotlib
from matplotlib import pyplot as plt
import csv

def plot_increasing_size_exp():
    data = []

    with open('increasing_size_experiments.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    x_axis = data.pop(0)
    x_axis.pop(0)

    # use total times only
    data = data[3:]

    x_axis = [int(x) for x in x_axis]
    #x_axis = list(range(2,51))
    times = []
    labels = ['DFA', 'Mealy', 'Moore']
    for r in data:
        row_name = r.pop(0)
        times.extend([float(i) for i in r])
        plt.plot(x_axis, [float(i) for i in r], label=labels.pop(0))

    plt.legend()
    plt.xticks([100, 1000, 2000, 3000, 4000, 5000,])
    plt.yticks([min(times), 0.5, 1, 1.5, max(times)])
    #plt.yticks([min(times), 1, 2, max(times)])
    #plt.yticks([min(times), 3, 6, 10 , 14])

    plt.ylabel("Time (s)")
    plt.xlabel("Automaton Size")
    #plt.grid(axis='y')
    plt.grid()

    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

    plt.savefig('state_increase_runtime.pgf')
    #plt.show()


def plot_increasing_alphabeth_exp():
    data = []

    with open('increasing_alphabet_experiments.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    x_axis = data.pop(0)[:-1]
    x_axis.pop(0)

    # use total times only

    x_axis = [int(x) for x in x_axis]
    #x_axis = list(range(2,51))
    times = []
    labels = ['DFA', 'Mealy', 'Moore']
    for r in data:
        row_name = r.pop(0)
        times.extend([float(i) for i in r])
        plt.plot(x_axis, [float(i) for i in r], label=labels.pop(0))

    plt.legend()
    plt.xticks([5,25,50,75,100])
    plt.yticks([min(times), 1, 2.5, 4, max(times)])
    #plt.yticks([min(times), 1, 2, max(times)])
    #plt.yticks([min(times), 3, 6, 10 , 14])

    plt.ylabel("Time (s)")
    plt.xlabel("Alphabet Size")
    #plt.grid(axis='y')
    plt.grid()

    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    plt.savefig('alphabet_increase_runtime.pgf')
    #plt.show()

def plot_together():
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(10, 3))
    
    
    with open('increasing_alphabet_experiments.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    x_axis = data.pop(0)[:-1]
    x_axis.pop(0)

    # use total times only

    x_axis = [int(x) for x in x_axis]
    #x_axis = list(range(2,51))
    times = []
    labels = ['DFA', 'Mealy', 'Moore']
    for r in data:
        row_name = r.pop(0)
        times.extend([float(i) for i in r])
        ax1.plot(x_axis, [float(i) for i in r], label=labels.pop(0))

    ax1.legend()
    ax1.set_xticks([5,25,50,75,100], minor=False)
    ax1.set_yticks([min(times), 1, 2.5, 4, max(times)], minor=False)
    #plt.yticks([min(times), 1, 2, max(times)])
    #plt.yticks([min(times), 3, 6, 10 , 14])

    ax1.set_ylabel("Time (s)")
    ax1.set_xlabel("Alphabet Size")
    #plt.grid(axis='y')
    ax1.grid()

    with open('increasing_size_experiments.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    x_axis = data.pop(0)
    x_axis.pop(0)

    # use total times only
    data = data[3:]

    x_axis = [int(x) for x in x_axis]
    #x_axis = list(range(2,51))
    times = []
    labels = ['DFA', 'Mealy', 'Moore']
    for r in data:
        row_name = r.pop(0)
        times.extend([float(i) for i in r])
        ax2.plot(x_axis, [float(i) for i in r], label=labels.pop(0))

    ax2.legend()
    ax2.set_xticks([100, 1000, 2000, 3000, 4000, 5000,], minor=False)
    ax2.set_yticks([min(times), 0.5, 1, 1.5, max(times)], minor=False)
    #plt.yticks([min(times), 1, 2, max(times)])
    #plt.yticks([min(times), 3, 6, 10 , 14])

    ax2.set_ylabel("Time (s)")
    ax2.set_xlabel("Automaton Size")
    #plt.grid(axis='y')
    ax2.grid()

    # matplotlib.use("pgf")
    # matplotlib.rcParams.update({
    #     "pgf.texsystem": "pdflatex",
    #     'font.family': 'serif',
    #     'text.usetex': True,
    #     'pgf.rcfonts': False,
    # })
    #
    # plt.savefig('state_increase_runtime.pgf')

    fig.tight_layout()
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    fig.savefig('both_images.pgf',bbox_inches='tight')


def plot_together():
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(10, 3))

    with open('increasing_alphabet_experiments.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    x_axis = data.pop(0)[:-1]
    x_axis.pop(0)

    # use total times only

    x_axis = [int(x) for x in x_axis]
    # x_axis = list(range(2,51))
    times = []
    labels = ['DFA', 'Mealy', 'Moore']
    for r in data:
        row_name = r.pop(0)
        times.extend([float(i) for i in r])
        ax1.plot(x_axis, [float(i) for i in r], label=labels.pop(0))

    ax1.legend()
    ax1.set_xticks([5, 25, 50, 75, 100], minor=False)
    ax1.set_yticks([min(times), 1, 2.5, 4, max(times)], minor=False)
    # plt.yticks([min(times), 1, 2, max(times)])
    # plt.yticks([min(times), 3, 6, 10 , 14])

    ax1.set_ylabel("Time (s)")
    ax1.set_xlabel("Alphabet Size")
    # plt.grid(axis='y')
    ax1.grid()

    with open('increasing_size_experiments.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    x_axis = data.pop(0)
    x_axis.pop(0)

    # use total times only
    data = data[3:]

    x_axis = [int(x) for x in x_axis]
    # x_axis = list(range(2,51))
    times = []
    labels = ['DFA', 'Mealy', 'Moore']
    for r in data:
        row_name = r.pop(0)
        times.extend([float(i) for i in r])
        ax2.plot(x_axis, [float(i) for i in r], label=labels.pop(0))

    ax2.legend()
    ax2.set_xticks([100, 1000, 2000, 3000, 4000, 5000, ], minor=False)
    ax2.set_yticks([min(times), 0.5, 1, 1.5, max(times)], minor=False)
    # plt.yticks([min(times), 1, 2, max(times)])
    # plt.yticks([min(times), 3, 6, 10 , 14])

    ax2.set_ylabel("Time (s)")
    ax2.set_xlabel("Automaton Size")
    # plt.grid(axis='y')
    ax2.grid()

    # matplotlib.use("pgf")
    # matplotlib.rcParams.update({
    #     "pgf.texsystem": "pdflatex",
    #     'font.family': 'serif',
    #     'text.usetex': True,
    #     'pgf.rcfonts': False,
    # })
    #
    # plt.savefig('state_increase_runtime.pgf')

    fig.tight_layout()
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    fig.savefig('both_images.pgf', bbox_inches='tight')

# queries_mealy_sizes, 100,500,1000,2000
# learnlib, 3356.14, 22316.74, 49037.52, 106613.32
# aalpy, 2255.1, 6025.8, 8037.85, 12372.5

def plot_together_learnlib_comp():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

    with open('learnlib_com.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    x_axis = data.pop(0)[:-1]
    x_axis.pop(0)

    x_axis = [int(x) for x in x_axis]
    # x_axis = list(range(2,51))
    times = []
    labels = ['DFA(AALpy)','DFA(LearnLib)',  'Mealy(AALpy)', 'Mealy(Learnlib)']
    for r in data:
        row_name = r.pop(0)
        times.extend([float(i) for i in r])
        ax1.plot(x_axis, [float(i) for i in r], label=labels.pop(0))

    ax1.legend()
    ax1.set_xticks([100, 1000, 2000, 3000, 4000, 5000], minor=False)
    ax1.set_yticks([min(times), 0.5, 1, 1.5, max(times)], minor=False)
    # plt.yticks([min(times), 1, 2, max(times)])
    # plt.yticks([min(times), 3, 6, 10 , 14])

    ax1.set_ylabel("Time (s)")
    ax1.set_xlabel("Automaton Size")
    # plt.grid(axis='y')
    ax1.grid()

    with open('learnlib_alph_comp.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    x_axis = data.pop(0)
    x_axis.pop(0)

    # use total times only
    x_axis = [int(x) for x in x_axis]
    # x_axis = list(range(2,51))
    times = []
    labels = ['DFA(AALpy)', 'DFA(LearnLib)', 'Mealy(AALpy)', 'Mealy(Learnlib)']
    for r in data:
        row_name = r.pop(0)
        times.extend([float(i) for i in r])
        ax2.plot(x_axis, [float(i) for i in r], label=labels.pop(0))

    ax2.legend()
    ax2.set_xticks([5, 25, 50, 75, 100], minor=False)
    ax2.set_yticks([min(times), 1, 2.5, 4, max(times)], minor=False)
    # plt.yticks([min(times), 1, 2, max(times)])
    # plt.yticks([min(times), 3, 6, 10 , 14])

    ax2.set_ylabel("Time (s)")
    ax2.set_xlabel("Alphabet Size")
    # plt.grid(axis='y')
    ax2.grid()

    # matplotlib.use("pgf")
    # matplotlib.rcParams.update({
    #     "pgf.texsystem": "pdflatex",
    #     'font.family': 'serif',
    #     'text.usetex': True,
    #     'pgf.rcfonts': False,
    # })
    #
    # plt.savefig('state_increase_runtime.pgf')

    fig.tight_layout()
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    fig.savefig('learnlib_comp.pgf', bbox_inches='tight')


def plot_runtime_steps():
    automaton_sizes = list(range(2,100,2))
    learnlib = [84.0, 84.0, 308.0, 452.0, 516.0, 564.0, 836.0, 996.0, 1188.0, 1268.0, 1870.5, 1604.0, 2358.46, 2004.0, 3189.12, 2699.32, 3756.52, 3617.64, 3925.64, 5144.98, 4278.78, 5198.06, 5452.32, 5272.12, 4841.46, 6046.92, 5864.2, 6043.34, 7501.06, 8291.58, 6976.76, 7722.48, 8498.26, 8021.4, 8409.82, 8427.62, 10348.0, 9017.5, 11266.08, 10496.48, 10948.76, 11697.94, 12539.88, 12573.5, 12266.44, 12788.36, 13619.44, 13204.7, 14218.14]
    aalpy = [80, 80, 304, 448, 512, 560, 832, 992, 1184, 1264, 1904.4666666666667, 1600, 2355, 2000, 3304.4, 2969, 3875.8, 3491.8, 3994.6666666666665, 5259.8, 4122.6, 5408.2, 5425.6, 5457.733333333334, 4700.6, 5575.266666666666, 5877.333333333333, 5643.333333333333, 7848, 8746.4, 6902, 6764.133333333333, 8968.266666666666, 8994.666666666666, 8548.066666666668, 8034.2, 10303.8, 9518.2, 11861.6, 10639.466666666667, 11049.466666666667, 12028.666666666666, 12798.4, 13055.733333333334, 12525.4, 13473.6, 13445.866666666667, 13654.866666666667, 15162.066666666668]

    learnlib_dfa_steps = [12, 176, 366, 817.517349244473, 729.3582188142311, 862.8326032572571, 1629.7225991534913, 2278.6612662357325, 1851.9775842649628, 2256.358673947518, 2877.186267318514, 3621.257762117678, 4309.52400476166, 3957.7026357704567, 5892.417280455997, 4340.934611960256, 5361.8053640093285, 5826.332838306764, 6170.173413693644, 6362.896456856769, 7467.5489502843675, 10647.813901891837, 7771.6990409704595, 10021.521929114147, 11682.322172805243, 13337.927737932912, 12739.746892560444, 14478.86166799318, 12871.644461597713, 13099.900598108483, 11801.712844210842, 14545.968533421541, 16248.345494832956, 16266.864422969868, 15963.905452213141, 19195.796973001674, 17252.921273408057, 18935.661631940457, 17732.29013734688, 19960.551029106155, 17375.00727356809, 18493.587556335762, 22476.68693665887, 20993.560784351062, 24736.797016941302, 23454.586313063137, 21647.451883062633, 29979.506261659953, 25118.61300496266]

    aalpy_dfa_steps = [12, 176.6, 366.93333333333334, 790.5333333333333, 840.5333333333333, 886.1333333333333, 1657.8,
                 2225.133333333333, 2076.3333333333335, 2649.3333333333335, 3065, 3218.0666666666666, 4783.466666666666,
                 3860.3333333333335, 5724.4, 4960.733333333334, 5226.733333333334, 6479.4, 5991.866666666667, 7096.2,
                 7241.866666666667, 9292.066666666668, 8487.6, 10789.4, 11765.2, 12693.666666666666, 11715.266666666666,
                 12860.6, 11887.466666666667, 12740.333333333334, 12882.133333333333, 15290.666666666666, 15572,
                 14563.066666666668, 14886.8, 17937.466666666667, 16332.533333333333, 20078.6, 18581.8,
                 18823.266666666666, 20331.733333333334, 20824.733333333334, 19940.8, 23263.666666666668,
                 21659.466666666667, 25069.933333333334, 24592.333333333332, 27892.2, 24161.066666666666]
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Automaton Size')
    ax1.set_ylabel('Number of Learning Steps')
    ax1.plot(automaton_sizes, aalpy_dfa_steps, label='DFA (AALpy)')
    ax1.plot(automaton_sizes, learnlib_dfa_steps, label='DFA (Learnlib)')
    ax1.plot(automaton_sizes, aalpy, label='Mealy (AALpy)')
    ax1.plot(automaton_sizes, learnlib, label='Mealy (Learnlib)')
    # ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    #
    ax2.set_ylabel('Total Learning Time (ms)')  # we already handled the x-label with ax1
    ax2.plot(automaton_sizes, [e * 50 for e in aalpy_dfa_steps], label='DFA (AALpy)')
    ax2.plot(automaton_sizes, [e * 50 for e in learnlib_dfa_steps], label='DFA (LearnLib)')
    ax2.plot(automaton_sizes, [e * 50 for e in aalpy], label='Mealy (AALpy)')
    ax2.plot(automaton_sizes, [e * 50 for e in learnlib], label='Mealy (LearnLib)')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid()
    plt.legend()
    # matplotlib.use("pgf")
    # matplotlib.rcParams.update({
    #     "pgf.texsystem": "pdflatex",
    #     'font.family': 'serif',
    #     'text.usetex': True,
    #     'pgf.rcfonts': False,
    #     #'font.size': 42
    # })

    import tikzplotlib
    tikzplotlib.save("test.txt")
    #fig.savefig('runtime_comp.pgf', bbox_inches='tight')

plot_runtime_steps()
#plot_runtime_steps()
# plot_together_learnlib_comp()
#plot_increasing_size_exp()
#plot_increasing_alphabeth_exp()