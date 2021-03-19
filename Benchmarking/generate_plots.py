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


plot_together()
#plot_increasing_size_exp()
#plot_increasing_alphabeth_exp()