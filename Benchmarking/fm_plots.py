

def plot_error():
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

    # MDP then SMM
    learning_time_data = [
        [68.19, 140.31, 154.35, 116.8],
        [27.5, 98.31, 30.87, 68]
    ]

    num_mq_data = [
        [81803.23, 153758.15, 560705.92, 248552.62, ],
        [36937.54, 91309.08, 51791.54, 92607]
    ]

    import numpy as np

    N = 4

    ind = np.arange(N)  # the x locations for the groups
    width = 0.25  # the width of the bars

    # fig = plt.figure()
    fig, (ax_time, ax_mq) = plt.subplots(1, 2, figsize=(10, 3))

    ax_time.bar(ind, learning_time_data[0], width, label='MDP')
    ax_time.bar(ind + width, learning_time_data[1], width, label='SMM')

    # add some
    ax_time.set_ylabel('Learning Time (s)')

    ax_time.set_xticks(ind + width / 2)
    ax_time.set_xticklabels(('35 State\nGridworld', '72 State\nGridworld', 'MQTT', 'TCP',))

    ax_time.grid(axis='y')
    ax_time.legend(loc='upper left')

    ax_mq.bar(ind, num_mq_data[0], width, label='MDP')
    ax_mq.bar(ind + width, num_mq_data[1], width, label='SMM')

    # add some
    ax_mq.set_ylabel('\# Membership Queries')
    ax_mq.ticklabel_format(axis='y', style='sci', scilimits=(1, 4))
    ax_mq.set_xticks(ind + width / 2)
    ax_mq.set_xticklabels(('35 State\nGridworld', '72 State\nGridworld', 'MQTT', 'TCP',))
    ax_mq.legend(loc='upper left')


    ax_mq.grid(axis='y')
    fig.tight_layout()

    # plt.show()

    plt.savefig("error_bench.pgf", bbox_inches='tight')

    import tikzplotlib

    tikzplotlib.save("error_bench.tex")

def plot_benchmarks():
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

    # MDP then SMM

    num_mq_data = [
        [81803.23, 153758.15, 560705.92, 248552.62, ],
        [36937.54, 91309.08, 51791.54, 92607]
    ]

    # TODO
    avr_cum_err = [
        
    ]

    import numpy as np

    N = 4

    ind = np.arange(N)  # the x locations for the groups
    width = 0.25  # the width of the bars

    # fig = plt.figure()
    fig, (ax_time, ax_mq) = plt.subplots(1, 2, figsize=(10, 3))

    ax_time.bar(ind, avr_cum_err[0], width, label='MDP')
    ax_time.bar(ind + width, avr_cum_err[1], width, label='SMM')

    # add some
    ax_time.set_ylabel('Learning Time (s)')

    ax_time.set_xticks(ind + width / 2)
    ax_time.set_xticklabels(('35 State\nGridworld', '72 State\nGridworld', 'MQTT', 'TCP',))

    ax_time.grid(axis='y')
    ax_time.legend(loc='upper left')

    ax_mq.bar(ind, num_mq_data[0], width, label='MDP')
    ax_mq.bar(ind + width, num_mq_data[1], width, label='SMM')

    # add some
    ax_mq.set_ylabel('\# Membership Queries')
    ax_mq.ticklabel_format(axis='y', style='sci', scilimits=(1, 4))
    ax_mq.set_xticks(ind + width / 2)
    ax_mq.set_xticklabels(('35 State\nGridworld', '72 State\nGridworld', 'MQTT', 'TCP',))

    ax_mq.legend(loc='upper left')

    ax_mq.grid(axis='y')
    fig.tight_layout()

    # plt.show()

    plt.savefig("benchmarking.pgf", bbox_inches='tight')

    import tikzplotlib

    tikzplotlib.save("benchmarking.tex")

if __name__ == '__main__':
    plot_error()