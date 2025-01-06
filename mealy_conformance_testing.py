import os
import sys
import pathlib
import numpy as np
import multiprocessing as mp

# import argument parser
import argparse

from aalpy.oracles.WMethodEqOracle import (
    WMethodEqOracle,
    WMethodDiffFirstEqOracle,
    RandomWMethodEqOracle,
)
from aalpy.oracles import PerfectKnowledgeEqOracle
from aalpy.oracles import StatePrefixEqOracle
from aalpy.oracles.StochasticStateCoverageEqOracle import (
    StochasticStateCoverageEqOracle,
)
from aalpy.SULs.AutomataSUL import AutomatonSUL
from aalpy.base.SUL import CacheSUL
from aalpy.learning_algs.deterministic.LStar import run_Lstar
from aalpy.learning_algs.deterministic.KV import run_KV
from aalpy.utils.FileHandler import load_automaton_from_file, save_automaton_to_file

# print up to 1 decimal point
np.set_printoptions(precision=1)
# do not print in scientific notation
np.set_printoptions(suppress=True)
# print up to 3 decimal point
# pd.options.display.float_format = '{:.3f}'.format

WALKS_PER_ROUND = {
    "TCP": 200000,  # tcp is large, it is learned in multiple rounds
    "TLS": 2000,  # tls is tiny, it is learned in one round
    "MQTT": 2000,  # this is also small, but it is not learned in one round
}
WALK_LEN = {"TCP": 70, "TLS": 50, "MQTT": 50}

METHOD_TO_ORACLES = {
    "wmethod": 2,
    "state_coverage": 5,
}

PROTOCOL_TO_MAX_MODEL_SIZE = {
    "TCP": 60,
    "TLS": 10,
    "MQTT": 20,
}


def process_oracle(alphabet, sul, oracle, correct_size, i):
    """
    Process the oracle and return the number of queries to the equivalence and membership oracles
    and whether the learned model has the correct size.

    Args:
        alphabet: input alphabet
        sul: system under learning
        oracle: equivalence oracle
        correct_size: correct size of the model
        i: index of the oracle
    """
    _, info = run_Lstar(alphabet, sul, oracle, "mealy", return_data=True, print_level=2)
    # _, info = run_KV(alphabet, sul, oracle, 'mealy', return_data=True, print_level=0)
    return (
        i,
        info["queries_eq_oracle"],
        info["queries_learning"],
        1 if info["automaton_size"] != correct_size else 0,
        info["intermediate_hypotheses"],
        info["counterexamples"],
    )


def do_learning_experiments(model, alphabet, correct_size, prot):
    """
    Perform the learning experiments for the given model and alphabet.

    Args:
        model: model to learn
        alphabet: input alphabet correct_size: correct size of the model
    """
    # create a copy of the SUL for each oracle
    suls = [AutomatonSUL(model) for _ in range(NUM_ORACLES)]
    wpr = WALKS_PER_ROUND[prot]
    wl = WALK_LEN[prot]
    # initialize the oracles
    if BASE_METHOD == "wmethod":
        max_size = PROTOCOL_TO_MAX_MODEL_SIZE[prot]
        eq_oracles = [
            WMethod(alphabet, suls[0], max_size),
            WMethodDiffFirst(alphabet, suls[1], max_size),
        ]
    elif BASE_METHOD == "state_coverage":
        eq_oracles = [
            StochasticRandom(alphabet, suls[0], wpr, wl),
            StochasticLinear(alphabet, suls[1], wpr, wl),
            StochasticSquare(alphabet, suls[2], wpr, wl),
            StochasticExponential(alphabet, suls[3], wpr, wl),
            StochasticInverse(alphabet, suls[4], wpr, wl),
        ]
    else:
        raise ValueError("Unknown base method")

    assert len(suls) == len(eq_oracles), "Number of oracles and SULs must be the same."
    assert NUM_ORACLES == len(
        eq_oracles
    ), "Number of oracles must be the same as the number of methods."

    if PARALLEL:
        # create the arguments for eache oracle's task
        tasks = [
            (alphabet, sul, oracle, correct_size, i)
            for i, (sul, oracle) in enumerate(zip(suls, eq_oracles))
        ]

        with mp.Pool(NUM_ORACLES) as pool:
            results = pool.starmap(process_oracle, tasks)
    else:
        results = [
            process_oracle(alphabet, sul, oracle, correct_size, i)
            for i, (sul, oracle) in enumerate(zip(suls, eq_oracles))
        ]

    return results


def main():
    ROOT = os.getcwd() + "/DotModels"
    # PROTOCOLS    = ["ASML", "TLS", "MQTT", "EMV", "TCP"]
    PROTOCOLS = ["TLS", "MQTT"]
    DIRS = [pathlib.Path(ROOT + "/" + prot) for prot in PROTOCOLS]
    FILES = [file for dir in DIRS for file in dir.iterdir()]
    FILES_PER_PROT = {
        prot: len([file for file in DIRS[i].iterdir()])
        for i, prot in enumerate(PROTOCOLS)
    }
    MODELS = (load_automaton_from_file(f, "mealy") for f in FILES)

    EQ_QUERIES = np.zeros((len(FILES), TIMES, NUM_ORACLES))
    MB_QUERIES = np.zeros((len(FILES), TIMES, NUM_ORACLES))
    FAILURES = np.zeros((len(FILES), TIMES, NUM_ORACLES))

    # iterate over the models
    for index, (model, file) in enumerate(zip(MODELS, FILES)):
        # these variables can be shared among the processes
        prot = file.parent.stem
        correct_size = model.size
        alphabet = list(model.get_input_alphabet())
        # repeat the experiments to gather statistics
        for trial in range(TIMES):

            results = do_learning_experiments(model, alphabet, correct_size, prot)

            for i, eq_queries, mb_queries, failure, hyps, cexs in results:
                EQ_QUERIES[index, trial, i] = eq_queries
                MB_QUERIES[index, trial, i] = mb_queries
                FAILURES[index, trial, i] = failure

                if SAVE_INTERMEDIATE_HYPOTHESES:
                    MODEL_RES_DIR = f"./results/{BASE_METHOD}/{prot}/{file.stem}/trial_{trial}/oracle_{i}"
                    if not os.path.exists(MODEL_RES_DIR):
                        os.makedirs(MODEL_RES_DIR)
                    for i, hyp, cex in enumerate(zip(hyps, cexs)):
                        save_automaton_to_file(hyp, f"{MODEL_RES_DIR}/h{i}.dot", "dot")
                        with open(f"{MODEL_RES_DIR}/cex{i}.txt", "w") as f:
                            f.write(str(cex))

    prev = 0
    for prot in PROTOCOLS:
        items = FILES_PER_PROT[prot]
        np.save(
            f"./results/{BASE_METHOD}/eq_queries_{prot}.npy",
            EQ_QUERIES[prev : prev + items, :, :],
        )
        np.save(
            f"./results/{BASE_METHOD}/mb_queries_{prot}.npy",
            MB_QUERIES[prev : prev + items, :, :],
        )
        np.save(
            f"./results/{BASE_METHOD}/failures_{prot}.npy",
            FAILURES[prev : prev + items, :, :],
        )
        prev += items

    for array, name in zip(
        [EQ_QUERIES, MB_QUERIES, FAILURES], ["eq_queries", "mb_queries", "failures"]
    ):
        averages = np.mean(array, axis=1)
        std_devs = np.std(array, axis=1)

        np.save(f"./results/{BASE_METHOD}/{name}.npy", array)
        np.save(f"./results/{BASE_METHOD}/{name}_averages.npy", averages)
        np.save(f"./results/{BASE_METHOD}/{name}_std_devs.npy", std_devs)
        if not "failures" == name:
            s1_scores = np.sum(averages, axis=0)
            maxima = np.max(averages, axis=1)
            s2_scores = np.sum(averages / maxima[:, np.newaxis], axis=0)

            np.save(f"./results/{BASE_METHOD}/{name}_s1_scores.npy", s1_scores)
            np.save(f"./results/{BASE_METHOD}/{name}_s2_scores.npy", s2_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse arguments for running learning experiments."
    )

    parser.add_argument(
        "-p",
        "--parallel",
        action="store_true",
        default=False,
        help="Run the experiments in parallel or not. Defaults to False.",
    )

    parser.add_argument(
        "-t",
        "--times",
        type=int,
        default=30,
        help="Number of times to run the stochastic experiments. Defaults to 30.",
    )

    parser.add_argument(
        "-b",
        "--base_method",
        type=str,
        choices=["state_coverage", "wmethod"],
        default="state_coverage",
        help="Base method to use. Can be 'state_coverage' or 'wmethod'. Defaults to 'state_coverage'.",
    )

    parser.add_argument(
        "-s",
        "--save_intermediate",
        action="store_true",
        default=False,
        help="Save intermediate results or not. Defaults to False.",
    )

    args = parser.parse_args()
    TIMES = args.times
    PARALLEL = args.parallel
    BASE_METHOD = args.base_method
    SAVE_INTERMEDIATE_HYPOTHESES = args.save_intermediate

    NUM_ORACLES = METHOD_TO_ORACLES[BASE_METHOD]

    if BASE_METHOD == "state_coverage":

        class StochasticRandom(StochasticStateCoverageEqOracle):
            def __init__(
                self, alphabet, sul, walks_per_round, walk_len, prob_function="random"
            ):
                super().__init__(
                    alphabet, sul, walks_per_round, walk_len, prob_function
                )

        class StochasticLinear(StochasticStateCoverageEqOracle):
            def __init__(
                self, alphabet, sul, walks_per_round, walk_len, prob_function="linear"
            ):
                super().__init__(
                    alphabet, sul, walks_per_round, walk_len, prob_function
                )

        class StochasticSquare(StochasticStateCoverageEqOracle):
            def __init__(
                self, alphabet, sul, walks_per_round, walk_len, prob_function="square"
            ):
                super().__init__(
                    alphabet, sul, walks_per_round, walk_len, prob_function
                )

        class StochasticExponential(StochasticStateCoverageEqOracle):
            def __init__(
                self,
                alphabet,
                sul,
                walks_per_round,
                walk_len,
                prob_function="exponential",
            ):
                super().__init__(
                    alphabet, sul, walks_per_round, walk_len, prob_function
                )

        def user(x, size):
            fundamental = 0.5 / (1 - 0.5**size)
            return fundamental * (0.5**x)

        class StochasticInverse(StochasticStateCoverageEqOracle):
            def __init__(
                self,
                alphabet,
                sul,
                walks_per_round,
                walk_len,
                prob_function="user",
                user=user,
            ):
                super().__init__(
                    alphabet, sul, walks_per_round, walk_len, prob_function, user
                )

    elif BASE_METHOD == "wmethod":
        TIMES = 1  # WMethod is deterministic

        class WMethod(WMethodEqOracle):
            def __init__(self, alphabet, sul, max_model_size):
                super().__init__(alphabet, sul, max_model_size)

        class WMethodDiffFirst(WMethodDiffFirstEqOracle):
            def __init__(self, alphabet, sul, max_model_size):
                super().__init__(alphabet, sul, max_model_size)

    main()
