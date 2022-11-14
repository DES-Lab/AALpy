import time

from aalpy.base import SUL, Oracle
from aalpy.learning_algs.non_deterministic.NonDeterministicSULWrapper import NonDeterministicSULWrapper
from aalpy.learning_algs.non_deterministic.OnfsmObservationTable import NonDetObservationTable
from aalpy.utils.HelperFunctions import print_learning_info, print_observation_table, \
    get_available_oracles_and_err_msg, all_suffixes

print_options = [0, 1, 2, 3]

available_oracles, available_oracles_error_msg = get_available_oracles_and_err_msg()


def run_non_det_Lstar(alphabet: list, sul: SUL, eq_oracle: Oracle, n_sampling=5, samples=None, stochastic=False,
                      max_learning_rounds=None, return_data=False, print_level=2):
    """
    A ONFSM learning algorithm that does not rely on all weather assumption (once an input is queried, all possible
    outputs are observed).

    Args:

        alphabet: input alphabet

        sul: system under learning

        eq_oracle: equivalence oracle

        n_sampling: number of times that each cell has to be updated. If this number is to low, all-weather condition
            will not hold and learning will not converge to the correct model. (Default value = 50)

        samples: input output sequances provided to learning algorithm. List of ((input sequence), (output sequence)).

        stochastic: if True, non deterministic learning will be performed but probabilities will be added to the
        returned model, making it a stochastic Mealy machine

        max_learning_rounds: if max_learning_rounds is reached, learning will stop (Default value = None)

        return_data: if True, map containing all information like number of queries... will be returned
            (Default value = False)

        print_level: 0 - None, 1 - just results, 2 - current round and hypothesis size, 3 - educational/debug
            (Default value = 2)

    Returns:
        learned ONFSM

    """

    start_time = time.time()
    eq_query_time = 0
    learning_rounds = 0

    sul = NonDeterministicSULWrapper(sul)

    if samples:
        for inputs, outputs in samples:
            sul.cache.add_trace(inputs, outputs)

    eq_oracle.sul = sul

    ot = NonDetObservationTable(alphabet, sul, n_sampling)

    # Keep track of last counterexample and last hypothesis size
    # With this data we can check if the extension of the E set lead to state increase
    last_cex = None

    hypothesis = None

    while True:
        if max_learning_rounds and learning_rounds - 1 == max_learning_rounds:
            break

        ot.S = list()
        ot.S.append((tuple(), tuple()))
        ot.query_missing_observations()

        row_to_close = ot.get_row_to_close()
        while row_to_close is not None:
            ot.query_missing_observations()
            row_to_close = ot.get_row_to_close()
            ot.clean_obs_table()

        hypothesis = ot.gen_hypothesis()

        if counterexample_not_valid(hypothesis, last_cex):
            cex = sul.cache.find_cex_in_cache(hypothesis)

            if cex is None:
                learning_rounds += 1
                # Find counterexample
                if print_level > 1:
                    print(f'Hypothesis {learning_rounds}: {len(hypothesis.states)} states.')

                if print_level == 3:
                    print_observation_table(ot, 'non-det')

                eq_query_start = time.time()
                cex = eq_oracle.find_cex(hypothesis)
                eq_query_time += time.time() - eq_query_start

            last_cex = cex
        else:
            cex = last_cex

        if cex is None:
            break
        else:
            cex_suffixes = all_suffixes(cex[0])
            for suffix in cex_suffixes:
                if suffix not in ot.E:
                    ot.E.append(suffix)
                    break

    if stochastic:
        hypothesis = ot.gen_hypothesis(stochastic=True)

    total_time = round(time.time() - start_time, 2)
    eq_query_time = round(eq_query_time, 2)
    learning_time = round(total_time - eq_query_time, 2)

    info = {
        'learning_rounds': learning_rounds,
        'automaton_size': len(hypothesis.states),
        'queries_learning': sul.num_queries,
        'steps_learning': sul.num_steps,
        'queries_eq_oracle': eq_oracle.num_queries,
        'steps_eq_oracle': eq_oracle.num_steps,
        'learning_time': learning_time,
        'eq_oracle_time': eq_query_time,
        'total_time': total_time
    }

    if print_level > 0:
        print_learning_info(info)

    if return_data:
        return hypothesis, info

    return hypothesis


def counterexample_not_valid(hypothesis, cex):
    if cex is None:
        return True
    hypothesis.reset_to_initial()
    for i, o in zip(cex[0], cex[1]):
        out = hypothesis.step_to(i, o)
        if out is None:
            return False
    return True
