import time

from aalpy.base import SUL, Oracle
from aalpy.learning_algs.stochastic.SamplingBasedObservationTable import SamplingBasedObservationTable
from aalpy.learning_algs.stochastic.StochasticTeacher import StochasticTeacher
from aalpy.utils.HelperFunctions import print_learning_info, print_observation_table


def get_cex_prefixes(cex, automaton_type):
    """

    Args:
        cex: counterexample
        automaton_type: `mdp` or `smm`

    Returns:

        all prefixes of the counterexample based on the `automaton_type`
    """
    if automaton_type == 'mdp':
        return [tuple(cex[:i + 1]) for i in range(0, len(cex), 2)]
    return [tuple(cex[:i]) for i in range(0, len(cex), 2)]


strategies = ['normal','no_cq']
print_options = [0, 1, 2, 3]


def run_stochastic_Lstar(input_alphabet, sul: SUL, eq_oracle: Oracle, n_c=20, n_resample=100, min_rounds=10,
                         max_rounds=200, automaton_type='mdp', strategy='normal', return_data=False, print_level=2):
    """
    Learning of Markov Decision Processes based on 'L*-Based Learning of Markov Decision Processes' by Tappler et al.

    Args:

        input_alphabet: input alphabet

        sul: system under learning

        eq_oracle: equivalence oracle

        n_c: cutoff for a cell to be considered complete (Default value = 20)

        n_resample: resampling size (Default value = 100)

        min_rounds: minimum number of learning rounds (Default value = 10)

        max_rounds: if learning_rounds >= max_rounds, learning will stop (Default value = 200)

        automaton_type: either 'mdp' or 'smm' (Default value = 'mdp')

        strategy: if no_cq, improved version of the algorithm will be used (Default value = 'normal')

        return_data: if True, map containing all information like number of queries... will be returned
            (Default value = False)

        print_level: 0 - None, 1 - just results, 2 - current round and hypothesis size, 3 - educational/debug
            (Default value = 2)


    Returns:
      learned MDP/SMM

    """

    assert strategy in strategies
    # Initialize teacher and observation table
    stochastic_teacher = StochasticTeacher(sul, n_c, eq_oracle, automaton_type)
    observation_table = SamplingBasedObservationTable(input_alphabet, automaton_type,
                                                      stochastic_teacher, strategy=strategy)

    start_time = time.time()
    eq_query_time = 0

    # Ask queries for non-completed cells and update the observation table
    observation_table.refine_not_completed_cells(n_resample, uniform=True)
    observation_table.update_obs_table_with_freq_obs()

    learning_rounds = 0
    while True:
        learning_rounds += 1

        observation_table.make_closed_and_consistent()

        hypothesis = observation_table.generate_hypothesis()

        observation_table.trim(hypothesis)

        # If there is no chaos state is not reachable, remove it from state set
        cex = observation_table.chaos_counterexample(hypothesis)
        chaos_cex_present = True if cex else False

        if not cex:
            if automaton_type == 'mdp':
                hypothesis.states.remove(next(state for state in hypothesis.states if state.output == 'chaos'))
            else:
                hypothesis.states.remove(next(state for state in hypothesis.states if state.state_id == 'chaos'))

        if print_level == 3:
            print_observation_table(observation_table.S,observation_table.get_extended_s(), observation_table.E,
                                observation_table.T, False)

        if print_level > 1:
            print(f'Hypothesis: {learning_rounds}: {len(hypothesis.states)} states.')

        # If there is a prefix leading to chaos state, use that as a counterexample, otherwise preform equivalence query
        eq_query_start = time.time()
        cex = stochastic_teacher.equivalence_query(hypothesis) if not cex else cex
        eq_query_time += time.time() - eq_query_start

        if cex:
            # get all prefixes and add them to the S set
            for p in get_cex_prefixes(cex, automaton_type):
                if p not in observation_table.S:
                    observation_table.S.append(p)

        # Ask queries for non-completed cells and update the observation table
        observation_table.refine_not_completed_cells(n_resample)
        observation_table.update_obs_table_with_freq_obs()

        if observation_table.stop(learning_rounds, chaos_present=chaos_cex_present, min_rounds=min_rounds,
                                  max_rounds=max_rounds, print_unambiguity=print_level > 1):
            break

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
