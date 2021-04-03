import time

from aalpy.base import SUL, Oracle
from aalpy.learning_algs.stochastic.DifferenceChecker import AdvancedHoeffdingChecker, HoeffdingChecker
from aalpy.learning_algs.stochastic.SamplingBasedObservationTable import SamplingBasedObservationTable
from aalpy.learning_algs.stochastic.StochasticCexProcessing import stochastic_longest_prefix, stochastic_rs
from aalpy.learning_algs.stochastic.StochasticTeacher import StochasticTeacher
from aalpy.utils.HelperFunctions import print_learning_info, print_observation_table, get_cex_prefixes

strategies = ['normal', 'no_cq']
cex_sampling_options = [None, 'bfs', 'random']
print_options = [0, 1, 2, 3]


def run_stochastic_Lstar(input_alphabet, sul: SUL, eq_oracle: Oracle, n_c=20, n_resample=100, min_rounds=10,
                         max_rounds=200, automaton_type='mdp', strategy='normal', cex_processing=None,
                         samples_cex_strategy=None, return_data=False, print_level=2):
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

        cex_processing: cex processing strategy, None , 'longest_prefix' or 'rs' (rs is experimental)

        samples_cex_strategy: strategy for finding counterexamples in the trace tree

        return_data: if True, map containing all information like number of queries... will be returned
            (Default value = False)

        print_level: 0 - None, 1 - just results, 2 - current round and hypothesis size, 3 - educational/debug
            (Default value = 2)


    Returns:
      learned MDP/SMM

    """

    assert strategy in strategies
    assert samples_cex_strategy in cex_sampling_options

    compatibility_checker = AdvancedHoeffdingChecker() if strategy != "normal" else HoeffdingChecker()

    stochastic_teacher = StochasticTeacher(sul, n_c, eq_oracle, automaton_type, compatibility_checker,
                                           samples_cex_strategy=samples_cex_strategy)

    # This way all steps from eq. oracle will be added to the tree
    eq_oracle.sul = stochastic_teacher.sul

    observation_table = SamplingBasedObservationTable(input_alphabet, automaton_type,
                                                      stochastic_teacher, compatibility_checker=compatibility_checker,
                                                      strategy=strategy,
                                                      cex_processing=cex_processing)

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
        # cex = observation_table.chaos_counterexample(hypothesis)
        chaos_cex_present = observation_table.chaos_counterexample(hypothesis)

        if not chaos_cex_present:
            if automaton_type == 'mdp':
                hypothesis.states.remove(next(state for state in hypothesis.states if state.output == 'chaos'))
            else:
                hypothesis.states.remove(next(state for state in hypothesis.states if state.state_id == 'chaos'))

        if print_level > 1:
            print(f'Hypothesis: {learning_rounds}: {len(hypothesis.states)} states.')

        if print_level == 3:
            print_observation_table(observation_table.S, observation_table.get_extended_s(), observation_table.E,
                                    observation_table.T, False)

        # If there is a prefix leading to chaos state, use that as a counterexample, otherwise preform equivalence query
        cex = None
        if not chaos_cex_present:
            eq_query_start = time.time()
            cex = stochastic_teacher.equivalence_query(hypothesis)
            eq_query_time += time.time() - eq_query_start

        if cex:
            if print_level == 3:
                print('Counterexample', cex)
            # get all prefixes and add them to the S set
            if cex_processing is None:
                for pre in get_cex_prefixes(cex, automaton_type):
                    if pre not in observation_table.S:
                        observation_table.S.append(pre)
            else:
                suffixes = None
                if cex_processing == 'longest_prefix':
                    prefixes = observation_table.S + list(observation_table.get_extended_s())
                    suffixes = stochastic_longest_prefix(cex, prefixes)
                elif cex_processing == 'rs':
                    suffixes = stochastic_rs(sul, cex, hypothesis)
                for suf in suffixes:
                    if suf not in observation_table.E:
                        observation_table.E.append(suf)

        # print('Size of E:', len(observation_table.E))
        # Ask queries for non-completed cells and update the observation table
        refined = observation_table.refine_not_completed_cells(n_resample)
        observation_table.update_obs_table_with_freq_obs()

        if observation_table.stop(learning_rounds, chaos_present=chaos_cex_present, min_rounds=min_rounds,
                                  max_rounds=max_rounds, print_unambiguity=print_level > 1):
            break

        if not refined:
            # If all cells were refined, but stopping did not happen, increase n_c
            stochastic_teacher.n_c *= 1.5
            stochastic_teacher.complete_query_cache.clear()

    total_time = round(time.time() - start_time, 2)
    eq_query_time = round(eq_query_time, 2)
    learning_time = round(total_time - eq_query_time, 2)

    info = {
        'learning_rounds': learning_rounds,
        'automaton_size': len(hypothesis.states),
        'queries_learning': stochastic_teacher.sul.num_queries - eq_oracle.num_queries,
        'steps_learning': stochastic_teacher.sul.num_steps - eq_oracle.num_queries,
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
