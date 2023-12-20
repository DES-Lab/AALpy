import time

from aalpy.base import Oracle, SUL
from aalpy.utils.HelperFunctions import extend_set, print_learning_info, print_observation_table, all_prefixes
from .CounterExampleProcessing import longest_prefix_cex_processing, rs_cex_processing, \
    counterexample_successfully_processed, linear_cex_processing, exponential_cex_processing
from .ObservationTable import ObservationTable
from ...base.SUL import CacheSUL

counterexample_processing_strategy = [None, 'rs', 'longest_prefix', 'linear_fwd', 'linear_bwd', 'exponential_fwd',
                                      'exponential_bwd']
closedness_options = ['suffix_all', 'suffix_single']
print_options = [0, 1, 2, 3]


def run_Lstar(alphabet: list, sul: SUL, eq_oracle: Oracle, automaton_type, samples=None,
              closing_strategy='shortest_first', cex_processing='rs',
              e_set_suffix_closed=False, all_prefixes_in_obs_table=True,
              max_learning_rounds=None, cache_and_non_det_check=True, return_data=False, print_level=2):
    """
    Executes L* algorithm.

    Args:

        alphabet: input alphabet

        sul: system under learning

        eq_oracle: equivalence oracle

        automaton_type: type of automaton to be learned. Either 'dfa', 'mealy' or 'moore'.

        samples: input output traces provided to the learning algorithm. They are added to cache and could reduce
        total interaction with the system. Syntax: list of [(input_sequence, output_sequence)] or None

        closing_strategy: closing strategy used in the close method. Either 'longest_first', 'shortest_first' or
            'single' (Default value = 'shortest_first')

        cex_processing: Counterexample processing strategy. Either None, 'rs' (Riverst-Schapire), 'longest_prefix'.
            (Default value = 'rs'), 'longest_prefix', 'linear_fwd', 'linear_bwd', 'exponential_fwd', 'exponential_bwd'

        e_set_suffix_closed: True option ensures that E set is suffix closed,
            False adds just a single suffix per counterexample.

        all_prefixes_in_obs_table: if True, entries of observation table will contain the whole output of the whole
            suffix, otherwise just the last output meaning that all prefixes of the suffix will be added.
            If False, just a single suffix will be added.

        max_learning_rounds: number of learning rounds after which learning will terminate (Default value = None)

        cache_and_non_det_check: Use caching and non-determinism checks (Default value = True)

        return_data: if True, a map containing all information(runtime/#queries/#steps) will be returned
            (Default value = False)

        print_level: 0 - None, 1 - just results, 2 - current round and hypothesis size, 3 - educational/debug
            (Default value = 2)

    Returns:

        automaton of type automaton_type (dict containing all information about learning if 'return_data' is True)

    """

    assert cex_processing in counterexample_processing_strategy
    assert print_level in print_options

    if cache_and_non_det_check or samples is not None:
        # Wrap the sul in the CacheSUL, so that all steps/queries are cached
        sul = CacheSUL(sul)
        eq_oracle.sul = sul

        if samples:
            for input_seq, output_seq in samples:
                sul.cache.add_to_cache(input_seq, output_seq)

    start_time = time.time()
    eq_query_time = 0
    learning_rounds = 0
    hypothesis = None

    observation_table = ObservationTable(alphabet, sul, automaton_type, all_prefixes_in_obs_table)

    # Initial update of observation table, for empty row
    observation_table.update_obs_table()
    cex = None

    while True:
        if max_learning_rounds and learning_rounds == max_learning_rounds:
            break

        # Make observation table consistent (iff there is no counterexample processing)
        if not cex_processing:
            inconsistent_rows = observation_table.get_causes_of_inconsistency()
            while inconsistent_rows is not None:
                added_suffix = extend_set(observation_table.E, inconsistent_rows)
                observation_table.update_obs_table(e_set=added_suffix)
                inconsistent_rows = observation_table.get_causes_of_inconsistency()

        # Close observation table
        rows_to_close = observation_table.get_rows_to_close(closing_strategy)
        while rows_to_close is not None:
            rows_to_query = []
            for row in rows_to_close:
                observation_table.S.append(row)
                rows_to_query.extend([row + (a,) for a in alphabet])
            observation_table.update_obs_table(s_set=rows_to_query)
            rows_to_close = observation_table.get_rows_to_close(closing_strategy)

        # Generate hypothesis
        hypothesis = observation_table.gen_hypothesis(no_cex_processing_used=cex_processing is None)
        # Find counterexample if none has previously been found (first round) and cex is successfully processed
        # (not a counterexample in the current hypothesis)
        if cex is None or counterexample_successfully_processed(sul, cex, hypothesis):
            learning_rounds += 1

            if print_level > 1:
                print(f'Hypothesis {learning_rounds}: {len(hypothesis.states)} states.')

            if print_level == 3:
                print_observation_table(observation_table, 'det')

            eq_query_start = time.time()
            cex = eq_oracle.find_cex(hypothesis)
            eq_query_time += time.time() - eq_query_start

        # If no counterexample is found, return the hypothesis
        if cex is None:
            break

        # make sure counterexample is a tuple in case oracle returns a list
        cex = tuple(cex)

        if print_level == 3:
            print('Counterexample', cex)

        # Process counterexample and ask membership queries
        if not cex_processing:
            s_to_update = []
            added_rows = extend_set(observation_table.S, all_prefixes(cex))
            s_to_update.extend(added_rows)
            for p in added_rows:
                s_to_update.extend([p + (a,) for a in alphabet])

            observation_table.update_obs_table(s_set=s_to_update)
            continue

        elif cex_processing == 'longest_prefix':
            cex_suffixes = longest_prefix_cex_processing(observation_table.S + list(observation_table.s_dot_a()),
                                                         cex, closedness='suffix')
        elif cex_processing == 'rs':
            cex_suffixes = rs_cex_processing(sul, cex, hypothesis, e_set_suffix_closed, closedness='suffix')
        else:
            direction = cex_processing[-3:]
            if 'linear' in cex_processing:
                cex_suffixes = linear_cex_processing(sul, cex, hypothesis, e_set_suffix_closed,
                                                     direction=direction, closedness='suffix')
            else:
                cex_suffixes = exponential_cex_processing(sul, cex, hypothesis, e_set_suffix_closed,
                                                          direction=direction, closedness='suffix')

        added_suffixes = extend_set(observation_table.E, cex_suffixes)
        observation_table.update_obs_table(e_set=added_suffixes)

    total_time = round(time.time() - start_time, 2)
    eq_query_time = round(eq_query_time, 2)
    learning_time = round(total_time - eq_query_time, 2)

    info = {
        'learning_rounds': learning_rounds,
        'automaton_size': hypothesis.size,
        'queries_learning': sul.num_queries,
        'steps_learning': sul.num_steps,
        'queries_eq_oracle': eq_oracle.num_queries,
        'steps_eq_oracle': eq_oracle.num_steps,
        'learning_time': learning_time,
        'eq_oracle_time': eq_query_time,
        'total_time': total_time,
        'characterization_set': observation_table.E
    }
    if cache_and_non_det_check:
        info['cache_saved'] = sul.num_cached_queries

    if print_level > 0:
        print_learning_info(info)

    if return_data:
        return hypothesis, info

    return hypothesis
