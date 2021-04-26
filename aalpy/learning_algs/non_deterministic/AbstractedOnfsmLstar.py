import time

from aalpy.base import SUL, Oracle
from aalpy.learning_algs.non_deterministic.AbstractedOnfsmObservationTable import AbstractedNonDetObservationTable
from aalpy.learning_algs.non_deterministic.TraceTree import SULWrapper
from aalpy.utils.HelperFunctions import print_learning_info, print_observation_table

print_options = [0, 1, 2, 3]


def run_abstracted_ONFSM_Lstar(alphabet: list, sul: SUL, eq_oracle: Oracle, abstraction_mapping: dict, n_sampling=100,
                               max_learning_rounds=None, return_data=False, print_level=2):
    """
    Based on ''Learning Abstracted Non-deterministic Finite State Machines'' from Pferscher and Aichernig.
    The algorithm learns an abstracted onfsm of a non-deterministic system. For the additional abstraction,
    equivalence classes for outputs are used.
    Learning ONFSM relies on all-weather assumption. If this assumption is not satisfied by sampling,
    learning might not converge to the minimal model and runtime could increase substantially.
    Note that this is the inherent flaw of the all-weather assumption. (All outputs will be seen)
    AALpy v.2.0 will try to solve that problem with a novel approach.

    Args:

        alphabet: input alphabet

        sul: system under learning

        eq_oracle: equivalence oracle

        abstraction_mapping: dictionary containing mappings from abstracted to concrete values (equivalence classes)

        n_sampling: number of times that membership/input queries will be asked for each cell in the observation
            (Default value = 100)

        max_learning_rounds: if max_learning_rounds is reached, learning will stop (Default value = None)

        return_data: if True, map containing all information like number of queries... will be returned
            (Default value = False)

        print_level: 0 - None, 1 - just results, 2 - current round and hypothesis size, 3 - educational/debug
            (Default value = 2)

    Returns:
        learned abstracted ONFSM

    """
    start_time = time.time()
    eq_query_time = 0
    learning_rounds = 0
    hypothesis = None

    sul = SULWrapper(sul)
    eq_oracle.sul = sul

    abstracted_observation_table = AbstractedNonDetObservationTable(alphabet, sul, abstraction_mapping, n_sampling)

    # We fist query the initial row. Then based on output in its cells, we generate new rows in S.A,
    # and then we perform membership/input queries for them.
    abstracted_observation_table.update_obs_table()
    new_rows = abstracted_observation_table.update_extended_S(abstracted_observation_table.S[0])
    abstracted_observation_table.update_obs_table(s_set=new_rows)

    while True:
        learning_rounds += 1
        if max_learning_rounds and learning_rounds - 1 == max_learning_rounds:
            break

        closed_complete_consistent = False
        while not closed_complete_consistent:
            closed_complete_consistent = True

            row_to_close = abstracted_observation_table.get_row_to_close()
            while row_to_close is not None:
                # First we add new rows to the S.A. They are added based on the values in the cells of the
                # rows that is to be closed. Once those rows are created, they are populated and closedness is checked
                # once again.
                closed_complete_consistent = False
                extended_rows = abstracted_observation_table.update_extended_S(row_to_close)
                abstracted_observation_table.update_obs_table(s_set=extended_rows)
                row_to_close = abstracted_observation_table.get_row_to_close()

            row_to_complete = abstracted_observation_table.get_row_to_complete()
            while row_to_complete is not None:
                closed_complete_consistent = False
                extended_rows = abstracted_observation_table.complete_extended_S(row_to_complete)
                abstracted_observation_table.update_obs_table(s_set=extended_rows)
                row_to_complete = abstracted_observation_table.get_row_to_complete()

            e_column_for_consistency = abstracted_observation_table.get_row_to_make_consistent()
            while e_column_for_consistency is not None:
                closed_complete_consistent = False
                extended_col = abstracted_observation_table.update_E(e_column_for_consistency)
                abstracted_observation_table.update_obs_table(e_set=extended_col)
                e_column_for_consistency = abstracted_observation_table.get_row_to_make_consistent()

        hypothesis = abstracted_observation_table.gen_hypothesis()

        if print_level == 3:
            print('Observation Table')
            print_observation_table(abstracted_observation_table.observation_table, 'non-det')

            print('Abstracted Observation Table')
            print_observation_table(abstracted_observation_table, 'non-det')

        if print_level > 1:
            print(f'Hypothesis {learning_rounds} has {len(hypothesis.states)} states.')

        # Find counterexample
        eq_query_start = time.time()
        cex = eq_oracle.find_cex(hypothesis)
        eq_query_time += time.time() - eq_query_start

        if cex is None:
            break

        if print_level >= 2:
            print('Counterexample', cex)

        # Process counterexample -> add cex to S.A or E
        abstracted_observation_table.cex_processing(cex, hypothesis)

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
