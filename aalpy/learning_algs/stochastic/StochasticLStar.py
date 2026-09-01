# Sampling-based stochastic L* algorithm for learning MDPs and stochastic Mealy machines.
import time

from aalpy.automata import Mdp, StochasticMealyMachine
from aalpy.base import SUL, Oracle
from aalpy.learning_algs.stochastic.DifferenceChecker import AdvancedHoeffdingChecker, HoeffdingChecker, \
    ChiSquareChecker, DifferenceChecker
from aalpy.learning_algs.stochastic.SamplingBasedObservationTable import SamplingBasedObservationTable
from aalpy.learning_algs.stochastic.StochasticCexProcessing import stochastic_longest_prefix, stochastic_rs
from aalpy.learning_algs.stochastic.StochasticTeacher import StochasticTeacher
from aalpy.utils.HelperFunctions import print_learning_info, print_observation_table, get_cex_prefixes, \
    get_available_oracles_and_err_msg

from aalpy.utils.ModelChecking import stop_based_on_confidence

strategies = ['classic', 'normal', 'chi2']
cex_sampling_options = [None, 'bfs']
cex_processing_options = [None, 'longest_prefix', 'rs']
print_options = [0, 1, 2, 3]
diff_checker_options = {'classic': HoeffdingChecker(),
                        'chi2': ChiSquareChecker(),
                        'normal': AdvancedHoeffdingChecker()}
available_oracles, available_oracles_error_msg = get_available_oracles_and_err_msg()


def run_stochastic_Lstar(input_alphabet: list, sul: SUL, eq_oracle: Oracle, target_unambiguity: float = 0.99,
                         min_rounds: int = 10, max_rounds: int | None = 200, automaton_type: str = 'mdp',
                         strategy: str | DifferenceChecker = 'normal',
                         cex_processing: str | None = None, samples_cex_strategy: str | None = None,
                         stopping_range_dict: dict | str = 'strict', custom_oracle: bool = False,
                         return_data: bool = False, property_based_stopping: tuple | None = None,
                         n_c: int = 20, n_resample: int = 100, print_level: int = 2) \
        -> 'Mdp | StochasticMealyMachine | tuple[Mdp | StochasticMealyMachine, dict]':
    """
    Learning of Markov Decision Processes and Stochastic Mealy machines based on 'L*-Based Learning of Markov Decision
    Processes' and 'Active Model Learning of Stochastic Reactive Systems' by Tappler et al.

    :param list input_alphabet: Input alphabet.
    :param SUL sul: System under learning.
    :param Oracle eq_oracle: Equivalence oracle.
    :param float target_unambiguity: Target unambiguity value.
    :param int min_rounds: Minimum number of learning rounds.
    :param int | None max_rounds: If learning_rounds >= max_rounds, learning will stop.
    :param str automaton_type: Either 'mdp' or 'smm'.
    :param str | DifferenceChecker strategy: Either one of ['classic', 'normal', 'chi2'] or an object implementing
        DifferenceChecker class. Classic strategy is the one presented in the seed paper, 'normal' is the updated
        version and chi2 is based on chi squared.
    :param str | None cex_processing: Cex processing strategy, None, 'longest_prefix' or 'rs' (rs is experimental).
    :param str | None samples_cex_strategy: Strategy for finding counterexamples in the trace tree. None, 'bfs' or
        "random:<#traces to check:int>:<stop probability for single trace in [0,1)>" eg. random:200:0.2.
    :param dict | str stopping_range_dict: Values in form of a dictionary, or 'strict', 'relaxed' to use predefined
        stopping criteria. Custom values: Dictionary where keys encode the last n unambiguity values which need to
        be in range of its value in order to perform early stopping. Eg. {5: 0.001, 10: 0.01} would stop if last 5
        hypothesis had unambiguity values when max(last_5_vals) - (last_5_vals) <= 0.001.
    :param bool custom_oracle: If True, warning about oracle type will be removed and custom oracle can be used.
    :param bool return_data: If True, map containing all information like number of queries... will be returned.
    :param tuple | None property_based_stopping: A tuple containing (path to the properties file, correct values of
        each property, allowed error for each property. Recommended one is 0.02 (2%)).
    :param int n_c: Cutoff for a cell to be considered complete, only used with 'classic' strategy.
    :param int n_resample: Resampling size, only used with 'classic' strategy.
    :param int print_level: 0 - None, 1 - just results, 2 - current round and hypothesis size, 3 - educational/debug.
    :return Mdp | StochasticMealyMachine | tuple[Mdp | StochasticMealyMachine, dict]: Learned MDP/SMM, or a
        (hypothesis, info) pair if return_data is True.
    """

    assert samples_cex_strategy in cex_sampling_options or samples_cex_strategy.startswith('random')
    assert cex_processing in cex_processing_options
    assert automaton_type in {'mdp', 'smm'}
    if not isinstance(stopping_range_dict, dict):
        assert stopping_range_dict in {'strict', 'relaxed'}
    if property_based_stopping:
        assert len(property_based_stopping) == 3

    if strategy in diff_checker_options:
        compatibility_checker = diff_checker_options[strategy]
    else:
        assert isinstance(strategy, DifferenceChecker)
        compatibility_checker = strategy

    if not custom_oracle and type(eq_oracle) not in available_oracles:
        raise SystemExit(available_oracles_error_msg)

    if stopping_range_dict == 'strict':
        stopping_range_dict = {12: 0.001, 18: 0.002, 25: 0.005, 30: 0.01, 35: 0.02}
    elif stopping_range_dict == 'relaxed':
        stopping_range_dict = {7: 0.001, 12: 0.003, 17: 0.005, 22: 0.01, 28: 0.02}

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
        chaos_cex_present = observation_table.chaos_counterexample(hypothesis)

        if not chaos_cex_present:
            if automaton_type == 'mdp':
                hypothesis.states.remove(next(state for state in hypothesis.states if state.output == 'chaos'))
            else:
                hypothesis.states.remove(next(state for state in hypothesis.states if state.state_id == 'chaos'))

        if print_level > 1:
            print(f'Hypothesis: {learning_rounds}: {len(hypothesis.states)} states.')

        if print_level == 3:
            print_observation_table(observation_table, 'stochastic')

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
                    suffixes = [stochastic_longest_prefix(cex, prefixes)[-1]]
                elif cex_processing == 'rs':
                    suffixes = stochastic_rs(sul, cex, hypothesis)
                for suf in suffixes:
                    if suf not in observation_table.E:
                        observation_table.E.append(suf)
                        break

        # Ask queries for non-completed cells and update the observation table
        refined = observation_table.refine_not_completed_cells(n_resample)
        observation_table.update_obs_table_with_freq_obs()

        if property_based_stopping and learning_rounds >= min_rounds:
            # stop based on maximum allowed error
            if stop_based_on_confidence(hypothesis, property_based_stopping, print_level):
                break
        else:
            # stop based on number of unambiguous rows
            stop_based_on_unambiguity = observation_table.stop(learning_rounds, chaos_cex_present, cex,
                                                               stopping_range_dict,
                                                               target_unambiguity=target_unambiguity,
                                                               min_rounds=min_rounds, max_rounds=max_rounds,
                                                               print_unambiguity=print_level > 1)
            if stop_based_on_unambiguity:
                break

        if not refined:
            break

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

