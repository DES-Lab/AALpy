import time

from aalpy.automata import Dfa, DfaState, MealyState, MealyMachine, MooreState, MooreMachine
from aalpy.base import Oracle, SUL
from aalpy.utils.HelperFunctions import print_learning_info
from .ClassificationTree import ClassificationTree
from .CounterExampleProcessing import counterexample_successfully_processed
from ...base.SUL import CacheSUL

print_options = [0, 1, 2, 3]
counterexample_processing_strategy = [None, 'rs']
automaton_class = {'dfa': Dfa, 'mealy': MealyMachine, 'moore': MooreMachine}


def run_KV(alphabet: list, sul: SUL, eq_oracle: Oracle, automaton_type, cex_processing='rs',
           max_learning_rounds=None, cache_and_non_det_check=True, return_data=False, print_level=2):
    """
    Executes the KV algorithm.

    Args:

        alphabet: input alphabet

        sul: system under learning

        eq_oracle: equivalence oracle

        automaton_type: type of automaton to be learned. One of 'dfa', 'mealy', 'moore'

        cex_processing: None for no counterexample processing, or 'rs' for Rivest & Schapire counterexample processing

        max_learning_rounds: number of learning rounds after which learning will terminate (Default value = None)

        cache_and_non_det_check: Use caching and non-determinism checks (Default value = True)

        return_data: if True, a map containing all information(runtime/#queries/#steps) will be returned
            (Default value = False)

        print_level: 0 - None, 1 - just results, 2 - current round and hypothesis size, 3 - educational/debug
            (Default value = 2)


    Returns:

        automaton of type automaton_type (dict containing all information about learning if 'return_data' is True)

    """

    assert print_level in print_options
    assert cex_processing in counterexample_processing_strategy
    assert automaton_type in [*automaton_class]

    start_time = time.time()
    eq_query_time = 0
    learning_rounds = 0

    if cache_and_non_det_check:
        # Wrap the sul in the CacheSUL, so that all steps/queries are cached
        sul = CacheSUL(sul)
        eq_oracle.sul = sul

    if automaton_type != 'mealy':
        # Do a membership query on the empty string to determine whether
        # the start state of the SUL is accepting or rejecting
        empty_string_mq = sul.query(tuple())[-1]

        # Construct a hypothesis automaton that consists simply of this
        # single (accepting or rejecting) state with self-loops for
        # all transitions.
        if automaton_type == 'dfa':
            initial_state = DfaState(state_id='s0', is_accepting=empty_string_mq)
        else:
            initial_state = MooreState(state_id='s0', output=empty_string_mq)
    else:
        initial_state = MealyState(state_id='s0')

    for a in alphabet:
        initial_state.transitions[a] = initial_state
        if automaton_type == 'mealy':
            initial_state.output_fun[a] = sul.query((a,))[-1]

    hypothesis = automaton_class[automaton_type](initial_state, [initial_state])

    # Perform an equivalence query on this automaton
    eq_query_start = time.time()
    cex = eq_oracle.find_cex(hypothesis)

    eq_query_time += time.time() - eq_query_start
    already_found = False
    if cex is None:
        already_found = True
    else:
        cex = tuple(cex)

    # initialise the classification tree to have a root
    # labeled with the empty word as the distinguishing string
    # and two leaves labeled with access strings cex and empty word
    classification_tree = ClassificationTree(alphabet=alphabet, sul=sul, automaton_type=automaton_type, cex=cex)

    while True and not already_found:
        learning_rounds += 1
        if max_learning_rounds and learning_rounds - 1 == max_learning_rounds:
            break

        hypothesis = classification_tree.gen_hypothesis()

        if print_level == 2:
            print(f'\rHypothesis {learning_rounds}: {hypothesis.size} states.', end="")

        if print_level == 3:
            # would be nice to have an option to print classification tree
            print(f'Hypothesis {learning_rounds}: {hypothesis.size} states.')

        if counterexample_successfully_processed(sul, cex, hypothesis):
            # Perform an equivalence query on this automaton
            eq_query_start = time.time()
            cex = eq_oracle.find_cex(hypothesis)
            eq_query_time += time.time() - eq_query_start

            if cex is None:
                break
            else:
                cex = tuple(cex)

            if print_level == 3:
                print('Counterexample', cex)

        if cex_processing == 'rs':
            classification_tree.update_rs(cex, hypothesis)
        else:
            classification_tree.update(cex, hypothesis)

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
        'classification_tree': classification_tree,
        'cache_saved': sul.num_cached_queries,
    }

    if print_level > 0:
        if print_level == 2:
            print("")
        print_learning_info(info)

    if return_data:
        return hypothesis, info

    return hypothesis

