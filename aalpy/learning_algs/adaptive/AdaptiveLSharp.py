import time

from aalpy.base import Oracle, SUL
from aalpy.utils.HelperFunctions import print_learning_info
from .AdaptiveObservationTree import AdaptiveObservationTree
from ...base.SUL import CacheSUL


def run_adaptive_Lsharp(alphabet: list, sul: SUL, references: list, eq_oracle: Oracle, automaton_type,
                        extension_rule=None, separation_rule="SepSeq",
                        rebuilding=True, state_matching="Approximate",
                        samples=None, max_learning_rounds=None,
                        cache_and_non_det_check=True, return_data=False, print_level=2):
    """
    Based on ''State Matching and Multiple References in Adaptive Active Automata Learning'' from Kruger, Junges and Rot.
    The algorithm learns a Mealy machine using a set of references. These references are used by two procedures 
    1) Rebuilding which kickstarts the learning process using the references and 2) State Matching which matches the
    basis states and references states to find new basis states faster.

    Args:

        alphabet: input alphabet

        sul: system under learning

        references: a list of references

        eq_oracle: equivalence oracle

        automaton_type: type of automaton to be learned. Either 'dfa', 'mealy' or 'moore'

        extension_rule: strategy used during the extension rule. Options: "Nothing" (default), "SepSeq" and "ADS".

        separation_rule: strategy used during the extension rule. Options: "SepSeq" (default) and "ADS".

        rebuilding: procedure that poses output queries to rebuild the observation tree based on prefixes and separating sequences from the reference(s).
        Only executes at the start of adaptive L#. default value: True. 

        state_matching: if not None, the learner maintains a matching relation between basis states (in the observation tree) and reference model states.
        This matching relation is used in three rules added on top of L# to either identify a frontier state faster or isolate it when the matching indicates
        that the frontier state corresponds to a reference model state not yet present in the basis. default value: "Approximate". 
        - Two states match according to "total matching" if all output over the defined and shared alphabet are exactly the same.
        - Two states match according to "approximate matching" if they have the highest ratio of equivalent outputs to defined outputs over the shared alphabet. 
        - None can be used if only the rebuilding procedure is needed.

        samples: input output traces provided to the learning algorithm. They are added to cache and could reduce
        total interaction with the system. Syntax: list of [(input_sequence, output_sequence)] or None

        max_learning_rounds: number of learning rounds after which learning will terminate (Default value = None)

        cache_and_non_det_check: Use caching and non-determinism checks (Default value = True)

        return_data: if True, a map containing all information(runtime/#queries/#steps) will be returned
            (Default value = False)

        print_level: 0 - None, 1 - just results, 2 - current round and hypothesis size, 3 - educational/debug
            (Default value = 2)

    Returns:

        automaton of type automaton_type (dict containing all information about learning if 'return_data' is True)

    """
    assert extension_rule in {None, "SepSeq", "ADS"}
    assert separation_rule in {"SepSeq", "ADS"}

    assert state_matching in {None, "Total", "Approximate"}
    assert references is not None, 'List of reference models is empty. Use L*, KV, L#, or provide models.'

    if not rebuilding and not state_matching:
        raise Exception(f"Use L# instead of Adaptive L# if rebuilding is set to False and state matching to None.")

    if cache_and_non_det_check or samples is not None:
        # Wrap the sul in the CacheSUL, so that all steps/queries are cached
        sul = CacheSUL(sul)
        eq_oracle.sul = sul

        if samples:
            for input_seq, output_seq in samples:
                sul.cache.add_to_cache(input_seq, output_seq)

    ob_tree = AdaptiveObservationTree(alphabet, sul, references, automaton_type,
                                      extension_rule, separation_rule,
                                      rebuilding, state_matching)
    start_time = time.time()
    eq_query_time = 0
    learning_rounds = 0
    hypothesis = None

    while True:
        learning_rounds += 1
        if max_learning_rounds and learning_rounds == max_learning_rounds:
            break

        # Building the hypothesis
        hypothesis = ob_tree.build_hypothesis()

        if print_level > 1:
            print(f'Hypothesis {learning_rounds}: {hypothesis.size} states.')
        if print_level == 3:
            print(hypothesis)
            if ob_tree.state_matching:
                ob_tree.state_matcher.print_match_table(ob_tree)

        # Pose Equivalence Query
        eq_query_start = time.time()
        cex = eq_oracle.find_cex(hypothesis)
        eq_query_time += time.time() - eq_query_start

        if print_level > 2:
            print(f'Counterexample: {cex}')

        if cex is None:
            break

        # Process the counterexample and start a new learning round
        cex_outputs = sul.query(cex)
        ob_tree.process_counter_example(hypothesis, cex, cex_outputs)

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
        'cache_saved': sul.num_cached_queries,
    }
    if rebuilding:
        info['rebuild_states'] = ob_tree.rebuild_states
    if state_matching != "None":
        info['matching_type'] = state_matching
        info['matching_states'] = ob_tree.matching_states

    if print_level > 0:
        print_learning_info(info)

    if return_data:
        return hypothesis, info

    return hypothesis
